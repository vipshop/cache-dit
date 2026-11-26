# Tensor Parallelism：planner 契约与切分规则

路径相对 `src/cache_dit/distributed/`。

## 1. planner 契约

```python
@TensorParallelismPlannerRegister.register("FluxTransformer")   # 类名前缀
class FluxTensorParallelismPlanner(TensorParallelismPlanner):

  def _apply(self, transformer, parallelism_config, **kwargs):
    tp_mesh = self.mesh(parallelism_config=parallelism_config)   # 基类提供
    transformer, layer_plans = self.parallelize_transformer(transformer, tp_mesh)
    return transformer, layer_plans
```

- 基类 `apply()` 调 `_apply()` 后统一 `record_plans()`：按 `hash(frozenset(layer_plan.keys()))` 去重，把 `RowwiseParallel` / `ColwiseParallel` 的模块名分别写进 `transformer._rowwise_layers` / `_colwise_layers`。
- `mesh()`：hybrid 时复用 `parallelism_config._tp_mesh`；否则 `init_device_mesh([tp_size])`。
- 注册表按「类名前缀 + 先注册先命中」匹配；`planners.py::_activate_tp_planners` 的 import 顺序就是优先级（`QwenImage21` 先于 `QwenImage`、`WanS2V` → `WanAnimate2` → `Wan`）。
- TP 只接受 `ModelMixin`（`_parallelize_transformer_tensor` 里的 assert）；额外依赖 `einops`（`cache-dit[parallelism]`），缺失时报 "parallelism functionality requires the 'parallelism' extra dependencies"。

## 2. 标准切分：FLUX 范例

```python
for _, block in transformer.transformer_blocks.named_children():
  shard_div_attr(block.attn, "heads", tp_mesh.size())
  layer_plan = {
    "attn.to_q": ColwiseParallel(),
    "attn.to_k": ColwiseParallel(),
    "attn.to_v": ColwiseParallel(),
    "attn.to_out.0": RowwiseParallel(),
    "ff.net.0.proj": ColwiseParallel(),
    "ff.net.2": RowwiseParallel(),
    ...
  }
  if getattr(block.norm1, "linear", None) is not None:
    layer_plan["norm1.linear"] = ColwiseParallel(output_layouts=Replicate())
  parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
```

规律：

1. **Q/K/V/FFN-in 列切，out/FFN-out 行切**，一列一行配对保证残差形状不变。
2. **head 数必须跟着切**：`shard_div_attr` 在不能整除时抛错并列出合法 tp_size（`heads=12` 时合法值 2/3/4/6/12），信息比 PyTorch 的 reshape 报错可读得多。
3. **adaLN / modulation 的 `linear`**：`ColwiseParallel(output_layouts=Replicate())`，因为它输出要 broadcast 到整条隐状态，行切会拿到残缺的 shift/scale。
4. **融合权重需要手工重排**：FLUX 的 `FluxSingleTransformerBlock.proj_out` 把 out 与 down 两个投影塞在一个矩阵里，必须先 `rearrange` 成 `(G D) C` 再 `copy_`，否则行切会把两种权重混在一起（`flux.py::rearrange_proj_out_weight`）。
5. **纯复制的层**（embedding、`proj_out` 前的小线性、time embedding）不进 plan，由 PyTorch 保持 replicated。

## 3. 三个必须手工处理的算子

| 情况 | 处理 |
|---|---|
| QK RMSNorm 覆盖完整投影宽度 | 用 `distributed/utils.py::_DistributedRMSNorm.from_rmsnorm(tp_mesh, norm)` 替换：每个 rank 只有部分特征，前向里对 `mean(x²)` 做 `AVG` all-reduce 还原全局统计量，weight 沿 dim0 切 |
| `torch.nn.RMSNorm` 与 diffusers `RMSNorm` 属性名不同 | `from_rmsnorm` 同时支持 `normalized_shape` 与 `dim`；bias 版直接 assert 不支持 |
| 需要 shard 的整型属性（heads、kv_heads、inner_dim） | `shard_div_attr(obj, attr, tp_size)`，不要手写 `// tp_size` |

`_DistributedRMSNorm.forward` 会断言 `hidden_states.shape[-1] == weight.shape[0]`，所以「忘记把 norm 换成分布式版本」通常在这里炸，而不是数值慢慢漂。

## 4. TP × 量化

- `record_plans` 写下 `_rowwise_layers` / `_colwise_layers` 就是为了让量化按布局选 kernel；少了这两个属性，量化会按未切分的形状假设走，容易在 `torch._scaled_mm` 处报 stride 错。
- 代码里有一条明确注释：`RowwiseParallel` 会把 linear 权重布局变成非连续（`_dispatch_get_local_results_slow_path`），这是 `float8` 路径必须先处理的已知问题。
- `float8_per_row` + TP 在带 bias 的 Linear 上会被 `_scaled_mm` 的 "DTensor + bias NOT supported" 过滤**整层跳过**（exit 0，日志里是 `Quantized Linear Layers: 0 / Skipped N`）；`float8_weight_only` 则会 `NotImplementedError: triton.quantize_fp8_row 无 sharding strategy`。要真实量化数字就单卡不 TP。

## 5. hybrid（CP + TP）

- 顺序固定：`_parallelize_transformer_hybrid` 先 `_parallelize_transformer_cp` 再 `_parallelize_transformer_tp`。
- mesh `[ring, ulysses, tp]`，tp 是最后一维（1D TP 要求），所以序列维先切、特征维后切。
- 至少 4 卡（`hybrid_enabled` 断言）；`usp_tp` 硬要求 8 卡。
- 两条整除约束同时成立才能跑：`heads % tp_size == 0`，并且 `(heads // tp_size) % ulysses_size == 0`。所以 4 卡 `ulysses_tp`（U1×T2）只是纯 TP；想 CP+TP 又只有 4 卡时只能 `tp_ulysses`（U2×T2）。
- `_tp_mesh` 在 hybrid 下已由 `ParallelismConfig` 建好，planner 的 `mesh()` 直接复用，不要重复 init。

## 6. 排障

| 症状 | 原因 |
|---|---|
| `[TP] ... must be divisible by tp_size. Valid tp_size (>1): ...` | `heads` 或 `inner_dim` 不能整除；换 tp_size 或换并行维度 |
| `No planner registered under name: XxxTransformer` | 该模型没有 TP planner（只有 CP），见 `references/planner-inventory.md` |
| `transformer must be an instance of diffusers' ModelMixin` | 自持/第三方模型走不了 TP，只能 CP |
| 输出全黑/全零 | 常见于 adaLN 行切、或 fused 权重没重排；先看 `_colwise_layers` 是否包含 `norm*.linear` |
| 量化后精度暴跌但层数正常 | 检查是否 TP 与 `float8_per_row` 同时开（bias 被静默跳过导致"没量化却以为量化了"） |
