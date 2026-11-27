# Context Parallelism：plan、hook、通信

路径相对 `src/cache_dit/distributed/`。

## 1. CP plan DSL

一个 plan 是 `dict[module_id, subplan]`：

```python
from ...distributed.parallel import (
  _ContextParallelInput, _ContextParallelModelPlan, _ContextParallelOutput,
)

_cp_plan: _ContextParallelModelPlan = {
  "": {
    "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
    "img_ids": _ContextParallelInput(split_dim=0, expected_dims=2),
  },
  "rope": {
    0: _ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
    1: _ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
  },
  "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
}
```

规则：

| 形态 | 语义 |
|---|---|
| `""` | 根模块（transformer 本身）的 forward 入参 |
| `"blocks.*"` | 通配符只能出现一次，且父模块必须是 `ModuleList` |
| 值为 `dict` | split 计划：键是 forward 参数名（或输出槽位索引），值是 `_ContextParallelInput` |
| 值为 `_ContextParallelOutput` | gather 计划 |
| 值为 `(out_a, out_b)` | **一个** gather hook，处理 tuple 返回的多个槽位；`None` 表示该槽不处理 |
| 值为 `[dict_plan, output_plan]` | **两个** hook（subplan）：先 split 输入再 gather 输出，名字带 `---0` / `---1` |
| `split_output=True` | 不是切输入，而是切**该模块的返回值**（例如 `rope` 返回 `(cos, sin)`） |
| `expected_dims` 不匹配 | 只 `warning_once` 并跳过 split，不报错（所以 shape 写错会静默不切） |

`_iter_cp_module_plans` 区分「subplan 容器」和「tuple 输出计划」的判据：list/tuple 里**全部**是 dict 或 output plan 才当 subplan；纯 output tuple 视为一个 gather hook。

## 2. split / gather 的真实行为

`_ContextParallelSplitHook.pre_forward`：

- 按 `inspect.signature(cls.forward)` 把参数名映射到位置/关键字；`kwargs` 优先。
- 计划里的参数这次调用没提供（例如 t2i 不传 `image_cond_hidden_states`）→ 只 warning 并跳过。**陷阱**：如果模型内部从已经切过的本地张量重新推导同一个输入，跳过就等于没切。
- 张量按 `ulysses_anything` 二选一：普通模式 `_EquipartitionSharder`（要求整除，`chunk`）；UAA 模式 `_PartitionAnythingSharder`（`tensor_split`，允许不整除）。
- list/tuple 输入逐个套对应计划；长度必须与计划长度一致。

`_ContextParallelGatherHook.post_forward`：

- 输出槽位数必须等于计划长度；计划为 `None` 的槽原样透传。
- 计划要求 gather 但该槽返回 `None` → 直接 raise（"A plan that skips an optional stream must carry `None` in that slot as well"）。
- gather 用 `_AllGatherFunction`（autograd-aware，反向按 rank 取 chunk）；UAA 用 `_AllGatherAnythingFunction`，先 `_gather_size()` 收各 rank 的实际长度再 `all_gather` + `cat`。`_gather_size` 有真实通信，已用 `@torch.compiler.disable` 排除在 dynamo 之外。

## 3. 挂载入口 `_enable_context_parallelism`

```text
assert torch.distributed.is_initialized()
config = _normalize_parallel_config(config)          # 接受 diffusers 风格配置
validate_context_parallel_attention_backend(model)   # 后端必须 CP-capable，否则 raise
config._mesh is None → init_device_mesh((ring, ulysses)) → config.setup(rank, world, device, mesh)
model._cp_config = model._parallel_config = config
每个 attention processor 也写 _cp_config / _parallel_config
cp_plan = 参数 或 model._cp_plan，两者都空 → ValueError
_apply_context_parallel(model, config, cp_plan)      # 逐模块注册 hook
```

`_ContextParallelConfig.setup()` 会校验 `ring_degree * ulysses_degree <= world_size`，并从 mesh 派生 `_ring_mesh` / `_ulysses_mesh` / 两个 local rank。hybrid 下由 `ParallelismConfig` 预先 `setup(mesh=_cp_mesh)`，runtime 不会重建。

## 4. 三种 attention 拓扑

### 4.1 Ulysses（`ulysses.py`）

```text
Q/K/V (B, S_LOCAL, H_GLOBAL, D)
  → all-to-all: (B, S_GLOBAL, H_LOCAL, D)     # 切 head，拼序列
  → forward_op 在整条序列上算
  → all-to-all back: (B, S_LOCAL, H_GLOBAL, D)
```

- 默认分别 `send_q/send_k/send_v`；`_FusedAll2AllComm.fused_enabled` 时改一次 `send_qkv`（融合只在「量化 + UAA」同时成立且未设 `CACHE_DIT_DISABLE_ULYSSES_FUSED_QKV=1` 时启用；BF16 下 FLUX 的 Q-first 调度重叠更好，所以不融合）。
- `cp_gqa_strategy == "replicate_kv_sequence"`：Q 走 a2a，KV 用 `gather_replicated_kv_for_local_q` 复制成 MHA 布局，并强制 `enable_gqa=False`。
- `cp_gqa_strategy == "group_aligned_flash_varlen"`：走 `send_group_aligned_gqa_qkv/o`，禁止 `return_lse`。
- Ulysses 自身不需要 LSE；只有 USP/Ring 需要。

### 4.2 Ring（`ring.py`）

```text
for step in range(world_size):
  如果还有下一跳：P2P 把当前 K/V 发给 next、从 prev 收下一对
  out, lse = forward_op(..., return_lse=True, _save_ctx=False)
  与 prev_out/prev_lse 做 online-softmax 合并
```

- **强制 `return_lse=True`**：backend 不支持 LSE 就不能用 Ring（`sol_attn` / `sage3` / `flash_varlen` / cudnn 路径都会在 forward op 里 raise）。
- 合并默认走 `fused_merge_attn_states`（`_C_ops`）；缺失时用 `sigmoid/logsigmoid` 的纯 torch 公式，`convert_to_fp32=True` 时先升 fp32。
- KV 收发用双缓冲的两个 slot（world_size>2 才建第二个）；`_RingP2PComm` 只支持 `p2p`，`rotate_method != "p2p"` 在 `_ContextParallelConfig.__post_init__` 就被拒。
- forward op 里的 `attn_mask` 永远是非空即报错（Ring 不做 mask 语义）。

### 4.3 USP（`usp.py`）

Ulysses 的 all-to-all 收 Q/K/V → 调 `RingAttention` → all-to-all 把输出换回来。LSE 同样走 a2a（`send_lse`），`lse` 形状 `[B,S,H,1]`。是 `--parallel usp` / `usp_tp` 的实现。

## 5. 通信原语（`communicate/primitives.py`）

### 5.1 选择矩阵

`_All2AllComm.__init__` 一次性选好 5 个实现，后续调用不再判断：

| 维度 | 取值 | 实现 |
|---|---|---|
| 序列/head 是否均匀 | `ulysses_anything=False` | `_all_to_all_single_qkv_async` / `_all_to_all_single_o_async` |
| | `ulysses_anything=True` | `_all_to_all_single_any_*_async`（外层 `_maybe_pad_qkv_head` 补齐） |
| 是否量化 | `bits ∈ {4,6,8}` | `_all_to_all_single_*_fN_async`（打包 + 量化 kernel） |
| | `bits=0` | BF16 路径 |
| K 单独控制 | `quantize_k=False` | K 用 `bits=0` 的实现，Q/V/O 仍量化 |
| O 的 LSE | 始终 `bits=0` | `send_lse` 不量化 |

元数据只有 `NUM_QO_HEAD` 与 `Q_S_LOCAL`（`_init_comm_metadata`），一旦 init 就复用；`send_q/k/v` 允许隐式 init，`send_o` 不允许（必须先有 metadata）。

### 5.2 低精度通信

```python
ParallelismConfig(
  ulysses_size=4,
  quant_communicate_kwargs={"bits": 8, "quant_type": "per_group_16", "quantize_k": True},
)
```

- `bits` 只能 4/6/8，默认 8；`quant_type` 只能 `per_token` / `per_group_16` / `per_group_32`，默认 `per_group_16`；**FP4 只允许 `per_group_16`**。
- 必须 `ulysses_size > 1`，否则 `__post_init__` 直接 raise。
- kernel 是 `_C_ops` 里的 `ulysses_fN_*_quant/dequant`；没编译时报 "Build cache_dit._C_ops with CACHE_DIT_BUILD_CUDA=1, or disable quant_communicate_kwargs"。
- 有专门的 `_FusedAll2AllComm` 融合 QKV 通道，但只在「量化 + UAA」时启用（见 §4.1）。

### 5.3 Ring P2P

`_RingP2PComm` 用 `batch_isend_irecv` 组一对 `isend`/`irecv`；`send_recv_kv` 一次提交 K 与 V，`batch_send_recv_kv` 把 K/V `cat` 成一条消息（返回时只做 view，**不要 contiguous**，否则会拷空张量导致数值错）。

## 6. Async Ulysses（`transformers/async_ulysses/`）

- `AsyncUlyssesRegistry` 保存 `(类, 方法名) → wrapper`，由 planner 在 `_apply` 里显式 `AsyncUlyssesRegistry.apply(transformer)` 触发（例如 FLUX 的 planner 在 `ulysses_async` 时调用）。
- 它替换的是**模型类的方法**（attention 的 qkv 投影与 attn 调用），用来把通信和计算重叠；只支持 diffusers 模型，非 diffusers 类直接 `ValueError`。
- 已注册：flux / flux2 / qwen_image / zimage / longcat_image / ovis_image / controlnets（含 zimage_controlnet）。
- 与 `_All2AllComm` 的关系：wrapper 内部仍调 `send_q/k/v`，只是把 wait 挪到计算之后。

## 7. CP × cache 的三条交互

1. **`Bn_compute_blocks` 下限**：planner 若在最后一块 gather（MOVA / Kandinsky6 / StableAudio3 设了 `_cp_requires_min_bn_blocks = 1`），`DBCacheConfig(Bn_compute_blocks=0)` 会直接 raise。原因：cache 命中时只执行 `[Fn, N-Bn)` 里的尾部块，`Bn=0` 意味着最后一块永不执行，gather 不发生。
2. **split/gather 只能有一个 owner**：hook 是模块级常驻，`mock.patch` 只换 block 列表不会摘 hook。既手工切又挂 hook = 双切；既手工 all-gather 又挂 gather hook = 双 gather。
3. **compile**：UAA 依赖 `capture_scalar_outputs`（torch≥2.10 在 `compile/utils.py` 自动开）；`_all_gather_anything` 被 `@torch.compiler.disable` 包住，因为它的 `_gather_size` 是真实通信、dynamo 无法 fake。

## 8. 排障表

| 症状 | 先查 |
|---|---|
| `Context parallelism requires a cache-dit registered attention backend` | `--attn-params type=X` 是不是 diffusers-only 名字（`_native_cudnn`）；换成 `native`/`flash`/`sage`/`ffpa*` |
| `... which does not support context parallelism` | 该 backend 的 `supports_context_parallel=False`（如 `_flash_3` 之外的假想 backend、或未注册的 proxy） |
| 形状对但数值错 | 双切/双 gather；processor 自己做了 a2a 又传了 `parallel_config` |
| `Tensor size along dimension to be sharded must be divisible by mesh size` | 非 UAA 下序列/head 不整除；改 `--uaa` 或换 mesh |
| `Cannot create a submesh from a submesh` | torch < 2.10 的 hybrid mesh |
| `_All2AllComm metadata is not initialized` | 直接调了 `send_o` 而没有先 `send_q/k/v` 或 `init_meta` |
| Ring 报 `return_lse is not supported` | backend 无 LSE，或 `cp_gqa_strategy` 用了 group-aligned |
| `Bn_compute_blocks >= 1` 报错 | 见 §7.1，设 `Bn_compute_blocks=1` |
