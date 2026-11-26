# 架构：配置 → 后端 → mesh → hook

所有路径相对 `src/cache_dit/distributed/`。

## 1. 文件职责

| 文件 | 行数 | 职责 | 关键符号 |
|---|---:|---|---|
| `backend.py` | 92 | 后端枚举与可用性 | `ParallelismBackend.AUTO / CACHE_DIT / NATIVE_PYTORCH / NATIVE_HYBRID`、`canonicalize`、`from_str`、`_check_diffusers_cp_support` |
| `config.py` | 516 | 用户入口配置 + 派生 mesh 状态 | `ParallelismConfig`、`__post_init__`、`_maybe_init_hybrid_meshes`、`_get_world_size`、`strify` |
| `dispatch.py` | 137 | 顶层编排 | `enable_parallelism`、`remove_parallelism_stats` |
| `utils.py` | 148 | TP 小工具 | `shard_div_attr`、`_DistributedRMSNorm` |
| `quant_utils.py` | 55 | 通信量化参数归一化 | `normalize_quant_communicate_kwargs`、`quant_communicate_group_size` |
| `topology.py` | 121 | NUMA 诊断 | `_parse_topology`、`warn_if_cross_numa` |
| `parallel/config.py` | 201 | 内部 CP 配置与 plan 类型 | `_ContextParallelConfig`、`_ContextParallelInput`、`_ContextParallelOutput`、`_ContextParallelModelPlan` |
| `parallel/runtime.py` | 98 | CP 挂载入口 | `_enable_context_parallelism` |
| `parallel/context_parallel.py` | 490 | plan → hook、split/gather 实现 | `_apply_context_parallel`、`_ContextParallelSplitHook`、`_ContextParallelGatherHook`、`_EquipartitionSharder`、`_PartitionAnythingSharder` |
| `parallel/hooks.py` | 169 | 前/后向 hook 栈 | `ModelHook`、`HookRegistry` |
| `parallel/ring.py` | 121 | Ring attention | `RingAttention` |
| `parallel/ulysses.py` | 120 | Ulysses attention | `UlyssesAttention` |
| `parallel/usp.py` | 83 | USP = Ulysses + Ring | `USPAttention` |
| `parallel/diffusers_bridge.py` | 137 | 配置归一化 + 后端校验 | `_normalize_parallel_config`、`get_diffusers_attention_classes`、`validate_context_parallel_attention_backend` |
| `communicate/primitives.py` | 1418 | 通信原语 | `_All2AllComm`、`_FusedAll2AllComm`、`_RingP2PComm`、`_gather_size`、一组 `_all_to_all_single_*_async` |

## 2. `ParallelismConfig` 字段

分四组读，不要把它们混在一起：

| 组 | 字段 | 说明 |
|---|---|---|
| 布局 | `ulysses_size`、`ring_size`、`tp_size` | >1 才生效；`None` 与 `1` 等价于关闭 |
| 布局 | `backend` | `AUTO` 由布局推导，见 §3 |
| CP 旋钮 | `attention_backend` | 传给 `set_attn_backend`；缺省时 CP/TP 路径会补 `"native"` |
| CP 旋钮 | `cp_plan` | 用户自定义 plan，绕过 planner 注册表 |
| CP 旋钮 | `ulysses_anything`（UAA） | 任意序列长度/head 数；禁止与 ring 共存 |
| CP 旋钮 | `ulysses_async` | 仅 diffusers 模型，靠 `async_ulysses` 打方法补丁 |
| CP 旋钮 | `quant_communicate_kwargs` | `{bits, quant_type, quantize_k}`，必须 `ulysses_size>1` |
| CP 旋钮 | `ring_rotate_method` / `ring_convert_to_fp32` | 实际只支持 `p2p`；fp32 默认开 |
| 额外模块 | `extra_parallel_modules` | 名字或模块实例；`enable_cache` 里解析成模块 |
| 额外模块 | `extra_parallel_kwargs` | 不透明字典，透传进 `_ContextParallelConfig.extra_kwargs` |
| Ray | `use_ray`、`ray_*` | 由 `ray/engine.py` 消费，worker 侧再走同一套 planner |
| 私有 | `_mesh`、`_cp_mesh`、`_tp_mesh`、`_flat_*_mesh`、`_rank`、`_cp_rank`、`_tp_rank`、`_world_size`、`_cp_world_size`、`_tp_world_size`、`_device` | **不要手写**，由 `_maybe_init_hybrid_meshes` / `setup()` 填 |

`_get_world_size()` 是额外模块（TE/VAE/ControlNet）的并行度来源：hybrid 取 `u*r*t`，usp 取 `u*r`，纯 CP 取 `max(u, r)`，纯 TP 取 `t`。所以「4 卡 ulysses_tp」的 TE 会用 4 卡做 TP，而不是 2 卡。

`strify()` 是日志与文件名后缀的唯一来源：`Ulysses2_TP2_TEP_VAEP_` 这种串就是它拼的；改并行字段后如果文件名没变，说明配置没生效。

## 3. backend 推导与断言

`__post_init__` 的固定顺序：

```text
quant_communicate_kwargs 归一化（要求 ulysses_size > 1）
backend = from_str(backend) → canonicalize（NATIVE_DIFFUSER 归入 CACHE_DIT）
AUTO → hybrid ? NATIVE_HYBRID : (cp||usp ? CACHE_DIT : (tp ? NATIVE_PYTORCH : NONE))
world_size = _get_world_size()
hybrid → assert world_size >= 4；hybrid+usp → assert >= 8；usp → assert >= 4
hybrid → assert backend == NATIVE_HYBRID
cp/usp → assert backend == CACHE_DIT
tp    → assert backend == NATIVE_PYTORCH
都没有 → ValueError("No parallelism is enabled")
hybrid 且非 Ray → _maybe_init_hybrid_meshes()
```

几条容易误解的点：

- `usp_enabled()` 是「ulysses>1 **且** ring>1」，不是"任意一个"。
- `hybrid_enabled()` 是「cp_enabled **且** tp_enabled」，纯 CP+TP；`USP+TP` 也算 hybrid。
- `NATIVE_DIFFUSER` 只为兼容旧名字，实际等同于 `CACHE_DIT`；不要新增分支。
- `ParallelismBackend.is_supported(CACHE_DIT/NATIVE_HYBRID)` 只检查 diffusers 的 `Attention` / `ModelMixin` / `AttentionModuleMixin` 是否可导入，不检查 torch 分布式是否初始化——后者在 `_enable_context_parallelism` 才报错。

## 4. mesh 与 rank

hybrid 用 3D mesh，**tp 永远在最后一维**：

```python
self._mesh = dist.device_mesh.init_device_mesh(
  device_type=self._device_type,
  mesh_shape=(ring_size, ulysses_size, tp_size),
  mesh_dim_names=("ring", "ulysses", "tp"),
)
self._cp_mesh = self._mesh["ring", "ulysses"]   # 2D 子 mesh
self._tp_mesh = self._mesh["tp"]                # 1D
self._flat_mesh = self._mesh._flatten()
```

- 纯 CP / 纯 TP 不建 3D mesh：`_ContextParallelConfig.setup()` 里自己 `init_device_mesh((ring, ulysses))`；TP planner 的 `mesh()` 自己 `init_device_mesh([tp_size])`。
- `_flat_tp_mesh` 有一处兜底：PyTorch ≤ 2.9 + SGLang 组合会报 `tp already exists for submesh`，此时直接复用 `_tp_mesh`（1D 不需要 flatten）。
- `ring/ulysses/tp` 三个 rank 与 world size 都从对应 flat mesh 取，不要用 `dist.get_rank()` 代替。
- 跨 NUMA 只告警（`warn_if_cross_numa`），不阻断；判断依据是 `nvidia-smi topo -m` 的 `SYS` 单元。

## 5. 状态属性（排查时先看这几个）

| 属性 | 挂在哪 | 含义 |
|---|---|---|
| `_is_parallelized` | transformer / TE / VAE / ControlNet | 幂等闸门；已置位时 `enable_parallelism` 只 warning |
| `_parallelism_config` | 同上 | 记录本次并行配置（Ray worker 也用它） |
| `_extra_parallel_modules` | transformer | `remove_parallelism_stats` 按它递归清理 |
| `_cp_config` / `_parallel_config` | transformer 与每个 attention processor | 同一个对象；processor 侧是 dispatch 时读 `_cp_config` 的依据 |
| `_cp_plan` | transformer | planner 产出的 plan（`ContextParallelismPlanner.apply` 写入） |
| `_cp_requires_min_bn_blocks` | transformer | planner 声明的最小尾部块数，由 `caching/cache_interface.py::check_min_bn_blocks` 消费 |
| `_rowwise_layers` / `_colwise_layers` | transformer / TE | TP planner 记录，供量化按布局选 kernel |

## 6. HookRegistry

CP 的 split/gather 不是 `module.register_forward_hook`，而是**替换 `forward`**：

- `HookRegistry.register_hook(hook, name)` 把 `module.forward` 包成 `pre_forward → 原 forward → post_forward`，并把 registry 存在 `module._cache_dit_hook`。
- 每个 hook 都有稳定名字：`cp_input---<module_id>` / `cp_output---<module_id>`，subplan 追加 `---<index>`；`remove_hook(name, recurse=True)` 按名字递归摘除。
- **签名会被保留**：`inspect.signature(原 forward)` 写回 `patched_forward.__signature__`。这不是洁癖——modular pipeline（FLUX.3）按 `transformer.forward` 的参数名拼 kwargs，`(*args, **kwargs)` 会让 `img_ids` 这类必填输入被静默丢掉。
- 后注册的 hook 包住先注册的：因此 `[input_plan, output_plan]` 这种 subplan 列表能自然得到「pre 里 split、post 里 gather」。
- 一个模块可以有多个 hook（subplan），也可以对 tuple 输出逐槽 gather；plan 里用 `None` 占位表示"这一槽不处理"，此时**模块返回也必须带 None**，否则 gather hook 直接 raise。

## 7. 入口一览

| 入口 | 行为 |
|---|---|
| `cache_dit.enable_cache(pipe, parallelism_config=...)` | 解析 extra modules → 对每个 transformer 调 `enable_parallelism` → `check_min_bn_blocks` → 可选量化 |
| `cache_dit.enable_parallelism(transformer, config)` | 单 transformer 编排（Ray worker 也用它） |
| `cache_dit.remove_parallelism_stats(module)` | 测试/多配置轮换时清理标记 |
| `torchrun` / `dist.init_process_group` | 必须先于任何 `enable_parallelism`；否则 `_enable_context_parallelism` 抛 "torch.distributed must be available and initialized" |
| Ray | `use_ray=True` 时由 `ray/engine.py` 在每个 worker 里建进程组再调 `enable_parallelism`；`_ray_skip_native_parallelism` 是测试钩子 |
