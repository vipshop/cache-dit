---
name: cache-dit-distributed
description: "Understand, extend or debug cache-dit's distributed module (src/cache_dit/distributed/): ParallelismConfig and ParallelismBackend selection, the CP plan DSL and split/gather hooks, Ring / Ulysses / USP attention and the all-to-all + P2P primitives (incl. FP4/FP6/FP8 Ulysses communication and async Ulysses), tensor parallelism planners and the CP+TP hybrid 3D mesh, and the text-encoder / VAE / ControlNet parallelism paths. Use when a --parallel / parallelism_config run fails or produces wrong output, when adding a CP or TP planner for a model, when choosing ulysses vs ring vs usp vs tp, when a cache plan and a CP plan interact (Bn_compute_blocks), when VAE-P tiles desync, or when reviewing such a PR. 触发词：distributed 模块、并行推理、CP/TP/USP 报错、UAA、ulysses、ring、TP planner、CP plan、VAE 并行、text encoder 并行、mesh 建错、all2all 通信。"
user-invocable: true
---

# cache-dit 分布式模块

一句话：**`distributed/` 只做三件事——把 `ParallelismConfig` 变成 mesh、给模型装 hook/换通信、为额外组件（TE/VAE/ControlNet）单独铺一套 planner。** 计算本身仍由 attention backend 与模型 forward 决定。

## ROUTE FIRST

| 你在做什么 | 读哪里 |
|---|---|
| 先建立整体地图（目录、字段、调用链、mesh） | `references/architecture.md` |
| 改/写 CP plan、调 Ulysses/Ring/USP、低精度通信、async Ulysses | `references/context-parallel.md` |
| 改/写 TP planner、查 head 切分与 CP+TP 顺序 | `references/tensor-parallel.md` |
| TE-P / VAE-P / ControlNet-P、tile P2P 协议 | `references/extra-modules.md` |
| 找某模型归哪个 planner、新增 planner 要改哪几处 | `references/planner-inventory.md` |
| 只想知道怎么开并行、参数怎么组合 | 本文件 §2 + `docs/user_guide/CONTEXT_PARALLEL.md` |

## 0. 五条硬规则

1. **后端由配置推导，不由用户随便指定**：`backend=AUTO` 时 `__post_init__` 按 `hybrid → cp/usp → tp` 推导成 `NATIVE_HYBRID / CACHE_DIT / NATIVE_PYTORCH`；三者与开关不匹配直接 assert（`context parallelism requires CACHE_DIT` 等）。改任何并行行为前先确认落在哪个 backend 分支。
2. **CP 与 TP 的顺序不可交换**：hybrid 里**先 CP 后 TP**（`_parallelize_transformer_hybrid`），因为 3D mesh 是 `[ring, ulysses, tp]`、tp 永远在最后一维，序列维先切、特征维后切。
3. **通信只有一个 owner**：templated CP 由 `UlyssesAttention` / `RingAttention` / `USPAttention` 负责 all-to-all 与 P2P；模型 processor 若自己做了 all-to-all，就必须把 `parallel_config` 传 `None`，否则通信两次、序列顺序静默错乱。
4. **planner 注册表按类名前缀匹配，先注册者优先**：`get_planner` 遍历 `_registry` 取第一个 `class_name.startswith(key)`。`QwenImage21` 必须先于 `QwenImage`、`WanS2V`/`WanAnimate2` 必须先于 `Wan`（`transformers/planners.py` 的 import 顺序就是靠这个）。
5. **并行会改模型状态，测试要能撤销**：`_is_parallelized` 是幂等闸门（重复 enable 只 warning），`remove_parallelism_stats()` 负责清 `_is_parallelized` / `_parallelism_config` / `_extra_parallel_modules`；同一进程里连跑多组并行配置时必须先 remove。

## 1. 模块地图

```text
src/cache_dit/distributed/
├── backend.py          ParallelismBackend: AUTO / CACHE_DIT / NATIVE_PYTORCH / NATIVE_HYBRID
├── config.py           ParallelismConfig：用户入口 + 私有 mesh/rank 状态
├── dispatch.py         enable_parallelism()：transformer → 额外模块（TE/VAE/ControlNet）
├── topology.py         nvidia-smi topo 解析，跨 NUMA 只告警
├── utils.py            shard_div_attr()、_DistributedRMSNorm（TP 用）
├── quant_utils.py      quant_communicate_kwargs 归一化（bits/quant_type/quantize_k）
├── parallel/           CP 运行时：config / context_parallel / hooks / ring / ulysses / usp
│                       + diffusers_bridge（后端校验）+ runtime（挂载入口）
├── communicate/        primitives.py：Ulysses all-to-all 全家桶 + Ring P2P
├── transformers/      每模型 CP/TP planner（61 个模型文件）+ register + planners（激活表）
├── text_encoders/      TE 的 TP planner（+ 少数 DP）
├── autoencoders/       VAE 的 DP planner（tiling + tile P2P）
└── controlnets/        ControlNet 的 CP planner
```

## 2. 三条调用链

```text
enable_cache(pipe, parallelism_config=...)
  → caching/cache_interface.py：解析 extra_parallel_modules（名字→模块）、check_min_bn_blocks
  → distributed/dispatch.py::enable_parallelism(transformer)
      ├─ attention 后端注册 + set_attn_backend（默认 native）
      ├─ transformers/dispatch.py::parallelize_transformer → 按 backend 分派
      │    ├─ CACHE_DIT      → _parallelize_transformer_cp
      │    │     ├─ 构造 _ContextParallelConfig
      │    │     ├─ cp_plan = 用户 cp_plan 或 ContextParallelismPlannerRegister.get_planner(t)()
      │    │     └─ parallel/runtime.py::_enable_context_parallelism
      │    │           ├─ validate_context_parallel_attention_backend（后端必须 CP-capable）
      │    │           ├─ init_device_mesh((ring, ulysses)) → config.setup()
      │    │           ├─ 把 _cp_config 写到 transformer 与每个 attention processor
      │    │           └─ _apply_context_parallel → HookRegistry 挂 split/gather hook
      │    ├─ NATIVE_PYTORCH → _parallelize_transformer_tp → TP planner 的 parallelize_module
      │    └─ NATIVE_HYBRID  → CP 再 TP
      └─ 额外模块：text_encoder → parallelize_text_encoder（TP）
                    vae          → parallelize_autoencoder（DP + tiling）
                    controlnet   → parallelize_controlnet（仅 CACHE_DIT CP）
```

`--parallel` 的字符串在 `cli/utils.py::_prepare_distributed_size` 里折算成三元组（真实语义，容易记错）：

| `--parallel` | 折算 |
|---|---|
| `ulysses` / `ring` / `tp` | 该维 = world_size |
| `usp` | ulysses = world//2，ring = 余数 |
| `ulysses_tp` | ulysses = world//2，tp = 余数（2 卡 = U1×T2，**纯 TP**） |
| `tp_ulysses` | tp = world//2，ulysses = 余数（4 卡 = U2×T2） |
| `ring_tp` / `tp_ring` | 同理，ring 与 tp 对半 |
| `usp_tp` | 硬断言 world_size == 8，U2×R2×T2 |

`--parallel-text` / `--parallel-vae` / `--parallel-controlnet` **单独用是 no-op**：它们只在 `--parallel` 非空时被装进 `extra_parallel_modules`。

## 3. 四个最常被问的问题

| 问题 | 答案落点 |
|---|---|
| CP 与 TP 怎么选？ | 序列长、attention 占比高、显存不够但不想切权重 → CP；只想省权重显存、heads 能整除 → TP；两者都要 → hybrid（先 CP 后 TP，4 卡起） |
| 为什么 CP 直接报错？ | 先看 `validate_context_parallel_attention_backend`：非 cache-dit 注册的后端或 non-CP 后端会在这里 raise；再看 mesh 大小与 `heads` 整除 |
| 为什么 CP 输出错但形状对？ | 90% 是 split/gather 双份或缺失：hook 与模型内部手工切分只能二选一；其次看 UAA 是否越界（`tensor_split` 不整除） |
| 为什么 VAE 并行没生效？ | `check_auto_encoder` 只认 `pipe.vae`；pipeline 把 VAE 挂在 `video_vae` 时 `--parallel-vae` 只 warning（"no VAE found"）不报错，需给 pipeline 加 `vae` 别名；另外 `--parallel-vae` 与 `--vae-tiling` / `--vae-slicing` 互斥（断言） |

## 4. 高频坑（详细版在各 reference）

1. `Bn_compute_blocks=0`（DBCache 默认）+ planner 声明了 `_cp_requires_min_bn_blocks=1` → `check_min_bn_blocks` 直接 raise：cache 命中时最后一块不执行，gather 永远不发生。
2. hybrid 需要 PyTorch ≥ 2.10：`init_device_mesh` 不支持从 submesh 再建 submesh，报 `Cannot create a submesh from a submesh`；`cache-dit` 会补一句升级提示。`flat_tp_mesh` 在 2.9 及以下还有 "tp already exists for submesh" 的兜底。
3. `quant_communicate_kwargs` 必须配 `ulysses_size > 1`；FP4 只允许 `per_group_16`；内核没编进 `_C_ops` 时报 "Build cache_dit._C_ops with CACHE_DIT_BUILD_CUDA=1"。
4. UAA（`ulysses_anything`）不能与 `ring_degree > 1` 共存，且要求 `ulysses_degree > 1`；compile 下依赖 `capture_scalar_outputs`（torch≥2.10 自动开）。
5. `_All2AllComm` 在构造时就选好 q/k/v/o/lse 的实现；`quantize_k=False` 会让 K 单独走 BF16 实现（默认 True）。
6. VAE tile 传输有两套实现：`TileBatchedP2PComm`（批量，单 tile 相位快）与 `TileSafeP2PComm`（一消息一完成，多 tile decode 必须用它）。混用或自己发 shape 会踩 "SymIntArrayRef expected to contain only concrete integers"。
7. TP 只接受 `ModelMixin`；非 diffusers 模型要么先 vendor 进 cache-dit，要么只走 CP。
8. 多卡组合的硬约束：`heads % tp_size == 0`，且 `(heads // tp_size) % ulysses_size == 0`（Ulysses 切 head）。

## 5. 验证

| 层次 | 判据 |
|---|---|
| 单卡等价 | `cp_config`/`tp` 关掉与开启（world_size=1 或走 native）输出逐比特；planner 本身用 `tests/parallelism/test_<model>_cp_plan.py` 断言 plan 字典 |
| 多卡数值 | `tests/parallelism/test_<model>_cp_tp_tiny.py`：tiny 模型 + `mp.spawn`，与单卡比 PSNR/相对 L2；端口必须取空闲端口并 `destroy_process_group()` |
| 通信原语 | `tests/parallelism/test_distributed_primitives.py`、`test_fN_fused_qkv_a2a.py`、`test_bf16_a2a_alignment.py`、`test_ulysses_gqa_multirank.py` |
| VAE-P | `test_*_vae_dp*.py`（含 `test_vae_p2p_order.py`：多 tile 顺序回归） |
| TE-P | `test_ming_image_te_p_tiny.py` |
| 关闭路径 | `remove_parallelism_stats()` 后重跑 baseline，确认没有残留 hook（`HookRegistry` 的 `remove_hook` 按名字递归摘） |

精度/性能口径与 cache 侧一致：同 seed、同 recipe、raw float 比较；`--compile` 行 run-to-run 不收敛只能报区间。

## 6. 文档同步

| 文件 | 改什么 |
|---|---|
| `docs/user_guide/CONTEXT_PARALLEL.md` | 用户可见的 `--parallel` 组合、UAA / async / FP4-8 通信、Ring / USP 说明 |
| `docs/user_guide/API_DOCS.md` | `ParallelismConfig` 字段语义 |
| `docs/developer_guide/AGENTS.md` | 只有新增/删除 skill 时才动 |
| `docs/EXAMPLES.md` | 代表性多卡命令 |

## 7. 参考文件

- `references/architecture.md`：目录职责、`ParallelismConfig` 全字段、backend 推导与断言、mesh/rank 派生、状态属性、HookRegistry 机制。
- `references/context-parallel.md`：CP plan DSL、split/gather 语义、Ring/Ulysses/USP 流程与 LSE 契约、all-to-all 选择矩阵、低精度通信、async Ulysses、CP×cache 交互与排障。
- `references/tensor-parallel.md`：TP planner 契约、FLUX 范例、colwise/rowwise 配对、RMSNorm/adain 特殊处理、hybrid 顺序与 mesh、TP×量化。
- `references/extra-modules.md`：TE-P / VAE-P / ControlNet-P 的 dispatcher、tile P2P 两套实现与已知顺序/ dtype 坑。
- `references/planner-inventory.md`：模型 → CP/TP planner 全量清单、五类 plan 形态、新增 planner 的 runbook 与测试对应关系。
