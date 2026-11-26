---
name: cache-dit-attention
description: "Add, fix, or review a cache-dit attention backend under src/cache_dit/attention/: the _AttnBackend enum, _AttnBackendRegistry registration and signature filtering, the NHD dispatch contract, the templated-CP forward/backward ops (Ulysses / Ring / USP), the diffusers bridge, set_attn_backend and ParallelismConfig.attention_backend wiring, the --attn CLI choices, torch.compile-safe torch.library wrapping, and the acceptance matrix. Use when integrating a vendor attention kernel (ffpa-attn, SageAttention-2/3, Sol-Attn, FlashAttention, flash-varlen, a sparse or quantized attention), when --attn appears to do nothing or silently falls back to SDPA, when context parallelism rejects or corrupts a backend, or when reviewing such a PR. 触发词：新增 attention backend、接入 attention 后端、--attn 不生效、attention 静默回退、CP 不支持 attention、attention backend 注册."
user-invocable: true
---

# 接入一个新的 attention backend

一句话：**backend 是"一次函数调用 + 一条注册表记录"，不是一套子系统。** 真正的难点全在契约边界（layout / dtype / mask / LSE / CP）与"怎么证明它真的被调用"。

## ROUTE FIRST

| 你在做什么 | 读哪里 |
|---|---|
| 接入新 backend，先看要改哪些文件 | 本文件 §1（架构）、§2（七步） |
| 写代码时查每个接触点的真实语义 | `references/dispatch-chain.md` |
| 不确定 layout / mask / LSE / GQA / CP 该怎么实现 | `references/backend-contract.md` |
| 想抄一个同类先例（ffpa / sage2 / sage3 / sol_attn / flash_varlen / cudnn / npu） | `references/precedents.md` |
| 写测试、跑验收、判定"有没有真的生效" | `references/acceptance.md` |
| 只是想让某个已有 backend 工作起来（装依赖、选名字） | `docs/user_guide/ATTENTION.md` + `references/dispatch-chain.md` §8 |

## 0. 四条硬规则

1. **不许静默回退。** backend 不满足条件时必须 raise。允许 fallback 的实现必须让它可观测——`sol_attn` 的默认策略是退 native，但每条 reason 都计数 + 首次 warning，`strict=True` 才改成 raise。最恶劣的形态是"看起来在跑，其实走的是 SDPA"：ffpa 必须在 import 期设 `FFPA_CUDA_ALLOW_SMALL_D=1`，否则 `ffpa_attn_func` 悄悄退 SDPA。
2. **必须证明 backend 真的被调用了。** "跑通 + PSNR 正常"不是证据。要么 spy vendor 函数计数，要么自带 stats（`sol_attn._get_sol_attn_stats()` 的 `kernel_calls` / `fallback_calls`）。
3. **CP 只能有一个通信 owner。** 走 dispatcher 的 templated CP（`_cp_config` 非空）就不要再在 backend 里做 all-to-all / all-gather；反之，自己管通信就必须把 `_cp_config` 视为 `None` 并声明处理方式。
4. **参数要么支持，要么 raise。** dispatcher 按函数签名过滤 kwargs，没声明的参数会被**静默丢弃**（只有 `CACHE_DIT_ATTN_CHECKS=1` 才打 warning）。所以 `attn_mask` / `dropout_p` / `return_lse` 即使不支持也必须出现在签名里并显式报错。

## 1. 架构：一个 enum + 一张注册表 + 一个 dispatcher

| 文件 | 职责 |
|---|---|
| `src/cache_dit/attention/backends/register.py` | `_AttnBackend` enum（所有合法名字）、`_AttnBackendRegistry`（`register` 装饰器、签名集合、约束、CP 能力、diffusers 桥接标记）、`_dispatch_attention_fn`、`_context_parallel_attention`、通用检查函数 |
| `src/cache_dit/attention/backends/<name>.py` | backend 本体；**import 即注册** |
| `src/cache_dit/attention/backends/__init__.py` | 包级 import，保证 `import cache_dit.attention` 时全部注册 |
| `src/cache_dit/attention/__init__.py` | `_maybe_register_custom_attn_backends()` 里**另有一份显式 import 列表**，两份都要加 |
| `src/cache_dit/attention/diffusers_bridge.py` | 把 cache-dit 的 backend 注入 diffusers 的 `AttentionBackendName` / `_AttentionBackendRegistry`（含枚举成员动态注入） |
| `src/cache_dit/attention/interface.py` | `set_attn_backend`：diffusers `ModelMixin` 走 `set_attention_backend`，普通 `nn.Module` 走 `processor._attention_backend` |
| `src/cache_dit/attention/backend_selector.py` | 平台自动选择（目前只有 NPU） |
| `src/cache_dit/cli/utils.py` | `--attn` 的 `choices` + 每个 backend 专属 flag 的交叉校验 |
| `src/cache_dit/distributed/dispatch.py` | 并行入口：`ParallelismConfig.attention_backend` → `set_attn_backend` |

调用链：

```text
--attn X
  → CLI choices 校验 / set_*_config 全局配置
  → set_attn_backend(module, "X") 或 parallelism_config.attention_backend="X"
  → 模块及其 processor 上写入 _attention_backend = "X"
  → 模型内部 _dispatch_attention_fn(..., backend="X")
  → _AttnBackendRegistry.get_backend("X")(**按签名过滤后的 kwargs)
```

## 2. 七步接入（最小可用 diff 就是这 7 个文件）

```text
src/cache_dit/attention/backends/register.py     # 1 行 enum 成员
src/cache_dit/attention/backends/<name>.py       # 新文件：注册 + 调用体
src/cache_dit/attention/backends/__init__.py     # 1 行 import
src/cache_dit/attention/__init__.py              # 显式 import 列表加一个名字
src/cache_dit/cli/utils.py                       # choices（+ 专属 flag 校验）
docs/user_guide/ATTENTION.md                     # 表格一行 + 专节
tests/api/test_<name>_backend.py                 # 新文件
```

1. **加 enum 成员**（`register.py::_AttnBackend`）。命名约定：与 diffusers 语义相同的实现复用同名（`flash` / `flash_varlen` / `sage` / `native`），避免两套同名不同实现；cache-dit 独有且对用户可见的用直白小写（`sol_attn`、`ffpa_fp8`），内部/特殊用途用下划线前缀（`_flash_3`、`_sdpa_cudnn`、`_native_npu`）。
2. **写 backend 文件**：vendor 导入用 `try/except Exception: func = None`，把"没装依赖"推迟到调用时报 `RuntimeError`（而不是 import 期炸掉整个包）；函数上挂 `@_AttnBackendRegistry.register(...)`。
3. **两处 import 都要加**：`backends/__init__.py` 与 `attention/__init__.py::_maybe_register_custom_attn_backends` 的显式 import 列表。漏后者会得到 `Backend X is not registered`——除非 diffusers 恰好同名，那会被 proxy 接住，掩盖漏注册。
4. **写形状/语义门**：`constraints=` 平时是空列表且只在 `CACHE_DIT_ATTN_CHECKS=1` 时执行，所以**门要写在调用体里**（`_require_supported_cuda` / `_ineligible_reason` 就是先例），`constraints` 只作可选加固。
5. **CP**：声明 `supports_context_parallel=True` 并写 `forward_op` / `backward_op`，最后走 `_context_parallel_attention`。没有 LSE 就只能纯 Ulysses（Ring 强制 `return_lse=True`）。
6. **编译安全**：vendor 是 Triton 或被 dynamo trace 会碎图的，用 `torch.library.define` + CUDA impl + `register_fake` 包成黑盒（`sol_attn` / `sage3` 先例），并补 `opcheck` + `fullgraph=True` 用例。
7. **对外接线**：CLI `choices=[...]`；backend 专属 flag 必须校验"只有对应 `--attn` 才能用"；需要全局旋钮的写 `set_<name>_config()`（在 compile 之前调用）；`docs/user_guide/ATTENTION.md` 补行与专节。

## 3. 契约决定工作量

先判自己属于哪一类，再决定写多少：

| 形态 | 例子 | 额外工作量 |
|---|---|---|
| 纯调用型：vendor 直接吃 NHD `[B,N,H,D]` | `sage`（`sageattn(..., tensor_layout="NHD")`） | 一个文件 + 一个 enum |
| 布局/形状转换型：vendor 要 BHND、固定 head_dim、固定 dtype | `sage3`（head_dim pad 到 64/128，>128 退 SDPA）、`ffpa`（BHND 回退 + sm_120 NHD 快路径） | 转换 + 回退分支 + 精度代价说明 |
| 多档变体族：同一 vendor 多个精度档 | `ffpa` / `ffpa_fp8` / `ffpa_fp8_per_block` / `ffpa_fp4` | 一个内部 impl + N 个薄 wrapper + 一个配置 setter |
| 近似/稀疏型：不逐比特、有 eligibility 集合 | `sol_attn`（head_dim=128、BF16、等长 Q/K/V、默认 scale） | eligibility 判定 + 显式 fallback + stats + strict |
| 平台原生 | `_native_npu` / `_npu_fia` | 设备探测（`AttnBackendSelector`） |

完整参数语义、mask/LSE/GQA 契约、以及现有 backend 的能力矩阵见 `references/backend-contract.md`。

## 4. 静默失败清单（按踩坑频率排序）

1. **漏声明参数** → mask / dropout / `return_lse` 被过滤掉，backend 拿到的是默认值。判断方法：`get_supported_arg_names(backend)` 必须包含全部语义参数。
2. **vendor 层静默回退**（ffpa 的 small-D、sage3 的 >128 SDPA）→ 用户以为在用 kernel。必须：能设 env 就设，能断言就断言，退路要打 `warning_once` 且统计。
3. **共享 backend 对象被改写状态**（ffpa 在缓存的 `CUDABackend` 上切 `tensor_layout`）→ 任何 fallback 分支都必须把状态还原成进入时的样子。
4. **CP + mask**：`_context_parallel_attention` 的 mask allowlist 按 `forward_op.__name__` 匹配（目前只有 native / cudnn / flash_varlen / npu / npu_fia），新 backend 默认不支持 mask+CP，请显式 raise，不要指望上游拒绝。
5. **Ring 需要 LSE**：无法返回 `[B,N,H]` LSE 的 backend 只能声明纯 Ulysses；`sol_attn` 直接在 `ring_degree>1` 时 `NotImplementedError`。
6. **`enable_gqa` 只在 torch>=2.5 透传**：签名里有也未必收得到，默认值必须是安全语义（通常 `False`）。
7. **fp32 组件也会命中**：`--attn` 是模块级设置，会落到目标 transformer 上所有注意力模块，包括跑 fp32 的投影/子模块（ffpa 的注释点名 MiniMax-H3 的 fp32 投影与视频 VAE decoder）。ffpa 的做法是"非 fp16/bf16 退 SDPA，但有 CP 就直接报错"，因为 CP 下 SDPA 只会算本地分片。
8. **diffusers-only 名字**：`_native_cudnn` 不在 cache-dit enum 里，diffusers 模型能设、非 diffusers 模块会直接 `ValueError`。新增 backend 若只打算给 diffusers 用，走 `_maybe_register_diffusers_backend_proxy`，不要塞进 enum。
9. **compile 碎图**：直接调 vendor Triton kernel 会被 dynamo trace 进去，inductor 重编译常失败（`tl.arange` 非 2 的幂等）。先包 `torch.library`。
10. **性能归因错位**：attention 占比决定收益上限（1024² 图像模型 attn 只占 18–25%，长视频序列才高），不要用 FLOPs 推；必须同 recipe、同 seed、CUDA event 计时。

## 5. 验收（细节与代码骨架见 `references/acceptance.md`）

最低集合，缺一条就不算接入完成：

- 注册与签名：`get_backend` 非空、`get_supported_arg_names` 覆盖全部语义参数、`is_context_parallel_available` 与声明一致。
- diffusers 桥：`_maybe_register_custom_attn_backends()` 后 `_AttentionBackendRegistry._backends[AttentionBackendName(x)]` 非空；`set_attn_backend(dummy ModelMixin)` 收到名字。
- 依赖缺失：monkeypatch 成 `None` 后是明确的 `RuntimeError`，不是 `AttributeError`。
- 语义拒绝：mask / causal / dropout / gqa / lse 逐条 raise。
- 数值：对独立 reference 或 dense SDPA 给 rel-L2 与 cos（`sol_attn` 用 3e-3 / 0.999；低精度档必须写清为什么到不了）。
- **真被调用的证据**：spy 计数或自带 stats，`fallback_calls` 必须为 0。
- CP：2/4 卡与单卡比，且断言 CP 分支看到的分片语义（`tests/parallelism/test_sol_attn_cp.py` 先例）。
- 端到端：同 seed、同 recipe 的 native vs 新 backend，PSNR/SSIM + 视觉 + attention 段耗时。

## 6. 文档同步清单

| 文件 | 改什么 |
|---|---|
| `docs/user_guide/ATTENTION.md` | 能力表一行（parallelism / attn_mask 两列必须真实）+ backend 专节（硬件范围、精度、限制） |
| `docs/user_guide/INSTALL.md` | 可选依赖安装命令（**不要**给 vendor 包加 `pyproject` extra，现有 flash/sage/ffpa/sol_attn 都没加） |
| `docs/user_guide/API_DOCS.md` | `attention_backend` 的候选名单 |
| `docs/user_guide/CONTEXT_PARALLEL.md` | CP 限制（如"只能纯 Ulysses"） |
| `docs/EXAMPLES.md` | 有代表性的 `--attn X` 示例命令 |
| `docs/developer_guide/AGENTS.md` | 只有**新增/删除 skill** 时才动，不要为单个 backend 改 |

## 7. 先例速查（完整分析见 `references/precedents.md`）

| backend | 抄什么 | 别抄什么 |
|---|---|---|
| `ffpa`（+fp8/fp4） | 多档变体族组织、能力门（sm_120 / sm_89 分开）、NHD 零拷贝快路径、全局 override + CLI 交叉校验、mask 归一化 | 不要在共享对象上留可变状态；不要依赖上游静默回退 |
| `sage`（SageAttention-2） | 最小接入样板：一个函数、`tensor_layout="NHD"`、CP 用 forward/backward op | 不要省掉 `attn_mask` 的显式 raise |
| `sol_attn` | 近似 backend 的完整样板：`torch.library` 黑盒 + fake、eligibility + fallback stats + strict、Ring 显式拒绝、opcheck/fullgraph 测试 | 不要把近似精度写成"无损" |
| `sage3` | head_dim padding 与 `>128` 退 SDPA、用 `torch.library` 挡住 inductor | 不要照抄它的精度（FP4 P 只有 {0,0.5,1} 三档，cos≈0.03） |
| `flash_varlen` | mask → 2D unpad 的 varlen 形态、CP mask allowlist 成员 | 不要假设它能吃任意 3D mask |
| `_sdpa_cudnn` / `_native_cudnn` | 同名 SDPA kernel 切换；cache-dit 版进 enum、diffusers 版走 proxy | 不要给两者都声明 CP 却不验证 |
