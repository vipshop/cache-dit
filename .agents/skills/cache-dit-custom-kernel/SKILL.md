---
name: cache-dit-custom-kernel
description: 'Use when adding, building, reviewing, profiling, or optimizing a Cache-DiT custom operator — SVDQ INT4/NVFP4 producers, fused LoRA, GEMM, CuTe rewrite, native/cute dispatch, SmoothQuant outlier migration and low-rank residual quantization — plus the full CUDA -> pybind -> Python -> torch.library -> internal-call chain, ports from nunchaku or deepcompressor, reproducible ccache builds, and layered validation. Every operator in the repo today is a CUDA/CuTe C++ kernel; CuTe DSL is a possible future backend, Triton is explicitly not supported.'
argument-hint: 'Describe the operator family or kernel, the target repo/extension, dtypes and architectures, and whether this is a new operator, a port, or a performance change.'
user-invocable: true
---

# Cache-DiT 自定义算子开发与优化

## 使用范围与路由

本 skill 处理 `csrc/kernels/` 下**所有**算子族（现有 `comm`、`attn`、`svdq`，目前全部是 CUDA/CuTe C++ kernel）的接入、移植、构建、profile 与优化。读完本节再动手：

| 任务 | 判据 | 读哪里 |
|---|---|---|
| 接入新算子 / 新算子族 | 要新建 `csrc/kernels/<族>/` 下的文件，或要动 `setup.py` 的源清单/扩展/schema 注册 | `references/kernel-integration-chain.md` |
| 从上游仓库移植算子 | 有 nunchaku / deepcompressor / flashinfer 参照实现 | `references/operator-migration.md` |
| 改现有算子实现 | 只改 `.cu/.cuh/.h`，schema、`setup.py`、Python 包装都不动 | 本文 §5 + 该族 Python 模块 + 对应测试 |
| 性能优化 | 数值契约与链路不变 | 本文 §6；SVDQ 另读下列三份 ref |
| SVDQuant 原理 | 推导、误差上界、scale 粒度、低秩残差 | `references/svdquant-mathematics.md` |

SVDQ 专项参考，修改 NVFP4 CuTe 路径前先读：

- `references/svdq-nvfp4-cute.md`：当前实现、有效/无效尝试和实测性能。
- `references/rfc-cuda-optimizations.md`：优化门槛、待办、状态更新规则。
- `references/svdq-nvfp4-cute-tma-sync-analysis.md`：TMA/mbarrier、warp/block 同步、K-block 软件流水及 CUTLASS/FlashInfer SM120 对照；做 SVDQ CuTe GEMM 调度与性能优化前先读。

原理路由：先读 `references/svdquant-mathematics.md` 的 §1–4 建立记号并复核推导与成立条件，需要落到当前实现时读 §5，评估 TurboQuant / DVD-Quant 是否可借鉴时读 §6–8；性能结论与数值契约仍以 `references/svdq-nvfp4-cute.md` 为准。

配合 `cuda-auto-tune`（NCU 驱动迭代优化）、`cuda-cpp-kernel`（CUDA/PTX/架构细节）、`cutlass-cpp-kernel`（CUTLASS/CuTe C++ 模板）、`cute-dsl-kernel`（CuTe DSL Python kernel 编写）使用。本仓目前没有 CuTe DSL 算子；将来引入时，kernel 编写走 `cute-dsl-kernel`，接入、注册与构建仍走本 skill。NCU 优化必须先 profile 后修改，再复采验证。

**算子形态现状与后端边界**：仓库当前**只有纯 CUDA/CuTe C++ kernel**（`csrc/kernels/`），没有 Triton，也没有 CuTe DSL 算子。后续**可能**引入 CuTe DSL（`@cute.kernel`）算子，但必须先补齐 Python 侧注册、扩展构建与打包这条链再落地；**Triton 明确不支持**，不要新增，也不要恢复 `kernels/backend.py` 之类后端选择层。上游只有 Triton 实现时，读算法后用 CUDA 重写。细则与反模式见 `references/operator-migration.md` §1、§9。

## 1. 分层架构（所有算子族共用）

```text
内部调用点
  -> torch.ops.cache_dit_cuda_ops.<op>        Python 侧 torch.library：dispatch / fake / compile
  -> kernels/_<族>.py  @torch.library.impl(..., "CUDA")   元数据校验 + 建输出
  -> _call_<op>()                            out-param 桥 + 签名归一化
  -> _C_<族>.ops.<op>                        pybind 子模块
  -> csrc/kernels/<族>/<TU>.cpp -> <kernel>.cu            TORCH_CHECK + launch
```

两条最容易找错地方的机制：

- `cache_dit_cuda_ops` 命名空间**只存在于 Python**（`torch.library.define` 在模块 import 时创建）；`csrc/` 里没有任何 `TORCH_LIBRARY`，pybind 只暴露 `ops`（SVDQ 另有 `utils`）子模块。两套注册表靠 `_call_*` 桥接。
- 三族现状、extension 名、namespace、arch 门槛、内部调用点、以及"没有统一文件名契约"这一事实，全部只在 `references/kernel-integration-chain.md` §2 维护，不要在这里重抄。

C++ 入口一律 `void` + out-param；schema 默认函数式，所以 out-param 对 torch 不可见、不需要 `mutates_args`。校验以 C++ 的 `TORCH_CHECK` 为权威（`svdq` 是历史例外，两侧都几乎不校验，见 ref §3.3）。

## 2. 新算子接入 checklist

细节、模板代码与实测陷阱见 `references/kernel-integration-chain.md`；这里只给可逐项打勾的顺序：

1. `csrc/kernels/<族>/<kernel>.cu`：kernel + 入口 `TORCH_CHECK`。
2. 声明头：`void <op>(at::Tensor ...)`，`namespace cache_dit::<族>`。
3. pybind TU：`PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` + `m.def_submodule("ops").def("<op>", ...)`。
4. `setup.py` 6 处：`_required_<族>_sources` / `_missing_<族>_sources` / `_get_<族>_extension` / `_get_extensions()` 追加 / 该函数里的 `"the three extensions"` 文案 / `IS_CU13_RELEASE` 分支；另有 4 个"族清单"同步点（ref §5.3）。
5. `src/cache_dit/kernels/_<族>.py` 六段：loader → availability → `_get_required_ops_module` → `_call_<op>` → `torch.library.define` + `impl(CUDA)` + `register_fake` → 公开包装。
6. `src/cache_dit/kernels/__init__.py`：re-export 公开包装，并建议一并 re-export `<族>_extension_is_available` / `<族>_get_load_error`（现状只有 svdq 导出了这两个）；`__all__` 等于全部非下划线成员。
7. 内部调用点：可用性门禁用 `<族>_extension_is_available()` 或捕获 `RuntimeError`；错误信息要带构建命令与 arch 门槛。
8. 测试：`test_ops_dispatch.py` 参数表条目、fake 用例、参考实现对比、`fullgraph=True` compile smoke（模板 `tests/kernels/test_cuda_fp8_comm_ops.py`）。

三条硬规则：schema 形参顺序即调用顺序（写反不报错、静默算错）；fake 的 shape/dtype 必须与 impl 一致（否则 eager 绿、compile 炸）；`torch.library.define`/`impl` 不幂等，不要 reload 这些模块。

## 3. SVDQ 专项：架构与命名

SVDQ 调用链：

```text
SVDQW4A4Linear.quantize / forward_quant
  -> src/cache_dit/kernels/_svdquant.py
  -> torch.library schema + fake/CUDA impl
  -> pybind.cpp -> ops.h
  -> svdq::native_kernels 或 svdq::cute_kernels
```

目录与命名契约：

- `csrc/kernels/svdq/native/`：旧 native 实现，使用 `svdq::native_kernels`。
- `csrc/kernels/svdq/cute/fp4/`：独立 CuTe 实现，使用 `svdq::cute_kernels`。
- cute 侧文件与 C++ kernel/host/launch 名不重复携带 `_cute`；但 `ops.h`/pybind 的**对外**入口按后端显式命名（`gemm_w4a4_cute`、`quantize_w4a4_native`、`quantize_w4a4_row_major`），Python 侧只有 `svdq_gemm_w4a4` 与 `svdq_quantize_w4a4` 两个公共入口，用 `runtime`（`native`/`cute`）× `lora_layout`（`native`/`row_major`）两个正交维度选择，不再有 `svdq_*_cute` / `_native_v2` 这类平行包装（PR #1259）。
- `gemm_w4a4.cu/.cuh/.h`：NVFP4 主 GEMM、BF16 LoRA-up 融合和 TMA pipeline。
- `quantize_w4a4.cu`：`quantize_w4a4_fused_lora_kernel`，SVDQ 上半段 activation quant + LoRA-down 融合。
- `common.h`、`Tensor.h`、`torch.h` 的共享类型仍在全局命名空间；host 侧用 `::Tensor` 避免与 `cute::Tensor` 歧义。

公开 `runtime_kernel` 只接受 `None/native/cute`。INT4 使用 native v2；NVFP4 当前默认 native，`cute` 为显式 opt-in，直到 GEMM 和 e2e 性能达标。deprecated `v1/v2` 配置字符串不作为公开 selector。

SVDQ 在通用模板之外的偏离（3 个 required getter、`utils` 子模块、无 `TORCH_CHECK`）见 `references/kernel-integration-chain.md` §2 与 §4.2，照抄前先看。

## 4. SVDQ 专项：独立 NVFP4 CuTe 工作流

保持以下数据流，不得重新引入旧工作流适配层：

```text
quantize_w4a4_fused_lora_kernel
  -> row-major E2M1 A + CUTLASS 128x4 SFA + row-major BF16 D[M,R]
  -> gemm_w4a4_kernel
  -> A/B/SFA/SFB TMA + NVFP4 MMA + BF16 LoRA-up MMA
```

禁止在 CuTe forward 中调用 native activation quantizer、读取 v1 FP32 LoRA scratch、执行 G2R 旧布局反解或增加 `lora_act_to_bf16`。旧 checkpoint 只能在加载时一次性转换 `qweight/wscales/proj_up/proj_down/smooth_factor`；state dict 保持原 native pack 格式。

当前 producer 使用手写 `mma.sync/ldmatrix` 和 shared staging。未来迁移到 CuTe `TiledMMA/TiledCopy` 时，必须保留现有数学、输出布局和 CTA 内归约契约。

## 5. 可重复构建

先初始化依赖并激活目标 Python 环境：

```bash
git submodule update --init --recursive
CACHE_DIT_BUILD_CUDA=1 \
CACHE_DIT_CUDA_ARCH_LIST=120a \
bash tools/build_fast.sh install -e . --no-build-isolation --no-deps
```

`csrc/third_party/cutlass` 是 NVIDIA 官方 v4.6.1 submodule，默认构建直接使用它。`CACHE_DIT_CUTLASS_DIR` 只用于显式开发覆盖；不要假设机器上存在同级 CUTLASS checkout。

`tools/build_fast.sh` 默认 `MAX_JOBS=32`，通过 `tools/nvcc` 将 nvcc 交给 ccache。当前环境已有 ccache；用 `bash tools/build_fast.sh` 调用，因为脚本没有可执行位。需要强制缓存能力时设 `CACHE_DIT_REQUIRE_CCACHE=1`。仅在确有必要时设 `CACHE_DIT_CLEAN=1`，它会删除构建目录。

每次 kernel 迭代使用同一入口。不要因为源码改动触发全量重编就放弃缓存；记录构建是否 cold、是否换过 compiler/shim，避免把首次 ccache 填充时间误报为加速收益。

**改共享 `.h` 必须显式清理**：`setup.py` 没有 `depends=`，`build_ext` 只比源文件 mtime，改头文件后 `.so` 会静默不变。新增算子族要动的 `setup.py` 位置见 `references/kernel-integration-chain.md` §5.1。

## 6. 优化工作流

1. 明确验收对象：producer、GEMM、完整 layer 或 e2e，不能混用。
2. 固定 shape、dtype、GPU、warmup、迭代数和输入；profile 前检查 GPU 占用。
3. 先取 nsys kernel-only 和 NCU full 基线。nsys 用于性能判定，NCU 用于归因。
4. 从 roofline、occupancy、寄存器/local memory、global sectors、shared bank conflict、PC sampling 和 warp stall 中确定一个主瓶颈。
5. 一次只做一个有假设支撑的改动；保持数值契约不变。
6. 重建后先跑数值回归，再用同口径 nsys 复测；NCU 复采只解释指标变化。
7. 将有效和无效尝试都写入性能参考；无效方案给出证据，不要反复尝试。
8. 更新 RFC 状态，只有完成独立 kernel 与 e2e 验收才可标完成。

### 测量口径

- **nsys kernel-only median/mean**：kernel 性能主指标；报告实例数、min/max 和稳态 warmup。
- **CUDA Event op-level**：包含分配、清零和 launch 的 Python 算子调用成本，不能替代 kernel-only。
- **NCU duration**：受 replay、时钟和 profiler 影响，只用于优化前后归因，不作为交付性能。
- **FLUX e2e**：固定模型、prompt、seed、分辨率、steps 和 `--compile`，至少多次重复；同时做图像检查和 PSNR。

不要把冷启动首轮离群值当成稳定收益，也不要用单 GEMM 收益推断 e2e 收益。

## 7. 验证与回归

通用测试矩阵（含每个用例的真实强度与局限：dispatch 参数表是手工维护的、source manifest 只覆盖 svdq、`tests/kernels/` 不进 CI）见 `references/kernel-integration-chain.md` §7。

SVDQ 回归命令：

```bash
python -m pytest -q \
  tests/kernels/test_ops_dispatch.py \
  tests/kernels/test_svdquant_cute.py \
  tests/kernels/test_svdquant_runtime.py \
  tests/kernels/test_svdquant_source_manifest.py \
  tests/quantization/test_quantize_config.py
```

`test_svdquant_source_manifest.py` 是 CPU-only 护栏（不 import torch、扩展或 setup.py）：它用 `ast` 抽取并执行 `setup.py` 的源清单函数，双向比对 `csrc/kernels/svdq/` 下的 `.cu/.cpp`。改动该目录的文件名或源清单后必须跑；CI 不收集它（workflows 不许改），只能本地执行。

CUDA 数值改动还需运行 producer、独立 GEMM、高层 Linear 和 INT4 隔离回归。改名时同时检查 Python 私有 pybind 调用、公开 torch.library 名称、fake 实现、CLI、文档和临时 harness。运行 `pre-commit run --files ...`、`git diff --check` 和 `read_lints`。

FLUX 验收使用同一 seed 对比默认 native、显式 cute 和 BF16 baseline。精度结论需说明量化差异；不能仅凭“正常出图”宣称无损。

## 8. SVDQ 当前性能结论与边界

截至 2026-11-05，SM120、M=4096/N=K=3072/rank=32：

- 当前 CuTe GEMM 已恢复为 A/SFA G2R + B/SFB/D/L1 TMA，当前 CuTe producer 直接输出 v1 A/SFA 与 BF16 D，无 finalize。
- 当前统一 CUDA Event 基线见 `references/svdq-nvfp4-cute.md` §17（GPU4，Event50×3轮）。M4096 rank32 kernel-only 120.22us vs native120.24us；完整层 M1024/M4096 分别 62.99/164.62us vs native63.65/168.34us。
- **rank32 大 shape 仍落后 native**：`8192x3072`/`4096x6144`/`8192x6144`/`16Kx8Kx8K` 的 cute/native 为1.026/1.020/1.047/1.064。rank128 在所有大 shape 略优于 native（0.32%–1.43%）；rank64 在 `8192x6144`/16K 退化。下一条主矛盾是 rank32/64 大 shape。
- hybrid 的机制是拆掉 A/SFA smem 中转：相对全 TMA，shared 指令/wavefront/bank conflict 降22%/18%/29%，`math_pipe_throttle` 6.00→4.61，静态 FENCE 9→2；代价是 A 重复 G2R load，global 请求升4倍、L1 hit 94.5→55.1%，但 L2 命中87%，总量仍远低于 native。
- 当前 hybrid 重测逐 k-block `A load -> B/SFB copy -> MMA`：168 regs、零 spill；12格净变化约0，2K rank32 +0.61%，8K/16K 小优0.2%–0.5%，在1%门槛内，已保留。旧“0.12%不保留”是旧 hybrid/nsys 口径。
- stages 仍只有 S3：S2 被 output overlay 64KiB > pipeline 55.3KiB 拒绝，S4 110.6KiB > 99KiB；G2R 只释放未使用的 SFA 3KiB，A/B 区仍被 D/L1 overlay 占用。要加 stages 必须先解耦 overlay。
- 剩余 0.737M bank conflict 已定位为 **LoRA BF16 D/L1 的标量 LDS**，不是 SFB。canonical `tile_to_shape(SW64 atom,(BM,rankTile,stages))` + LDSM 已将它降到 **483**，零 spill；M4096 约快0.45%、16K 约快0.42%，已保留。BK64 fallback 必须用 SW32 atom；不要重做 conflict 消除。
- FlashInfer CuTeDSL 四 shape 比 native 慢9%–15%，不移植。A-only TMA 与 v1 pack 不兼容（rel_l2=1.30）；producer coalesced 写回无收益（46.81→47.01us）。
- 16K NCU 排除 wave 尾部和 L2（两边37.24 waves、cute L2 95.3%）；剩余差距在 shared 装载（16.4×指令）和 TMA/mbarrier sleeping/barrier。删除3条 load-side `fence_view_async_shared` 仅±0.2%，已保留为等价简化，但没有缩小大 shape 差距。
- 三条新主循环方案均失败并回退：A/SFA前移±0.4%噪声；A/SFA双缓冲SASS过闸（R221/224、零spill）但全矩阵慢2.4%–6.1%；BK64 S5/S6/S7在16K慢26%–27%。不要再重复这三条。
- G2R 的 v1 pack 映射已实证：identity smooth 下 producer 的 v1 qact 与 native 逐字节一致，ascales 值集合一致；不要让 producer 再写 standard row-major A/SFA。
- persistent 在当前 hybrid 上按 ffpa grid-stride + epilogue_done 重测：94 tests通过，但12格慢3.0%–6.7%；light producer 288T 与 warpgroup 协议不兼容（setmaxnreg死锁/去后STACK264、慢150%–198%）；Non-WS 256T 慢1.6%–4.3%。均已回退。
- LoRA-first + 首K tile A/SFA 预取已保留：≤4K rank32/64/128 受益1%–2.6%，8K rank128/64 受益，16K rank32/64 回退约1.4%；长序列建议 native。wait 内移和 BK64/S3 无收益，未保留。
- 16K 已增加 `kLoRAFirst=false` 的 main-first 专用实例：local load/store sectors 归零，16K 相对 LoRA-first 快1.5%，2/4/8K在±0.5%内。剩余差距仍是 long_scoreboard/sleeping；下一步全 G2R 专用 kernel 或更大 tile/more-warp 结构，统一数据见 §17/§19/§21/§22。
- FLUX e2e（GPU4、1024²、28 steps、`--compile`、单次）native 6.06s / cute 6.19s / BF16 base 16.99s，即 2.80x/2.74x 于 base，cute/native=1.021。**SVDQ 路径出图不可复现**：同配置两次运行仅 34–36dB，与 native↔cute 的 33.9dB 同量级，故 e2e PSNR 不能判别 native/cute 数值差异，只能判别量化与否；kernel 等价性看 parity 用例。见 §24。
- **PRO 6000（SM120）四方对比**（native / cute+native D / cute+row_major D / FlashInfer cute-dsl）：`cute`+`row_major` D 在 kernel-only 15 格里 14 格最快（相对 native −0.5%~−10.3%），e2e 在 rank≤64 的 8/8 格最优（−3.7%~−12.6%）；rank128 时 CuTe producer 退化 22%–45%，e2e 应回退 `cute`+native D 或 native；FlashInfer cute-dsl 在 ≤8K 慢 5%–33%，只在 16K rank32 快 1.9%。数据与 dispatch 规则见 §25，未经 PRO 5000/H800 复测前不要写成默认。
- **PRO 5000 rank128 专项（2026-11-09）**：统一 `W4A4QuantizeKernel` 后，rank32 多数快 2%–16%、rank64 16K快14%/8K持平但2K–4K慢5%–6%、rank128 16K持平但2K/4K/8K慢10%–14.5%、长K慢16%。保留：rank128 D双缓冲shared归约、小M rank split2、movmatrix B装载、按M的warp/buffer分派。已证伪：16-row CTA、16-warp分组、强制2CTA、B/A swizzle、rank64 split4。完整矩阵与阈值理由见 references/svdq-nvfp4-cute.md §26，未完成全格≤1.00。
- `quantize` 占 e2e 的 14%–31%，且随 rank 上升；只用 kernel-only 的数字会高估收益。两套 producer 只能吃各自的权重/平滑张量（`row_major` 用 `_cute_proj_down`/`_cute_smooth`，`native` 用 fragment-packed 的 `proj_down`/`smooth_factor`），混用是静默数值损坏。
- e2e 最终目标是完整 CuTe 工作流相对旧 native 稳定降低至少 5% 延迟。

详细数据和历史负结果见参考文档。修改性能结论时同步更新本节的摘要和 RFC 状态。
