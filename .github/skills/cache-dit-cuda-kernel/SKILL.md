---
name: cache-dit-cuda-kernel
description: 'This skill should be used when adding, building, reviewing, profiling, or optimizing any Cache-DiT CUDA/CuTe C++ operator family — SVDQ INT4/NVFP4 producers, fused LoRA, GEMM, CuTe rewrite, native/cute dispatch, SmoothQuant outlier migration and low-rank residual quantization — plus the full CUDA -> pybind -> Python -> torch.library -> internal-call chain, porting operators from nunchaku or deepcompressor, reproducible ccache builds, and layered validation.'
argument-hint: 'Describe the operator family or kernel, the target repo/extension, dtypes and architectures, and whether this is a new operator, a port, or a performance change.'
user-invocable: true
---

# Cache-DiT CUDA 算子开发与优化

## 使用范围与路由

本 skill 处理 `csrc/kernels/` 下**所有**算子族（现有 `comm`、`attn`、`svdq`）的接入、移植、构建、profile 与优化。读完本节再动手：

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

配合 `cuda-auto-tune`（NCU 驱动迭代优化）、`cuda-cpp-kernel`（CUDA/PTX/架构细节）、`cutlass-cpp-kernel`（CUTLASS/CuTe C++ 模板）使用；CuTe DSL Python（`@cute.kernel`）走 `cute-dsl-kernel`，本仓不维护 CuTe DSL 算子。NCU 优化必须先 profile 后修改，再复采验证。

**后端政策（硬约束）**：cache-dit 长期只维护 CUDA 算子，不引入 Triton / CuTe DSL 实现，不恢复 `kernels/backend.py` 之类后端选择层。若上游只有 Triton 实现，读算法后用 CUDA 重写。细则与反模式见 `references/operator-migration.md` §1、§9。

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
- cute 文件和 C++ kernel/host/launch 名不重复携带 `_cute`；Python 的 `svdq_*_cute` 公开包装名可保留以区分公开 runtime。
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

- producer 已基本追平 native，但最终稳态 median 仍略慢约 0.8%；继续保留优化项。
- CuTe GEMM 比 native 慢约 16%，尚未达到“快至少 5%”目标。
- stages/tile 扫描已完成：最佳 `128x256/BK128/S2` 仍慢 15.89%；**除非有新 NCU 证据，否则不要再次盲目扫描 stages/smem**。
- direct g2r 原型已回退：native pack 按 MMA fragment 直接排布，CuTe standard row-major pack 不匹配；继续该路线前必须先做 fragment-aligned pack。
- NVFP4 默认 runtime 因而保持 native；不能将 producer 进展描述成完整工作流已达标。
- e2e 最终目标是完整 CuTe 工作流相对旧 native 稳定降低至少 5% 延迟。

详细数据和历史负结果见参考文档。修改性能结论时同步更新本节的摘要和 RFC 状态。
