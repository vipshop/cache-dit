---
name: cache-dit-cuda-kernel
description: This skill should be used when developing, building, reviewing, profiling, or optimizing Cache-DiT CUDA/CuTe kernels, especially SVDQ INT4/NVFP4 producers, fused LoRA paths, GEMM, pybind/torch.library integration, and FLUX end-to-end validation. It also routes questions about understanding SVDQuant mathematical principles, from SmoothQuant outlier migration to low-rank residual quantization. It records the native/cute dispatch contract, reproducible ccache builds, measurement discipline, completed SVDQ CuTe rewrite, performance evidence, failed attempts, and future optimization RFC.
---

# Cache-DiT CUDA Kernel 开发与优化

## 使用范围

用本 skill 处理 `csrc/kernels/`、SVDQ Python runtime、CUDA 扩展构建、nsys/NCU profile 和量化模型 e2e 验收。修改 SVDQ NVFP4 CuTe 路径前，先读：

- `references/svdq-nvfp4-cute.md`：当前实现、有效/无效尝试和实测性能。
- `references/rfc-cuda-optimizations.md`：优化门槛、待办、状态更新规则。

### 原理理解路由

任务涉及理解 SVDQ/SVDQuant 原理、SmoothQuant → SVDQuant 推导、异常值迁移、量化误差上界、scale 粒度或低秩残差分解时：

1. 先读 `references/svdquant-mathematics.md` 的 §1–4，建立统一记号并复核完整推导与成立条件。
2. 需要落到当前实现时继续读 §5；评估 TurboQuant、DVD-Quant 是否可借鉴时读 §6–8。
3. 当前 kernel 的性能结论和数值契约仍以 `references/svdq-nvfp4-cute.md` 为准。

配合 `cuda-auto-tune`、`cuda-cpp-kernel`、`cutlass-cpp-kernel` 使用。NCU 优化必须先 profile 后修改，再复采验证。

## 1. 当前架构与命名

SVDQ 调用链：

```text
SVDQW4A4Linear.quantize / forward_quant
  -> src/cache_dit/kernels/_svdquant.py
  -> torch.library schema + fake/CUDA impl
  -> pybind.cpp -> ops.h
  -> svdq::native_kernels 或 svdq::cute_kernels
```

目录与命名契约：

- `csrc/kernels/svdq/natitve/`：旧 native 实现，使用 `svdq::native_kernels`。
- `csrc/kernels/svdq/cute/fp4/`：独立 CuTe 实现，使用 `svdq::cute_kernels`。
- cute 文件和 C++ kernel/host/launch 名不重复携带 `_cute`；Python 的 `svdq_*_cute` 公开包装名可保留以区分公开 runtime。
- `gemm_w4a4.cu/.cuh/.h`：NVFP4 主 GEMM、BF16 LoRA-up 融合和 TMA pipeline。
- `quantize_w4a4.cu`：`quantize_w4a4_fused_lora_kernel`，SVDQ 上半段 activation quant + LoRA-down 融合。
- `common.h`、`Tensor.h`、`torch.h` 的共享类型仍在全局命名空间；host 侧用 `::Tensor` 避免与 `cute::Tensor` 歧义。

公开 `runtime_kernel` 只接受 `None/native/cute`。INT4 使用 native v2；NVFP4 当前默认 native，`cute` 为显式 opt-in，直到 GEMM 和 e2e 性能达标。deprecated `v1/v2` 配置字符串不作为公开 selector。

## 2. 独立 NVFP4 CuTe 工作流

保持以下数据流，不得重新引入旧工作流适配层：

```text
quantize_w4a4_fused_lora_kernel
  -> row-major E2M1 A + CUTLASS 128x4 SFA + row-major BF16 D[M,R]
  -> gemm_w4a4_kernel
  -> A/B/SFA/SFB TMA + NVFP4 MMA + BF16 LoRA-up MMA
```

禁止在 CuTe forward 中调用 native activation quantizer、读取 v1 FP32 LoRA scratch、执行 G2R 旧布局反解或增加 `lora_act_to_bf16`。旧 checkpoint 只能在加载时一次性转换 `qweight/wscales/proj_up/proj_down/smooth_factor`；state dict 保持原 native pack 格式。

当前 producer 使用手写 `mma.sync/ldmatrix` 和 shared staging。未来迁移到 CuTe `TiledMMA/TiledCopy` 时，必须保留现有数学、输出布局和 CTA 内归约契约。

## 3. 可重复构建

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

## 4. 优化工作流

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

## 5. 最小验证集

按改动面选择：

```bash
python -m pytest -q \
  tests/kernels/test_ops_dispatch.py \
  tests/kernels/test_svdquant_cute.py \
  tests/quantization/test_quantize_config.py
```

CUDA 数值改动还需运行 producer、独立 GEMM、高层 Linear 和 INT4 隔离回归。改名时同时检查 Python 私有 pybind 调用、公开 torch.library 名称、fake 实现、CLI、文档和临时 harness。运行 `pre-commit run --files ...`、`git diff --check` 和 `read_lints`。

FLUX 验收使用同一 seed 对比默认 native、显式 cute 和 BF16 baseline。精度结论需说明量化差异；不能仅凭“正常出图”宣称无损。

## 6. 当前性能结论与边界

截至 2026-11-04，SM120、M=4096/N=K=3072/rank=32：

- producer 已基本追平 native，但最终稳态 median 仍略慢约 0.8%；继续保留优化项。
- CuTe GEMM 比 native 慢约 16%，尚未达到“快至少 5%”目标。
- NVFP4 默认 runtime 因而保持 native；不能将 producer 进展描述成完整工作流已达标。
- e2e 最终目标是完整 CuTe 工作流相对旧 native 稳定降低至少 5% 延迟。

详细数据和历史负结果见参考文档。修改性能结论时同步更新本节的摘要和 RFC 状态。
