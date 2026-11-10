#pragma once

#include "Tensor.h"

#include <vector>

namespace svdq::cute_kernels {

// Public entry points for the row-major-LoRA NVFP4 activation producer.
//
// The operator produces two logically independent outputs from the same input tile:
//   1. Activation quantization: (input / smooth) is reduced to per-K32 scales and E2M1 values.
//      `output` and `ascales` use the native 256-row MMA-fragment pack consumed by the CuTe GEMM;
//      they are not ordinary row-major [M,K/2] and [M,K/32] tensors. `output` is exposed as INT8
//      storage and `ascales` as FP8_E4M3 storage, matching the existing native producer bit layout.
//   2. LoRA-down projection: D = input @ lora_down. `lora_down` is contiguous row-major [K,R]
//      BF16, and `lora_act` is contiguous row-major [M,R] BF16. The R16 scale in `lora_scales`
//      is applied after FP32 accumulation and before the final BF16 rounding.
//
// Schedule selection is part of the call contract:
//   - default/invalid `lora_scratch` selects the fused producer, which reduces D inside the CTA;
//   - a valid FP32 `lora_scratch` with at least M*R elements selects split-K. The workspace is
//     cleared and overwritten by the launch, then converted to `lora_act` by a finalize kernel.
//     Split-K uses relaxed FP32 atomics across K CTAs, so D addition order and bitwise
//     reproducibility are not guaranteed; use numerical tolerances when comparing D across
//     schedules or runs.
// All work is issued on the current CUDA stream. Host-side validation failures are reported as
// std::runtime_error before launch.
//
// quantize_r{32,64,128} are separate translation-unit entry points. Each instantiates only its
// corresponding rank in quantize_impl.cuh, reducing template compile time and keeping rank-specific
// tuning independent. quantize_uni is the general implementation, including rank == 0 and the
// non-specialized ranks. The actual ops-level dispatch may still choose quantize_uni for a
// specialized rank when that implementation is faster for a particular shape.

/// Return whether a rank has a dedicated quantize_r{32,64,128} implementation.
/// This only reports implementation availability; it does not prescribe ops-level dispatch.
constexpr bool is_specialized_rank(int rank) { return rank == 32 || rank == 64 || rank == 128; }

/// General producer. Supports rank == 0 and all rank values accepted by quantize_uni.cu.
/// For rank == 0, LoRA tensors are not required and only qact/ascales are produced.
void quantize_uni(svdq::Tensor input, svdq::Tensor output, svdq::Tensor ascales,
                  svdq::Tensor lora_down, svdq::Tensor lora_act, svdq::Tensor smooth,
                  std::vector<float> lora_scales, svdq::Tensor lora_scratch = {});

/// Rank-32 CuTe TiledMMA/TiledCopy producer. Requires two values in `lora_scales` (R32/R16).
void quantize_r32(svdq::Tensor input, svdq::Tensor output, svdq::Tensor ascales,
                  svdq::Tensor lora_down, svdq::Tensor lora_act, svdq::Tensor smooth,
                  std::vector<float> lora_scales, svdq::Tensor lora_scratch = {});

/// Rank-64 CuTe TiledMMA/TiledCopy producer. Requires four values in `lora_scales` (R64/R16).
void quantize_r64(svdq::Tensor input, svdq::Tensor output, svdq::Tensor ascales,
                  svdq::Tensor lora_down, svdq::Tensor lora_act, svdq::Tensor smooth,
                  std::vector<float> lora_scales, svdq::Tensor lora_scratch = {});

/// Rank-128 CuTe TiledMMA/TiledCopy producer. Requires eight values in `lora_scales` (R128/R16).
void quantize_r128(svdq::Tensor input, svdq::Tensor output, svdq::Tensor ascales,
                   svdq::Tensor lora_down, svdq::Tensor lora_act, svdq::Tensor smooth,
                   std::vector<float> lora_scales, svdq::Tensor lora_scratch = {});

}  // namespace svdq::cute_kernels
