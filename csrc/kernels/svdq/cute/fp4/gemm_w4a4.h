#pragma once

#include "Tensor.h"

#include <vector>

namespace svdq::cute_kernels {

// SM120 CuTe NVFP4 (W4A4) GEMM with a fused BF16 LoRA-up epilogue.
//
// Mathematical contract:
//   P = dequant_fp4(act, ascales) @ dequant_fp4(wgt, wscales)^T
//   out = BF16(alpha * P * wcscales + bias + lora_act_in @ lora_up^T)
// Every 16 K elements share one FP8_E4M3 scale. SF means Scale Factor: SFA/ascales belong to A,
// SFB/wscales belong to B. D is the LoRA-down output and L1 is the LoRA-up weight (not L1 cache).
// alpha/wcscales apply only to P, never to the LoRA term. Both products use FP32 accumulation and
// the result is rounded to BF16 exactly once.
//
// Physical tensor contract (all tensors are contiguous CUDA tensors):
//   act            [M, K/2] v1 256-row interleaved E2M1 activation pack, low nibble is even K
//   wgt            [N, K/2] row-major packed E2M1 bytes, same nibble order
//   ascales        FP8_E4M3 activation scales in the matching v1 interleaved pack
//   wscales        FP8_E4M3 flat buffer of CUTLASS 128x4 512B atoms for wgt (N rows)
//   out            [M, N] row-major BF16
//   lora_act_in    [M, R] row-major BF16 D; per-rank scales are already folded in
//   lora_up        [N, R] row-major BF16 L1
//   bias           [N] BF16, optional (an invalid tensor means zero)
//   wcscales       [N] BF16, optional (an invalid tensor means one)
//
// M must be a multiple of 256 for the v1 activation pack; N must be a multiple of 128 for the
// CUTLASS weight SF atom; K must be a multiple of 64. R is zero or a multiple of 16, and the
// default BN=256/BK=128 configuration additionally requires R to be a multiple of 32.
//
// The kernel is warp-specialized: a 128-thread producer warpgroup feeds B/SFB/D/L1 through TMA,
// while activation A/SFA are loaded gmem->register (G2R) and 256 consumer threads execute the
// NVFP4/BF16 MMAs and epilogue. D/L1 reuse the A/B stage storage through swizzle-free TMA layouts.
//
// BF16 is the only LoRA/output dtype implemented by this CuTe path. FP16 LoRA remains on the
// old native path and is rejected by the public module rather than silently converted.
void gemm_w4a4_f32d(::Tensor act, ::Tensor wgt, ::Tensor out, ::Tensor ascales, ::Tensor wscales,
                    ::Tensor lora_act_in, ::Tensor lora_up, ::Tensor bias, ::Tensor wcscales,
                    std::vector<float> lora_scales, float alpha);

void gemm_w4a4(::Tensor act, ::Tensor wgt, ::Tensor out, ::Tensor ascales, ::Tensor wscales,
               ::Tensor lora_act_in, ::Tensor lora_up, ::Tensor bias, ::Tensor wcscales,
               float alpha);

// Produce v1 interleaved E2M1 activation, matching v1 SFA and row-major BF16 LoRA-down D
// directly. No FP32 LoRA scratch, finalize kernel or activation repack is involved. The producer
// completes its full K reduction in warp registers and emits the GEMM's D TMA input in one launch.
void quantize_w4a4(::Tensor input, ::Tensor output, ::Tensor ascales, ::Tensor lora_down,
                   ::Tensor lora_act, ::Tensor smooth, std::vector<float> lora_scales);

}  // namespace svdq::cute_kernels
