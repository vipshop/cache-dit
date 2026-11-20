#pragma once

#include "Tensor.h"

namespace svdq::cute_kernels {

// SM120 CuTe SVDQ W8A8 GEMM with a fused BF16 LoRA-up epilogue.
//
// Mathematical contract:
//   out = BF16((qA @ qB^T) * (ascale[m] * wscale[n]) + bias[n] + lora_act @ lora_up^T)
// Both scales are constant along K and are applied to the E4M3 product only; the LoRA term is
// accumulated afterwards into the same FP32 fragment and the result is rounded to BF16 once.
//
// Physical tensor contract (all tensors are contiguous CUDA tensors):
//   act         [M, K] E4M3 activation codes, row-major
//   wgt         [N, K] E4M3 residual codes, row-major
//   out         [M, N] BF16 row-major
//   ascales     [M] FP32 per-token activation scales
//   wscales     [N] FP32 per-output-channel residual scales
//   lora_act_in [M, R] BF16 LoRA-down output D (invalid when R == 0)
//   lora_up     [N, R] BF16 LoRA-up weight L1 (invalid when R == 0)
//   bias        [N] BF16, optional
// M/N/K must be multiples of 128 and R must be 0, 32, 64 or 128.
void gemm_w8a8(Tensor act, Tensor wgt, Tensor out, Tensor ascales, Tensor wscales,
               Tensor lora_act_in, Tensor lora_up, Tensor bias);

}  // namespace svdq::cute_kernels
