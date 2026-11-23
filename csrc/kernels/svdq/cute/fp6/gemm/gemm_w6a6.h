#pragma once

#include "Tensor.h"

namespace svdq::cute_kernels {

// SVDQ W6A6 GEMM: packed E2M3 residual weights and activation, MX block-scaled MMA.
//
//   P[m,n] = sum_g 2^(asf[m,g]-127) * 2^(wsf[n,g]-127) * (qA[m,:] @ qB[n,:]^T),  g = k / 32
//   out    = BF16(P * (ascales[m] * wscales[n]) + bias[n] + D[M,R] @ L1[N,R]^T)
//
// The micro-scales are consumed inside `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X`
// so no blockwise promotion happens in the mainloop. `ascales`/`wscales` are the optional FP32
// second-level scales; they are constant along K and are applied once in the epilogue.
//
// Layouts (all contiguous CUDA tensors):
//   act    [M, K*3/4] uint8, four packed E2M3 codes per three bytes
//   wgt    [N, K*3/4] uint8, same packing
//   asf    [M, K/32]  uint8, UE8M0 (value = 2^(code-127))
//   wsf    [N, K/32]  uint8, UE8M0
//   out    [M, N]     BF16
// M and N must be multiples of 128, K a multiple of 128, and the rank 0/32/64/128.
void gemm_w6a6(Tensor act, Tensor wgt, Tensor out, Tensor ascales, Tensor wscales, Tensor asf,
               Tensor wsf, Tensor lora_act_in, Tensor lora_up, Tensor bias);

}  // namespace svdq::cute_kernels
