#pragma once

#include "Tensor.h"

namespace svdq::cute_kernels {

// SM120 CuTe SVDQ W4A8 GEMM with a fused BF16 LoRA-up epilogue.
//   out = BF16((qA @ unpack(qB)^T) * (ascale[m] * wscale[n]) + bias + lora_act @ lora_up^T)
// act is E4M3 [M,K]; wgt is packed E2M1 uint8 [N,K/2] with even-k in the low nibble. M/N/K must be
// multiples of 128. All tensors are contiguous CUDA tensors and rank is 0, 32, 64 or 128.
void gemm_w4a8(Tensor act, Tensor wgt, Tensor out, Tensor ascales, Tensor wscales,
               Tensor lora_act_in, Tensor lora_up, Tensor bias);

}  // namespace svdq::cute_kernels
