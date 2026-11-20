#pragma once

#include "Tensor.h"

namespace svdq::cute_kernels {

// Fused SVDQ-FP8 activation producer: per-token E4M3 quantization plus the BF16 LoRA-down
// projection in one kernel.
//
//   ascales[m] = max(amax_k |input[m,k] / smooth[k]|, eps) / 448
//   output[m,k] = satfinite_e4m3((input[m,k] / smooth[k]) / ascales[m])
//   lora_act[m,r] = BF16(sum_k input[m,k] * lora_down[r,k])
//
// `lora_down` is L2/s, so the low-rank branch consumes the raw activation while the quantized
// output consumes the smoothed one. M must be a multiple of 128, K a multiple of 64, and the rank
// must be 0, 32, 64 or 128. All tensors are contiguous CUDA tensors; `lora_down`/`lora_act` are
// invalid exactly when the rank is 0. `lora_scratch` is an optional zero-initialized FP32 [M,rank]
// buffer that selects the split-K schedule; the entry point rounds it into `lora_act` itself.
/// Host-side mirror of the split-K schedule decision; used to allocate the FP32 scratch.
int quantize_w8a8_k_splits(int M, int K);

void quantize_w8a8(Tensor input, Tensor output, Tensor ascales, Tensor lora_down, Tensor lora_act,
                   Tensor smooth, Tensor lora_scratch);

}  // namespace svdq::cute_kernels
