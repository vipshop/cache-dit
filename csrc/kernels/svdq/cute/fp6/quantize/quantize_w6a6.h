#pragma once

#include "Tensor.h"

namespace svdq::cute_kernels {

// Fused SVDQ W6A6 activation producer: MX block-scaled E2M3 quantization plus the BF16 LoRA-down
// projection in one kernel.
//
//   asf[m,g]     = ue8m0(2^ceil(log2(amax_{k in g} |input[m,k] / smooth[k]| / 7.5))),  g = k / 32
//   output[m,k]  = satfinite_e2m3((input[m,k] / smooth[k]) / 2^(asf[m,g]-127))
//   lora_act[m,r] = BF16(sum_k input[m,k] * lora_down[r,k])
//
// `lora_down` is L2/s, so the low-rank branch consumes the raw activation while the quantized
// output consumes the smoothed one. M must be a multiple of 128, K a multiple of 128, and the rank
// must be 0, 32, 64 or 128. All tensors are contiguous CUDA tensors; `lora_down`/`lora_act` are
// invalid exactly when the rank is 0. `lora_scratch` is an optional zero-initialized FP32 [M,rank]
// buffer that selects the split-K schedule; the entry point rounds it into `lora_act` itself.

/// Host-side mirror of the split-K schedule decision; used to allocate the FP32 scratch.
int quantize_w6a6_k_splits(int M, int K);

void quantize_w6a6(Tensor input, Tensor output, Tensor asf, Tensor lora_down, Tensor lora_act,
                   Tensor smooth, Tensor lora_scratch);

}  // namespace svdq::cute_kernels
