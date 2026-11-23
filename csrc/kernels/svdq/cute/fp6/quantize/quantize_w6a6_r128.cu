#include "quantize_w6a6.cuh"

namespace svdq::cute_kernels {

// Rank-128 instantiation of the fused W6A6 producer, compiled independently of the other ranks.
void quantize_w6a6_rank128(void const *input, void const *smooth, void const *lora_down,
                           void *output, void *asf, void *lora_act, float *lora_scratch, int M,
                           int K, cudaStream_t stream) {
  launch_quantize_w6a6<128>(input, smooth, lora_down, output, asf, lora_act, lora_scratch, M, K,
                            stream);
}

}  // namespace svdq::cute_kernels
