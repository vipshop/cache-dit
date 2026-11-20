#include "quantize_w8a8.cuh"

namespace svdq::cute_kernels {

// Rank-64 instantiation of the fused activation producer, compiled independently of the other
// ranks.
void quantize_w8a8_rank64(void const *input, void const *smooth, void const *lora_down,
                          void *output, float *ascales, void *lora_act, float *lora_scratch, int M,
                          int K, cudaStream_t stream) {
  launch_quantize_w8a8<64>(input, smooth, lora_down, output, ascales, lora_act, lora_scratch, M, K,
                           stream);
}

}  // namespace svdq::cute_kernels
