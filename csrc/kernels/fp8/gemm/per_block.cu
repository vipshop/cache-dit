#include <c10/cuda/CUDAException.h>

#include "gemm.cuh"

namespace cache_dit::fp8 {

void launch_fp8_gemm_per_block(const void* a, const void* b, const float* a_scale,
                               const float* b_scale, const void* bias, void* out, int M, int N,
                               int K, int num_k_blocks, cudaStream_t stream) {
  fp8_gemm_cutlass_blockwise(a, b, a_scale, b_scale, bias, out, M, N, K, num_k_blocks, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_fp8_gemm_per_block_activation(const void* a, const void* b, const float* a_scale,
                                          const float* b_scale, const void* bias, void* out, int M,
                                          int N, int K, int num_k_blocks, int64_t activation,
                                          cudaStream_t stream) {
  if (activation == kActivationGelu) {
    fp8_gemm_cutlass_blockwise_activation<cutlass::epilogue::thread::GELU>(
      a, b, a_scale, b_scale, bias, out, M, N, K, num_k_blocks, stream);
  } else if (activation == kActivationGeluTanh) {
    fp8_gemm_cutlass_blockwise_activation<cutlass::epilogue::thread::Fp8GeluTanh>(
      a, b, a_scale, b_scale, bias, out, M, N, K, num_k_blocks, stream);
  } else {
    TORCH_CHECK(false, "Unknown FP8 activation id: ", activation);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::fp8
