#include <c10/cuda/CUDAException.h>

#include "gemm.cuh"

namespace cache_dit::fp8 {

void launch_fp8_gemm_per_tensor(const void* a, const void* b, const float* a_scale,
                                const float* b_scale, const void* bias, void* out, int M, int N,
                                int K, cudaStream_t stream) {
  fp8_gemm_cutlass_rowwise<true>(a, b, a_scale, b_scale, bias, out, M, N, K, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::fp8
