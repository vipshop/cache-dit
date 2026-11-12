#pragma once

#include <cuda_runtime.h>

namespace cache_dit::fp8 {

// Host launch declarations use void pointers so the torch extension entry TU
// does not need the CUTLASS/CuTe include tree.
void launch_fp8_gemm_per_row(const void* a, const void* b, const float* a_scale,
                             const float* b_scale, const void* bias, void* out, int M, int N, int K,
                             bool use_ws, cudaStream_t stream);
void launch_fp8_gemm_per_tensor(const void* a, const void* b, const float* a_scale,
                                const float* b_scale, const void* bias, void* out, int M, int N,
                                int K, bool use_ws, cudaStream_t stream);
void launch_fp8_gemm_per_block(const void* a, const void* b, const float* a_scale,
                               const float* b_scale, const void* bias, void* out, int M, int N,
                               int K, int num_k_blocks, bool use_ws, cudaStream_t stream);

}  // namespace cache_dit::fp8
