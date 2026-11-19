#pragma once

#include <cuda_runtime.h>

namespace cache_dit::fp8 {

constexpr int64_t kActivationGelu = 1;
constexpr int64_t kActivationGeluTanh = 2;

// Host launch ABI kept separate from the CUTLASS implementation. A is E4M3
// row-major [M,K], B is E4M3 row-major [N,K] (math B.T), out is BF16 [M,N].
// Per-row scales are FP32 [M]/[N], per-tensor scales are scalars, and per-block
// scales are [M,K/128]/[N/128,K/128]. The Python public wrapper has already
// padded shapes and scale tables before reaching this strict C++ interface.
// Void pointers keep the torch extension entry TU independent of CUTLASS/CuTe;
// the matching definitions in gemm/per_*.cu cast them back to collective types.
void launch_fp8_gemm_per_row(const void* a, const void* b, const float* a_scale,
                             const float* b_scale, const void* bias, void* out, int M, int N, int K,
                             cudaStream_t stream);
void launch_fp8_gemm_per_tensor(const void* a, const void* b, const float* a_scale,
                                const float* b_scale, const void* bias, void* out, int M, int N,
                                int K, cudaStream_t stream);
void launch_fp8_gemm_per_block(const void* a, const void* b, const float* a_scale,
                               const float* b_scale, const void* bias, void* out, int M, int N,
                               int K, int num_k_blocks, cudaStream_t stream);
void launch_fp8_gemm_per_row_activation(const void* a, const void* b, const float* a_scale,
                                        const float* b_scale, const void* bias, void* out, int M,
                                        int N, int K, int64_t activation, cudaStream_t stream);
void launch_fp8_gemm_per_tensor_activation(const void* a, const void* b, const float* a_scale,
                                           const float* b_scale, const void* bias, void* out, int M,
                                           int N, int K, int64_t activation, cudaStream_t stream);
void launch_fp8_gemm_per_block_activation(const void* a, const void* b, const float* a_scale,
                                          const float* b_scale, const void* bias, void* out, int M,
                                          int N, int K, int num_k_blocks, int64_t activation,
                                          cudaStream_t stream);

}  // namespace cache_dit::fp8
