#include <c10/cuda/CUDAException.h>

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include "gemm.cuh"

namespace cache_dit::fp8 {
namespace {

template <bool kWS>
void launch_impl(const void* a, const void* b, const float* a_scale, const float* b_scale,
                 const void* bias, void* out, int M, int N, int K, int num_k_blocks,
                 cudaStream_t stream) {
  using Traits = Fp8GemmTraits<128, 128, 128, 2>;
  const auto* a_ptr = static_cast<const cutlass::float_e4m3_t*>(a);
  const auto* b_ptr = static_cast<const cutlass::float_e4m3_t*>(b);
  const auto* bias_ptr = static_cast<const __nv_bfloat16*>(bias);
  auto* out_ptr = static_cast<cutlass::bfloat16_t*>(out);
  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  constexpr int kBM = Traits::kBM;
  constexpr int kBN = Traits::kBN;
  constexpr int kBK = Traits::kBK;

  auto mA = make_tensor(make_gmem_ptr(a_ptr), make_shape(M, K), make_stride(K, _1{}));
  auto mB = make_tensor(make_gmem_ptr(b_ptr), make_shape(N, K), make_stride(K, _1{}));
  auto mC = make_tensor(make_gmem_ptr(out_ptr), make_shape(M, N), make_stride(N, _1{}));
  auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, mA, SmemLayoutA{}, Shape<Int<kBM>, Int<kBK>>{}, _1{});
  auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, mB, SmemLayoutB{}, Shape<Int<kBN>, Int<kBK>>{}, _1{});
  auto tma_c =
    make_tma_copy(SM90_TMA_STORE{}, mC, SmemLayoutO{}, Shape<Int<kBM>, Int<kBN>>{}, _1{});

  const dim3 grid((N + kBN - 1) / kBN, (M + kBM - 1) / kBM);
  if constexpr (kWS) {
    auto kernel =
      fp8_gemm_block_ws_kernel<Traits, decltype(tma_a), decltype(tma_b), decltype(tma_c)>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Traits::kSmemBytes));
    kernel<<<grid, Traits::kNumThreads + 128, Traits::kSmemBytes, stream>>>(
      tma_a, tma_b, tma_c, a_scale, b_scale, bias_ptr, M, N, K, num_k_blocks);
  } else {
    auto kernel = fp8_gemm_block_kernel<Traits, decltype(tma_a), decltype(tma_b), decltype(tma_c)>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Traits::kSmemBytes));
    kernel<<<grid, Traits::kNumThreads, Traits::kSmemBytes, stream>>>(
      tma_a, tma_b, tma_c, a_scale, b_scale, bias_ptr, M, N, K, num_k_blocks);
  }
}

}  // namespace

void launch_fp8_gemm_per_block(const void* a, const void* b, const float* a_scale,
                               const float* b_scale, const void* bias, void* out, int M, int N,
                               int K, int num_k_blocks, bool use_ws, cudaStream_t stream) {
  if (use_ws) {
    launch_impl<true>(a, b, a_scale, b_scale, bias, out, M, N, K, num_k_blocks, stream);
  } else {
    launch_impl<false>(a, b, a_scale, b_scale, bias, out, M, N, K, num_k_blocks, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::fp8
