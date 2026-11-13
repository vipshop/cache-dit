#include <c10/cuda/CUDAException.h>

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include "gemm.cuh"

namespace cache_dit::fp8 {
namespace {

// Per-row shape dispatch. All configurations use BK=128, two A/B pipeline stages, and M4N2 warps.
// Smaller BM/BN tiles increase CTA-level parallelism for short sequences; larger BN improves weight
// reuse for long sequences. BN=192 is a separate medium-width regime: it reduces accumulator/smem
// pressure relative to BN=256 without paying the full B-reread cost of BN=128. These boundaries are
// empirical SM120 choices, not numerical requirements; every instantiation implements the same ABI.
using DefaultTraits = Fp8GemmTraits<128, 256, 128, 2, 2>;
using BalancedTraits = Fp8GemmTraits<128, 192, 128, 2, 2>;
using SmallMTraits = Fp8GemmTraits<128, 128, 128, 2, 2>;
using TinyMTraits = Fp8GemmTraits<64, 64, 128, 2, 2>;

template <typename Traits, bool kWS>
void launch_impl(const cutlass::float_e4m3_t* a, const cutlass::float_e4m3_t* b,
                 const float* a_scale, const float* b_scale, const __nv_bfloat16* bias,
                 cutlass::bfloat16_t* out, int M, int N, int K, cudaStream_t stream) {
  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  constexpr int kBM = Traits::kBM;
  constexpr int kBN = Traits::kBN;
  constexpr int kBK = Traits::kBK;

  auto mA = make_tensor(make_gmem_ptr(a), make_shape(M, K), make_stride(K, _1{}));
  auto mB = make_tensor(make_gmem_ptr(b), make_shape(N, K), make_stride(K, _1{}));
  auto mC = make_tensor(make_gmem_ptr(out), make_shape(M, N), make_stride(N, _1{}));
  auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, mA, SmemLayoutA{}, Shape<Int<kBM>, Int<kBK>>{}, _1{});
  auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, mB, SmemLayoutB{}, Shape<Int<kBN>, Int<kBK>>{}, _1{});
  auto tma_c =
    make_tma_copy(SM90_TMA_STORE{}, mC, SmemLayoutO{}, Shape<Int<kBM>, Int<kBN>>{}, _1{});
  const dim3 grid((N + kBN - 1) / kBN, (M + kBM - 1) / kBM);

  if constexpr (kWS) {
    auto kernel =
      fp8_gemm_scale_ws_kernel<Traits, decltype(tma_a), decltype(tma_b), decltype(tma_c), false>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Traits::kSmemBytes));
    kernel<<<grid, Traits::kNumThreads + 128, Traits::kSmemBytes, stream>>>(
      tma_a, tma_b, tma_c, a_scale, b_scale, bias, M, N, K);
  } else {
    auto kernel =
      fp8_gemm_scale_kernel<Traits, decltype(tma_a), decltype(tma_b), decltype(tma_c), false>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Traits::kSmemBytes));
    kernel<<<grid, Traits::kNumThreads, Traits::kSmemBytes, stream>>>(tma_a, tma_b, tma_c, a_scale,
                                                                      b_scale, bias, M, N, K);
  }
}

}  // namespace

void launch_fp8_gemm_per_row(const void* a, const void* b, const float* a_scale,
                             const float* b_scale, const void* bias, void* out, int M, int N, int K,
                             bool use_ws, cudaStream_t stream) {
  const auto* a_ptr = static_cast<const cutlass::float_e4m3_t*>(a);
  const auto* b_ptr = static_cast<const cutlass::float_e4m3_t*>(b);
  const auto* bias_ptr = static_cast<const __nv_bfloat16*>(bias);
  auto* out_ptr = static_cast<cutlass::bfloat16_t*>(out);
  auto launch = [&](auto traits_tag) {
    using Traits = decltype(traits_tag);
    if (use_ws) {
      launch_impl<Traits, true>(a_ptr, b_ptr, a_scale, b_scale, bias_ptr, out_ptr, M, N, K, stream);
    } else {
      launch_impl<Traits, false>(a_ptr, b_ptr, a_scale, b_scale, bias_ptr, out_ptr, M, N, K,
                                 stream);
    }
  };
  if (M <= 128) {
    launch(TinyMTraits{});
  } else if (M <= 512 || (M <= 1024 && N >= 4096)) {
    launch(SmallMTraits{});
  } else if (M > 2048 && M <= 4096 && N < 4096) {
    launch(BalancedTraits{});
  } else {
    launch(DefaultTraits{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::fp8
