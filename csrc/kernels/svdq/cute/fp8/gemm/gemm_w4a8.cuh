#pragma once

#include <cstdint>
#include <unordered_map>
#include <unordered_set>

#include "gemm_w8a8.cuh"

namespace svdq::cute_kernels {

using namespace cute;

// W4A8 differs from W8A8 only on the residual-weight path. Activation A, both scale vectors, the
// BF16 LoRA branch and the output tile retain the W8A8 layouts and synchronization. The B source
// is packed E2M1 [N,K/2]; TMA 16U4 expands it into 8-bit smem containers before the SU4 LDSM.
template <int kBN = 128, int kStages = 2, int kBK = 128>
using W4A8CuteTraits = W8A8CuteTraits<kBN, kStages, kBK, true>;

/**
 * Launch one fused W4A8 + BF16 LoRA GEMM on the current CUDA stream.
 *
 * :param act: E4M3 activation codes [M,K].
 * :param wgt: Packed E2M1 residual codes [N,K/2], even-k in the low nibble.
 * :param out: BF16 output [M,N].
 * :param ascales: FP32 per-token activation scales [M].
 * :param wscales: FP32 per-output-channel residual scales [N].
 * :param lora_act: Optional BF16 D [M,rank].
 * :param lora_up: Optional BF16 L1 [N,rank].
 * :param bias: Optional BF16 bias [N].
 * :param M/N/K: Logical GEMM extents; K refers to unpacked elements, not packed bytes.
 * :param rank: LoRA rank, 0/32/64/128.
 * :param stream: Launch stream.
 */
template <int kBN = 128, int kStages = 2, int kBK = 128>
void launch_gemm_w4a8(void const *act, void const *wgt, void *out, float const *ascales,
                      float const *wscales, cutlass::bfloat16_t const *lora_act,
                      cutlass::bfloat16_t const *lora_up, cutlass::bfloat16_t const *bias, int M,
                      int N, int K, int rank, cudaStream_t stream) {
  using Traits = W4A8CuteTraits<kBN, kStages, kBK>;
  using ElementAB = typename Traits::ElementAB;
  using ElementB = cutlass::detail::float_e2m1_unpacksmem_t;
  using ElementO = typename Traits::ElementO;
  constexpr int kBM = Traits::kBM;

  auto mA = make_tensor(make_gmem_ptr(static_cast<ElementAB const *>(act)), make_shape(M, K),
                        make_stride(K, _1{}));
  auto mB = make_tensor(make_gmem_ptr(recast_ptr<ElementB const>(wgt)), make_shape(N, K),
                        make_stride(K, _1{}));
  auto mC = make_tensor(make_gmem_ptr(static_cast<ElementO *>(out)), make_shape(M, N),
                        make_stride(N, _1{}));
  auto build_params = [&]() {
    auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, mA, typename Traits::SmemLayoutA{}(_, _, Int<0>{}),
                               Shape<Int<kBM>, Int<kBK>>{}, _1{});
    auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, mB, typename Traits::SmemLayoutB{}(_, _, Int<0>{}),
                               Shape<Int<kBN>, Int<kBK>>{}, _1{});
    auto tma_c = make_tma_copy(SM90_TMA_STORE{}, mC, typename Traits::SmemLayoutO{},
                               Shape<Int<kBM>, Int<kBN>>{}, _1{});
    using Params = W8A8CuteTma<decltype(tma_a), decltype(tma_b), decltype(tma_c)>;
    return Params{tma_a, tma_b, tma_c};
  };
  using TmaParams = decltype(build_params());
  auto address = [](const void *ptr) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
  };
  W8A8CuteCacheKey key{{address(act), address(wgt), address(out), address(lora_act),
                        address(lora_up), static_cast<uint64_t>(M) << 32 | static_cast<uint32_t>(N),
                        static_cast<uint64_t>(K) << 32 | static_cast<uint32_t>(rank), 1}};
  thread_local std::unordered_map<W8A8CuteCacheKey, TmaParams, W8A8CuteCacheHash> cache;
  auto it = cache.find(key);
  if (it == cache.end()) {
    if (cache.size() >= 32) cache.clear();
    it = cache.emplace(key, build_params()).first;
  }

  auto kernel = gemm_w8a8_kernel<Traits, TmaParams>;
  static thread_local std::unordered_set<int> configured_devices;
  int device = 0;
  checkCUDA(cudaGetDevice(&device));
  if (configured_devices.insert(device).second) {
    checkCUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   Traits::kSmemBytes));
  }
  kernel<<<dim3(N / kBN, M / kBM), Traits::kNumThreads + 128, Traits::kSmemBytes, stream>>>(
    it->second, ascales, wscales, lora_act, lora_up, bias, M, N, K, rank);
}

}  // namespace svdq::cute_kernels
