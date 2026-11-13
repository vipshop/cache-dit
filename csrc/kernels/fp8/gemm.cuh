#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/device_kernel.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <cstdint>

#include "gemm.h"

namespace cache_dit::fp8 {

using namespace cute;

// Reading guide for the scale kernels in this header:
//   A      = E4M3 activation [M,K], B = E4M3 weight [N,K], C = BF16 output [M,N].
//   a_scale = per-row dequantization scales [M], b_scale = per-output-channel scales [N].
//   mX/sX/tCrX = global-memory, shared-memory, and per-thread register tensors, respectively.
//   TMA    = asynchronous bulk copy engine used for A/B loads and the final C store.
//   full/empty = the transaction/CTA barriers that publish and recycle each pipeline stage.
//   stage  = physical ring slot; kt = logical K tile; phase = mbarrier phase for that slot.
//
// Mathematical contract of fp8_gemm_scale_kernel and fp8_gemm_scale_ws_kernel:
//
//   P[m,n] = sum_k E4M3(A[m,k]) * E4M3(B[n,k])             (FP32 accumulation)
//   C[m,n] = BF16(P[m,n] * a_scale[m] * b_scale[n] + bias[n])
//
// The activation/weight scales are applied once in the epilogue, not inside the K loop. This keeps
// the SM89 FP8 MMA atom on a single FP32 accumulator and fixes K accumulation order for a given
// tile. kPerTensor selects the scalar variant P * a_scale[0] * b_scale[0]; per-block scaling has a
// different K-coupled contract and is implemented by fp8_gemm_block_kernel below, not by this path.
//
// Tile and memory contract:
//   * One CTA owns a [kBM,kBN] output tile. kBK=128 FP8 elements = 128 B per A/B row, matching the
//     SW128 shared-memory atom. M4N2 (kWarpsN_=2) divides the same warps across M and N so B
//     fragments are reused by more than one M warp without changing the output coordinates.
//   * TMA descriptors retain the logical [M,K]/[N,K]/[M,N] extents and perform boundary handling.
//     Register epilogue loads additionally guard m<M/n<N; tail CTAs must not fetch a scale or bias
//     through an out-of-range coordinate even though their accumulator storage exists.
//   * A/B stages occupy kStages * (kAElems+kBElems) bytes. After the mainloop, the same allocation
//     is reinterpreted as BF16 output staging. This aliasing is legal only because all consumer
//     threads finish their K work and synchronize before the first STSM output write.
//   * The epilogue performs scale/bias in FP32, converts to BF16 once, uses STSM to produce the
//     swizzled smem tile, and lets one thread issue the TMA store. A direct register-to-global
//     scatter is intentionally not used: it was both slower and less coalesced on the target SM120.
//
// Non-WS and WS share this numerical contract but schedule differently:
//   * non-WS: one thread issues TMA for kt+kStages while all threads consume kt. `gemm_ss`
//   prefetches
//     the next SMEM->RF fragment before invoking the next MMA, overlapping copies and tensor math.
//   * WS: 128 producer threads run at a reduced register allocation; only producer lane zero issues
//     TMA. Consumer warps wait on full[], execute MMA, and publish empty[] for the producer.
//     Register reallocation and the consumer-only named barrier replace the non-WS CTA-wide
//     epilogue sync.
// Both variants must preserve bit-identical output for identical traits, inputs, scales, and bias.

template <typename Layout>
CUTE_DEVICE auto convert_layout_acc_rowcol(Layout acc_layout) {
  auto divided = logical_divide(acc_layout, Shape<_2>{});
  return make_layout(make_layout(get<0, 1>(divided), get<1>(divided)),
                     make_layout(get<0, 0>(divided), get<2>(divided)));
}

template <typename To, typename Engine, typename Layout>
CUTE_DEVICE auto convert_type(Tensor<Engine, Layout> const& tensor) {
  using From = typename Engine::value_type;
  constexpr int kElements = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To, From, kElements> convert;
  auto fragment = convert(*reinterpret_cast<cutlass::Array<From, kElements> const*>(tensor.data()));
  return make_tensor(make_rmem_ptr<To>(&fragment), tensor.layout());
}

template <typename TensorC, typename TensorA, typename TensorB, typename TensorSA,
          typename TensorSB, typename TiledMma, typename TiledCopyA, typename TiledCopyB,
          typename ThreadCopyA, typename ThreadCopyB>
CUTE_DEVICE void gemm_ss(TensorC& acc, TensorA& fragment_a, TensorB& fragment_b,
                         TensorSA const& shared_a, TensorSB const& shared_b, TiledMma tiled_mma,
                         TiledCopyA tiled_copy_a, TiledCopyB tiled_copy_b,
                         ThreadCopyA thread_copy_a, ThreadCopyB thread_copy_b) {
  auto copy_view_a = thread_copy_a.retile_D(fragment_a);
  auto copy_view_b = thread_copy_b.retile_D(fragment_b);
  copy(tiled_copy_a, shared_a(_, _, _0{}), copy_view_a(_, _, _0{}));
  copy(tiled_copy_b, shared_b(_, _, _0{}), copy_view_b(_, _, _0{}));
#pragma unroll
  for (int tile_k = 0; tile_k < size<2>(fragment_a); ++tile_k) {
    if (tile_k + 1 < size<2>(fragment_a)) {
      copy(tiled_copy_a, shared_a(_, _, tile_k + 1), copy_view_a(_, _, tile_k + 1));
      copy(tiled_copy_b, shared_b(_, _, tile_k + 1), copy_view_b(_, _, tile_k + 1));
    }
    gemm(tiled_mma, fragment_a(_, _, tile_k), fragment_b(_, _, tile_k), acc);
  }
}

template <int kBM_ = 128, int kBN_ = 256, int kBK_ = 128, int kStages_ = 2, int kWarpsN_ = 1>
struct Fp8GemmTraits {
  static_assert(kBK_ == 128, "SW128 atom requires 128B rows (1B elements)");
  static_assert(kBM_ % 16 == 0, "BM must be a multiple of MMA_M");
  static_assert(kBN_ % 64 == 0, "BN must cover a full SW128 bf16 atom");
  static_assert(kStages_ >= 2, "the epilogue reuses the A+B pipeline storage");
  static constexpr int kBM = kBM_;
  static constexpr int kBN = kBN_;
  static constexpr int kBK = kBK_;
  static constexpr int kStages = kStages_;
  static constexpr int kNumWarps = kBM / 16;
  static_assert(kWarpsN_ >= 1 && kNumWarps % kWarpsN_ == 0,
                "the N warp factor must divide the total warp count");
  static constexpr int kNumWarpsM = kNumWarps / kWarpsN_;
  static constexpr int kNumThreads = kNumWarps * 32;

  using Element = cutlass::float_e4m3_t;
  using ElementO = cutlass::bfloat16_t;
  using SmemAtom = GMMA::Layout_K_SW128_Atom<Element>;
  using SmemLayoutA = decltype(tile_to_shape(SmemAtom{}, Shape<Int<kBM>, Int<kBK>>{}));
  using SmemLayoutB = decltype(tile_to_shape(SmemAtom{}, Shape<Int<kBN>, Int<kBK>>{}));
  using SmemAtomO = GMMA::Layout_K_SW128_Atom<ElementO>;
  using SmemLayoutO = decltype(tile_to_shape(SmemAtomO{}, Shape<Int<kBM>, Int<kBN>>{}));
  static constexpr int kSmemBytes =
    kStages * (cosize(SmemLayoutA{}) + cosize(SmemLayoutB{})) * sizeof(Element);
  static_assert(cosize(SmemLayoutO{}) * sizeof(ElementO) <= kSmemBytes,
                "the epilogue staging buffer must fit in the freed A+B pipeline storage");

  using MmaAtom = MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>;
  using TiledMma =
    decltype(make_tiled_mma(MmaAtom{}, Layout<Shape<Int<kNumWarpsM>, Int<kWarpsN_>, _1>>{},
                            Tile<Int<kBM>, Int<kBN>, _32>{}));
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
};

// Non-WS execution flow:
//   1. Build TMA tensor views, allocate FP32 accumulators, and initialize full/empty barriers.
//   2. Prime at most kStages K tiles. Each full barrier tracks the exact A+B transaction bytes.
//   3. For every kt, wait for its full stage, run SMEM->RF copies and FP8 MMA, then mark the stage
//      empty. Thread zero may refill kt+kStages only after all consumer threads released that slot.
//   4. After a CTA-wide sync, map accumulator coordinates to logical rows/columns, load guarded
//      row/column scales, and apply scale+bias in FP32.
//   5. Convert once to BF16, reuse the now-idle A/B smem through STSM, synchronize again, and issue
//      the single-thread TMA store. The final store wait prevents returning with smem still in use.
template <typename Traits, typename TmaA, typename TmaB, typename TmaC, bool kPerTensor>
__global__ void __launch_bounds__(Traits::kNumThreads, 1)
  fp8_gemm_scale_kernel(CUTLASS_GRID_CONSTANT TmaA const tma_a,
                        CUTLASS_GRID_CONSTANT TmaB const tma_b,
                        CUTLASS_GRID_CONSTANT TmaC const tma_c, const float* __restrict__ a_scale,
                        const float* __restrict__ b_scale, const __nv_bfloat16* __restrict__ bias,
                        int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
  using CtaBarrier = cutlass::arch::ClusterBarrier;
  using Element = typename Traits::Element;
  using ElementO = typename Traits::ElementO;
  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using TiledMma = typename Traits::TiledMma;
  using SmemCopyAtom = typename Traits::SmemCopyAtom;

  constexpr int kBM = Traits::kBM;
  constexpr int kBN = Traits::kBN;
  constexpr int kBK = Traits::kBK;
  constexpr int kStages = Traits::kStages;
  constexpr int kNumThreads = Traits::kNumThreads;
  constexpr int kAElems = cosize(SmemLayoutA{});
  constexpr int kBElems = cosize(SmemLayoutB{});

  const int tid = threadIdx.x;
  const int m0 = blockIdx.y * kBM;
  const int n0 = blockIdx.x * kBN;

  auto mA = tma_a.get_tma_tensor(make_shape(M, K));
  auto mB = tma_b.get_tma_tensor(make_shape(N, K));
  auto mC = tma_c.get_tma_tensor(make_shape(M, N));
  auto a_slice = tma_a.get_slice(_0{});
  auto b_slice = tma_b.get_slice(_0{});
  auto c_slice = tma_c.get_slice(_0{});

  extern __shared__ __align__(1024) Element smem[];
  Element* a_base = smem;
  Element* b_base = smem + kStages * kAElems;

  __shared__ uint64_t full[kStages];
  __shared__ uint64_t empty[kStages];
  if (tid == 0) {
    for (int stage = 0; stage < kStages; ++stage) {
      TmaBarrier::init(&full[stage], 1);
      CtaBarrier::init(&empty[stage], kNumThreads);
    }
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto s2r_copy_a = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto s2r_copy_b = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto s2r_thr_a = s2r_copy_a.get_thread_slice(tid);
  auto s2r_thr_b = s2r_copy_b.get_thread_slice(tid);

  auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<kBM>, Int<kBN>>{});
  clear(tCrC);
  auto cC = make_identity_tensor(Shape<Int<kBM>, Int<kBN>>{});
  auto tScC = thr_mma.partition_C(cC);
  auto tScC_rc = make_tensor(tScC.data(), convert_layout_acc_rowcol(tScC.layout()));
  constexpr int kCRows = decltype(size<0>(tScC_rc))::value;
  constexpr int kCCols = decltype(size<1>(tScC_rc))::value;

  auto issue_tma = [&](int kt, int stage) {
    cutlass::arch::fence_view_async_shared();
    auto sA = make_tensor(make_smem_ptr(a_base + stage * kAElems), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(b_base + stage * kBElems), SmemLayoutB{});
    auto gA = local_tile(mA, Shape<Int<kBM>, Int<kBK>>{}, make_coord(blockIdx.y, kt));
    auto gB = local_tile(mB, Shape<Int<kBN>, Int<kBK>>{}, make_coord(blockIdx.x, kt));
    TmaBarrier::arrive_and_expect_tx(&full[stage], sizeof(Element) * (size(sA) + size(sB)));
    copy(tma_a.with(full[stage]), a_slice.partition_S(gA), a_slice.partition_D(sA));
    copy(tma_b.with(full[stage]), b_slice.partition_S(gB), b_slice.partition_D(sB));
  };

  const int nKt = (K + kBK - 1) / kBK;
  for (int stage = 0; stage < kStages; ++stage) {
    CtaBarrier::arrive(&empty[stage]);
  }
  if (tid == 0) {
    for (int kt = 0; kt < kStages && kt < nKt; ++kt) {
      CtaBarrier::wait(&empty[kt], 0);
      issue_tma(kt, kt);
    }
  }

#pragma unroll 1
  for (int kt = 0; kt < nKt; ++kt) {
    const int stage = kt % kStages;
    const int phase = (kt / kStages) & 1;
    TmaBarrier::wait(&full[stage], phase);
    cutlass::arch::fence_view_async_shared();

    auto sA = make_tensor(make_smem_ptr(a_base + stage * kAElems), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(b_base + stage * kBElems), SmemLayoutB{});
    auto tCrA = thr_mma.partition_fragment_A(sA);
    auto tCrB = thr_mma.partition_fragment_B(sB);
    auto tAsA = s2r_thr_a.partition_S(sA);
    auto tBsB = s2r_thr_b.partition_S(sB);
    gemm_ss(tCrC, tCrA, tCrB, tAsA, tBsB, tiled_mma, s2r_copy_a, s2r_copy_b, s2r_thr_a, s2r_thr_b);

    CtaBarrier::arrive(&empty[stage]);
    if (tid == 0) {
      const int kt_next = kt + kStages;
      if (kt_next < nKt) {
        const int s_next = kt_next % kStages;
        const int ph_next = (kt_next / kStages) & 1;
        CtaBarrier::wait(&empty[s_next], ph_next);
        issue_tma(kt_next, s_next);
      }
    }
  }

  __syncthreads();
  auto tCrC_rc = make_tensor(tCrC.data(), convert_layout_acc_rowcol(tCrC.layout()));
  float row_scale[kCRows];
  float col_scale[kCCols];
  const float tensor_scale = kPerTensor ? a_scale[0] * b_scale[0] : 1.0f;
#pragma unroll
  for (int r = 0; r < kCRows; ++r) {
    const int m = m0 + get<0>(tScC_rc(r, 0));
    row_scale[r] = kPerTensor ? tensor_scale : (m < M ? a_scale[m] : 0.0f);
  }
#pragma unroll
  for (int c = 0; c < kCCols; ++c) {
    const int n = n0 + get<1>(tScC_rc(0, c));
    col_scale[c] = kPerTensor ? 1.0f : (n < N ? b_scale[n] : 0.0f);
  }
#pragma unroll
  for (int r = 0; r < kCRows; ++r) {
#pragma unroll
    for (int c = 0; c < kCCols; ++c) {
      const int n = n0 + get<1>(tScC_rc(r, c));
      const float bias_value = (bias != nullptr && n < N) ? __bfloat162float(bias[n]) : 0.0f;
      tCrC_rc(r, c) = tCrC_rc(r, c) * row_scale[r] * col_scale[c] + bias_value;
    }
  }

  auto tCrCh = convert_type<ElementO>(tCrC);
  auto r2s_copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, ElementO>{}, tiled_mma);
  auto r2s_thr = r2s_copy.get_thread_slice(tid);
  auto sC = make_tensor(make_smem_ptr(reinterpret_cast<ElementO*>(smem)), SmemLayoutO{});
  auto tCrCh_src = r2s_thr.retile_S(tCrCh);
  auto tCsC_dst = r2s_thr.partition_D(sC);
  copy(r2s_copy, tCrCh_src, tCsC_dst);
  cutlass::arch::fence_view_async_shared();
  __syncthreads();
  auto gC = local_tile(mC, Shape<Int<kBM>, Int<kBN>>{}, make_coord(blockIdx.y, blockIdx.x));
  if (tid == 0) {
    copy(tma_c, c_slice.partition_S(sC), c_slice.partition_D(gC));
  }
  tma_store_arrive();
  tma_store_wait<0>();
#endif
}

// WS execution flow:
//   1. Initialize the same pipeline contract as non-WS, then split at kProducerThreads=128.
//   2. Producer warps donate registers and wait for empty[] before issuing A/B TMA. Only producer
//      lane zero launches copies; full[] still accounts for both A and B transaction bytes.
//   3. Consumer warps receive a larger register budget, wait for full[], and run the identical MMA
//      and FP32 accumulation sequence. Releasing empty[] hands the physical slot back to the
//      producer.
//   4. A consumer-only named barrier replaces the non-WS CTA sync. Scales, bias, BF16 conversion,
//      and the smem overlay follow exactly the non-WS numerical and memory-safety rules.
//   5. Only consumer lane zero issues the TMA store. Producer warps never touch the output tile.
template <typename Traits, typename TmaA, typename TmaB, typename TmaC, bool kPerTensor>
__global__ void __launch_bounds__(Traits::kNumThreads + 128, 1)
  fp8_gemm_scale_ws_kernel(CUTLASS_GRID_CONSTANT TmaA const tma_a,
                           CUTLASS_GRID_CONSTANT TmaB const tma_b,
                           CUTLASS_GRID_CONSTANT TmaC const tma_c,
                           const float* __restrict__ a_scale, const float* __restrict__ b_scale,
                           const __nv_bfloat16* __restrict__ bias, int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#if defined(CUDA_CTA_RECONFIG_ACTIVATED)
  static_assert(CUDA_CTA_RECONFIG_ACTIVATED,
                "FP8 warp specialization requires an architecture-specific -arch=sm_*a build.");
#endif
  using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
  using CtaBarrier = cutlass::arch::ClusterBarrier;
  using NamedBarrier = cutlass::arch::NamedBarrier;
  using Element = typename Traits::Element;
  using ElementO = typename Traits::ElementO;
  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using TiledMma = typename Traits::TiledMma;
  using SmemCopyAtom = typename Traits::SmemCopyAtom;

  constexpr int kBM = Traits::kBM;
  constexpr int kBN = Traits::kBN;
  constexpr int kBK = Traits::kBK;
  constexpr int kStages = Traits::kStages;
  constexpr int kProducerThreads = 128;
  constexpr int kConsumerThreads = Traits::kNumThreads;
  constexpr int kAElems = cosize(SmemLayoutA{});
  constexpr int kBElems = cosize(SmemLayoutB{});
  static_assert(kProducerThreads * 32 + kConsumerThreads * 232 <= 65536,
                "WS register budget: 128 x 32 + consumer x 232 must fit 64K");

  const int tid = threadIdx.x;
  const int m0 = blockIdx.y * kBM;
  const int n0 = blockIdx.x * kBN;
  const int nKt = (K + kBK - 1) / kBK;

  extern __shared__ __align__(1024) Element smem[];
  Element* a_base = smem;
  Element* b_base = smem + kStages * kAElems;
  __shared__ uint64_t full[kStages];
  __shared__ uint64_t empty[kStages];
  if (tid == 0) {
    for (int stage = 0; stage < kStages; ++stage) {
      TmaBarrier::init(&full[stage], 1);
      CtaBarrier::init(&empty[stage], kConsumerThreads);
    }
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();

  if (tid < kProducerThreads) {
    cutlass::arch::warpgroup_reg_dealloc<32>();
    if (tid == 0) {
      auto mA = tma_a.get_tma_tensor(make_shape(M, K));
      auto mB = tma_b.get_tma_tensor(make_shape(N, K));
      auto a_slice = tma_a.get_slice(_0{});
      auto b_slice = tma_b.get_slice(_0{});
      for (int kt = 0; kt < nKt; ++kt) {
        const int stage = kt % kStages;
        const int phase = (kt / kStages) & 1;
        CtaBarrier::wait(&empty[stage], phase);
        cutlass::arch::fence_view_async_shared();
        auto sA = make_tensor(make_smem_ptr(a_base + stage * kAElems), SmemLayoutA{});
        auto sB = make_tensor(make_smem_ptr(b_base + stage * kBElems), SmemLayoutB{});
        auto gA = local_tile(mA, Shape<Int<kBM>, Int<kBK>>{}, make_coord(blockIdx.y, kt));
        auto gB = local_tile(mB, Shape<Int<kBN>, Int<kBK>>{}, make_coord(blockIdx.x, kt));
        TmaBarrier::arrive_and_expect_tx(&full[stage], sizeof(Element) * (size(sA) + size(sB)));
        copy(tma_a.with(full[stage]), a_slice.partition_S(gA), a_slice.partition_D(sA));
        copy(tma_b.with(full[stage]), b_slice.partition_S(gB), b_slice.partition_D(sB));
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<232>();
    const int ctid = tid - kProducerThreads;
    for (int stage = 0; stage < kStages; ++stage) {
      CtaBarrier::arrive(&empty[stage]);
    }

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(ctid);
    auto s2r_copy_a = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto s2r_copy_b = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto s2r_thr_a = s2r_copy_a.get_thread_slice(ctid);
    auto s2r_thr_b = s2r_copy_b.get_thread_slice(ctid);
    auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<kBM>, Int<kBN>>{});
    clear(tCrC);
    auto cC = make_identity_tensor(Shape<Int<kBM>, Int<kBN>>{});
    auto tScC = thr_mma.partition_C(cC);
    auto tScC_rc = make_tensor(tScC.data(), convert_layout_acc_rowcol(tScC.layout()));
    constexpr int kCRows = decltype(size<0>(tScC_rc))::value;
    constexpr int kCCols = decltype(size<1>(tScC_rc))::value;

#pragma unroll 1
    for (int kt = 0; kt < nKt; ++kt) {
      const int stage = kt % kStages;
      const int phase = (kt / kStages) & 1;
      TmaBarrier::wait(&full[stage], phase);
      cutlass::arch::fence_view_async_shared();
      auto sA = make_tensor(make_smem_ptr(a_base + stage * kAElems), SmemLayoutA{});
      auto sB = make_tensor(make_smem_ptr(b_base + stage * kBElems), SmemLayoutB{});
      auto tCrA = thr_mma.partition_fragment_A(sA);
      auto tCrB = thr_mma.partition_fragment_B(sB);
      auto tAsA = s2r_thr_a.partition_S(sA);
      auto tBsB = s2r_thr_b.partition_S(sB);
      gemm_ss(tCrC, tCrA, tCrB, tAsA, tBsB, tiled_mma, s2r_copy_a, s2r_copy_b, s2r_thr_a,
              s2r_thr_b);
      CtaBarrier::arrive(&empty[stage]);
    }

    NamedBarrier::arrive_and_wait(kConsumerThreads, 0);
    auto tCrC_rc = make_tensor(tCrC.data(), convert_layout_acc_rowcol(tCrC.layout()));
    float row_scale[kCRows];
    float col_scale[kCCols];
    const float tensor_scale = kPerTensor ? a_scale[0] * b_scale[0] : 1.0f;
#pragma unroll
    for (int r = 0; r < kCRows; ++r) {
      const int m = m0 + get<0>(tScC_rc(r, 0));
      row_scale[r] = kPerTensor ? tensor_scale : (m < M ? a_scale[m] : 0.0f);
    }
#pragma unroll
    for (int c = 0; c < kCCols; ++c) {
      const int n = n0 + get<1>(tScC_rc(0, c));
      col_scale[c] = kPerTensor ? 1.0f : (n < N ? b_scale[n] : 0.0f);
    }
#pragma unroll
    for (int r = 0; r < kCRows; ++r) {
#pragma unroll
      for (int c = 0; c < kCCols; ++c) {
        const int n = n0 + get<1>(tScC_rc(r, c));
        const float bias_value = (bias != nullptr && n < N) ? __bfloat162float(bias[n]) : 0.0f;
        tCrC_rc(r, c) = tCrC_rc(r, c) * row_scale[r] * col_scale[c] + bias_value;
      }
    }

    auto tCrCh = convert_type<ElementO>(tCrC);
    auto r2s_copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, ElementO>{}, tiled_mma);
    auto r2s_thr = r2s_copy.get_thread_slice(ctid);
    auto sC = make_tensor(make_smem_ptr(reinterpret_cast<ElementO*>(smem)), SmemLayoutO{});
    auto tCrCh_src = r2s_thr.retile_S(tCrCh);
    auto tCsC_dst = r2s_thr.partition_D(sC);
    copy(r2s_copy, tCrCh_src, tCsC_dst);
    cutlass::arch::fence_view_async_shared();
    NamedBarrier::arrive_and_wait(kConsumerThreads, 0);
    auto mC = tma_c.get_tma_tensor(make_shape(M, N));
    auto c_slice = tma_c.get_slice(_0{});
    auto gC = local_tile(mC, Shape<Int<kBM>, Int<kBN>>{}, make_coord(blockIdx.y, blockIdx.x));
    if (ctid == 0) {
      copy(tma_c, c_slice.partition_S(sC), c_slice.partition_D(gC));
    }
    tma_store_arrive();
    tma_store_wait<0>();
  }
#endif
}

template <typename Traits, typename TmaA, typename TmaB, typename TmaC>
__global__ void __launch_bounds__(Traits::kNumThreads, 1)
  fp8_gemm_block_kernel(CUTLASS_GRID_CONSTANT TmaA const tma_a,
                        CUTLASS_GRID_CONSTANT TmaB const tma_b,
                        CUTLASS_GRID_CONSTANT TmaC const tma_c, const float* __restrict__ a_scale,
                        const float* __restrict__ b_scale, const __nv_bfloat16* __restrict__ bias,
                        int M, int N, int K, int num_k_blocks) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
  using CtaBarrier = cutlass::arch::ClusterBarrier;
  using Element = typename Traits::Element;
  using ElementO = typename Traits::ElementO;
  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using TiledMma = typename Traits::TiledMma;
  using SmemCopyAtom = typename Traits::SmemCopyAtom;

  constexpr int kBM = Traits::kBM;
  constexpr int kBN = Traits::kBN;
  constexpr int kBK = Traits::kBK;
  constexpr int kStages = Traits::kStages;
  constexpr int kNumThreads = Traits::kNumThreads;
  constexpr int kAElems = cosize(SmemLayoutA{});
  constexpr int kBElems = cosize(SmemLayoutB{});

  const int tid = threadIdx.x;
  const int m0 = blockIdx.y * kBM;
  const int n0 = blockIdx.x * kBN;

  auto mA = tma_a.get_tma_tensor(make_shape(M, K));
  auto mB = tma_b.get_tma_tensor(make_shape(N, K));
  auto mC = tma_c.get_tma_tensor(make_shape(M, N));
  auto a_slice = tma_a.get_slice(_0{});
  auto b_slice = tma_b.get_slice(_0{});
  auto c_slice = tma_c.get_slice(_0{});

  extern __shared__ __align__(1024) Element smem[];
  Element* a_base = smem;
  Element* b_base = smem + kStages * kAElems;

  __shared__ uint64_t full[kStages];
  __shared__ uint64_t empty[kStages];
  if (tid == 0) {
    for (int stage = 0; stage < kStages; ++stage) {
      TmaBarrier::init(&full[stage], 1);
      CtaBarrier::init(&empty[stage], kNumThreads);
    }
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto s2r_copy_a = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto s2r_copy_b = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto s2r_thr_a = s2r_copy_a.get_thread_slice(tid);
  auto s2r_thr_b = s2r_copy_b.get_thread_slice(tid);

  auto tCrPartial = partition_fragment_C(tiled_mma, Shape<Int<kBM>, Int<kBN>>{});
  auto tCrTotal = partition_fragment_C(tiled_mma, Shape<Int<kBM>, Int<kBN>>{});
  clear(tCrPartial);
  clear(tCrTotal);

  auto cC = make_identity_tensor(Shape<Int<kBM>, Int<kBN>>{});
  auto tScC = thr_mma.partition_C(cC);
  auto tScC_rc = make_tensor(tScC.data(), convert_layout_acc_rowcol(tScC.layout()));
  constexpr int kCRows = decltype(size<0>(tScC_rc))::value;
  constexpr int kCCols = decltype(size<1>(tScC_rc))::value;

  auto issue_tma = [&](int kt, int stage) {
    cutlass::arch::fence_view_async_shared();
    auto sA = make_tensor(make_smem_ptr(a_base + stage * kAElems), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(b_base + stage * kBElems), SmemLayoutB{});
    auto gA = local_tile(mA, Shape<Int<kBM>, Int<kBK>>{}, make_coord(blockIdx.y, kt));
    auto gB = local_tile(mB, Shape<Int<kBN>, Int<kBK>>{}, make_coord(blockIdx.x, kt));
    TmaBarrier::arrive_and_expect_tx(&full[stage], sizeof(Element) * (size(sA) + size(sB)));
    copy(tma_a.with(full[stage]), a_slice.partition_S(gA), a_slice.partition_D(sA));
    copy(tma_b.with(full[stage]), b_slice.partition_S(gB), b_slice.partition_D(sB));
  };

  const int nKt = (K + kBK - 1) / kBK;
  for (int stage = 0; stage < kStages; ++stage) {
    CtaBarrier::arrive(&empty[stage]);
  }
  if (tid == 0) {
    for (int kt = 0; kt < kStages && kt < nKt; ++kt) {
      CtaBarrier::wait(&empty[kt], 0);
      issue_tma(kt, kt);
    }
  }

#pragma unroll 1
  for (int kt = 0; kt < nKt; ++kt) {
    const int stage = kt % kStages;
    const int phase = (kt / kStages) & 1;
    TmaBarrier::wait(&full[stage], phase);
    cutlass::arch::fence_view_async_shared();

    auto sA = make_tensor(make_smem_ptr(a_base + stage * kAElems), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(b_base + stage * kBElems), SmemLayoutB{});
    auto tCrA = thr_mma.partition_fragment_A(sA);
    auto tCrB = thr_mma.partition_fragment_B(sB);
    auto tAsA = s2r_thr_a.partition_S(sA);
    auto tBsB = s2r_thr_b.partition_S(sB);
    gemm_ss(tCrPartial, tCrA, tCrB, tAsA, tBsB, tiled_mma, s2r_copy_a, s2r_copy_b, s2r_thr_a,
            s2r_thr_b);

    auto partial_rc =
      make_tensor(tCrPartial.data(), convert_layout_acc_rowcol(tCrPartial.layout()));
    auto total_rc = make_tensor(tCrTotal.data(), convert_layout_acc_rowcol(tCrTotal.layout()));
    float row_scale[kCRows];
    float col_scale[kCCols];
#pragma unroll
    for (int r = 0; r < kCRows; ++r) {
      const int m = m0 + get<0>(tScC_rc(r, 0));
      row_scale[r] = m < M ? a_scale[static_cast<int64_t>(m) * num_k_blocks + kt] : 0.0f;
    }
#pragma unroll
    for (int c = 0; c < kCCols; ++c) {
      const int n = n0 + get<1>(tScC_rc(0, c));
      col_scale[c] = n < N ? b_scale[static_cast<int64_t>(n / 128) * num_k_blocks + kt] : 0.0f;
    }
#pragma unroll
    for (int r = 0; r < kCRows; ++r) {
#pragma unroll
      for (int c = 0; c < kCCols; ++c) {
        total_rc(r, c) += partial_rc(r, c) * row_scale[r] * col_scale[c];
      }
    }
    clear(tCrPartial);

    CtaBarrier::arrive(&empty[stage]);
    if (tid == 0) {
      const int kt_next = kt + kStages;
      if (kt_next < nKt) {
        const int s_next = kt_next % kStages;
        const int ph_next = (kt_next / kStages) & 1;
        CtaBarrier::wait(&empty[s_next], ph_next);
        issue_tma(kt_next, s_next);
      }
    }
  }

  __syncthreads();
  auto total_rc = make_tensor(tCrTotal.data(), convert_layout_acc_rowcol(tCrTotal.layout()));
#pragma unroll
  for (int r = 0; r < kCRows; ++r) {
#pragma unroll
    for (int c = 0; c < kCCols; ++c) {
      const int n = n0 + get<1>(tScC_rc(r, c));
      const float bias_value = (bias != nullptr && n < N) ? __bfloat162float(bias[n]) : 0.0f;
      total_rc(r, c) += bias_value;
    }
  }

  auto tCrCh = convert_type<ElementO>(tCrTotal);
  auto r2s_copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, ElementO>{}, tiled_mma);
  auto r2s_thr = r2s_copy.get_thread_slice(tid);
  auto sC = make_tensor(make_smem_ptr(reinterpret_cast<ElementO*>(smem)), SmemLayoutO{});
  auto tCrCh_src = r2s_thr.retile_S(tCrCh);
  auto tCsC_dst = r2s_thr.partition_D(sC);
  copy(r2s_copy, tCrCh_src, tCsC_dst);
  cutlass::arch::fence_view_async_shared();
  __syncthreads();
  auto gC = local_tile(mC, Shape<Int<kBM>, Int<kBN>>{}, make_coord(blockIdx.y, blockIdx.x));
  if (tid == 0) {
    copy(tma_c, c_slice.partition_S(sC), c_slice.partition_D(gC));
  }
  tma_store_arrive();
  tma_store_wait<0>();
#endif
}

template <typename Traits, typename TmaA, typename TmaB, typename TmaC>
__global__ void __launch_bounds__(Traits::kNumThreads + 128, 1)
  fp8_gemm_block_ws_kernel(CUTLASS_GRID_CONSTANT TmaA const tma_a,
                           CUTLASS_GRID_CONSTANT TmaB const tma_b,
                           CUTLASS_GRID_CONSTANT TmaC const tma_c,
                           const float* __restrict__ a_scale, const float* __restrict__ b_scale,
                           const __nv_bfloat16* __restrict__ bias, int M, int N, int K,
                           int num_k_blocks) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#if defined(CUDA_CTA_RECONFIG_ACTIVATED)
  static_assert(CUDA_CTA_RECONFIG_ACTIVATED,
                "FP8 warp specialization requires an architecture-specific -arch=sm_*a build.");
#endif
  using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
  using CtaBarrier = cutlass::arch::ClusterBarrier;
  using NamedBarrier = cutlass::arch::NamedBarrier;
  using Element = typename Traits::Element;
  using ElementO = typename Traits::ElementO;
  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using TiledMma = typename Traits::TiledMma;
  using SmemCopyAtom = typename Traits::SmemCopyAtom;

  constexpr int kBM = Traits::kBM;
  constexpr int kBN = Traits::kBN;
  constexpr int kBK = Traits::kBK;
  constexpr int kStages = Traits::kStages;
  constexpr int kProducerThreads = 128;
  constexpr int kConsumerThreads = Traits::kNumThreads;
  constexpr int kAElems = cosize(SmemLayoutA{});
  constexpr int kBElems = cosize(SmemLayoutB{});
  static_assert(kProducerThreads * 32 + kConsumerThreads * 232 <= 65536,
                "WS register budget: 128 x 32 + consumer x 232 must fit 64K");

  const int tid = threadIdx.x;
  const int m0 = blockIdx.y * kBM;
  const int n0 = blockIdx.x * kBN;
  const int nKt = (K + kBK - 1) / kBK;

  extern __shared__ __align__(1024) Element smem[];
  Element* a_base = smem;
  Element* b_base = smem + kStages * kAElems;
  __shared__ uint64_t full[kStages];
  __shared__ uint64_t empty[kStages];
  if (tid == 0) {
    for (int stage = 0; stage < kStages; ++stage) {
      TmaBarrier::init(&full[stage], 1);
      CtaBarrier::init(&empty[stage], kConsumerThreads);
    }
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();

  if (tid < kProducerThreads) {
    cutlass::arch::warpgroup_reg_dealloc<32>();
    if (tid == 0) {
      auto mA = tma_a.get_tma_tensor(make_shape(M, K));
      auto mB = tma_b.get_tma_tensor(make_shape(N, K));
      auto a_slice = tma_a.get_slice(_0{});
      auto b_slice = tma_b.get_slice(_0{});
      for (int kt = 0; kt < nKt; ++kt) {
        const int stage = kt % kStages;
        const int phase = (kt / kStages) & 1;
        CtaBarrier::wait(&empty[stage], phase);
        cutlass::arch::fence_view_async_shared();
        auto sA = make_tensor(make_smem_ptr(a_base + stage * kAElems), SmemLayoutA{});
        auto sB = make_tensor(make_smem_ptr(b_base + stage * kBElems), SmemLayoutB{});
        auto gA = local_tile(mA, Shape<Int<kBM>, Int<kBK>>{}, make_coord(blockIdx.y, kt));
        auto gB = local_tile(mB, Shape<Int<kBN>, Int<kBK>>{}, make_coord(blockIdx.x, kt));
        TmaBarrier::arrive_and_expect_tx(&full[stage], sizeof(Element) * (size(sA) + size(sB)));
        copy(tma_a.with(full[stage]), a_slice.partition_S(gA), a_slice.partition_D(sA));
        copy(tma_b.with(full[stage]), b_slice.partition_S(gB), b_slice.partition_D(sB));
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<232>();
    const int ctid = tid - kProducerThreads;
    for (int stage = 0; stage < kStages; ++stage) {
      CtaBarrier::arrive(&empty[stage]);
    }

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(ctid);
    auto s2r_copy_a = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto s2r_copy_b = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto s2r_thr_a = s2r_copy_a.get_thread_slice(ctid);
    auto s2r_thr_b = s2r_copy_b.get_thread_slice(ctid);
    auto tCrPartial = partition_fragment_C(tiled_mma, Shape<Int<kBM>, Int<kBN>>{});
    auto tCrTotal = partition_fragment_C(tiled_mma, Shape<Int<kBM>, Int<kBN>>{});
    clear(tCrPartial);
    clear(tCrTotal);
    auto cC = make_identity_tensor(Shape<Int<kBM>, Int<kBN>>{});
    auto tScC = thr_mma.partition_C(cC);
    auto tScC_rc = make_tensor(tScC.data(), convert_layout_acc_rowcol(tScC.layout()));
    constexpr int kCRows = decltype(size<0>(tScC_rc))::value;
    constexpr int kCCols = decltype(size<1>(tScC_rc))::value;

#pragma unroll 1
    for (int kt = 0; kt < nKt; ++kt) {
      const int stage = kt % kStages;
      const int phase = (kt / kStages) & 1;
      TmaBarrier::wait(&full[stage], phase);
      cutlass::arch::fence_view_async_shared();
      auto sA = make_tensor(make_smem_ptr(a_base + stage * kAElems), SmemLayoutA{});
      auto sB = make_tensor(make_smem_ptr(b_base + stage * kBElems), SmemLayoutB{});
      auto tCrA = thr_mma.partition_fragment_A(sA);
      auto tCrB = thr_mma.partition_fragment_B(sB);
      auto tAsA = s2r_thr_a.partition_S(sA);
      auto tBsB = s2r_thr_b.partition_S(sB);
      gemm_ss(tCrPartial, tCrA, tCrB, tAsA, tBsB, tiled_mma, s2r_copy_a, s2r_copy_b, s2r_thr_a,
              s2r_thr_b);

      auto partial_rc =
        make_tensor(tCrPartial.data(), convert_layout_acc_rowcol(tCrPartial.layout()));
      auto total_rc = make_tensor(tCrTotal.data(), convert_layout_acc_rowcol(tCrTotal.layout()));
      float row_scale[kCRows];
      float col_scale[kCCols];
#pragma unroll
      for (int r = 0; r < kCRows; ++r) {
        const int m = m0 + get<0>(tScC_rc(r, 0));
        row_scale[r] = m < M ? a_scale[static_cast<int64_t>(m) * num_k_blocks + kt] : 0.0f;
      }
#pragma unroll
      for (int c = 0; c < kCCols; ++c) {
        const int n = n0 + get<1>(tScC_rc(0, c));
        col_scale[c] = n < N ? b_scale[static_cast<int64_t>(n / 128) * num_k_blocks + kt] : 0.0f;
      }
#pragma unroll
      for (int r = 0; r < kCRows; ++r) {
#pragma unroll
        for (int c = 0; c < kCCols; ++c) {
          total_rc(r, c) += partial_rc(r, c) * row_scale[r] * col_scale[c];
        }
      }
      clear(tCrPartial);
      CtaBarrier::arrive(&empty[stage]);
    }

    NamedBarrier::arrive_and_wait(kConsumerThreads, 0);
    auto total_rc = make_tensor(tCrTotal.data(), convert_layout_acc_rowcol(tCrTotal.layout()));
#pragma unroll
    for (int r = 0; r < kCRows; ++r) {
#pragma unroll
      for (int c = 0; c < kCCols; ++c) {
        const int n = n0 + get<1>(tScC_rc(r, c));
        const float bias_value = (bias != nullptr && n < N) ? __bfloat162float(bias[n]) : 0.0f;
        total_rc(r, c) += bias_value;
      }
    }

    auto tCrCh = convert_type<ElementO>(tCrTotal);
    auto r2s_copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, ElementO>{}, tiled_mma);
    auto r2s_thr = r2s_copy.get_thread_slice(ctid);
    auto sC = make_tensor(make_smem_ptr(reinterpret_cast<ElementO*>(smem)), SmemLayoutO{});
    auto tCrCh_src = r2s_thr.retile_S(tCrCh);
    auto tCsC_dst = r2s_thr.partition_D(sC);
    copy(r2s_copy, tCrCh_src, tCsC_dst);
    cutlass::arch::fence_view_async_shared();
    NamedBarrier::arrive_and_wait(kConsumerThreads, 0);
    auto mC = tma_c.get_tma_tensor(make_shape(M, N));
    auto c_slice = tma_c.get_slice(_0{});
    auto gC = local_tile(mC, Shape<Int<kBM>, Int<kBN>>{}, make_coord(blockIdx.y, blockIdx.x));
    if (ctid == 0) {
      copy(tma_c, c_slice.partition_S(sC), c_slice.partition_D(gC));
    }
    tma_store_arrive();
    tma_store_wait<0>();
  }
#endif
}

}  // namespace cache_dit::fp8
