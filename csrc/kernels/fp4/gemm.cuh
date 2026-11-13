#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/device_kernel.h>
#include <cutlass/numeric_conversion.h>

#include <cstdint>

#include "quantize.cuh"

namespace cache_dit::fp4 {
using namespace cute;

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

namespace sf_partition {

template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto thrfrg_sfa(SFATensor&& tensor,
                                           TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
  CUTE_STATIC_ASSERT_V(rank(tensor) >= Int<2>{});
  using AtomShape_MNK = typename Atom::Shape_MNK;
  using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;
  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto t_tile = make_tile(get<0>(permutation_mnk), get<2>(permutation_mnk));
  auto t_tensor = logical_divide(tensor, t_tile);
  auto a_tile =
    make_tile(make_layout(size<0>(AtomShape_MNK{})), make_layout(size<2>(AtomShape_MNK{})));
  auto a_tensor = zipped_divide(t_tensor, a_tile);
  auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{}, _);
  auto thr_tile = make_tile(
    _, make_tile(make_layout(size<1>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
  return zipped_divide(tv_tensor, thr_tile);
}

template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto thrfrg_sfb(SFBTensor&& tensor,
                                           TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
  CUTE_STATIC_ASSERT_V(rank(tensor) >= Int<2>{});
  using AtomShape_MNK = typename Atom::Shape_MNK;
  using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;
  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto t_tile = make_tile(get<1>(permutation_mnk), get<2>(permutation_mnk));
  auto t_tensor = logical_divide(tensor, t_tile);
  auto b_tile =
    make_tile(make_layout(size<1>(AtomShape_MNK{})), make_layout(size<2>(AtomShape_MNK{})));
  auto b_tensor = zipped_divide(t_tensor, b_tile);
  auto tv_tensor = b_tensor.compose(AtomLayoutSFB_TV{}, _);
  auto thr_tile = make_tile(
    _, make_tile(make_layout(size<2>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
  return zipped_divide(tv_tensor, thr_tile);
}

template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_fragment_sfa(SFATensor&& tensor, ThrMma& thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  auto thr_tensor =
    make_tensor(static_cast<SFATensor&&>(tensor).data(), thrfrg_sfa(tensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
  auto part = thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
  return make_fragment_like<ValTypeSF>(part);
}

template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_fragment_sfb(SFBTensor&& tensor, ThrMma& thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  auto thr_tensor =
    make_tensor(static_cast<SFBTensor&&>(tensor).data(), thrfrg_sfb(tensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
  auto part = thr_tensor(thr_vnk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
  return make_fragment_like<ValTypeSF>(part);
}

template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto layout_sfa_tv(TiledMma& mma) {
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_a = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto a_tile = make_tile(
    _, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                             make_stride(Int<1>{}, Int<0>{})),
                 _));
  auto thridx_to_thrid = right_inverse(thr_layout_vmnk);
  return thrfrg_sfa(ref_a, mma).compose(a_tile, _).compose(thridx_to_thrid, _);
}

template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto layout_sfb_tv(TiledMma& mma) {
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_b = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto b_tile = make_tile(
    _, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                             make_stride(Int<0>{}, Int<1>{})),
                 _));
  auto thridx_to_thrid = right_inverse(thr_layout_vmnk);
  return thrfrg_sfb(ref_b, mma).compose(b_tile, _).compose(thridx_to_thrid, _);
}

}  // namespace sf_partition

template <class TiledMma_, int kMN, int kBK_>
struct SfSmemAtom {
  static constexpr int kMMA_NSF = size<2>(typename TiledMma_::AtomShape_MNK{}) / kSFVec;
  using Config = cutlass::detail::Sm1xxBlockScaledConfig<kSFVec>;
  using Blk_Elems = decltype(typename Config::Blk_MN{} * typename Config::Blk_SF{});
  using mnBlockShape = Shape<_32, _4>;
  using mnBlockStride = Stride<_16, _4>;
  using kBlockShape = Shape<Int<kSFVec>, Int<kMMA_NSF>>;
  using kBlockStride = Stride<_0, _1>;
  using shapeM = decltype(prepend(Int<kMN / kRowBlock>{}, mnBlockShape{}));
  using strideM = decltype(prepend(Blk_Elems{}, mnBlockStride{}));
  using shapeK = decltype(prepend(
    make_shape(Int<kSFVecPerBlk / kMMA_NSF>{}, Int<kBK_ / kSFVec / kSFVecPerBlk>{}),
    kBlockShape{}));
  using strideK =
    decltype(prepend(make_stride(Int<kMMA_NSF>{}, Int<kMN / kRowBlock * 512>{}), kBlockStride{}));
  using type =
    decltype(make_layout(make_shape(shapeM{}, shapeK{}), make_stride(strideM{}, strideK{})));
};

template <int kBN_ = 128, int kStages_ = 4, int kBK_ = 64>
struct Fp4GemmTraits {
  static constexpr int kBM = 128;
  static constexpr int kBN = kBN_;
  static constexpr int kBK = kBK_;
  static constexpr int kStages = kStages_;
  static constexpr int kNumThreads = kBM / 16 * 32;
  static_assert(kBN % 128 == 0, "BN must be a multiple of 128 for the SF atom");
  static_assert(kBK % 64 == 0, "BK must contain whole K64 block-scaled MMA atoms");
  static_assert(kStages >= 2, "At least two stages are required");

  using Element = cutlass::float_e2m1_t;
  using ElementSF = cutlass::float_ue4m3_t;
  using ElementO = cutlass::bfloat16_t;
  using ElementAcc = float;
  using MmaAtom =
    SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<Element, Element, ElementAcc, ElementSF, kSFVec>;
  using AtomLayoutMNK = Layout<Shape<_4, _2, _1>>;
  using PermTileM = Int<kBM>;
  using PermTileN = Layout<Shape<_8, _2, _2>, Stride<_1, _16, _8>>;
  using PermTileK = _64;
  using TiledMma =
    decltype(make_tiled_mma(MmaAtom{}, AtomLayoutMNK{}, Tile<PermTileM, PermTileN, PermTileK>{}));
  static_assert(size<0>(typename TiledMma::AtomShape_MNK{}) == 16 &&
                  size<2>(typename TiledMma::AtomShape_MNK{}) == 64,
                "The block-scaled atom must be 16x8x64");

  using SmemAlloc = uint4_t;
  using SmemLayoutAtomA = UMMA::Layout_K_SW32_Atom<SmemAlloc>;
  using SmemLayoutAtomB = UMMA::Layout_K_SW32_Atom<SmemAlloc>;
  using SmemLayoutA =
    decltype(tile_to_shape(SmemLayoutAtomA{}, Shape<Int<kBM>, Int<kBK>, Int<kStages>>{}));
  using SmemLayoutB =
    decltype(tile_to_shape(SmemLayoutAtomB{}, Shape<Int<kBN>, Int<kBK>, Int<kStages>>{}));
  using SmemLayoutAtomSFA = typename SfSmemAtom<TiledMma, kBM, kBK>::type;
  using SmemLayoutAtomSFB =
    typename SfSmemAtom<TiledMma, (kBN > kRowBlock ? kBN : kRowBlock), kBK>::type;
  using SmemLayoutSFA = decltype(make_layout(
    append(shape(SmemLayoutAtomSFA{}), Int<kStages>{}),
    append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))));
  using SmemLayoutSFB = decltype(make_layout(
    append(shape(SmemLayoutAtomSFB{}), Int<kStages>{}),
    append(stride(SmemLayoutAtomSFB{}), size(filter_zeros(SmemLayoutAtomSFB{})))));
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, SmemAlloc>;
  using SmemAtomO = GMMA::Layout_K_SW128_Atom<ElementO>;
  using SmemLayoutO = decltype(tile_to_shape(SmemAtomO{}, Shape<Int<kBM>, Int<kBN>>{}));

  static constexpr int kAStageBytes = cosize(SmemLayoutA{}) / kStages / 2;
  static constexpr int kBStageBytes = cosize(SmemLayoutB{}) / kStages / 2;
  static constexpr int kSFAStageBytes = cosize(SmemLayoutSFA{}) / kStages;
  static constexpr int kSFBStageBytes = cosize(SmemLayoutSFB{}) / kStages;
  static constexpr int kStageBytes = kAStageBytes + kBStageBytes + kSFAStageBytes + kSFBStageBytes;
  static constexpr int kSmemBytes = kStages * kStageBytes;
  static constexpr uint32_t kTxBytes =
    kBM * kBK / 2 + kBN * kBK / 2 + kSFAStageBytes + kSFBStageBytes;
  static constexpr bool kOStagingFits = cosize(SmemLayoutO{}) * sizeof(ElementO) <= kSmemBytes;
  static_assert(kOStagingFits, "Output staging must fit in the reused mainloop shared memory");
};

template <typename Traits, typename TmaA, typename TmaB, typename TmaSFA, typename TmaSFB,
          typename TmaC, bool kLevel1A, bool kLevel1B>
__global__ void __launch_bounds__(Traits::kNumThreads, 1)
  fp4_gemm_tma_kernel(CUTLASS_GRID_CONSTANT TmaA const tma_a,
                      CUTLASS_GRID_CONSTANT TmaB const tma_b,
                      CUTLASS_GRID_CONSTANT TmaSFA const tma_sfa,
                      CUTLASS_GRID_CONSTANT TmaSFB const tma_sfb,
                      CUTLASS_GRID_CONSTANT TmaC const tma_c, const float* __restrict__ sa,
                      const float* __restrict__ sb, const cutlass::bfloat16_t* __restrict__ bias,
                      int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  using cute::tma_store_arrive;
  using cute::tma_store_wait;
  using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
  using CtaBarrier = cutlass::arch::ClusterBarrier;
  using Element = typename Traits::Element;
  using ElementSF = typename Traits::ElementSF;
  using ElementO = typename Traits::ElementO;
  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutSFA = typename Traits::SmemLayoutSFA;
  using SmemLayoutSFB = typename Traits::SmemLayoutSFB;
  using SmemLayoutAtomSFA = typename Traits::SmemLayoutAtomSFA;
  using SmemLayoutAtomSFB = typename Traits::SmemLayoutAtomSFB;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using TiledMma = typename Traits::TiledMma;
  using SmemCopyAtom = typename Traits::SmemCopyAtom;

  constexpr int kBM = Traits::kBM;
  constexpr int kBN = Traits::kBN;
  constexpr int kBK = Traits::kBK;
  constexpr int kStages = Traits::kStages;
  constexpr int kNumThreads = Traits::kNumThreads;
  constexpr int kAStageBytes = Traits::kAStageBytes;
  constexpr int kBStageBytes = Traits::kBStageBytes;
  constexpr int kSFAStageBytes = Traits::kSFAStageBytes;

  const int tid = threadIdx.x;
  const int m0 = blockIdx.y * kBM;
  const int n0 = blockIdx.x * kBN;
  auto mA = tma_a.get_tma_tensor(make_shape(M, K));
  auto mB = tma_b.get_tma_tensor(make_shape(N, K));
  auto mC = tma_c.get_tma_tensor(make_shape(M, N));
  auto mSFA = tma_sfa.get_tma_tensor(shape(sf_gmem_layout(M, K)));
  auto mSFB = tma_sfb.get_tma_tensor(shape(sf_gmem_layout(N, K)));
  auto a_slice = tma_a.get_slice(_0{});
  auto b_slice = tma_b.get_slice(_0{});
  auto sfa_slice = tma_sfa.get_slice(_0{});
  auto sfb_slice = tma_sfb.get_slice(_0{});
  auto c_slice = tma_c.get_slice(_0{});

  extern __shared__ __align__(1024) uint8_t shm[];
  uint8_t* a_bytes = shm;
  uint8_t* b_bytes = a_bytes + kStages * kAStageBytes;
  ElementSF* sfa_base = reinterpret_cast<ElementSF*>(b_bytes + kStages * kBStageBytes);
  ElementSF* sfb_base = sfa_base + kStages * kSFAStageBytes;

  __shared__ uint64_t full[kStages];
  __shared__ uint64_t empty[kStages];
  if (tid == 0) {
    for (int stage = 0; stage < kStages; ++stage) {
      TmaBarrier::init(&full[stage], 1);
      CtaBarrier::init(&empty[stage], kNumThreads / 32);
    }
  }
  __syncthreads();

  auto sA = make_tensor(make_smem_ptr(recast_ptr<uint4_t>(a_bytes)), SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(recast_ptr<uint4_t>(b_bytes)), SmemLayoutB{});
  auto sSFA = make_tensor(make_smem_ptr(sfa_base), SmemLayoutSFA{});
  auto sSFB = make_tensor(make_smem_ptr(sfb_base), SmemLayoutSFB{});

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto s2r_copy_a = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto s2r_copy_b = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto s2r_thr_a = s2r_copy_a.get_thread_slice(tid);
  auto s2r_thr_b = s2r_copy_b.get_thread_slice(tid);
  auto tCsA = s2r_thr_a.partition_S(as_position_independent_swizzle_tensor(sA));
  auto tCsB = s2r_thr_b.partition_S(as_position_independent_swizzle_tensor(sB));
  auto tCrA = thr_mma.partition_fragment_A(sA(_, _, Int<0>{}));
  auto tCrB = thr_mma.partition_fragment_B(sB(_, _, Int<0>{}));
  auto tCrA_view = s2r_thr_a.retile_D(tCrA);
  auto tCrB_view = s2r_thr_b.retile_D(tCrB);

  using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;
  auto tCrSFA = sf_partition::partition_fragment_sfa(sSFA(_, _, Int<0>{}), thr_mma);
  auto tCrSFB = sf_partition::partition_fragment_sfb(sSFB(_, _, Int<0>{}), thr_mma);
  auto copy_sfa = make_tiled_copy_impl(
    SmemCopyAtomSF{}, sf_partition::layout_sfa_tv(tiled_mma),
    make_shape(size<0>(tile_shape(tiled_mma)), size<2>(tile_shape(tiled_mma))));
  auto copy_sfb = make_tiled_copy_impl(
    SmemCopyAtomSF{}, sf_partition::layout_sfb_tv(tiled_mma),
    make_shape(size<1>(tile_shape(tiled_mma)), size<2>(tile_shape(tiled_mma))));
  auto thr_copy_sfa = copy_sfa.get_thread_slice(tid);
  auto thr_copy_sfb = copy_sfb.get_thread_slice(tid);
  auto tCsSFA = thr_copy_sfa.partition_S(as_position_independent_swizzle_tensor(sSFA));
  auto tCsSFB = thr_copy_sfb.partition_S(as_position_independent_swizzle_tensor(sSFB));
  auto tCrSFA_view = thr_copy_sfa.retile_D(tCrSFA);
  auto tCrSFB_view = thr_copy_sfb.retile_D(tCrSFB);

  auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<kBM>, Int<kBN>>{});
  clear(tCrC);
  auto cC = make_identity_tensor(Shape<Int<kBM>, Int<kBN>>{});
  auto tScC = thr_mma.partition_C(cC);
  auto tScC_rc = make_tensor(tScC.data(), convert_layout_acc_rowcol(tScC.layout()));
  constexpr int kCRows = decltype(size<0>(tScC_rc))::value;
  constexpr int kCCols = decltype(size<1>(tScC_rc))::value;

  constexpr uint32_t kTxBytes = Traits::kTxBytes;
  auto issue_tma = [&](int kt, int stage) {
    cutlass::arch::fence_view_async_shared();
    auto gA = local_tile(mA, Shape<Int<kBM>, Int<kBK>>{}, make_coord(blockIdx.y, kt));
    auto gB = local_tile(mB, Shape<Int<kBN>, Int<kBK>>{}, make_coord(blockIdx.x, kt));
    auto gSFA = local_tile(mSFA, Shape<Int<kBM>, Int<kBK>>{}, make_coord(blockIdx.y, kt));
    auto gSFB = local_tile(mSFB, Shape<Int<kBN>, Int<kBK>>{}, make_coord(blockIdx.x, kt));
    TmaBarrier::arrive_and_expect_tx(&full[stage], kTxBytes);
    copy(tma_a.with(full[stage]), a_slice.partition_S(gA), a_slice.partition_D(sA(_, _, stage)));
    copy(tma_b.with(full[stage]), b_slice.partition_S(gB), b_slice.partition_D(sB(_, _, stage)));
    copy(tma_sfa.with(full[stage]), sfa_slice.partition_S(gSFA),
         sfa_slice.partition_D(sSFA(_, _, stage)));
    copy(tma_sfb.with(full[stage]), sfb_slice.partition_S(gSFB),
         sfb_slice.partition_D(sSFB(_, _, stage)));
  };

  const int num_k_tiles = K / kBK;
  if (tid % 32 == 0) {
    for (int stage = 0; stage < kStages; ++stage) {
      CtaBarrier::arrive(&empty[stage]);
    }
  }
  if (tid == 0) {
    for (int kt = 0; kt < kStages && kt < num_k_tiles; ++kt) {
      CtaBarrier::wait(&empty[kt], 0);
      issue_tma(kt, kt);
    }
  }
#pragma unroll 1
  for (int kt = 0; kt < num_k_tiles; ++kt) {
    const int stage = kt % kStages;
    const int phase = (kt / kStages) & 1;
    TmaBarrier::wait(&full[stage], phase);
    cutlass::arch::fence_view_async_shared();
#pragma unroll
    for (int k_block = 0; k_block < kBK / 64; ++k_block) {
      copy(s2r_copy_a, tCsA(_, _, k_block, stage), tCrA_view(_, _, k_block));
      copy(tCsSFA(_, _, k_block, stage), tCrSFA_view(_, _, k_block));
      copy(s2r_copy_b, tCsB(_, _, k_block, stage), tCrB_view(_, _, k_block));
      copy(tCsSFB(_, _, k_block, stage), tCrSFB_view(_, _, k_block));
      cute::gemm(tiled_mma, make_zip_tensor(tCrA(_, _, k_block), tCrSFA(_, _, k_block)),
                 make_zip_tensor(tCrB(_, _, k_block), tCrSFB(_, _, k_block)), tCrC);
    }
    __syncwarp();
    if (tid % 32 == 0) {
      CtaBarrier::arrive(&empty[stage]);
    }
    if (tid == 0) {
      const int next = kt + kStages;
      if (next < num_k_tiles) {
        const int next_stage = next % kStages;
        CtaBarrier::wait(&empty[next_stage], (next / kStages) & 1);
        issue_tma(next, next_stage);
      }
    }
  }

  __syncthreads();
  auto tCrC_rc = make_tensor(tCrC.data(), convert_layout_acc_rowcol(tCrC.layout()));
  float col_scale[kCCols];
  float col_bias[kCCols];
#pragma unroll
  for (int col = 0; col < kCCols; ++col) {
    const int n = n0 + get<1>(tScC_rc(0, col));
    col_scale[col] = kLevel1B ? (n < N ? sb[n] : 0.0f) : 1.0f;
    col_bias[col] = bias != nullptr && n < N ? static_cast<float>(bias[n]) : 0.0f;
  }
#pragma unroll
  for (int row = 0; row < kCRows; ++row) {
    const int m = m0 + get<0>(tScC_rc(row, 0));
    const float scale_a = kLevel1A ? (m < M ? sa[m] : 0.0f) : 1.0f;
#pragma unroll
    for (int col = 0; col < kCCols; ++col) {
      tCrC_rc(row, col) = tCrC_rc(row, col) * scale_a * col_scale[col] + col_bias[col];
    }
  }
  auto tCrCh = convert_type<ElementO>(tCrC);
  auto r2s_copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, ElementO>{}, tiled_mma);
  auto r2s_thr = r2s_copy.get_thread_slice(tid);
  auto sC = make_tensor(make_smem_ptr(reinterpret_cast<ElementO*>(shm)), SmemLayoutO{});
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
#else
  asm volatile("trap;");
#endif
}

template <typename Traits, typename TmaA, typename TmaB, typename TmaSFA, typename TmaSFB,
          typename TmaC, bool kLevel1A, bool kLevel1B>
__global__ void __launch_bounds__(Traits::kNumThreads + 128, 1)
  fp4_gemm_tma_ws_kernel(CUTLASS_GRID_CONSTANT TmaA const tma_a,
                         CUTLASS_GRID_CONSTANT TmaB const tma_b,
                         CUTLASS_GRID_CONSTANT TmaSFA const tma_sfa,
                         CUTLASS_GRID_CONSTANT TmaSFB const tma_sfb,
                         CUTLASS_GRID_CONSTANT TmaC const tma_c, const float* __restrict__ sa,
                         const float* __restrict__ sb, const cutlass::bfloat16_t* __restrict__ bias,
                         int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  using cute::tma_store_arrive;
  using cute::tma_store_wait;
  using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
  using CtaBarrier = cutlass::arch::ClusterBarrier;
  using Element = typename Traits::Element;
  using ElementSF = typename Traits::ElementSF;
  using ElementO = typename Traits::ElementO;
  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutSFA = typename Traits::SmemLayoutSFA;
  using SmemLayoutSFB = typename Traits::SmemLayoutSFB;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using TiledMma = typename Traits::TiledMma;
  using SmemCopyAtom = typename Traits::SmemCopyAtom;

  constexpr int kBM = Traits::kBM;
  constexpr int kBN = Traits::kBN;
  constexpr int kBK = Traits::kBK;
  constexpr int kStages = Traits::kStages;
  constexpr int kNumThreads = Traits::kNumThreads;
  constexpr int kProducerThreads = 128;
  constexpr int kAStageBytes = Traits::kAStageBytes;
  constexpr int kBStageBytes = Traits::kBStageBytes;
  constexpr int kSFAStageBytes = Traits::kSFAStageBytes;
  static_assert(kProducerThreads * 32 + kNumThreads * 224 <= 65536, "WS register budget exceeded");

  const int tid = threadIdx.x;
  const int m0 = blockIdx.y * kBM;
  const int n0 = blockIdx.x * kBN;
  auto mA = tma_a.get_tma_tensor(make_shape(M, K));
  auto mB = tma_b.get_tma_tensor(make_shape(N, K));
  auto mC = tma_c.get_tma_tensor(make_shape(M, N));
  auto mSFA = tma_sfa.get_tma_tensor(shape(sf_gmem_layout(M, K)));
  auto mSFB = tma_sfb.get_tma_tensor(shape(sf_gmem_layout(N, K)));
  auto a_slice = tma_a.get_slice(_0{});
  auto b_slice = tma_b.get_slice(_0{});
  auto sfa_slice = tma_sfa.get_slice(_0{});
  auto sfb_slice = tma_sfb.get_slice(_0{});
  auto c_slice = tma_c.get_slice(_0{});

  extern __shared__ __align__(1024) uint8_t shm[];
  uint8_t* a_bytes = shm;
  uint8_t* b_bytes = a_bytes + kStages * kAStageBytes;
  ElementSF* sfa_base = reinterpret_cast<ElementSF*>(b_bytes + kStages * kBStageBytes);
  ElementSF* sfb_base = sfa_base + kStages * kSFAStageBytes;
  auto sA = make_tensor(make_smem_ptr(recast_ptr<uint4_t>(a_bytes)), SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(recast_ptr<uint4_t>(b_bytes)), SmemLayoutB{});
  auto sSFA = make_tensor(make_smem_ptr(sfa_base), SmemLayoutSFA{});
  auto sSFB = make_tensor(make_smem_ptr(sfb_base), SmemLayoutSFB{});

  __shared__ uint64_t full[kStages];
  __shared__ uint64_t empty[kStages];
  if (tid == 0) {
    for (int stage = 0; stage < kStages; ++stage) {
      TmaBarrier::init(&full[stage], 1);
      CtaBarrier::init(&empty[stage], kNumThreads / 32);
    }
  }
  __syncthreads();

  constexpr uint32_t kTxBytes = Traits::kTxBytes;
  auto issue_tma = [&](int kt, int stage) {
    cutlass::arch::fence_view_async_shared();
    auto gA = local_tile(mA, Shape<Int<kBM>, Int<kBK>>{}, make_coord(blockIdx.y, kt));
    auto gB = local_tile(mB, Shape<Int<kBN>, Int<kBK>>{}, make_coord(blockIdx.x, kt));
    auto gSFA = local_tile(mSFA, Shape<Int<kBM>, Int<kBK>>{}, make_coord(blockIdx.y, kt));
    auto gSFB = local_tile(mSFB, Shape<Int<kBN>, Int<kBK>>{}, make_coord(blockIdx.x, kt));
    TmaBarrier::arrive_and_expect_tx(&full[stage], kTxBytes);
    copy(tma_a.with(full[stage]), a_slice.partition_S(gA), a_slice.partition_D(sA(_, _, stage)));
    copy(tma_b.with(full[stage]), b_slice.partition_S(gB), b_slice.partition_D(sB(_, _, stage)));
    copy(tma_sfa.with(full[stage]), sfa_slice.partition_S(gSFA),
         sfa_slice.partition_D(sSFA(_, _, stage)));
    copy(tma_sfb.with(full[stage]), sfb_slice.partition_S(gSFB),
         sfb_slice.partition_D(sSFB(_, _, stage)));
  };

  const int num_k_tiles = K / kBK;
  if (tid >= kProducerThreads && tid % 32 == 0) {
    for (int stage = 0; stage < kStages; ++stage) {
      CtaBarrier::arrive(&empty[stage]);
    }
  }
  if (tid < kProducerThreads) {
    cutlass::arch::warpgroup_reg_dealloc<32>();
    if (tid == 0) {
      for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int stage = kt % kStages;
        CtaBarrier::wait(&empty[stage], (kt / kStages) & 1);
        issue_tma(kt, stage);
      }
    }
    return;
  }

  cutlass::arch::warpgroup_reg_alloc<224>();
  const int ctid = tid - kProducerThreads;
  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(ctid);
  auto s2r_copy_a = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto s2r_copy_b = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto s2r_thr_a = s2r_copy_a.get_thread_slice(ctid);
  auto s2r_thr_b = s2r_copy_b.get_thread_slice(ctid);
  auto tCsA = s2r_thr_a.partition_S(as_position_independent_swizzle_tensor(sA));
  auto tCsB = s2r_thr_b.partition_S(as_position_independent_swizzle_tensor(sB));
  auto tCrA = thr_mma.partition_fragment_A(sA(_, _, Int<0>{}));
  auto tCrB = thr_mma.partition_fragment_B(sB(_, _, Int<0>{}));
  auto tCrA_view = s2r_thr_a.retile_D(tCrA);
  auto tCrB_view = s2r_thr_b.retile_D(tCrB);

  using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;
  auto tCrSFA = sf_partition::partition_fragment_sfa(sSFA(_, _, Int<0>{}), thr_mma);
  auto tCrSFB = sf_partition::partition_fragment_sfb(sSFB(_, _, Int<0>{}), thr_mma);
  auto copy_sfa = make_tiled_copy_impl(
    SmemCopyAtomSF{}, sf_partition::layout_sfa_tv(tiled_mma),
    make_shape(size<0>(tile_shape(tiled_mma)), size<2>(tile_shape(tiled_mma))));
  auto copy_sfb = make_tiled_copy_impl(
    SmemCopyAtomSF{}, sf_partition::layout_sfb_tv(tiled_mma),
    make_shape(size<1>(tile_shape(tiled_mma)), size<2>(tile_shape(tiled_mma))));
  auto thr_copy_sfa = copy_sfa.get_thread_slice(ctid);
  auto thr_copy_sfb = copy_sfb.get_thread_slice(ctid);
  auto tCsSFA = thr_copy_sfa.partition_S(as_position_independent_swizzle_tensor(sSFA));
  auto tCsSFB = thr_copy_sfb.partition_S(as_position_independent_swizzle_tensor(sSFB));
  auto tCrSFA_view = thr_copy_sfa.retile_D(tCrSFA);
  auto tCrSFB_view = thr_copy_sfb.retile_D(tCrSFB);

  auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<kBM>, Int<kBN>>{});
  clear(tCrC);
  auto cC = make_identity_tensor(Shape<Int<kBM>, Int<kBN>>{});
  auto tScC = thr_mma.partition_C(cC);
  auto tScC_rc = make_tensor(tScC.data(), convert_layout_acc_rowcol(tScC.layout()));
  constexpr int kCRows = decltype(size<0>(tScC_rc))::value;
  constexpr int kCCols = decltype(size<1>(tScC_rc))::value;

#pragma unroll 1
  for (int kt = 0; kt < num_k_tiles; ++kt) {
    const int stage = kt % kStages;
    const int phase = (kt / kStages) & 1;
    TmaBarrier::wait(&full[stage], phase);
    cutlass::arch::fence_view_async_shared();
#pragma unroll
    for (int k_block = 0; k_block < kBK / 64; ++k_block) {
      copy(s2r_copy_a, tCsA(_, _, k_block, stage), tCrA_view(_, _, k_block));
      copy(tCsSFA(_, _, k_block, stage), tCrSFA_view(_, _, k_block));
      copy(s2r_copy_b, tCsB(_, _, k_block, stage), tCrB_view(_, _, k_block));
      copy(tCsSFB(_, _, k_block, stage), tCrSFB_view(_, _, k_block));
      cute::gemm(tiled_mma, make_zip_tensor(tCrA(_, _, k_block), tCrSFA(_, _, k_block)),
                 make_zip_tensor(tCrB(_, _, k_block), tCrSFB(_, _, k_block)), tCrC);
    }
    __syncwarp();
    if (ctid % 32 == 0) {
      CtaBarrier::arrive(&empty[stage]);
    }
  }

  __syncthreads();
  auto tCrC_rc = make_tensor(tCrC.data(), convert_layout_acc_rowcol(tCrC.layout()));
  float col_scale[kCCols];
  float col_bias[kCCols];
#pragma unroll
  for (int col = 0; col < kCCols; ++col) {
    const int n = n0 + get<1>(tScC_rc(0, col));
    col_scale[col] = kLevel1B ? (n < N ? sb[n] : 0.0f) : 1.0f;
    col_bias[col] = bias != nullptr && n < N ? static_cast<float>(bias[n]) : 0.0f;
  }
#pragma unroll
  for (int row = 0; row < kCRows; ++row) {
    const int m = m0 + get<0>(tScC_rc(row, 0));
    const float scale_a = kLevel1A ? (m < M ? sa[m] : 0.0f) : 1.0f;
#pragma unroll
    for (int col = 0; col < kCCols; ++col) {
      tCrC_rc(row, col) = tCrC_rc(row, col) * scale_a * col_scale[col] + col_bias[col];
    }
  }
  auto tCrCh = convert_type<ElementO>(tCrC);
  auto r2s_copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, ElementO>{}, tiled_mma);
  auto r2s_thr = r2s_copy.get_thread_slice(ctid);
  auto sC = make_tensor(make_smem_ptr(reinterpret_cast<ElementO*>(shm)), SmemLayoutO{});
  auto tCrCh_src = r2s_thr.retile_S(tCrCh);
  auto tCsC_dst = r2s_thr.partition_D(sC);
  copy(r2s_copy, tCrCh_src, tCsC_dst);
  cutlass::arch::fence_view_async_shared();
  cutlass::arch::NamedBarrier::arrive_and_wait(kNumThreads, 0);
  auto gC = local_tile(mC, Shape<Int<kBM>, Int<kBN>>{}, make_coord(blockIdx.y, blockIdx.x));
  if (ctid == 0) {
    copy(tma_c, c_slice.partition_S(sC), c_slice.partition_D(gC));
  }
  tma_store_arrive();
  tma_store_wait<0>();
#else
  asm volatile("trap;");
#endif
}

template <bool kLevel1A, bool kLevel1B, bool kWS, typename Traits = Fp4GemmTraits<>>
void fp4_gemm_tma_fwd(const cutlass::float_e2m1_t* a, const cutlass::float_e2m1_t* b,
                      const cutlass::float_ue4m3_t* sfa, const cutlass::float_ue4m3_t* sfb,
                      const float* sa, const float* sb, const cutlass::bfloat16_t* bias,
                      cutlass::bfloat16_t* out, int M, int N, int K, cudaStream_t stream) {
  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutAtomA = typename Traits::SmemLayoutAtomA;
  using SmemLayoutAtomB = typename Traits::SmemLayoutAtomB;
  using SmemLayoutAtomSFA = typename Traits::SmemLayoutAtomSFA;
  using SmemLayoutAtomSFB = typename Traits::SmemLayoutAtomSFB;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  constexpr int kBM = Traits::kBM;
  constexpr int kBN = Traits::kBN;
  constexpr int kBK = Traits::kBK;
  constexpr int kSmemBytes = Traits::kSmemBytes;

  auto mA =
    make_tensor(recast_ptr<cutlass::float_e2m1_t>(a), make_shape(M, K), make_stride(K, _1{}));
  auto mB =
    make_tensor(recast_ptr<cutlass::float_e2m1_t>(b), make_shape(N, K), make_stride(K, _1{}));
  auto mC = make_tensor(make_gmem_ptr(out), make_shape(M, N), make_stride(N, _1{}));
  auto mSFA = make_tensor(recast_ptr<cutlass::float_ue4m3_t>(sfa), sf_gmem_layout(M, K));
  auto mSFB = make_tensor(recast_ptr<cutlass::float_ue4m3_t>(sfb), sf_gmem_layout(N, K));
  auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, mA, SmemLayoutA{}(_, _, Int<0>{}),
                             Shape<Int<kBM>, Int<kBK>>{}, _1{});
  auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, mB, SmemLayoutB{}(_, _, Int<0>{}),
                             Shape<Int<kBN>, Int<kBK>>{}, _1{});
  auto tma_sfa = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFA, SmemLayoutAtomSFA{},
                                         Shape<Int<kBM>, Int<kBK>>{}, _1{});
  auto tma_sfb = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFB, SmemLayoutAtomSFB{},
                                         Shape<Int<kBN>, Int<kBK>>{}, _1{});
  auto tma_c =
    make_tma_copy(SM90_TMA_STORE{}, mC, SmemLayoutO{}, Shape<Int<kBM>, Int<kBN>>{}, _1{});
  dim3 grid((N + kBN - 1) / kBN, (M + kBM - 1) / kBM);

  if constexpr (kWS) {
    auto kernel =
      fp4_gemm_tma_ws_kernel<Traits, decltype(tma_a), decltype(tma_b), decltype(tma_sfa),
                             decltype(tma_sfb), decltype(tma_c), kLevel1A, kLevel1B>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    kernel<<<grid, Traits::kNumThreads + 128, kSmemBytes, stream>>>(tma_a, tma_b, tma_sfa, tma_sfb,
                                                                    tma_c, sa, sb, bias, M, N, K);
  } else {
    auto kernel = fp4_gemm_tma_kernel<Traits, decltype(tma_a), decltype(tma_b), decltype(tma_sfa),
                                      decltype(tma_sfb), decltype(tma_c), kLevel1A, kLevel1B>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    kernel<<<grid, Traits::kNumThreads, kSmemBytes, stream>>>(tma_a, tma_b, tma_sfa, tma_sfb, tma_c,
                                                              sa, sb, bias, M, N, K);
  }
}

inline bool fp4_gemm_use_wide_tile(int M, int N, int K) {
  return M >= 4096 && N >= 4096 && K >= 4096;
}

}  // namespace cache_dit::fp4
