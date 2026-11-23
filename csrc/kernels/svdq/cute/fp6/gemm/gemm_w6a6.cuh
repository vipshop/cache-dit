#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

#include <cuda_bf16.h>

#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/device_kernel.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "common.h"
#include "runtime.h"
#include "sf_partition.cuh"

namespace svdq::cute_kernels {

using namespace cute;

// Reinterpret the per-thread accumulator fragment as (row, column) coordinates. The atom's C layout
// interleaves two rows per register pair; this regrouping exposes row and column as separate modes
// so the epilogue can load one activation scale per row and one weight scale per column.
template <typename Layout>
CUTE_DEVICE auto w6a6_acc_rowcol(Layout acc_layout) {
  auto divided = logical_divide(acc_layout, Shape<_2>{});
  return make_layout(make_layout(get<0, 1>(divided), get<1>(divided)),
                     make_layout(get<0, 0>(divided), get<2>(divided)));
}

// SVDQ W6A6 GEMM with a fused BF16 LoRA-up epilogue.
//
// Mathematical contract:
//   P   = sum_g (sfa[m,g] * sfb[n,g]) * (qA[m,:] @ qB[n,:]^T) over the 32-wide group g
//   out = BF16(P * (ascale[m] * wscale[n]) + bias[n] + D[M,R] @ L1[N,R]^T)
//
// qA/qB are E2M3 codes; sfa/sfb are UE8M0 micro-scales shared by 32 consecutive K elements and are
// consumed inside the MMA (kind::mxf8f6f4.block_scale.scale_vec::1X, e2m3.e2m3, K32 atom). ascale
// and wscale are the FP32 second-level scales from the SVDQuant split. They are constant along K,
// so they still factor out of the reduction: the epilogue multiplies them once, after the whole
// mainloop, and before the LoRA-up MMA, which keeps them out of D @ L1^T.
//
// Physical tensors (all contiguous CUDA):
//   act      [M, K*3/4]  uint8, four packed E2M3 codes per three bytes, row-major
//   wgt      [N, K*3/4]  uint8, same packing for the residual weight
//   asf      [M, K/32]   uint8 UE8M0 (value = 2^(code-127))
//   wsf      [N, K/32]   uint8 UE8M0
//   ascales  [M] FP32, wscales [N] FP32
//   out      [M, N] BF16 row-major
//   lora_act [M, R] BF16 LoRA-down output D, row-major
//   lora_up  [N, R] BF16 L1, row-major
//   bias     [N] BF16, optional
// K must be a multiple of 128 because the packed row stride and the 16U6 TMA inner extent are both
// expressed in 96-byte units. M and N must be multiples of the tile extents.

template <int kBN_ = 128, int kStages_ = 2, int kBK_ = 128>
struct W6A6CuteTraits {
  static constexpr int kBM = 128;
  static constexpr int kBN = kBN_;
  static constexpr int kBK = kBK_;
  static constexpr int kStages = kStages_;
  // LoRA rank is streamed in 32-wide slices, exactly like the W8A8/W4A8 kernels, so one D/L1 slice
  // is 8 KiB and always fits inside one A/B stage.
  static constexpr int kRankTile = 32;
  static constexpr int kNumThreads = kBM / 16 * 32;

  // One MX block-scaled instruction covers 32 K elements with a single UE8M0 factor per side.
  static constexpr int kSfVec = 32;
  static constexpr int kSfPerTile = kBK / kSfVec;

  static_assert(kBK % kSfVec == 0, "BK must contain whole MX scale groups");
  static_assert(kStages >= 2, "kStages >= 2");

  using ElementData = cutlass::detail::float_e2m3_unpacksmem_t;
  // TMA keeps the packed 6-bit stream in gmem, while smem and the MMA consume one 8-bit container
  // per code; the SU6 LDSM expands the packed words while loading fragments.
  using ElementAB = uint8_t;
  using ElementSF = cutlass::float_ue8m0_t;
  using ElementLora = cutlass::bfloat16_t;
  using ElementO = cutlass::bfloat16_t;
  using ElementAcc = float;

  using MmaAtom =
    MMA_Atom<SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<cutlass::float_e2m3_t, cutlass::float_e2m3_t,
                                                     ElementAcc, ElementSF, kSfVec>>;
  using AtomLayoutMNK = Layout<Shape<_4, _2, _1>>;
  using PermutationMNK = Tile<Int<kBM>, _32, _32>;
  using TiledMma = decltype(make_tiled_mma(MmaAtom{}, AtomLayoutMNK{}, PermutationMNK{}));

  // The LoRA-up MMA differs only in K; keeping M/N identical is what lets both products share one
  // accumulator without a retile or a cross-warpgroup reduction.
  using LoraAtom = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
  using LoraPermutationMNK = Tile<Int<kBM>, _32, Int<kRankTile>>;
  using LoraMma = decltype(make_tiled_mma(LoraAtom{}, AtomLayoutMNK{}, LoraPermutationMNK{}));

  static_assert(size(LoraMma{}) == size(TiledMma{}),
                "the LoRA MMA must use the same thread count as the E2M3 MMA");
  static_assert(
    is_same_v<decltype(partition_shape_C(TiledMma{}, make_shape(Int<kBM>{}, Int<kBN>{}))),
              decltype(partition_shape_C(LoraMma{}, make_shape(Int<kBM>{}, Int<kBN>{})))>,
    "the LoRA MMA accumulator partition must match the E2M3 MMA one");

  using SmemLayoutAtomA = UMMA::Layout_K_SW128_Atom<ElementAB>;
  using SmemLayoutAtomB = UMMA::Layout_K_SW128_Atom<ElementAB>;
  using SmemLayoutA =
    decltype(tile_to_shape(SmemLayoutAtomA{}, Shape<Int<kBM>, Int<kBK>, Int<kStages>>{}));
  using SmemLayoutB =
    decltype(tile_to_shape(SmemLayoutAtomB{}, Shape<Int<kBN>, Int<kBK>, Int<kStages>>{}));

  static constexpr int kLoraStages = 2;
  using LoraSmemAtom = UMMA::Layout_K_SW64_Atom<ElementLora>;
  using SmemLayoutD = decltype(tile_to_shape(
    LoraSmemAtom{}, make_shape(Int<kBM>{}, Int<kRankTile>{}, Int<kLoraStages>{})));
  using SmemLayoutL1 = decltype(tile_to_shape(
    LoraSmemAtom{}, make_shape(Int<kBN>{}, Int<kRankTile>{}, Int<kLoraStages>{})));

  // Packed 6-bit operands are expanded by the SU6 LDSM, so A and B share one copy atom.
  using SmemCopyAtomAB = Copy_Atom<SM100_SU6_DU8x16_x4_LDSM_N, ElementAB>;
  using SmemCopyAtomLora = Copy_Atom<SM75_U32x4_LDSM_N, ElementLora>;
  using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;

  using SmemAtomO = GMMA::Layout_K_SW128_Atom<ElementO>;
  using SmemLayoutO = decltype(tile_to_shape(SmemAtomO{}, Shape<Int<kBM>, Int<kBN>>{}));

  static constexpr int kAStageBytes = kBM * kBK;
  static constexpr int kBStageBytes = kBN * kBK;
  static constexpr int kDStageBytes = kBM * kRankTile * static_cast<int>(sizeof(ElementLora));
  static constexpr int kL1StageBytes = kBN * kRankTile * static_cast<int>(sizeof(ElementLora));
  static constexpr int kStageBytes = kAStageBytes + kBStageBytes;
  static constexpr int kMainSmemBytes = kStages * kStageBytes;
  static constexpr int kLoraSmemBytes = kLoraStages * (kDStageBytes + kL1StageBytes);
  static constexpr int kSmemBytes = kMainSmemBytes + kLoraSmemBytes;

  static_assert(kAStageBytes % 1024 == 0 && kBStageBytes % 1024 == 0,
                "the stage stride must preserve the 1024 B smem base alignment");
  static_assert(cosize_v<SmemLayoutO> * static_cast<int>(sizeof(ElementO)) <= kMainSmemBytes,
                "the BF16 output staging tile must fit into the pipeline storage");
  static_assert(kSmemBytes <= 99 * 1024, "pipeline plus LoRA storage exceeds the SM120 budget");

  // TMA counts the packed source bytes (three per four codes); the smem layout reserves one 8-bit
  // container per code.
  static constexpr uint32_t kTxBytesMain =
    static_cast<uint32_t>((kAStageBytes + kBStageBytes) / 4 * 3);
};

template <class TmaA_, class TmaB_, class TmaC_>
struct W6A6CuteTma {
  TmaA_ a;
  TmaB_ b;
  TmaC_ c;
};

struct W6A6CuteCacheKey {
  std::array<uint64_t, 8> data;
  bool operator==(const W6A6CuteCacheKey &) const = default;
};

struct W6A6CuteCacheHash {
  size_t operator()(const W6A6CuteCacheKey &key) const {
    uint64_t hash = 0x9e3779b97f4a7c15ULL;
    for (const uint64_t value : key.data) {
      hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    }
    return static_cast<size_t>(hash);
  }
};

/**
 * Logical scale-factor view of one packed UE8M0 row-major tile.
 *
 * The MMA's SF layouts are indexed by logical (M/N, K) coordinates with a zero stride across each
 * 32-element group, so the packed [rows, K/32] tensor is presented as a (rows, (32, K/32)) tile. No
 * data is replicated: only the address arithmetic of the broadcast is expressed in the layout.
 *
 * :param ptr: First byte of the tile (already offset to the CTA row/column and K tile).
 * :param rows: Tile height in output rows/columns (kBM or kBN).
 * :param row_stride: Distance between two consecutive packed scale rows, in bytes.
 * :return: CuTe tensor view usable by sf_partition::partition_fragment_sf* and the SF TiledCopy.
 */
template <int kRows, int kGroups>
CUTE_HOST_DEVICE auto make_sf_tile_view(typename W6A6CuteTraits<>::ElementSF const *ptr,
                                        int row_stride) {
  return make_tensor(
    make_gmem_ptr(ptr),
    make_layout(
      make_shape(Int<kRows>{}, make_shape(Int<W6A6CuteTraits<>::kSfVec>{}, Int<kGroups>{})),
      make_stride(row_stride, make_stride(_0{}, _1{}))));
}

template <class Traits, class TmaParams>
class W6A6GemmKernel {
  using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
  using CtaBarrier = cutlass::arch::ClusterBarrier;
  static constexpr int kBM = Traits::kBM;
  static constexpr int kBN = Traits::kBN;
  static constexpr int kBK = Traits::kBK;
  static constexpr int kStages = Traits::kStages;
  static constexpr int kRankTile = Traits::kRankTile;
  static constexpr int kSfVec = Traits::kSfVec;
  static constexpr int kSfPerTile = Traits::kSfPerTile;

 public:
  // Asynchronous 16 B gmem -> smem copy; the destination address comes from the swizzled smem
  // layout, so the copy preserves the LDSM fragment pattern without any register roundtrip.
  __device__ __forceinline__ static void cp_async_16(void *dst, void const *src) {
    const uint32_t smem_addr = cute::cast_smem_ptr_to_uint(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(src));
  }

  /**
   * Issue one main-K tile's asynchronous A and B TMA copies.
   *
   * A and B are packed E2M3 rows; the shared transaction barrier counts packed source bytes while
   * the destination smem stage reserves one 8-bit container per code. The caller must already own
   * an empty stage, and the scales are loaded separately through the SF TiledCopies.
   *
   * @param tma     Descriptor bundle; only tma.a and tma.b are used.
   * @param full    Transaction-barrier ring; full[stage] is armed by this call.
   * @param kt      Logical K-tile index in units of BK elements.
   * @param stage   Physical smem ring slot.
   * @param mA/mB   CuTe coordinate tensors for the complete activation/weight tensors.
   * @param sA/sB   Multi-stage smem views; this call writes only (..., stage).
   * @param a_slice/b_slice TMA thread slices connecting global source and smem destination.
   * @return        void; completion is asynchronous and observed through full[stage].
   */
  template <class GmemA, class GmemB, class SmemA, class SmemB, class ASlice, class BSlice>
  __device__ __forceinline__ static void tma_load_A_B(TmaParams const &tma, uint64_t *full, int kt,
                                                      int stage, GmemA const &mA, GmemB const &mB,
                                                      SmemA &sA, SmemB &sB, ASlice const &a_slice,
                                                      BSlice const &b_slice) {
    auto gA = local_tile(mA, Shape<Int<kBM>, Int<kBK>>{}, make_coord(blockIdx.y, kt));
    auto gB = local_tile(mB, Shape<Int<kBN>, Int<kBK>>{}, make_coord(blockIdx.x, kt));
    TmaBarrier::arrive_and_expect_tx(&full[stage], Traits::kTxBytesMain);
    copy(tma.a.with(full[stage]), a_slice.partition_S(gA), a_slice.partition_D(sA(_, _, stage)));
    copy(tma.b.with(full[stage]), b_slice.partition_S(gB), b_slice.partition_D(sB(_, _, stage)));
  }

  /**
   * Execute the block-scaled E2M3 x E2M3 main reduction into the shared FP32 accumulator.
   *
   * Each K tile is split into BK/32 sub-blocks so only one A/B/SFA/SFB atom set is live at a time.
   * The micro-scales enter through make_zip_tensor, which is why this helper never multiplies any
   * scale into tCrC: the MX instruction applies sfa*sfb per 32-element group itself.
   *
   * The scale rows are double buffered: while tile kt runs its MMAs, the two byte-wide scale loads
   * of tile kt+1 are already in flight. Loading them on the critical path instead costs about 15%
   * of the tensor-pipe utilization, because a CTA issues only 24 of these loads per K extent and
   * each one would otherwise stall the first MMA of its tile.
   *
   * @param full/empty Shared barrier rings; warp leaders release the stage after the last MMA.
   * @param tid        Consumer thread index.
   * @param nKt        Number of main K tiles.
   * @param sfa/sfb    Packed UE8M0 rows for the whole problem.
   * @param m0/n0      Global row/column offsets of this CTA tile.
   * @param tiled_mma  Block-scaled tiled MMA descriptor.
   * @param copy_sfa/copy_sfb SF TiledCopies built from sf_partition::layout_sf*_tv.
   * @param tCrSFA0/tCrSFA1 Double-buffered activation scale fragments.
   * @param tCrSFB0/tCrSFB1 Double-buffered weight scale fragments.
   * @return           void; tCrC holds the scaled product on return.
   */
  template <class TiledMma, class CopyA, class CopyB, class CopySFA, class CopySFB, class SrcA,
            class SrcB, class DstA, class DstB, class FragA, class FragB, class FragSFA,
            class FragSFB, class Acc>
  __device__ __forceinline__ static void gemm_fp6_main(
    uint64_t *full, uint64_t *empty, int tid, int nKt, uint8_t const *sfa, uint8_t const *sfb,
    int nSfK, int m0, int n0, TiledMma const &tiled_mma, CopyA const &s2r_copy_a,
    CopyB const &s2r_copy_b, CopySFA const &copy_sfa, CopySFB const &copy_sfb, SrcA const &tCsA,
    SrcB const &tCsB, DstA &tCrA_view, DstB &tCrB_view, FragA &tCrA, FragB &tCrB, FragSFA &tCrSFA0,
    FragSFA &tCrSFA1, FragSFB &tCrSFB0, FragSFB &tCrSFB1, Acc &tCrC) {
    auto sfa_thr_copy = copy_sfa.get_thread_slice(tid);
    auto sfb_thr_copy = copy_sfb.get_thread_slice(tid);
    auto sfa_view0 = sfa_thr_copy.retile_D(tCrSFA0);
    auto sfa_view1 = sfa_thr_copy.retile_D(tCrSFA1);
    auto sfb_view0 = sfb_thr_copy.retile_D(tCrSFB0);
    auto sfb_view1 = sfb_thr_copy.retile_D(tCrSFB1);
    // One SF load covers the whole BK tile: BK/32 bytes per thread per side.
    auto load_sf = [&](int kt, int buf) {
      auto gSFA = make_sf_tile_view<kBM, kSfPerTile>(
        reinterpret_cast<typename Traits::ElementSF const *>(sfa + static_cast<int64_t>(m0) * nSfK +
                                                             kt * kSfPerTile),
        nSfK);
      auto gSFB = make_sf_tile_view<kBN, kSfPerTile>(
        reinterpret_cast<typename Traits::ElementSF const *>(sfb + static_cast<int64_t>(n0) * nSfK +
                                                             kt * kSfPerTile),
        nSfK);
      if (buf == 0) {
        copy(copy_sfa, sfa_thr_copy.partition_S(gSFA), sfa_view0);
        copy(copy_sfb, sfb_thr_copy.partition_S(gSFB), sfb_view0);
      } else {
        copy(copy_sfa, sfa_thr_copy.partition_S(gSFA), sfa_view1);
        copy(copy_sfb, sfb_thr_copy.partition_S(gSFB), sfb_view1);
      }
    };
    load_sf(0, 0);
#pragma unroll 1
    for (int kt = 0; kt < nKt; ++kt) {
      const int stage = kt % kStages;
      const int phase = (kt / kStages) & 1;
      TmaBarrier::wait(&full[stage], phase);
      if (kt + 1 < nKt) load_sf(kt + 1, (kt + 1) & 1);
      // The buffer choice must stay a compile-time constant: binding a reference to a
      // runtime-selected fragment would push both scale buffers into local memory.
      auto run_tile = [&](auto &fSFA, auto &fSFB) {
#pragma unroll
        for (int k_block = 0; k_block < kBK / kSfVec; ++k_block) {
          copy(s2r_copy_a, tCsA(_, _, k_block, stage), tCrA_view(_, _, k_block));
          copy(s2r_copy_b, tCsB(_, _, k_block, stage), tCrB_view(_, _, k_block));
          cute::gemm(tiled_mma, make_zip_tensor(tCrA(_, _, k_block), fSFA(_, _, k_block)),
                     make_zip_tensor(tCrB(_, _, k_block), fSFB(_, _, k_block)), tCrC);
        }
      };
      if (kt & 1) {
        run_tile(tCrSFA1, tCrSFB1);
      } else {
        run_tile(tCrSFA0, tCrSFB0);
      }
      __syncwarp();
      if (tid % 32 == 0) CtaBarrier::arrive(&empty[stage]);
    }
  }

  /**
   * Apply the per-token and per-output-channel scales plus bias to the FP32 accumulator.
   *
   * Math per element: acc = fmaf(acc * ascale[m], wscale[n], bias[n]). This runs before the LoRA
   * MMA, so the scales apply to the E2M3 product only and never to D @ L1^T. ascale depends on the
   * row and wscale on the column, so each thread loads its own small set of both.
   *
   * @param ascales/wscales FP32 scale vectors in gmem.
   * @param bias            Optional BF16 bias [N]; nullptr means zero.
   * @param m0/n0           Global row/column offsets of this CTA tile.
   * @param tScC_rc         Row/column coordinate view of the accumulator partition.
   * @return                void; tCrC is scaled and biased in place.
   */
  template <class CoordC, class Acc>
  __device__ __forceinline__ static void apply_scales(float const *ascales, float const *wscales,
                                                      cutlass::bfloat16_t const *bias, int m0,
                                                      int n0, CoordC const &tScC_rc, Acc &tCrC) {
    constexpr int kCRows = decltype(size<0>(tScC_rc))::value;
    constexpr int kCCols = decltype(size<1>(tScC_rc))::value;
    float row_scale[kCRows];
    float col_scale[kCCols];
    float col_bias[kCCols];
#pragma unroll
    for (int r = 0; r < kCRows; ++r) {
      row_scale[r] = ascales != nullptr ? __ldg(ascales + m0 + get<0>(tScC_rc(r, 0))) : 1.0f;
    }
#pragma unroll
    for (int c = 0; c < kCCols; ++c) {
      const int n = n0 + get<1>(tScC_rc(0, c));
      col_scale[c] = wscales != nullptr ? __ldg(wscales + n) : 1.0f;
      col_bias[c] = bias != nullptr ? static_cast<float>(bias[n]) : 0.0f;
    }
    auto acc = make_tensor(tCrC.data(), w6a6_acc_rowcol(tCrC.layout()));
#pragma unroll
    for (int r = 0; r < kCRows; ++r)
#pragma unroll
      for (int c = 0; c < kCCols; ++c)
        acc(r, c) = fmaf(acc(r, c) * row_scale[r], col_scale[c], col_bias[c]);
  }

  /**
   * Issue the asynchronous copies for one D/L1 rank slice.
   *
   * @param lora_act/lora_up Global BF16 D [M,rank] and L1 [N,rank].
   * @param rank             Total rank; 0, 32, 64 or 128.
   * @param rt               Rank-slice index; covered columns are [rt*32, rt*32+32).
   * @param m0/n0            Global row/column offsets of this CTA tile.
   * @param tid              Participating thread index within the filling group.
   * @param stage            Destination slot in the dedicated LoRA ring.
   * @param num_threads      Number of threads that take part in the fill.
   * @return                 void; completion is observed through cp.async waits, not a barrier.
   */
  template <class SmemD, class SmemL1>
  __device__ __forceinline__ static void fill_lora_stage(cutlass::bfloat16_t const *lora_act,
                                                         cutlass::bfloat16_t const *lora_up,
                                                         int rank, int rt, int m0, int n0, int tid,
                                                         int stage, SmemD &sD, SmemL1 &sL1,
                                                         int num_threads) {
    constexpr int kChunksPerRow = kRankTile * 2 / 16;
    constexpr int kDChunks = kBM * kChunksPerRow;
    constexpr int kL1Chunks = kBN * kChunksPerRow;
    const int col_base = rt * kRankTile;
    auto sD_stage = sD(_, _, stage);
    auto sL1_stage = sL1(_, _, stage);
    for (int chunk = tid; chunk < kDChunks; chunk += num_threads) {
      const int row = chunk / kChunksPerRow;
      const int col = (chunk % kChunksPerRow) * 8;
      cp_async_16(&sD_stage(row, col),
                  lora_act + static_cast<int64_t>(m0 + row) * rank + col_base + col);
    }
    for (int chunk = tid; chunk < kL1Chunks; chunk += num_threads) {
      const int row = chunk / kChunksPerRow;
      const int col = (chunk % kChunksPerRow) * 8;
      cp_async_16(&sL1_stage(row, col),
                  lora_up + static_cast<int64_t>(n0 + row) * rank + col_base + col);
    }
  }

  /**
   * Accumulate the BF16 LoRA-up correction into the same FP32 accumulator.
   *
   * Math per rank slice: tCrC += D_rt @ L1_rt^T. Both operands are BF16 and accumulation is FP32.
   *
   * @param lora_act/lora_up Global BF16 D [M,rank] and L1 [N,rank].
   * @param rank             Total rank; 0, 32, 64 or 128.
   * @param m0/n0            Global row/column offsets of this CTA tile.
   * @param nRt              Number of 32-wide rank slices.
   * @return                 void; all rank slices have been accumulated on return.
   */
  template <class SmemD, class SmemL1, class Acc>
  __device__ __forceinline__ static void gemm_bf16_lora(cutlass::bfloat16_t const *lora_act,
                                                        cutlass::bfloat16_t const *lora_up,
                                                        int rank, int m0, int n0, int tid, int nRt,
                                                        SmemD &sD, SmemL1 &sL1, Acc &tCrC) {
    if (nRt == 0) return;
    typename Traits::LoraMma lora_mma;
    auto thr_lora = lora_mma.get_thread_slice(tid);
    auto tCrD = thr_lora.partition_fragment_A(sD(_, _, Int<0>{}));
    auto tCrL1 = thr_lora.partition_fragment_B(sL1(_, _, Int<0>{}));
    auto s2r_copy_d = make_tiled_copy_A(typename Traits::SmemCopyAtomLora{}, lora_mma);
    auto s2r_copy_l1 = make_tiled_copy_B(typename Traits::SmemCopyAtomLora{}, lora_mma);
    auto s2r_thr_d = s2r_copy_d.get_thread_slice(tid);
    auto s2r_thr_l1 = s2r_copy_l1.get_thread_slice(tid);
    auto tCsD = s2r_thr_d.partition_S(as_position_independent_swizzle_tensor(sD));
    auto tCsL1 = s2r_thr_l1.partition_S(as_position_independent_swizzle_tensor(sL1));
    auto tCrD_view = s2r_thr_d.retile_D(tCrD);
    auto tCrL1_view = s2r_thr_l1.retile_D(tCrL1);
    constexpr int kLoraStages = Traits::kLoraStages;
#pragma unroll 1
    for (int rt = 0; rt < nRt; ++rt) {
      const int stage = rt % kLoraStages;
      if (rt >= kLoraStages) {
        // Ranks beyond the prefetched ring refill on demand; only rank 128 reaches this path.
        cutlass::arch::NamedBarrier::sync(Traits::kNumThreads, 1);
        fill_lora_stage(lora_act, lora_up, rank, rt, m0, n0, tid, stage, sD, sL1,
                        Traits::kNumThreads);
        cute::cp_async_fence();
        cute::cp_async_wait<0>();
        cutlass::arch::NamedBarrier::sync(Traits::kNumThreads, 1);
      }
      copy(s2r_copy_d, tCsD(_, _, _, stage), tCrD_view);
      copy(s2r_copy_l1, tCsL1(_, _, _, stage), tCrL1_view);
      cute::gemm(lora_mma, tCrD, tCrL1, tCrC);
    }
  }

  /**
   * Convert the final FP32 accumulator once and stage the BF16 tile in smem.
   *
   * @return void; the BF16 tile is visible in smem after the trailing barrier.
   */
  template <class TiledMma, class Acc>
  __device__ __forceinline__ static void stage_output_tile(void *shm, int tid,
                                                           TiledMma const &tiled_mma,
                                                           Acc const &tCrC) {
    using ElementO = typename Traits::ElementO;
    using SmemLayoutO = typename Traits::SmemLayoutO;
    constexpr int kFrag = decltype(size(tCrC))::value;
    cutlass::Array<ElementO, kFrag> acc_bf16;
    cutlass::NumericArrayConverter<ElementO, float, kFrag> convert;
    acc_bf16 = convert(*reinterpret_cast<cutlass::Array<float, kFrag> const *>(tCrC.data()));
    auto tCrCh = make_tensor(make_rmem_ptr(acc_bf16.data()), tCrC.layout());
    __syncthreads();
    auto r2s_copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, ElementO>{}, tiled_mma);
    auto r2s_thr = r2s_copy.get_thread_slice(tid);
    auto sC = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(shm)), SmemLayoutO{});
    copy(r2s_copy, r2s_thr.retile_S(tCrCh), r2s_thr.partition_D(sC));
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
  }

  /**
   * Publish the staged smem tile through the output TMA descriptor and drain the store.
   *
   * @return void; the output tile is visible in global memory after the drain.
   */
  template <class GmemC, class CSlice>
  __device__ __forceinline__ static void tma_store_O(TmaParams const &tma, GmemC const &mC,
                                                     CSlice const &c_slice, void *shm, int tid) {
    using ElementO = typename Traits::ElementO;
    using SmemLayoutO = typename Traits::SmemLayoutO;
    auto sC = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(shm)), SmemLayoutO{});
    auto gC = local_tile(mC, Shape<Int<kBM>, Int<kBN>>{}, make_coord(blockIdx.y, blockIdx.x));
    if (tid == 0) copy(tma.c, c_slice.partition_S(sC), c_slice.partition_D(gC));
    cute::tma_store_arrive();
    cute::tma_store_wait<0>();
  }
};

template <class Traits, class TmaParams>
__global__ void __launch_bounds__(Traits::kNumThreads + 128, 1)
  gemm_w6a6_kernel(CUTLASS_GRID_CONSTANT TmaParams const tma, float const *__restrict__ ascales,
                   float const *__restrict__ wscales, uint8_t const *__restrict__ sfa,
                   uint8_t const *__restrict__ sfb,
                   cutlass::bfloat16_t const *__restrict__ lora_act,
                   cutlass::bfloat16_t const *__restrict__ lora_up,
                   cutlass::bfloat16_t const *__restrict__ bias, int M, int N, int K, int rank) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
  using CtaBarrier = cutlass::arch::ClusterBarrier;
  using ElementAB = typename Traits::ElementAB;
  using ElementLora = typename Traits::ElementLora;
  using ElementSF = typename Traits::ElementSF;
  using TiledMma = typename Traits::TiledMma;
  using Gemm = W6A6GemmKernel<Traits, TmaParams>;
  constexpr int kBM = Traits::kBM;
  constexpr int kBN = Traits::kBN;
  constexpr int kBK = Traits::kBK;
  constexpr int kStages = Traits::kStages;
  constexpr int kLoraStages = Traits::kLoraStages;
  constexpr int kNumThreads = Traits::kNumThreads;
  constexpr int kSfPerTile = Traits::kSfPerTile;

  const int raw_tid = threadIdx.x;
  const bool is_producer = raw_tid < 128;
  const int tid = is_producer ? 0 : raw_tid - 128;
  const int m0 = blockIdx.y * kBM;
  const int n0 = blockIdx.x * kBN;
  const int nKt = K / kBK;
  const int nSfK = K / Traits::kSfVec;
  const int nRt = rank / Traits::kRankTile;

  auto mA = tma.a.get_tma_tensor(make_shape(M, K));
  auto mB = tma.b.get_tma_tensor(make_shape(N, K));
  auto mC = tma.c.get_tma_tensor(make_shape(M, N));
  auto a_slice = tma.a.get_slice(_0{});
  auto b_slice = tma.b.get_slice(_0{});
  auto c_slice = tma.c.get_slice(_0{});

  extern __shared__ __align__(1024) uint8_t shm[];
  uint8_t *a_bytes = shm;
  uint8_t *b_bytes = a_bytes + Traits::kAStageBytes * kStages;
  uint8_t *d_bytes = shm + Traits::kMainSmemBytes;
  uint8_t *l1_bytes = d_bytes + kLoraStages * Traits::kDStageBytes;

  __shared__ uint64_t full[kStages];
  __shared__ uint64_t empty[kStages];
  if (raw_tid == 0) {
    for (int s = 0; s < kStages; ++s) {
      TmaBarrier::init(&full[s], 1);
      CtaBarrier::init(&empty[s], kNumThreads / 32);
    }
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();

  auto sA_full = make_tensor(make_smem_ptr(reinterpret_cast<ElementAB *>(a_bytes)),
                             typename Traits::SmemLayoutA{});
  auto sB_full = make_tensor(make_smem_ptr(reinterpret_cast<ElementAB *>(b_bytes)),
                             typename Traits::SmemLayoutB{});
  auto sD_full = make_tensor(make_smem_ptr(reinterpret_cast<ElementLora *>(d_bytes)),
                             typename Traits::SmemLayoutD{});
  auto sL1_full = make_tensor(make_smem_ptr(reinterpret_cast<ElementLora *>(l1_bytes)),
                              typename Traits::SmemLayoutL1{});

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto s2r_copy_a = make_tiled_copy_A(typename Traits::SmemCopyAtomAB{}, tiled_mma);
  auto s2r_copy_b = make_tiled_copy_B(typename Traits::SmemCopyAtomAB{}, tiled_mma);
  auto s2r_thr_a = s2r_copy_a.get_thread_slice(tid);
  auto s2r_thr_b = s2r_copy_b.get_thread_slice(tid);
  auto tCsA = s2r_thr_a.partition_S(as_position_independent_swizzle_tensor(sA_full));
  auto tCsB = s2r_thr_b.partition_S(as_position_independent_swizzle_tensor(sB_full));
  auto tCrA = thr_mma.partition_fragment_A(sA_full(_, _, Int<0>{}));
  auto tCrB = thr_mma.partition_fragment_B(sB_full(_, _, Int<0>{}));
  auto tCrA_view = s2r_thr_a.retile_D(tCrA);
  auto tCrB_view = s2r_thr_b.retile_D(tCrB);

  // SF fragments are partitioned from a reference tile whose only data is the packed scale row; the
  // fragment shapes come from the atom's SF layouts, and the values are filled by the SF copies.
  auto sf_ref_a = make_sf_tile_view<kBM, kSfPerTile>(
    reinterpret_cast<ElementSF const *>(sfa + static_cast<int64_t>(m0) * nSfK), nSfK);
  auto sf_ref_b = make_sf_tile_view<kBN, kSfPerTile>(
    reinterpret_cast<ElementSF const *>(sfb + static_cast<int64_t>(n0) * nSfK), nSfK);
  auto tCrSFA = sf_partition::partition_fragment_sfa(sf_ref_a, thr_mma);
  auto tCrSFB = sf_partition::partition_fragment_sfb(sf_ref_b, thr_mma);
  // Second scale buffer: same fragment geometry, separate registers, so tile kt+1 can be fetched
  // while tile kt is still issuing MMAs.
  auto tCrSFA_alt = tCrSFA;
  auto tCrSFB_alt = tCrSFB;
  auto copy_sfa = make_tiled_copy_impl(
    typename Traits::SmemCopyAtomSF{}, sf_partition::layout_sfa_tv(tiled_mma),
    make_shape(size<0>(tile_shape(tiled_mma)), size<2>(tile_shape(tiled_mma))));
  auto copy_sfb = make_tiled_copy_impl(
    typename Traits::SmemCopyAtomSF{}, sf_partition::layout_sfb_tv(tiled_mma),
    make_shape(size<1>(tile_shape(tiled_mma)), size<2>(tile_shape(tiled_mma))));

  auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<kBM>, Int<kBN>>{});
  clear(tCrC);
  auto cC = make_identity_tensor(Shape<Int<kBM>, Int<kBN>>{});
  auto tScC = thr_mma.partition_C(cC);
  auto tScC_rc = make_tensor(tScC.data(), w6a6_acc_rowcol(tScC.layout()));

  if (is_producer) {
    cutlass::arch::warpgroup_reg_dealloc<32>();
    if (raw_tid == 0) {
      for (int g = 0; g < nKt; ++g) {
        const int stage = g % kStages;
        const int phase = (g / kStages) & 1;
        CtaBarrier::wait(&empty[stage], phase);
        Gemm::tma_load_A_B(tma, full, g, stage, mA, mB, sA_full, sB_full, a_slice, b_slice);
      }
    }
    return;
  }

  cutlass::arch::warpgroup_reg_alloc<224>();
  if (tid % 32 == 0) {
    for (int s = 0; s < kStages; ++s) CtaBarrier::arrive(&empty[s]);
  }

  // A single consumer warp asynchronously prefetches the first LoRA rank slices; the other warps
  // enter the mainloop immediately. The dedicated LoRA ring is disjoint from the main pipeline, so
  // the copies overlap the reduction and are waited on only before the low-rank epilogue.
  const int nPrefetch = nRt < kLoraStages ? nRt : kLoraStages;
  if (tid < 32) {
    for (int rt = 0; rt < nPrefetch; ++rt) {
      Gemm::fill_lora_stage(lora_act, lora_up, rank, rt, m0, n0, tid, rt, sD_full, sL1_full, 32);
    }
    if (nPrefetch > 0) cute::cp_async_fence();
  }

  Gemm::gemm_fp6_main(full, empty, tid, nKt, sfa, sfb, nSfK, m0, n0, tiled_mma, s2r_copy_a,
                      s2r_copy_b, copy_sfa, copy_sfb, tCsA, tCsB, tCrA_view, tCrB_view, tCrA, tCrB,
                      tCrSFA, tCrSFA_alt, tCrSFB, tCrSFB_alt, tCrC);
  Gemm::apply_scales(ascales, wscales, bias, m0, n0, tScC_rc, tCrC);
  if (nPrefetch > 0) {
    cute::cp_async_wait<0>();
    __syncthreads();
  }
  Gemm::gemm_bf16_lora(lora_act, lora_up, rank, m0, n0, tid, nRt, sD_full, sL1_full, tCrC);
  Gemm::stage_output_tile(shm, tid, tiled_mma, tCrC);
  Gemm::tma_store_O(tma, mC, c_slice, shm, tid);
#endif
}

/**
 * Launch one fused W6A6 + LoRA GEMM.
 *
 * Descriptors are cached per exact pointer/shape/rank key because they are pure copy recipes. The
 * packed E2M3 operands use the 16U6 TMA data type, which keeps gmem at three bytes per four codes.
 *
 * :param act: Packed E2M3 activation codes [M, K*3/4].
 * :param wgt: Packed E2M3 residual codes [N, K*3/4].
 * :param out: BF16 output [M,N].
 * :param ascales: FP32 per-token activation scales [M].
 * :param wscales: FP32 per-output-channel residual scales [N].
 * :param asf: UE8M0 activation micro-scales [M, K/32].
 * :param wsf: UE8M0 weight micro-scales [N, K/32].
 * :param lora_act: BF16 LoRA-down output [M,rank], may be null when rank is 0.
 * :param lora_up: BF16 LoRA-up weight [N,rank], may be null when rank is 0.
 * :param bias: Optional BF16 bias [N].
 * :param M/N/K: Logical extents; K refers to unpacked elements, not packed bytes.
 * :param rank: LoRA rank, 0/32/64/128.
 * :param stream: Launch stream.
 */
template <int kBN = 128, int kStages = 2, int kBK = 128>
void launch_gemm_w6a6(void const *act, void const *wgt, void *out, float const *ascales,
                      float const *wscales, void const *asf, void const *wsf,
                      cutlass::bfloat16_t const *lora_act, cutlass::bfloat16_t const *lora_up,
                      cutlass::bfloat16_t const *bias, int M, int N, int K, int rank,
                      cudaStream_t stream) {
  using Traits = W6A6CuteTraits<kBN, kStages, kBK>;
  using ElementData = typename Traits::ElementData;
  using ElementO = typename Traits::ElementO;
  constexpr int kBM = Traits::kBM;

  auto mA = make_tensor(make_gmem_ptr(recast_ptr<ElementData const>(act)), make_shape(M, K),
                        make_stride(K, _1{}));
  auto mB = make_tensor(make_gmem_ptr(recast_ptr<ElementData const>(wgt)), make_shape(N, K),
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
    using Params = W6A6CuteTma<decltype(tma_a), decltype(tma_b), decltype(tma_c)>;
    return Params{tma_a, tma_b, tma_c};
  };
  using TmaParams = decltype(build_params());
  auto address = [](const void *ptr) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
  };
  W6A6CuteCacheKey key{{address(act), address(wgt), address(out), address(asf), address(wsf),
                        static_cast<uint64_t>(M) << 32 | static_cast<uint32_t>(N),
                        static_cast<uint64_t>(K) << 32 | static_cast<uint32_t>(rank), 6}};
  thread_local std::unordered_map<W6A6CuteCacheKey, TmaParams, W6A6CuteCacheHash> cache;
  auto it = cache.find(key);
  if (it == cache.end()) {
    if (cache.size() >= 32) cache.clear();
    it = cache.emplace(key, build_params()).first;
  }

  auto kernel = gemm_w6a6_kernel<Traits, TmaParams>;
  static thread_local std::unordered_set<int> configured_devices;
  int device = 0;
  checkCUDA(cudaGetDevice(&device));
  if (configured_devices.insert(device).second) {
    checkCUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   Traits::kSmemBytes));
  }
  kernel<<<dim3(N / kBN, M / kBM), Traits::kNumThreads + 128, Traits::kSmemBytes, stream>>>(
    it->second, ascales, wscales, static_cast<uint8_t const *>(asf),
    static_cast<uint8_t const *>(wsf), lora_act, lora_up, bias, M, N, K, rank);
}

}  // namespace svdq::cute_kernels
