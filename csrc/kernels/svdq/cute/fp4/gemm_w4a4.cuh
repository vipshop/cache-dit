#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

#include <cuda_bf16.h>

#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/tensor_zip.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/device_kernel.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "common.h"

namespace svdq::cute_kernels {

using namespace cute;

// Reading guide for the CuTe terms used below:
//   gmem     = device global memory; smem = CTA shared memory; RF = per-thread registers.
//   TMA      = asynchronous bulk copy engine between gmem and smem.
//   fragment = the small per-thread register tile consumed or produced by one MMA atom.
//   Layout   = a compile-time map from logical coordinates to physical addresses, not a transpose.
//   partition_S/partition_D = create the source/destination per-thread views of a TiledCopy.
//   mbarrier = shared-memory arrival/transaction barrier used to publish one pipeline stage.
//
// Names used by the math and layouts:
//   SF      = Scale Factor, the FP8_E4M3 micro-scale shared by 16 NVFP4 codes; it is NOT an
//             abbreviation for shared memory or shared fragment. SFA is activation SF and SFB is
//             weight/B-side SF. ascales/wscales are the corresponding Python/native tensor names.
//   A/B     = activation and weight matrices of the main NVFP4 GEMM P=A@B^T.
//   D       = BF16 result of the LoRA-down projection, with shape [M,R].
//   L1      = BF16 LoRA-up weight, with shape [N,R]. L1 is a matrix name, NOT the L1 cache.
//   O       = final BF16 output tile in smem; C/tCrC is the FP32 accumulator before conversion.
//   mX      = a gmem tensor view, sX = an smem tensor view, tCrX = a thread register fragment.
//   tCsX    = a per-thread copy-source view; g = global pipeline step, stage = physical ring slot.
//
// The comments in gemm_w4a4_kernel are organized as Stage 1..9 and follow one CTA from
// descriptor-backed gmem views to the final BF16 output tile.

// =================================================================================================
// SVDQ NVFP4 + BF16 LoRA-up: mathematical and dataflow contract
// =================================================================================================
// For each 16-element K group, NVFP4 stores E2M1 codes plus one FP8_E4M3 scale:
//
//   dequant(code[k]) = E2M1(code[k]) * FP8_E4M3(scale[k / 16])
//   P = dequant(A) [M,K] @ dequant(B)^T [K,N]
//   Y = BF16((alpha * P) * wcscales[N] + bias[N] + D [M,R] @ L1^T [R,N])
//
// D is the LoRA-down result and L1 is the LoRA-up weight. lora_scales are folded into D by the
// independent CuTe activation producer before this GEMM starts. alpha and wcscales apply only to
// the NVFP4 product; neither may scale the LoRA term. Both matrix products accumulate into the
// same FP32 C fragment, so there is no cross-warpgroup reduction and only one final BF16 rounding.
//
// The implementation uses one standard TMA input path for all four NVFP4 operands:
//   A/B/SFA/SFB: standard packed gmem -> TMA -> smem -> register fragment
//   D/L1:        row-major BF16 gmem -> TMA -> A/B smem overlay -> BF16 register fragments
//
// The independent producer emits standard A/SFA and row-major BF16 D directly, so no v1 fragment,
// FP32 LoRA scratch, G2R activation load, or finalize kernel remains. The two CuTe TiledMMAs
// have different input types and K extents, but statically identical C partitions.
//
// Only BF16 D/L1 and BF16 output are instantiated here. FP16 LoRA remains on the old native
// path. Adding FP16 is not a pointer cast: it needs an F32F16F16F32 atom, matching TMA/copy
// types, an FP16 output ABI, and a separately verified rounding contract.

// NVFP4 block scaling carries one FP8_E4M3 factor per 16 K elements. E2M1 is the signed 4-bit
// code type; UE4M3 is the unsigned 8-bit scale type (scales are non-negative). CUTLASS stores
// these scales in 128-row x 4-factor (512 B) atoms, exactly the box of one BK=64 K step.
constexpr int kSFVec = 16;
constexpr int kSFVecPerBlk = 4;
constexpr int kRowBlock = 128;

// Express the flat scale-factor allocation in logical data coordinates (mn,K), not in a second
// packed-byte coordinate system. For row=n and scale group g=k/16, the physical byte offset is
//
//   (n/128) * (K/16 * 128) + (g/4) * 512
//     + 16 * (n % 32) + 4 * ((n/32) % 4) + (g % 4).
//
// The K mode has shape 16 with stride zero inside a group: all 16 codes logically reference the
// same scale, but the allocation contains only one byte, not 16 copies. Step<_2,_1> tiles the
// atom K-first and then advances to the next 128-row block. This layout is used for SFB TMA;
// ascales has the same byte count but intentionally retains the different v1 producer ordering.
CUTE_HOST_DEVICE constexpr auto sf_gmem_layout(int mn, int K) {
  using Config = cutlass::detail::Sm1xxBlockScaledConfig<kSFVec>;
  return tile_to_shape(typename Config::SfAtom{}, make_shape(mn, K), Step<_2, _1>{});
}

constexpr size_t sf_buffer_bytes(int mn, int K) {
  const int mn_padded = (mn + kRowBlock - 1) / kRowBlock * kRowBlock;
  return static_cast<size_t>(mn_padded / kRowBlock) * (K / 64) * 512;
}

// Smem counterpart of the 512 B scale atom. ((32,4)):(16,4) maps 128 logical rows onto four
// 32-row panels, while (16,4):(0,1) retains the K broadcast and four consecutive FP8 factors.
// kMN may be 128 or 256; kMN/128 prepends complete 512 B row blocks. kBK/64 similarly prepends
// whole K atoms when a pipeline stage contains two m16n8k64 instructions. The smem and gmem
// orderings therefore match, so TMA can copy the scales without a runtime shuffle. The later
// partition helpers, rather than a flat smem pointer cast, recover each thread's SF register.
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
  using strideK = decltype(prepend(make_stride(Int<kMMA_NSF>{}, Int<kMN / kRowBlock * 512>{}),
                                   kBlockStride{}));
  using type = decltype(make_layout(make_shape(shapeM{}, shapeK{}),
                                    make_stride(strideM{}, strideK{})));
};

// Scale-factor operands are neither plain smem tensors nor ordinary row-major fragments. The
// MMA atom's SFALayout/SFBLayout maps (thread,value) to logical (M/N,K), including zero-stride
// broadcasts and the 32-wide N permutation. The helpers below mirror the CUTLASS SM120
// collective in three stages:
//   1. thrfrg_sf*: compose the logical tensor with the atom's SF TV layout;
//   2. partition_fragment_sf*: select this thread's slice and materialize one packed u32;
//   3. layout_sf*_tv: construct the inverse thread map used by the smem->register tiled copy.
// A direct linear load would put another lane's scales into the MMA operand. The zip tensor in
// the mainloop then presents {data,SF} as the two operands required by block-scaled CuTe MMA.
namespace sf_partition {

// thrfrg_sfa = "thread-fragment view for SFA". It answers: if the activation SF tensor is viewed
// through this TiledMMA, which logical (M,K) coordinates belong to each thread and each value?
// The function only composes layouts; it neither allocates registers nor loads any scale.
//   logical_divide: split the full SF tensor at the TiledMMA's M/K permutation boundaries.
//   zipped_divide:  separate one m16n8k64 atom from the remaining M/K atom repetitions.
//   compose(SFALayout): apply the hardware atom's special (thread,value)->(M,K) map.
//   final zipped_divide: expose a hierarchical (thread/value, remaining M/K tiles) view.
// partition_fragment_sfa later uses this view to select one physical thread's register fragment.
template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto thrfrg_sfa(SFATensor &&sfatensor,
                                           TiledMMA<Atom, TiledThr, TiledPerm> &mma) {
  CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});
  using AtomShape_MNK = typename Atom::Shape_MNK;
  using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;
  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto t_tile = make_tile(get<0>(permutation_mnk), get<2>(permutation_mnk));
  auto t_tensor = logical_divide(sfatensor, t_tile);
  auto a_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                          make_layout(size<2>(AtomShape_MNK{})));
  auto a_tensor = zipped_divide(t_tensor, a_tile);
  auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{}, _);
  auto thr_tile = make_tile(_, make_tile(make_layout(size<1>(thr_layout_vmnk)),
                                        make_layout(size<3>(thr_layout_vmnk))));
  return zipped_divide(tv_tensor, thr_tile);
}

// B-side counterpart of thrfrg_sfa. The structure is the same, but the logical coordinates are
// (N,K), the atom tile uses size<1>(AtomShape_MNK), and thread tiling uses the N/K modes of
// thr_layout_vmnk. SFBLayout is not interchangeable with SFALayout: an m16n8k64 instruction owns
// activation and weight scales in different lane patterns, so sharing the A helper would assign
// weight scales to the wrong output columns.
template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto thrfrg_sfb(SFBTensor &&sfbtensor,
                                           TiledMMA<Atom, TiledThr, TiledPerm> &mma) {
  CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});
  using AtomShape_MNK = typename Atom::Shape_MNK;
  using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;
  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto t_tile = make_tile(get<1>(permutation_mnk), get<2>(permutation_mnk));
  auto t_tensor = logical_divide(sfbtensor, t_tile);
  auto a_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})),
                          make_layout(size<2>(AtomShape_MNK{})));
  auto a_tensor = zipped_divide(t_tensor, a_tile);
  auto tv_tensor = a_tensor.compose(AtomLayoutSFB_TV{}, _);
  auto thr_tile = make_tile(_, make_tile(make_layout(size<2>(thr_layout_vmnk)),
                                        make_layout(size<3>(thr_layout_vmnk))));
  return zipped_divide(tv_tensor, thr_tile);
}

// Materialize this thread's SFA register fragment from the abstract thrfrg view.
// thread_mma.thr_vmnk_ is this lane's position in the TiledMMA. For A we keep its V/M/K position
// and drop N: activation scales do not change when the same A row is reused across output columns.
// thr_tensor still points at the original SF storage. Indexing it with thr_vmk selects the source
// values; make_fragment_like then allocates the register fragment with the atom-required layout.
// Per m16n8k64 atom this is one u32 containing four UE4M3 bytes for four consecutive K groups.
// The actual smem -> register read happens later through the SF TiledCopy, not here.
template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_fragment_sfa(SFATensor &&sfatensor,
                                                      ThrMma &thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  auto thr_tensor = make_tensor(static_cast<SFATensor &&>(sfatensor).data(),
                                thrfrg_sfa(sfatensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vmk =
    make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
  auto part = thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
  return make_fragment_like<ValTypeSF>(part);
}

// SFB fragment construction mirrors partition_fragment_sfa, except thr_vnk keeps V/N/K and drops
// M. Weight scales depend on output column and K group, but are unchanged when the same B tile is
// reused across output rows. The returned u32 fragment is paired with tCrB by make_zip_tensor in
// the NVFP4 mainloop; no scale values are fetched until the explicit copy is issued.
template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_fragment_sfb(SFBTensor &&sfbtensor,
                                                      ThrMma &thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  auto thr_tensor = make_tensor(static_cast<SFBTensor &&>(sfbtensor).data(),
                                thrfrg_sfb(sfbtensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vnk =
    make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
  auto part = thr_tensor(thr_vnk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
  return make_fragment_like<ValTypeSF>(part);
}

// layout_sfa_tv builds the TV (Thread-Value) layout used to construct an SFA TiledCopy.
// It is a compile-time answer to "lane L's value V must read which logical (M,K) scale?".
// It is not the gmem 512 B atom layout, does not allocate a fragment, and performs no copy itself.
//   ref_A: an identity logical (M,K) reference covering the whole TiledMMA.
//   atile: adds atom repetition/broadcast information; the zero-stride N mode means the same SFA
//          values are reused as the TiledMMA moves across output columns.
//   thrfrg_sfa(ref_A,mma): applies the block-scaled atom's SFA mapping to that reference.
//   right_inverse(thr_layout_vmnk): converts hierarchical VMNK coordinates back to a linear lane.
// The two compose calls chain those maps into one TV layout. make_tiled_copy_impl then combines
// it with UniversalCopy<UE4M3> to generate the instructions that fill tCrSFA_view.
template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto layout_sfa_tv(TiledMma &mma) {
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto atile = make_tile(_, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk),
                                                            size<2>(thr_layout_vmnk)),
                                                  make_stride(Int<1>{}, Int<0>{})),
                                      _));
  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  return thrfrg_sfa(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
}

// layout_sfb_tv is the B-side TV copy plan: (lane,value)->logical (N,K). ref_B covers output
// columns rather than rows, and btile's zero stride is on the M mode because a weight scale is
// reused when the same B tile advances across output rows. The returned layout drives the SFB
// TiledCopy that loads tCrSFB_view. Keeping this separate from layout_sfa_tv is what preserves
// the different activation/weight scale ownership rules of the block-scaled MMA atom.
template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto layout_sfb_tv(TiledMma &mma) {
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto btile = make_tile(_, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk),
                                                            size<2>(thr_layout_vmnk)),
                                                  make_stride(Int<0>{}, Int<1>{})),
                                      _));
  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  return thrfrg_sfb(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
}

}  // namespace sf_partition

// Convert an accumulator layout from CuTe's hierarchical (MMA,MMA_M,MMA_N) form to this
// thread's (row,column) form. The first MMA mode packs two output columns per register pair;
// logical_divide separates that pair from the M-atom and N-atom repetition modes. This changes
// only the coordinate view: no registers move and no values are converted. Stage 7 uses it to
// apply the correct per-column scale/bias to every accumulator value.
template <typename Layout>
CUTE_DEVICE auto convert_layout_acc_rowcol(Layout acc_layout) {
  auto divided = logical_divide(acc_layout, Shape<_2>{});
  return make_layout(make_layout(get<0, 1>(divided), get<1>(divided)),
                     make_layout(get<0, 0>(divided), get<2>(divided)));
}

// Compile-time tile and MMA selection.
//
// One CTA owns a BM x BN output tile. BM=128 maps to four M atoms and two N atoms per 32-wide
// permutation, i.e. AtomLayoutMNK=(4,2,1) and 8 consumer warps. BK=64 is one block-scaled MMA;
// BK=128 places two independent K atoms in the same smem stage. The default configuration is
// BN=256/BK=128/stages=3, with BN=128/BK=64/stages=4 as the narrower fallback.
//
// NVFP4 uses SM120_16x8x64_TN_VS<E2M1,E2M1,F32,UE4M3,16>. "TN" means A[M,K] and B[N,K] are
// both K-major, so B is logically transposed by the GEMM, not physically transposed in gmem.
// scale_vec::4X consumes four FP8 scales per instruction, covering 4*16=64 K elements.
//
// LoRA uses the SM80-compatible BF16 m16n8k16 tensor-core atom with FP32 accumulation. It keeps
// exactly the same atom layout and M/N permutation as the NVFP4 TiledMMA; only K changes to
// kRankTile. Consequently partition_shape_C is identical and D@L1^T can accumulate directly
// into tCrC without retile, smem reduction, or a second accumulator. Executing both products on
// the same consumer warps is intentional: separate BF16/NVFP4 warpgroups would need an expensive
// cross-warpgroup FP32 reduction and would duplicate accumulator storage.
// TODO(cute-gemm-performance): evaluate persistent CTA scheduling, stage-count/smem-overlay
// tuning, and separate NVFP4 vs BF16-LoRA consumer WGs (one producer WG plus two consumer WGs).
// The WG split must preserve the single FP32 accumulator contract or use a measured low-cost
// reduction; it is not a drop-in scheduling change.
template <int kBN_ = 128, int kStages_ = 4, int kBK_ = 64>
struct W4A4CuteTraits {
  static constexpr int kBM = 128;
  static constexpr int kBN = kBN_;
  static constexpr int kBK = kBK_;
  static constexpr int kStages = kStages_;
  // A BF16 rank tile of BM x (BK/4) is byte-identical to a BM x BK E2M1 stage.
  static constexpr int kRankTile = kBK / 4;
  static constexpr int kNumThreads = kBM / 16 * 32;

  static_assert(kBN % kRowBlock == 0,
                "kBN must be a multiple of 128: the scale-factor smem/TMA atom spans 128 rows");
  static_assert(kBK % 64 == 0, "kBK must contain whole 64-element block-scaled MMA atoms");
  static_assert(kStages >= 2, "kStages >= 2");
  static_assert(kRankTile % 16 == 0, "one rank tile must contain whole BF16 MMA atoms");

  using Element = cutlass::float_e2m1_t;
  using ElementSF = cutlass::float_ue4m3_t;
  using ElementLora = cutlass::bfloat16_t;
  using ElementO = cutlass::bfloat16_t;
  using ElementAcc = float;

  using MmaAtom = SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<Element, Element, ElementAcc,
                                                         ElementSF, kSFVec>;
  using AtomLayoutMNK = Layout<Shape<_4, _2, _1>>;
  // The N permutation ((8,2,2):(1,16,8)) spans 32 output columns and is part of the atom's
  // thread/value mapping. It is a compile-time coordinate recipe, not a runtime N transpose.
  // Reusing this exact permutation for LoRA is what makes the two accumulator layouts compatible.
  using PermutationMNK = Tile<Int<kBM>, Layout<Shape<_8, _2, _2>, Stride<_1, _16, _8>>, _64>;
  using TiledMma = decltype(make_tiled_mma(MmaAtom{}, AtomLayoutMNK{}, PermutationMNK{}));

  // The LoRA-up MMA only differs in K: same atom layout and same M/N permutation, so both
  // tiled MMAs cut the CTA accumulator into identical per-thread fragments.
  using LoraAtom = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
  using LoraPermutationMNK =
    decltype(make_tile(get<0>(PermutationMNK{}), get<1>(PermutationMNK{}), Int<kRankTile>{}));
  using LoraMma = decltype(make_tiled_mma(LoraAtom{}, AtomLayoutMNK{}, LoraPermutationMNK{}));

  static_assert(size(LoraMma{}) == size(TiledMma{}),
                "the LoRA MMA must use the same thread count as the NVFP4 MMA");
  static_assert(is_same_v<decltype(partition_shape_C(TiledMma{}, make_shape(Int<kBM>{},
                                                                          Int<kBN>{}))),
                          decltype(partition_shape_C(LoraMma{}, make_shape(Int<kBM>{},
                                                                         Int<kBN>{})))>,
                "the LoRA MMA accumulator partition must match the NVFP4 MMA one");
  static_assert(kRankTile % size<2>(typename LoraAtom::Shape_MNK{}) == 0,
                "one rank tile must be a whole number of BF16 MMA atoms");

  // SW32 is the smem address-swizzle atom used for K-major FP4 data. TMA writes packed bytes in
  // this order and LDSM later reads the matching register pattern; the swizzle changes addresses,
  // not the logical (M/N,K) values. tile_to_shape repeats the atom over BM/BN, BK and stages.
  // uint4_t is only the 16-byte allocation/copy unit, not a claim that one element has 128 bits.
  using SmemAlloc = uint4_t;
  using SmemLayoutAtomA = UMMA::Layout_K_SW32_Atom<SmemAlloc>;
  using SmemLayoutAtomB = UMMA::Layout_K_SW32_Atom<SmemAlloc>;
  // SmemLayoutA: logical (BM,BK,stages) address map for packed activation A. AtomA is one small
  // repeating SW32 tile; SmemLayoutA is the full multi-stage tensor. tma.a writes this region
  // during the K loop, and LDSM reads it into tCrA. D and output staging reuse the same bytes only
  // after the empty-stage barrier proves the last consumer finished the A tile in that stage.
  using SmemLayoutA =
    decltype(tile_to_shape(SmemLayoutAtomA{}, Shape<Int<kBM>, Int<kBK>, Int<kStages>>{}));
  // SmemLayoutB: logical (BN,BK,stages) map for packed weight B. This is the actual destination
  // of tma.b. LDSM reads sB through this layout into tCrB; SFB has its own non-swizzled SF layout.
  using SmemLayoutB =
    decltype(tile_to_shape(SmemLayoutAtomB{}, Shape<Int<kBN>, Int<kBK>, Int<kStages>>{}));

  // D/L1 use swizzle-free row-major linear layouts and independent TMA descriptors. They share
  // only storage addresses and stage strides with A/B, never a CuTe layout or copy descriptor.
  // The overlay is byte-exact because BF16 needs 2 bytes/element while packed E2M1 needs 0.5:
  //
  //   BM * (BK/4) * 2 = BM * BK / 2  =>  kRankTile = BK/4
  //
  // Once the K loop drains, D and output staging reuse the A/B bytes. The empty-stage barrier
  // proves all consumer warps finished both A and B before D/L1 overwrite a stage. Reusing A's
  // swizzled FP4 layout for BF16 would still be incorrect even though the byte count matches.
  // SmemLayoutD: BF16 LoRA-down result D, logical (BM,kRankTile,stages), with strides
  // (kRankTile,1,BM*kRankTile). Thus one stage is an ordinary row-major [BM,kRankTile] matrix;
  // kRankTile=BK/4 is 32 for the default BK=128. tma.d writes it over the A byte allocation.
  using SmemLayoutD = decltype(make_layout(
    make_shape(Int<kBM>{}, Int<kRankTile>{}, Int<kStages>{}),
    make_stride(Int<kRankTile>{}, _1{}, Int<kBM * kRankTile>{})));
  // SmemLayoutL1: BF16 LoRA-up weight L1, logical (BN,kRankTile,stages), with analogous
  // row-major strides. tma.l1 writes it over the B allocation after empty[stage] opens. Again,
  // L1 names the second low-rank factor and has no relationship to the hardware L1 cache.
  using SmemLayoutL1 = decltype(make_layout(
    make_shape(Int<kBN>{}, Int<kRankTile>{}, Int<kStages>{}),
    make_stride(Int<kRankTile>{}, _1{}, Int<kBN * kRankTile>{})));

  // Append the pipeline-stage mode to each SF atom. filter_zeros removes the logical broadcast
  // modes before computing the stage stride: 16 codes share one stored scale, so advancing to the
  // next stage must skip real allocated bytes, not the 16x larger logical coordinate space.
  // SmemLayoutAtomSFA/SFB describe one SF atom before the stage dimension is appended.
  using SmemLayoutAtomSFA = typename SfSmemAtom<TiledMma, kBM, kBK>::type;
  using SmemLayoutAtomSFB =
    typename SfSmemAtom<TiledMma, (kBN > kRowBlock ? kBN : kRowBlock), kBK>::type;
  // SmemLayoutSFA: logical (BM,BK,stages) activation-SF view, including stride-0 K broadcasts.
  // It is the destination of tma.sfa. The independent CuTe producer emits SFA in exactly this
  // 128x4 atom order, so no v1 fragment repacking or G2R scale load remains in the main GEMM.
  using SmemLayoutSFA = decltype(make_layout(
    append(shape(SmemLayoutAtomSFA{}), Int<kStages>{}),
    append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))));
  // SmemLayoutSFB: logical (BN,BK,stages) weight-SF view. It is the actual destination of
  // tma.sfb. The appended stage stride uses filter_zeros so stage s+1 starts after the real SF
  // bytes of stage s, not after the broadcast-expanded logical element count.
  using SmemLayoutSFB = decltype(make_layout(
    append(shape(SmemLayoutAtomSFB{}), Int<kStages>{}),
    append(stride(SmemLayoutAtomSFB{}), size(filter_zeros(SmemLayoutAtomSFB{})))));

  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, SmemAlloc>;

  // SmemAtomO is one BF16 SW128 output atom; SmemLayoutO is the full (BM,BN) output tile, with
  // no pipeline-stage dimension. O means final output after FP32->BF16 conversion, not the FP32
  // tCrC accumulator. This region reuses pipeline storage only after every consumer has finished
  // the mainloop and LoRA MMA. SW128 gives STSM and TMA store one shared swizzled address recipe
  // and must not be confused with the FP4 SW32 data layout. Its byte extent is checked below.
  using SmemAtomO = GMMA::Layout_K_SW128_Atom<ElementO>;
  using SmemLayoutO = decltype(tile_to_shape(SmemAtomO{}, Shape<Int<kBM>, Int<kBN>>{}));

  // size(layout) counts logical elements, while cosize(layout) measures the physical address
  // span. They differ for SF because stride-0 broadcasts expose many logical reads of one stored
  // byte. A/B cosize is divided by 2 because each E2M1 code is 4 bits; SF cosize is already bytes
  // because each scale is FP8. All four regions are real TMA destinations during the K loop and
  // are reused as D/L1/output only after the pipeline drains.
  static constexpr int kAStageBytes = cosize_v<SmemLayoutA> / kStages / 2;
  static constexpr int kBStageBytes = cosize_v<SmemLayoutB> / kStages / 2;
  static constexpr int kSFAStageBytes = cosize_v<SmemLayoutSFA> / kStages;
  static constexpr int kSFBStageBytes = cosize_v<SmemLayoutSFB> / kStages;
  static constexpr int kDStageBytes = kBM * kRankTile * static_cast<int>(sizeof(ElementLora));
  static constexpr int kL1StageBytes = kBN * kRankTile * static_cast<int>(sizeof(ElementLora));
  static constexpr int kStageBytes =
    kAStageBytes + kBStageBytes + kSFAStageBytes + kSFBStageBytes;
  static constexpr int kSmemBytes = kStages * kStageBytes;

  static_assert(kDStageBytes == kAStageBytes, "a D rank tile must overlay one A stage");
  static_assert(kL1StageBytes == kBStageBytes, "an L1 rank tile must overlay one B stage");
  static_assert(kAStageBytes % 1024 == 0 && kBStageBytes % 1024 == 0,
                "the stage stride must preserve the 1024 B smem base alignment");
  static_assert(cosize_v<SmemLayoutO> * static_cast<int>(sizeof(ElementO)) <= kSmemBytes,
                "the BF16 output staging tile must fit into the pipeline storage");
  static_assert(kSmemBytes <= 99 * 1024, "pipeline storage exceeds the SM120 smem budget");

  // TMA transaction sizes feed mbarrier.expect_tx; they count bytes, not CuTe's 4-bit logical
  // elements. Main stages transfer A+B+SFA+SFB; LoRA stages replace that transaction with exactly
  // D+L1 bytes. Mixing these budgets would leave the full barrier waiting forever or release a
  // stage before its writes complete.
  static constexpr uint32_t kTxBytesMain =
    kAStageBytes + kBStageBytes + kSFAStageBytes + kSFBStageBytes;
  static constexpr uint32_t kTxBytesLora = kDStageBytes + kL1StageBytes;
};

template <class TmaA_, class TmaSFA_, class TmaB_, class TmaSFB_, class TmaD_,
          class TmaL1_, class TmaC_>
struct W4A4CuteTma {
  TmaA_ a;
  TmaSFA_ sfa;
  TmaB_ b;
  TmaSFB_ sfb;
  TmaD_ d;
  TmaL1_ l1;
  TmaC_ c;
};

struct W4A4CuteCacheKey {
  std::array<uint64_t, 11> data;
  bool operator==(const W4A4CuteCacheKey &) const = default;
};

struct W4A4CuteCacheHash {
  size_t operator()(const W4A4CuteCacheKey &key) const {
    uint64_t hash = 0x9e3779b97f4a7c15ULL;
    for (const uint64_t value : key.data) {
      hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    }
    return static_cast<size_t>(hash);
  }
};

// Warp-specialized multi-stage mainloop. Each CTA computes one output tile; this is not a
// persistent-CTA scheduler. Threads [0,128) form the TMA producer warpgroup and [128,384) eight
// MMA/epilogue consumer warps. Only producer thread 0 issues copies; the other producer threads
// retain the warpgroup role and participate in setmaxnreg. Register budgets are 32 per producer
// thread and 224 per consumer thread, keeping the 384-thread CTA inside the 64K-register SM.
//
// full[s] is a transaction barrier: producer arms the exact A+B+SFA+SFB or D+L1 byte count and
// TMA completion releases consumers. empty[s] is a CTA barrier arrived at by each consumer warp
// leader after its copy+MMA work, preventing the producer from overwriting a live stage.
// phase=(g/kStages)&1 is the mbarrier parity for reuse of the same physical barrier.
//
// Producer and consumers share one global step g over [0,nKt+nRt), so stage index and parity
// never reset at the LoRA boundary. Main g steps TMA A/B/SFA/SFB; trailing g steps TMA D/L1 into
// the overlay. This lets LoRA loads enter the ring up to kStages steps before the main K loop ends
// instead of exposing a serial LoRA load tail.
template <class Traits, class TmaParams>
__global__ void __launch_bounds__(Traits::kNumThreads + 128, 1)
gemm_w4a4_kernel(CUTLASS_GRID_CONSTANT TmaParams const tma,
                      cutlass::bfloat16_t const *__restrict__ bias,
                      cutlass::bfloat16_t const *__restrict__ wcscales, float alpha, int M,
                      int N, int K, int rank) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  using cute::tma_store_arrive;
  using cute::tma_store_wait;
  using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
  using CtaBarrier = cutlass::arch::ClusterBarrier;

  using Element = typename Traits::Element;
  using ElementSF = typename Traits::ElementSF;
  using ElementLora = typename Traits::ElementLora;
  using ElementO = typename Traits::ElementO;
  using TiledMma = typename Traits::TiledMma;
  using LoraMma = typename Traits::LoraMma;
  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutD = typename Traits::SmemLayoutD;
  using SmemLayoutL1 = typename Traits::SmemLayoutL1;
  using SmemLayoutSFA = typename Traits::SmemLayoutSFA;
  using SmemLayoutSFB = typename Traits::SmemLayoutSFB;
  using SmemLayoutAtomSFA = typename Traits::SmemLayoutAtomSFA;
  using SmemLayoutAtomSFB = typename Traits::SmemLayoutAtomSFB;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using SmemCopyAtom = typename Traits::SmemCopyAtom;

  constexpr int kBM = Traits::kBM;
  constexpr int kBN = Traits::kBN;
  constexpr int kBK = Traits::kBK;
  constexpr int kStages = Traits::kStages;
  constexpr int kRankTile = Traits::kRankTile;
  constexpr int kNumThreads = Traits::kNumThreads;

  const int raw_tid = threadIdx.x;
  const bool is_producer = raw_tid < 128;
  const int tid = is_producer ? 0 : raw_tid - 128;
  const int m0 = blockIdx.y * kBM;
  const int n0 = blockIdx.x * kBN;

  // -----------------------------------------------------------------------------------------------
  // Stage 1: establish this CTA's logical problem and gmem views.
  // -----------------------------------------------------------------------------------------------
  // blockIdx selects the output tile (m0,n0), not an input element. nKt counts NVFP4 K stages and
  // nRt counts LoRA rank stages. They share g so that the same ring can carry two different data
  // types. rankTma is only a descriptor extent; rank 0 uses a harmless placeholder and nRt=0 makes
  // every D/L1 access unreachable. get_tma_tensor converts a descriptor into logical coordinates;
  // it does not copy anything at this point.
  const int nKt = K / kBK;
  const int nRt = rank / kRankTile;
  const int gTotal = nKt + nRt;
  // The rank-0 path never dispatches a rank tile, so a placeholder rank extent keeps the
  // unused D/L1 descriptors well-formed.
  const int rankTma = rank > 0 ? rank : kRankTile;

  auto mA = tma.a.get_tma_tensor(make_shape(M, K));
  auto mSFA = tma.sfa.get_tma_tensor(shape(sf_gmem_layout(M, K)));
  auto mB = tma.b.get_tma_tensor(make_shape(N, K));
  auto mC = tma.c.get_tma_tensor(make_shape(M, N));
  auto mSFB = tma.sfb.get_tma_tensor(shape(sf_gmem_layout(N, K)));
  auto mD = tma.d.get_tma_tensor(make_shape(M, rankTma));
  auto mL1 = tma.l1.get_tma_tensor(make_shape(N, rankTma));
  auto a_slice = tma.a.get_slice(_0{});
  auto sfa_slice = tma.sfa.get_slice(_0{});
  auto b_slice = tma.b.get_slice(_0{});
  auto c_slice = tma.c.get_slice(_0{});
  auto sfb_slice = tma.sfb.get_slice(_0{});
  auto d_slice = tma.d.get_slice(_0{});
  auto l1_slice = tma.l1.get_slice(_0{});

  // -----------------------------------------------------------------------------------------------
  // Stage 2: allocate stage storage and initialize its synchronization objects.
  // -----------------------------------------------------------------------------------------------
  // The byte pointer is intentional: E2M1 is sub-byte, while D/output are BF16. Each region's
  // base is therefore computed in bytes first, then recast into the CuTe element view required by
  // that region. sA/sB/sD/sL1 are different layout views over reusable storage, not four separate
  // allocations. full/empty are real shared-memory barriers, so all threads observe their
  // initialized state after __syncthreads. empty expects one arrival per consumer warp, not 256
  // arrivals, which is why only warp leaders arrive later.
  extern __shared__ __align__(1024) uint8_t shm[];
  uint8_t *a_bytes = shm;
  uint8_t *b_bytes = a_bytes + Traits::kAStageBytes * kStages;
  ElementSF *sfa_base = reinterpret_cast<ElementSF *>(b_bytes + Traits::kBStageBytes * kStages);
  ElementSF *sfb_base = sfa_base + Traits::kSFAStageBytes * kStages;

  __shared__ uint64_t full[kStages];
  __shared__ uint64_t empty[kStages];
  if (tid == 0) {
    for (int s = 0; s < kStages; ++s) {
      TmaBarrier::init(&full[s], 1);
      CtaBarrier::init(&empty[s], kNumThreads / 32);
    }
  }
  __syncthreads();

  auto sA_full = make_tensor(make_smem_ptr(recast_ptr<uint4_t>(a_bytes)), SmemLayoutA{});
  auto sB_full = make_tensor(make_smem_ptr(recast_ptr<uint4_t>(b_bytes)), SmemLayoutB{});
  auto sD_full = make_tensor(make_smem_ptr(recast_ptr<ElementLora>(a_bytes)), SmemLayoutD{});
  auto sL1_full = make_tensor(make_smem_ptr(recast_ptr<ElementLora>(b_bytes)), SmemLayoutL1{});
  auto sSFA_full = make_tensor(make_smem_ptr(sfa_base), SmemLayoutSFA{});
  auto sSFB_full = make_tensor(make_smem_ptr(sfb_base), SmemLayoutSFB{});

  // -----------------------------------------------------------------------------------------------
  // Stage 3: build register fragments and the copy views that will fill them.
  // -----------------------------------------------------------------------------------------------
  // thr_mma converts a CTA-wide tile into this thread's tiny A/B/C fragments. partition_fragment_*
  // allocates register storage with the exact layout expected by the hardware atom. A TiledCopy
  // has a different iteration order, so retile_D presents that same register storage in the copy's
  // order; it does not allocate a second fragment or reorder values at runtime.
  // tCrC is the only accumulator for both GEMMs. cC is an identity-coordinate tensor: it stores
  // logical (m,n) coordinates rather than output values, allowing the epilogue to find the correct
  // bias/wcscale for every FP32 accumulator element.
  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  auto s2r_copy_a = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto s2r_copy_b = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto s2r_thr_a = s2r_copy_a.get_thread_slice(tid);
  auto s2r_thr_b = s2r_copy_b.get_thread_slice(tid);
  auto tCsA = s2r_thr_a.partition_S(as_position_independent_swizzle_tensor(sA_full));
  auto tCsB = s2r_thr_b.partition_S(as_position_independent_swizzle_tensor(sB_full));
  auto tCrA = thr_mma.partition_fragment_A(sA_full(_, _, Int<0>{}));
  auto tCrB = thr_mma.partition_fragment_B(sB_full(_, _, Int<0>{}));
  auto tCrA_view = s2r_thr_a.retile_D(tCrA);
  auto tCrB_view = s2r_thr_b.retile_D(tCrB);

  using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;
  auto tCrSFA = sf_partition::partition_fragment_sfa(sSFA_full(_, _, Int<0>{}), thr_mma);
  auto tCrSFB = sf_partition::partition_fragment_sfb(sSFB_full(_, _, Int<0>{}), thr_mma);
  auto copy_sfa = make_tiled_copy_impl(
    SmemCopyAtomSF{}, sf_partition::layout_sfa_tv(tiled_mma),
    make_shape(size<0>(tile_shape(tiled_mma)), size<2>(tile_shape(tiled_mma))));
  auto copy_sfb = make_tiled_copy_impl(
    SmemCopyAtomSF{}, sf_partition::layout_sfb_tv(tiled_mma),
    make_shape(size<1>(tile_shape(tiled_mma)), size<2>(tile_shape(tiled_mma))));
  auto thr_copy_sfa = copy_sfa.get_thread_slice(tid);
  auto thr_copy_sfb = copy_sfb.get_thread_slice(tid);
  auto tCsSFA = thr_copy_sfa.partition_S(as_position_independent_swizzle_tensor(sSFA_full));
  auto tCsSFB = thr_copy_sfb.partition_S(as_position_independent_swizzle_tensor(sSFB_full));
  auto tCrSFA_view = thr_copy_sfa.retile_D(tCrSFA);
  auto tCrSFB_view = thr_copy_sfb.retile_D(tCrSFB);

  auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<kBM>, Int<kBN>>{});
  clear(tCrC);
  auto cC = make_identity_tensor(Shape<Int<kBM>, Int<kBN>>{});
  auto tScC = thr_mma.partition_C(cC);
  auto tScC_rc = make_tensor(tScC.data(), convert_layout_acc_rowcol(tScC.layout()));
  constexpr int kCRows = decltype(size<0>(tScC_rc))::value;
  constexpr int kCCols = decltype(size<1>(tScC_rc))::value;

  // -----------------------------------------------------------------------------------------------
  // Stage 4: define one producer step, without executing it yet.
  // -----------------------------------------------------------------------------------------------
  // local_tile selects the gmem tile for this CTA and K/rank step. partition_S(gmem) and
  // partition_D(smem) then describe the two ends of the TMA copy. arrive_and_expect_tx says how
  // many bytes must arrive before full[stage] opens; each copy(...with(full)) contributes to that
  // same transaction barrier. The main branch transfers A/B/SFA/SFB, while the LoRA branch uses
  // different descriptors to put D/L1 into the overlay. All inputs now share the standard
  // row-major/128x4 producer layout; no v1 fragment or G2R path remains.
  auto issue_tma = [&](int g, int stage) {
    cutlass::arch::fence_view_async_shared();
    if (g < nKt) {
      auto gA = local_tile(mA, Shape<Int<kBM>, Int<kBK>>{}, make_coord(blockIdx.y, g));
      auto gSFA = local_tile(mSFA, Shape<Int<kBM>, Int<kBK>>{}, make_coord(blockIdx.y, g));
      auto gB = local_tile(mB, Shape<Int<kBN>, Int<kBK>>{}, make_coord(blockIdx.x, g));
      auto gSFB = local_tile(mSFB, Shape<Int<kBN>, Int<kBK>>{}, make_coord(blockIdx.x, g));
      TmaBarrier::arrive_and_expect_tx(&full[stage], Traits::kTxBytesMain);
      copy(tma.a.with(full[stage]), a_slice.partition_S(gA),
           a_slice.partition_D(sA_full(_, _, stage)));
      copy(tma.sfa.with(full[stage]), sfa_slice.partition_S(gSFA),
           sfa_slice.partition_D(sSFA_full(_, _, stage)));
      copy(tma.b.with(full[stage]), b_slice.partition_S(gB),
           b_slice.partition_D(sB_full(_, _, stage)));
      copy(tma.sfb.with(full[stage]), sfb_slice.partition_S(gSFB),
           sfb_slice.partition_D(sSFB_full(_, _, stage)));
    } else {
      const int rt = g - nKt;
      auto gD = local_tile(mD, Shape<Int<kBM>, Int<kRankTile>>{}, make_coord(blockIdx.y, rt));
      auto gL1 = local_tile(mL1, Shape<Int<kBN>, Int<kRankTile>>{}, make_coord(blockIdx.x, rt));
      TmaBarrier::arrive_and_expect_tx(&full[stage], Traits::kTxBytesLora);
      copy(tma.d.with(full[stage]), d_slice.partition_S(gD),
           d_slice.partition_D(sD_full(_, _, stage)));
      copy(tma.l1.with(full[stage]), l1_slice.partition_S(gL1),
           l1_slice.partition_D(sL1_full(_, _, stage)));
    }
  };

  // -----------------------------------------------------------------------------------------------
  // Stage 5: split producer and consumer roles.
  // -----------------------------------------------------------------------------------------------
  // Producer waits until a stage is empty, then performs Stage 4 and immediately moves to the next
  // g. Consumers initially mark every stage empty so the producer can start. setmaxnreg changes
  // the per-warpgroup register budget after launch: the TMA side needs few registers, while the
  // consumer needs space for two operand types and a large FP32 accumulator. This is why WS is
  // useful here even though only one producer thread actually issues TMA instructions.
  if (is_producer) {
    cutlass::arch::warpgroup_reg_dealloc<32>();
    if (raw_tid == 0) {
      for (int g = 0; g < gTotal; ++g) {
        const int stage = g % kStages;
        const int phase = (g / kStages) & 1;
        CtaBarrier::wait(&empty[stage], phase);
        issue_tma(g, stage);
      }
    }
    return;
  }

  cutlass::arch::warpgroup_reg_alloc<224>();
  if (tid % 32 == 0) {
    for (int s = 0; s < kStages; ++s) CtaBarrier::arrive(&empty[s]);
  }

  // -----------------------------------------------------------------------------------------------
  // Stage 6: consume all NVFP4 K stages and accumulate P in FP32.
  // -----------------------------------------------------------------------------------------------
  // Waiting on full makes the A/B/SFA/SFB bytes visible; the async-shared fence orders subsequent
  // smem reads after that wait. A/B use LDSM because their swizzled FP4 layouts match the copy
  // atom; SFA/SFB use the special SF copy views described above. make_zip_tensor pairs data and
  // scale fragments without combining their storage. cute::gemm expands the tiled atoms and
  // updates tCrC in place. Only after the warp's copy/MMA work is done may its leader mark the
  // stage empty for the producer's next reuse.
#pragma unroll 1
  for (int g = 0; g < nKt; ++g) {
    const int stage = g % kStages;
    const int phase = (g / kStages) & 1;
    TmaBarrier::wait(&full[stage], phase);
    cutlass::arch::fence_view_async_shared();
    copy(s2r_copy_a, tCsA(_, _, _, stage), tCrA_view);
    copy(s2r_copy_b, tCsB(_, _, _, stage), tCrB_view);
    copy(tCsSFA(_, _, _, stage), tCrSFA_view);
    copy(tCsSFB(_, _, _, stage), tCrSFB_view);
    cute::gemm(tiled_mma, make_zip_tensor(tCrA, tCrSFA), make_zip_tensor(tCrB, tCrSFB), tCrC);
    __syncwarp();
    if (tid % 32 == 0) CtaBarrier::arrive(&empty[stage]);
  }

  // -----------------------------------------------------------------------------------------------
  // Stage 7: apply residual-only scale and bias in FP32.
  // -----------------------------------------------------------------------------------------------
  // The C fragment is still indexed by the MMA's thread/value layout, so convert_layout_acc_rowcol
  // gives each value a logical (row,column) view. One thread owns only a few output columns; load
  // those column scales/biases once and reuse them across its rows. alpha is applied first, then
  // fmaf(P,wcscales,bias). We deliberately do this before the LoRA MMA so wcscales cannot leak
  // into D@L1^T. Everything remains FP32; converting here would add an unnecessary rounding.
  {
    float col_scale[kCCols];
    float col_bias[kCCols];
#pragma unroll
    for (int c = 0; c < kCCols; ++c) {
      const int n = n0 + get<1>(tScC_rc(0, c));
      col_scale[c] = wcscales != nullptr ? static_cast<float>(wcscales[n]) : 1.0f;
      col_bias[c] = bias != nullptr ? static_cast<float>(bias[n]) : 0.0f;
    }
    auto tCrC_rc = make_tensor(tCrC.data(), convert_layout_acc_rowcol(tCrC.layout()));
    if (alpha != 1.0f) {
#pragma unroll
      for (int r = 0; r < kCRows; ++r)
#pragma unroll
        for (int c = 0; c < kCCols; ++c) tCrC_rc(r, c) *= alpha;
    }
#pragma unroll
    for (int r = 0; r < kCRows; ++r)
#pragma unroll
      for (int c = 0; c < kCCols; ++c)
        tCrC_rc(r, c) = fmaf(tCrC_rc(r, c), col_scale[c], col_bias[c]);
  }

  // -----------------------------------------------------------------------------------------------
  // Stage 8: accumulate the BF16 LoRA-up correction into the same C fragment.
  // -----------------------------------------------------------------------------------------------
  // D/L1 already have row-major TMA-compatible layouts. This second TiledMMA partitions them with
  // BF16 m16n8k16 atoms, but its C partition is statically identical to the NVFP4 one (Stage 3).
  // Therefore cute::gemm(lora_mma,...,tCrC) needs no accumulator remap. g continues from nKt, so
  // full/empty phase and stage reuse follow the same rules as Stage 6. rank 0 skips the entire
  // block and also skips every D/L1 TMA in the producer.
  if (nRt > 0) {
    LoraMma lora_mma;
    auto thr_lora = lora_mma.get_thread_slice(tid);
    auto tCrD = thr_lora.partition_fragment_A(sD_full(_, _, Int<0>{}));
    auto tCrL1 = thr_lora.partition_fragment_B(sL1_full(_, _, Int<0>{}));
    using LoraCopyAtom = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, ElementLora>;
    auto s2r_copy_d = make_tiled_copy_A(LoraCopyAtom{}, lora_mma);
    auto s2r_copy_l1 = make_tiled_copy_B(LoraCopyAtom{}, lora_mma);
    auto s2r_thr_d = s2r_copy_d.get_thread_slice(tid);
    auto s2r_thr_l1 = s2r_copy_l1.get_thread_slice(tid);
    auto tCsD = s2r_thr_d.partition_S(sD_full);
    auto tCsL1 = s2r_thr_l1.partition_S(sL1_full);
    auto tCrD_view = s2r_thr_d.retile_D(tCrD);
    auto tCrL1_view = s2r_thr_l1.retile_D(tCrL1);

#pragma unroll 1
    for (int g = nKt; g < gTotal; ++g) {
      const int stage = g % kStages;
      const int phase = (g / kStages) & 1;
      TmaBarrier::wait(&full[stage], phase);
      cutlass::arch::fence_view_async_shared();
      copy(s2r_copy_d, tCsD(_, _, _, stage), tCrD_view);
      copy(s2r_copy_l1, tCsL1(_, _, _, stage), tCrL1_view);
      cute::gemm(lora_mma, tCrD, tCrL1, tCrC);
      __syncwarp();
      if (tid % 32 == 0) CtaBarrier::arrive(&empty[stage]);
    }
  }

  // -----------------------------------------------------------------------------------------------
  // Stage 9: round once and publish the output tile through smem + TMA store.
  // -----------------------------------------------------------------------------------------------
  // NumericArrayConverter performs the sole FP32->BF16 conversion, after both GEMMs and all
  // scales. STSM then writes the register fragment into a swizzled smem output layout; the first
  // barrier proves no mainloop stage is still using that storage. fence_view_async_shared makes
  // generic smem stores visible to the asynchronous TMA engine, and the second barrier ensures
  // every consumer finished its STSM before thread 0 starts the store. tma_store_wait<0> drains
  // outstanding stores before this CTA can reuse the output smem or exit.
  {
    constexpr int kFrag = kCRows * kCCols;
    cutlass::Array<ElementO, kFrag> acc_bf16;
    cutlass::NumericArrayConverter<ElementO, float, kFrag> convert;
    acc_bf16 = convert(*reinterpret_cast<cutlass::Array<float, kFrag> const *>(tCrC.data()));
    auto tCrCh = make_tensor(make_rmem_ptr(acc_bf16.data()), tCrC.layout());

    __syncthreads();
    auto r2s_copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, ElementO>{}, tiled_mma);
    auto r2s_thr = r2s_copy.get_thread_slice(tid);
    auto sC = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(shm)), SmemLayoutO{});
    auto tCrCh_src = r2s_thr.retile_S(tCrCh);
    auto tCsC_dst = r2s_thr.partition_D(sC);
    copy(r2s_copy, tCrCh_src, tCsC_dst);
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    auto gC = local_tile(mC, Shape<Int<kBM>, Int<kBN>>{}, make_coord(blockIdx.y, blockIdx.x));
    if (tid == 0) copy(tma.c, c_slice.partition_S(sC), c_slice.partition_D(gC));
    tma_store_arrive();
    tma_store_wait<0>();
  }
#else
  asm volatile("trap;");
#endif  // __CUDA_ARCH__ >= 1200
}

// Host launch has three jobs:
//   A. Describe each gmem tensor and its matching smem stage to create TMA descriptors. A TMA
//      descriptor is a reusable copy recipe (address, shape, stride, box and smem layout), not
//      the copied data itself. D/L1 descriptors intentionally differ from A/B descriptors.
//   B. Cache those recipes per host thread and per exact pointer/shape/rank key. This avoids
//      rebuilding them on every linear-layer call while remaining safe when output or D pointers
//      change. The cache holds at most 32 entries and is cleared wholesale when full; it is not
//      an LRU cache. Device-specific dynamic-smem opt-in is likewise performed only once.
//   C. Launch one CTA per BM x BN output tile with 128 extra producer threads.
// The template is instantiated only from gemm_w4a4.cu, keeping SM120 atoms out of other TUs.
template <int kBN = 128, int kStages = 4, int kBK_ = 64>
void launch_gemm_w4a4(void const *act, void const *wgt, void *out,
                           cutlass::float_ue4m3_t const *act_sf,
                           cutlass::float_ue4m3_t const *wgt_sf,
                           cutlass::bfloat16_t const *lora_act_in,
                           cutlass::bfloat16_t const *lora_up, cutlass::bfloat16_t const *bias,
                           cutlass::bfloat16_t const *wcscales, float alpha, int M, int N,
                           int K, int rank, cudaStream_t stream) {
  using Traits = W4A4CuteTraits<kBN, kStages, kBK_>;
  using Element = typename Traits::Element;
  using ElementSF = typename Traits::ElementSF;
  using ElementLora = typename Traits::ElementLora;
  using ElementO = typename Traits::ElementO;
  constexpr int kBM = Traits::kBM;
  constexpr int kBK = Traits::kBK;
  constexpr int kRankTile = Traits::kRankTile;

  auto mA = make_tensor(recast_ptr<Element>(static_cast<uint8_t const *>(act)),
                        make_shape(M, K), make_stride(K, _1{}));
  auto mSFA = make_tensor(recast_ptr<ElementSF>(act_sf), sf_gmem_layout(M, K));
  auto mB = make_tensor(recast_ptr<Element>(static_cast<uint8_t const *>(wgt)),
                        make_shape(N, K), make_stride(K, _1{}));
  auto mC = make_tensor(make_gmem_ptr(static_cast<ElementO *>(out)), make_shape(M, N),
                        make_stride(N, _1{}));
  auto mSFB = make_tensor(recast_ptr<ElementSF>(wgt_sf), sf_gmem_layout(N, K));

  // With rank 0 the rank tiles are never dispatched (nRt == 0), so the D/L1 descriptors are
  // built over an unused [M,kRankTile] / [N,kRankTile] view of the output and stay unreachable.
  // The placeholder merely satisfies descriptor construction; it is never read by a TMA.
  const int rankTma = rank > 0 ? rank : kRankTile;
  auto const *d_ptr = rank > 0 ? lora_act_in : reinterpret_cast<ElementLora const *>(out);
  auto const *l1_ptr = rank > 0 ? lora_up : reinterpret_cast<ElementLora const *>(out);
  auto mD = make_tensor(make_gmem_ptr(d_ptr), make_shape(M, rankTma),
                        make_stride(rankTma, _1{}));
  auto mL1 = make_tensor(make_gmem_ptr(l1_ptr), make_shape(N, rankTma),
                         make_stride(rankTma, _1{}));

  auto build_params = [&]() {
    auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, mA, typename Traits::SmemLayoutA{}(_, _, Int<0>{}),
                               Shape<Int<kBM>, Int<kBK>>{}, _1{});
    auto tma_sfa = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFA,
                                           typename Traits::SmemLayoutAtomSFA{},
                                           Shape<Int<kBM>, Int<kBK>>{}, _1{});
    auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, mB, typename Traits::SmemLayoutB{}(_, _, Int<0>{}),
                               Shape<Int<kBN>, Int<kBK>>{}, _1{});
    auto tma_sfb = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFB,
                                          typename Traits::SmemLayoutAtomSFB{},
                                          Shape<Int<kBN>, Int<kBK>>{}, _1{});
    auto tma_d = make_tma_copy(SM90_TMA_LOAD{}, mD, typename Traits::SmemLayoutD{}(_, _, Int<0>{}),
                               Shape<Int<kBM>, Int<kRankTile>>{}, _1{});
    auto tma_l1 =
      make_tma_copy(SM90_TMA_LOAD{}, mL1, typename Traits::SmemLayoutL1{}(_, _, Int<0>{}),
                    Shape<Int<kBN>, Int<kRankTile>>{}, _1{});
    auto tma_c = make_tma_copy(SM90_TMA_STORE{}, mC, typename Traits::SmemLayoutO{},
                               Shape<Int<kBM>, Int<kBN>>{}, _1{});
    using Params = W4A4CuteTma<decltype(tma_a), decltype(tma_sfa), decltype(tma_b),
                               decltype(tma_sfb), decltype(tma_d), decltype(tma_l1),
                               decltype(tma_c)>;
    return Params{tma_a, tma_sfa, tma_b, tma_sfb, tma_d, tma_l1, tma_c};
  };
  using TmaParams = decltype(build_params());
  auto address = [](const void *ptr) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
  };
  W4A4CuteCacheKey key{{address(act), address(act_sf), address(wgt), address(out),
                        address(wgt_sf), address(d_ptr), address(l1_ptr),
                        static_cast<uint64_t>(M), static_cast<uint64_t>(N),
                        static_cast<uint64_t>(K), static_cast<uint64_t>(rank)}};
  thread_local std::unordered_map<W4A4CuteCacheKey, TmaParams, W4A4CuteCacheHash> cache;
  auto it = cache.find(key);
  if (it == cache.end()) {
    if (cache.size() >= 32) {
      cache.clear();
    }
    it = cache.emplace(key, build_params()).first;
  }

  auto kernel = gemm_w4a4_kernel<Traits, TmaParams>;
  static thread_local std::unordered_set<int> configured_devices;
  int device = 0;
  checkCUDA(cudaGetDevice(&device));
  if (configured_devices.insert(device).second) {
    checkCUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   Traits::kSmemBytes));
  }
  kernel<<<dim3(N / kBN, M / kBM), Traits::kNumThreads + 128, Traits::kSmemBytes, stream>>>(
    it->second, bias, wcscales, alpha, M, N, K, rank);
}

}  // namespace svdq::cute_kernels
