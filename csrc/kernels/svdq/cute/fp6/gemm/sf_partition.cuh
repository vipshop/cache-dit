#pragma once

// Scale-factor (SF) fragment plumbing for the block-scaled SVDQ W6A6 GEMM.
//
// The MX block-scaled atom owns its scales in exactly the same (thread,value)->(M/N,K) shape as the
// NVFP4 atom, so these helpers are the same construction the fp4 GEMM uses inline. They live here,
// next to the fp6 kernels, so this format can be read and changed without touching fp4.
//
// The MMA atom's SFALayout/SFBLayout maps (thread,value) to logical (M/N,K), including zero-stride
// broadcasts and the 32-wide N permutation. A block-scaled atom therefore cannot consume a plain
// row-major fragment: the helpers below mirror the CUTLASS SM120 collective in three stages:
//   1. thrfrg_sf*: compose the logical tensor with the atom's SF TV layout;
//   2. partition_fragment_sf*: select this thread's slice and materialize the atom-sized fragment;
//   3. layout_sf*_tv: construct the inverse thread map used by the smem/gmem->register tiled copy.
// A direct linear load would put another lane's scales into the MMA operand. The mainloop then
// presents {data,SF} through make_zip_tensor as the two operands of a block-scaled CuTe MMA.

#include <cute/tensor.hpp>

namespace svdq::cute_kernels {

using namespace cute;

namespace sf_partition {

// thrfrg_sfa = "thread-fragment view for SFA". It answers: if the activation SF tensor is viewed
// through this TiledMMA, which logical (M,K) coordinates belong to each thread and each value?
// The function only composes layouts; it neither allocates registers nor loads any scale.
//   logical_divide: split the full SF tensor at the TiledMMA's M/K permutation boundaries.
//   zipped_divide:  separate one MMA atom from the remaining M/K atom repetitions.
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
  auto a_tile =
    make_tile(make_layout(size<0>(AtomShape_MNK{})), make_layout(size<2>(AtomShape_MNK{})));
  auto a_tensor = zipped_divide(t_tensor, a_tile);
  auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{}, _);
  auto thr_tile = make_tile(
    _, make_tile(make_layout(size<1>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
  return zipped_divide(tv_tensor, thr_tile);
}

// B-side counterpart of thrfrg_sfa. The structure is the same, but the logical coordinates are
// (N,K), the atom tile uses size<1>(AtomShape_MNK), and thread tiling uses the N/K modes of
// thr_layout_vmnk. SFBLayout is not interchangeable with SFALayout: a block-scaled instruction owns
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
  auto a_tile =
    make_tile(make_layout(size<1>(AtomShape_MNK{})), make_layout(size<2>(AtomShape_MNK{})));
  auto a_tensor = zipped_divide(t_tensor, a_tile);
  auto tv_tensor = a_tensor.compose(AtomLayoutSFB_TV{}, _);
  auto thr_tile = make_tile(
    _, make_tile(make_layout(size<2>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
  return zipped_divide(tv_tensor, thr_tile);
}

// Materialize this thread's SFA register fragment from the abstract thrfrg view.
// thread_mma.thr_vmnk_ is this lane's position in the TiledMMA. For A we keep its V/M/K position
// and drop N: activation scales do not change when the same A row is reused across output columns.
// thr_tensor still points at the original SF storage. Indexing it with thr_vmk selects the source
// values; make_fragment_like then allocates the register fragment with the atom-required layout.
// The actual gmem/smem read happens later through a TiledCopy, not inside this function.
template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_fragment_sfa(SFATensor &&sfatensor, ThrMma &thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  auto thr_tensor = make_tensor(static_cast<SFATensor &&>(sfatensor).data(),
                                thrfrg_sfa(sfatensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
  auto part = thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
  return make_fragment_like<ValTypeSF>(part);
}

// SFB fragment construction mirrors partition_fragment_sfa, except thr_vnk keeps V/N/K and drops
// M. Weight scales depend on output column and K group, but are unchanged when the same B tile is
// reused across output rows. The returned fragment is paired with tCrB by make_zip_tensor in the
// mainloop; no scale values are fetched until the explicit copy is issued.
template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_fragment_sfb(SFBTensor &&sfbtensor, ThrMma &thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  auto thr_tensor = make_tensor(static_cast<SFBTensor &&>(sfbtensor).data(),
                                thrfrg_sfb(sfbtensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
  auto part = thr_tensor(thr_vnk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
  return make_fragment_like<ValTypeSF>(part);
}

// layout_sfa_tv builds the TV (Thread-Value) layout used to construct an SFA TiledCopy.
// It is a compile-time answer to "lane L's value V must read which logical (M,K) scale?".
// It is not the gmem packed layout, does not allocate a fragment, and performs no copy itself.
//   ref_A: an identity logical (M,K) reference covering the whole TiledMMA.
//   atile: adds atom repetition/broadcast information; the zero-stride N mode means the same SFA
//          values are reused as the TiledMMA moves across output columns.
//   thrfrg_sfa(ref_A,mma): applies the block-scaled atom's SFA mapping to that reference.
//   right_inverse(thr_layout_vmnk): converts hierarchical VMNK coordinates back to a linear lane.
// The two compose calls chain those maps into one TV layout. make_tiled_copy_impl then combines
// it with UniversalCopy<ValTypeSF> to generate the instructions that fill tCrSFA_view.
template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto layout_sfa_tv(TiledMma &mma) {
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto atile = make_tile(
    _, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                             make_stride(Int<1>{}, Int<0>{})),
                 _));
  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  return thrfrg_sfa(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
}

// layout_sfb_tv is the B-side TV copy plan: (lane,value)->logical (N,K). ref_B covers output
// columns rather than rows, and btile's zero stride is on the M mode because a weight scale is
// reused when the same B tile advances across output rows. The returned layout drives the SFB
// TiledCopy that loads tCrSFB_view. Keeping this separate from layout_sfa_tv is what preserves the
// different activation/weight scale ownership rules of the block-scaled MMA atom.
template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto layout_sfb_tv(TiledMma &mma) {
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto btile = make_tile(
    _, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                             make_stride(Int<0>{}, Int<1>{})),
                 _));
  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  return thrfrg_sfb(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
}

}  // namespace sf_partition

}  // namespace svdq::cute_kernels
