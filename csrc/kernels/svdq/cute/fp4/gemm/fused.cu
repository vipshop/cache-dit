#include "gemm_w4a4.h"

#include <type_traits>

#include "gemm_w4a4.cuh"

namespace svdq::cute_kernels {

// Compile the fused epilogue independently of gemm_w4a4.cu. kFuseDown separates qout-only from
// qout+D so the former does not carry the down-reduction fragments or their register pressure.
void launch_gemm_w4a4_fused(void const *act, void const *wgt, void *out,
                            cutlass::float_ue4m3_t const *act_sf,
                            cutlass::float_ue4m3_t const *wgt_sf, void const *lora_act_in,
                            cutlass::bfloat16_t const *lora_up, cutlass::bfloat16_t const *bias,
                            cutlass::bfloat16_t const *wcscales, F32DLoraScales scales, float alpha,
                            int M, int N, int K, int rank, LoraLayout lora_layout,
                            FusedLaunchArgs const &fused, cudaStream_t stream) {
  using ElementLora = cutlass::bfloat16_t;
  auto launch = [&](auto native_fragment_tag, auto fuse_down_tag) {
    constexpr bool kNativeFragment = decltype(native_fragment_tag)::value;
    constexpr bool kFuseDown = decltype(fuse_down_tag)::value;
    launch_gemm_w4a4<256, 3, 128, false, kNativeFragment, true, kFuseDown>(
      act, wgt, out, act_sf, wgt_sf,
      kNativeFragment ? nullptr : static_cast<ElementLora const *>(lora_act_in),
      kNativeFragment ? static_cast<float const *>(lora_act_in) : nullptr, lora_up, bias, wcscales,
      scales, alpha, M, N, K, rank, fused.qout, fused.oscales, fused.smooth, fused.lora_down,
      fused.d_atomic, fused.d_final, fused.down_scales, fused.rank_down, fused.write_dense,
      fused.fuse_silu, stream);
  };

  const bool has_down = fused.rank_down > 0;
  if (lora_layout == LoraLayout::kNative) {
    if (has_down) {
      launch(std::true_type{}, std::true_type{});
    } else {
      launch(std::true_type{}, std::false_type{});
    }
  } else {
    if (has_down) {
      launch(std::false_type{}, std::true_type{});
    } else {
      launch(std::false_type{}, std::false_type{});
    }
  }
}

}  // namespace svdq::cute_kernels
