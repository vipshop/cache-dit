#include "gemm_w4a4.h"

#include <stdexcept>
#include <type_traits>

#include "common.h"
#include "gemm_w4a4.cuh"

namespace svdq::cute_kernels {
namespace {

// Row-major schedule thresholds measured on BN=256/BK=128. The BN=128 fallback shares the rule but
// was not part of the A/B matrix.
constexpr int kMainFirstMinM = 16384;
constexpr int kMainFirstWideM = 8192;
constexpr int kMainFirstWideN = 6144;

// Dispatches one compiled tile configuration. `LoraLayout::kNative` always runs main-first because
// its gather reads gmem directly into MMA registers and never enters the residual inverse-scale
// division. For row-major BF16, the measured split is: 16K rows always run main-first; at 8K rows,
// N>=6144 runs main-first; all smaller row/column combinations keep LoRA-first so its term can
// overlap the A/SFA prefetch.
//
// The fused variant is compiled for the default tile only and pins main-first, because its
// epilogue reads the accumulator after the residual scale. Both D layouts are supported: the
// native fragment one is written in place, the row-major one accumulates in FP32 and is rounded by
// `launch_finalize_lora_rows`.
template <int kBN, int kStages, int kBK>
void launch_gemm_w4a4_config(void const *act, void const *wgt, void *out,
                             cutlass::float_ue4m3_t const *act_sf,
                             cutlass::float_ue4m3_t const *wgt_sf, void const *lora_act_in,
                             cutlass::bfloat16_t const *lora_up, cutlass::bfloat16_t const *bias,
                             cutlass::bfloat16_t const *wcscales, F32DLoraScales scales,
                             float alpha, int M, int N, int K, int rank, LoraLayout lora_layout,
                             FusedLaunchArgs const &fused, cudaStream_t stream) {
  using ElementLora = cutlass::bfloat16_t;
  auto launch = [&](auto native_fragment_tag, auto lora_first_tag) {
    constexpr bool kNativeFragment = decltype(native_fragment_tag)::value;
    constexpr bool kLoRAFirst = decltype(lora_first_tag)::value;
    launch_gemm_w4a4<kBN, kStages, kBK, kLoRAFirst, kNativeFragment, false>(
      act, wgt, out, act_sf, wgt_sf,
      kNativeFragment ? nullptr : static_cast<ElementLora const *>(lora_act_in),
      kNativeFragment ? static_cast<float const *>(lora_act_in) : nullptr, lora_up, bias, wcscales,
      scales, alpha, M, N, K, rank, fused.qout, fused.oscales, fused.smooth, fused.lora_down,
      fused.d_atomic, fused.d_final, fused.down_scales, fused.rank_down, fused.write_dense,
      fused.fuse_silu, stream);
  };
  if constexpr (kBN == 256 && kBK == 128) {
    if (fused.qout != nullptr || fused.fuse_silu) {
      launch_gemm_w4a4_fused(act, wgt, out, act_sf, wgt_sf, lora_act_in, lora_up, bias, wcscales,
                             scales, alpha, M, N, K, rank, lora_layout, fused, stream);
      return;
    }
  } else if (fused.qout != nullptr || fused.fuse_silu) {
    // This tile configuration has no fused instance, and running the plain kernel would leave
    // qout/oscales/lora_act_out untouched. Fail loudly instead of returning garbage.
    throw std::runtime_error(
      "gemm_w4a4: the fused epilogue is compiled for the BN=256/BK=128 tile "
      "only; use N % 256 == 0 and K % 128 == 0.");
  }
  if (lora_layout == LoraLayout::kNative) {
    launch(std::true_type{}, std::false_type{});
  } else if (M >= kMainFirstMinM || (M >= kMainFirstWideM && N >= kMainFirstWideN)) {
    launch(std::false_type{}, std::false_type{});
  } else {
    launch(std::false_type{}, std::true_type{});
  }
}

}  // namespace

// Host entry for the SM120 CuTe path. Validation happens before any launch so unsupported dtypes
// or layouts fail instead of being misinterpreted by a TMA descriptor. Activation A/SFA keep the
// native 256-row MMA-fragment pack and are loaded G2R; weights use standard row-major E2M1 codes
// plus CUTLASS 128x4 SF atoms and are loaded by TMA. The independent CuTe activation producer emits
// the native pack plus BF16 D directly, while the native producer emits FP32 fragment D that
// `LoraLayout::kNative` lets the GEMM consume in place, so neither workflow needs a LoRA finalize
// or extra repack step.
//
// Two tile specializations are compiled:
//   BN=256, stages=3, BK=128: default when N/K alignment allows it; rank tiles have 32 columns.
//   BN=128, stages=4, BK=64:  fallback for narrower geometry; rank tiles have 16 columns.
// BK controls both the number of NVFP4 atoms per stage and the LoRA overlay width (BK/4), which
// is why rank divisibility is part of configuration selection. No runtime repack is performed.
void gemm_w4a4(svdq::Tensor act, svdq::Tensor wgt, svdq::Tensor out, svdq::Tensor ascales,
               svdq::Tensor wscales, svdq::Tensor lora_act_in, svdq::Tensor lora_up,
               svdq::Tensor bias, svdq::Tensor wcscales, std::vector<float> lora_scales,
               float alpha, LoraLayout lora_layout, svdq::Tensor qout, svdq::Tensor oscales,
               svdq::Tensor lora_down, svdq::Tensor lora_act_out, svdq::Tensor lora_scratch,
               svdq::Tensor smooth_factor, bool fuse_silu) {
  auto require = [](bool condition, const char *message) {
    if (!condition) {
      throw std::runtime_error(spdlog::fmt_lib::format("gemm_w4a4: {}", message));
    }
  };

  require(act.valid() && act.ndims() == 2, "act must be a valid 2D tensor");
  require(wgt.valid() && wgt.ndims() == 2, "wgt must be a valid 2D tensor");
  require(ascales.valid() && wscales.valid(), "ascales/wscales must be valid tensors");

  const int M = act.size(0);
  const int K = act.size(1) * 2;
  const int N = wgt.size(0);
  const int rank = lora_up.valid() ? lora_up.size(1) : 0;
  const bool fused = qout.valid() || fuse_silu;

  require(act.dtype() == svdq::Tensor::INT8 && wgt.dtype() == svdq::Tensor::INT8,
          "act/wgt must be packed int8/uint8 bytes");
  require(ascales.dtype() == svdq::Tensor::FP8_E4M3 && wscales.dtype() == svdq::Tensor::FP8_E4M3,
          "ascales/wscales must be FP8_E4M3");
  require(act.is_cuda() && wgt.is_cuda() && ascales.is_cuda() && wscales.is_cuda(),
          "act/wgt/ascales/wscales must be CUDA tensors");
  require(act.is_contiguous() && wgt.is_contiguous() && ascales.is_contiguous() &&
            wscales.is_contiguous(),
          "act/wgt/ascales/wscales must be contiguous");
  require(wgt.size(1) == act.size(1), "wgt must have the same packed K extent as act");
  require(
    M % 256 == 0 && N % 128 == 0,
    "M must be a multiple of 256 for the native activation pack; N must be a multiple of 128");
  require(K % 64 == 0, "K must be a multiple of 64");
  require(ascales.numel() == sf_buffer_bytes(M, K),
          "ascales must be the native interleaved activation scale pack for [M, K]");
  require(wscales.numel() == sf_buffer_bytes(N, K),
          "wscales must be a flat 128x4 512B atom buffer for [N, K]");

  // The dense output is optional only on the fused path, which can emit the packed activation
  // instead. Every other caller still has to provide it.
  require(out.valid() || fused, "out may only be omitted when the fused qout path is requested");
  if (out.valid()) {
    require(out.ndims() == 2 && out.size(0) == M && out.size(1) == N, "out must be [M, N]");
    require(out.dtype() == svdq::Tensor::BF16,
            "out must be BF16; only the BF16 epilogue is wired up");
    require(out.is_cuda() && out.is_contiguous(), "out must be a contiguous CUDA tensor");
  }

  require(lora_act_in.valid() == lora_up.valid(), "lora_act_in and lora_up must agree");
  if (rank > 0) {
    require(rank % 16 == 0, "R must be a multiple of 16");
    require(lora_up.ndims() == 2 && lora_up.size(0) == N && lora_act_in.ndims() == 2 &&
              lora_act_in.size(0) == M,
            "LoRA tensors must be [N, R] and [M, R]");
    require(lora_up.dtype() == svdq::Tensor::BF16, "lora_up must be BF16");
    require(lora_act_in.dtype() ==
              (lora_layout == LoraLayout::kNative ? svdq::Tensor::FP32 : svdq::Tensor::BF16),
            lora_layout == LoraLayout::kNative
              ? "lora_act_in must be the packed FP32 fragment D for LoraLayout::kNative"
              : "lora_act_in must be BF16 for LoraLayout::kRowMajor");
    require(lora_up.is_cuda() && lora_act_in.is_cuda() && lora_up.is_contiguous() &&
              lora_act_in.is_contiguous(),
            "LoRA tensors must be contiguous CUDA tensors");
  }
  if (bias.valid()) {
    require(bias.numel() == static_cast<size_t>(N) && bias.dtype() == svdq::Tensor::BF16 &&
              bias.is_cuda() && bias.is_contiguous(),
            "bias must be a contiguous BF16 [N] tensor");
  }
  if (wcscales.valid()) {
    require(wcscales.numel() == static_cast<size_t>(N) && wcscales.dtype() == svdq::Tensor::BF16 &&
              wcscales.is_cuda() && wcscales.is_contiguous(),
            "wcscales must be a contiguous BF16 [N] tensor");
  }

  const cudaDeviceProp *prop = getCurrentDeviceProperties();
  require(prop->major * 10 + prop->minor >= 120, "gemm_w4a4 requires an SM120 (or newer) device");

  // The native fragment path folds one scale per 16-rank group into its gmem gather, so the scale
  // table has to cover the group count that this rank implies.
  F32DLoraScales scales{};
  if (rank > 0) {
    const int groups = rank / 16;
    // Packed W1/W2 call this launcher with rank256/512 after their own host validation. Public
    // INT4/NVFP4 configs are capped by validate_linear_geometry and never reach r512 here.
    require(groups <= 32, "the CuTe LoRA path supports at most 32 rank groups (R <= 512)");
    require(static_cast<int>(lora_scales.size()) >= groups,
            "lora_scales must provide one value per 16-rank group");
    for (int i = 0; i < groups; ++i) scales.values[i] = lora_scales[i];
  }

  using ElementSF = cutlass::float_ue4m3_t;
  using ElementLora = cutlass::bfloat16_t;

  FusedLaunchArgs fused_args{};
  if (fused) {
    require(N % 256 == 0 && K % 128 == 0,
            "the fused epilogue is compiled for the BN=256/BK=128 tile only");
    // GELU belongs to the packed-activation path and SiLU to the dense-only one; accepting both
    // would silently drop the SiLU request.
    require(!(qout.valid() && fuse_silu),
            "fuse_silu is the dense-only activation; do not combine it with qout");
    require(out.valid() || qout.valid(),
            "the fused epilogue needs an output: either the dense buffer or qout");
    fused_args.write_dense = out.valid();
    fused_args.fuse_silu = fuse_silu;
    fused_args.qout = qout.valid() ? static_cast<uint8_t *>(qout.data_ptr()) : nullptr;
    if (qout.valid()) {
      require(
        qout.ndims() == 2 && qout.size(0) == M && qout.size(1) == N / 2 && qout.is_contiguous(),
        "qout must be a contiguous [M, N / 2] packed activation buffer");
      require(oscales.valid() && oscales.is_contiguous(), "qout requires oscales");
      require(oscales.dtype() == svdq::Tensor::FP8_E4M3,
              "oscales must be an FP8_E4M3 buffer of one byte per 16 packed activation codes");
      require(oscales.numel() == static_cast<size_t>(M) * N / 16,
              "oscales must hold one FP8 byte per 16 packed activation codes");
      require(smooth_factor.valid() && smooth_factor.dtype() == svdq::Tensor::BF16 &&
                smooth_factor.numel() == static_cast<size_t>(N) && smooth_factor.is_contiguous() &&
                smooth_factor.is_cuda(),
              "the fused quantizer requires a contiguous BF16 smooth_factor [N]");
      fused_args.oscales = static_cast<uint32_t *>(oscales.data_ptr());
      fused_args.smooth = static_cast<ElementLora const *>(smooth_factor.data_ptr());
    }
    if (lora_down.valid()) {
      require(qout.valid(), "lora_down is only meaningful together with qout");
      require(lora_down.ndims() == 2 && lora_down.size(0) == N &&
                lora_down.dtype() == svdq::Tensor::BF16 && lora_down.is_contiguous() &&
                lora_down.is_cuda(),
              "lora_down must be a contiguous BF16 [N, R_down] tensor");
      const int rank_down = lora_down.size(1);
      require(rank_down % 32 == 0 && rank_down <= 128,
              "the fused LoRA-down rank must be a multiple of 32 and at most 128");
      require(lora_act_out.valid() && lora_act_out.ndims() == 2 && lora_act_out.size(0) == M &&
                lora_act_out.size(1) == rank_down && lora_act_out.is_contiguous() &&
                lora_act_out.is_cuda(),
              "lora_act_out must be a contiguous [M, R_down] CUDA tensor");
      fused_args.lora_down = static_cast<ElementLora const *>(lora_down.data_ptr());
      fused_args.rank_down = rank_down;
      if (lora_layout == LoraLayout::kNative) {
        require(lora_act_out.dtype() == svdq::Tensor::FP32,
                "kNative writes the packed FP32 fragment D into lora_act_out");
        fused_args.d_atomic = static_cast<float *>(lora_act_out.data_ptr());
      } else {
        require(lora_act_out.dtype() == svdq::Tensor::BF16,
                "kRowMajor writes BF16 into lora_act_out");
        require(lora_scratch.valid() && lora_scratch.ndims() == 2 && lora_scratch.size(0) == M &&
                  lora_scratch.size(1) == rank_down && lora_scratch.dtype() == svdq::Tensor::FP32 &&
                  lora_scratch.is_contiguous() && lora_scratch.is_cuda(),
                "kRowMajor needs a zeroed FP32 [M, R_down] lora_scratch for the reduction");
        fused_args.d_atomic = static_cast<float *>(lora_scratch.data_ptr());
        fused_args.d_final = static_cast<ElementLora *>(lora_act_out.data_ptr());
        // The row-major finalize folds one scale per 16-rank group into the *next* layer's D. The
        // ABI has no channel for that layer's scales, so only the identity is well defined here;
        // accepting the up branch's list would fold the wrong coefficients silently.
        for (int i = 0; i < static_cast<int>(lora_scales.size()); ++i) {
          require(lora_scales[i] == 1.0f,
                  "the fused row-major D cannot fold per-rank scales from another layer; pass "
                  "lora_scales=None when the fused epilogue writes a row-major D");
        }
        const int groups_down = rank_down / 16;
        for (int i = 0; i < groups_down; ++i) fused_args.down_scales.values[i] = 1.0f;
      }
    } else {
      require(!lora_act_out.valid() && !lora_scratch.valid(), "lora_act_out requires lora_down");
    }
  } else {
    require(!qout.valid() && !oscales.valid() && !lora_down.valid() && !lora_act_out.valid() &&
              !smooth_factor.valid() && !fuse_silu,
            "the fused arguments require qout or fuse_silu");
  }

  const void *lora_act_ptr = rank > 0 ? lora_act_in.data_ptr() : nullptr;
  auto const *lora_up_ptr =
    rank > 0 ? static_cast<ElementLora const *>(lora_up.data_ptr()) : nullptr;
  void *out_ptr = out.valid() ? out.data_ptr() : nullptr;
  auto const *bias_ptr = bias.valid() ? static_cast<ElementLora const *>(bias.data_ptr()) : nullptr;
  auto const *wcscales_ptr =
    wcscales.valid() ? static_cast<ElementLora const *>(wcscales.data_ptr()) : nullptr;

  const bool native_fragment_pack = wscales.ndims() == 2;
  const bool use_large_k = native_fragment_pack && !fused && lora_layout == LoraLayout::kRowMajor &&
                           bias_ptr != nullptr && wcscales_ptr != nullptr && M >= 4096 &&
                           N <= 4096 && K >= 8192 && M % 256 == 0 && N % 128 == 0 && K % 64 == 0;
  if (native_fragment_pack) {
    require(wscales.size(0) * 16 == K && wscales.size(1) == N,
            "native fragment-packed wscales must have shape [K/16, N]");
    require(use_large_k,
            "native fragment-packed weights on the CuTe runtime require the large-K non-fused "
            "row-major-D path with bias and wcscales");
    launch_gemm_w4a4_large_k(act.data_ptr(), wgt.data_ptr(), out_ptr, ascales.data_ptr(),
                             wscales.data_ptr(), lora_act_ptr, lora_up_ptr, bias_ptr, wcscales_ptr,
                             alpha, M, N, K, rank, getCurrentCUDAStream());
    checkCUDA(cudaGetLastError());
    return;
  }
  if (N % 256 == 0 && K % 128 == 0) {
    require(rank % 32 == 0, "rank must be a multiple of 32 when BN=256 and BK=128");
    launch_gemm_w4a4_config<256, 3, 128>(
      act.data_ptr(), wgt.data_ptr(), out_ptr, static_cast<ElementSF const *>(ascales.data_ptr()),
      static_cast<ElementSF const *>(wscales.data_ptr()), lora_act_ptr, lora_up_ptr, bias_ptr,
      wcscales_ptr, scales, alpha, M, N, K, rank, lora_layout, fused_args, getCurrentCUDAStream());
  } else {
    launch_gemm_w4a4_config<128, 4, 64>(
      act.data_ptr(), wgt.data_ptr(), out_ptr, static_cast<ElementSF const *>(ascales.data_ptr()),
      static_cast<ElementSF const *>(wscales.data_ptr()), lora_act_ptr, lora_up_ptr, bias_ptr,
      wcscales_ptr, scales, alpha, M, N, K, rank, lora_layout, fused_args, getCurrentCUDAStream());
  }
  checkCUDA(cudaGetLastError());
}

}  // namespace svdq::cute_kernels
