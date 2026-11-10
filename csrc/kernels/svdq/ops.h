// Adapted from nunchaku/nunchaku/csrc/ops.h
#pragma once

// SVDQ source map
// ----------------
// Public host surface:
//   ops.h                  This file. Owns the only two public runtime dispatchers, `gemm_w4a4`
//                          and `quantize_w4a4`. It normalizes RuntimeParams/LoraLayout and selects
//                          native or CuTe.
//   pybind.cpp              Binds only the two public dispatchers (and the logging utility).
//   runtime.h               Host-only RuntimeKernel/RuntimeParams (runtime/fp4/alpha/
//                          act_unsigned) plus the shared LoraLayout enum; no device kernel body
//                          depends on it.
//   torch.h / torch.cpp     Torch tensor adapters, TorchOpContext, CUDA stream and logging glue.
//   Tensor.h / common.h     Lightweight tensor descriptor, shape/stride helpers and common CUDA
//                          error/assert utilities shared by native and CuTe kernels.
//
// Native runtime:
//   native/zgemm.h          Native entry declarations: `gemm_w4a4` is the single W4A4 GEMM shared
//                          by INT4 and NVFP4, `quantize_w4a4` the activation + LoRA-down producer.
//   native/dispatch.h       Shared dtype/dispatch helpers for native template configuration.
//   native/gemm_w4a4.{cu,cuh}
//                          Native GEMM implementation, including FP4 and the fused MLP,
//                          quantized-output and smooth-output paths.
//   native/gemm_w4a4_impl.cuh / gemm_w4a4_launch.cuh
//                          Kernel body/launch wiring and explicit-instantiation helpers.
//   native/gemm_base.cuh, mma.cuh, lora.cuh, epilogues.cuh, gemm_utils.cuh, utils.cuh
//                          Shared native GEMM building blocks: tile/host config, MMA atoms, LoRA
//                          epilogue, fusion epilogues, quantization/vector utilities.
//   native/gemm_w4a4_int4.cu
//                          Explicit INT4 instantiations of the native GEMM.
//   native/gemm_w4a4_fp4.cu
//                          Explicit NVFP4 instantiations of the native GEMM.
//
// CuTe runtime:
//   cute/fp4/gemm/gemm_w4a4.{h,cuh,cu}
//                          SM120 CuTe NVFP4 TMA/MMA implementation for native and row-major D.
//   cute/fp4/quantize/quantize_uni.cu
//                          Universal producer for rank 0 and all non-specialized rank values.
//   cute/fp4/quantize/quantize_impl.cuh
//                          Shared TiledMMA/TiledCopy implementation for rank-specialized TUs.
//   cute/fp4/quantize/quantize_r{32,64,128}.cu
//                          Independent rank-specific fused and split-K instantiations.
//
// Selection summary:
//   native + native layout -> native gemm_w4a4 (INT4 and NVFP4 share one kernel family)
//   cute + FP4 -> cute gemm_w4a4 with LoraLayout kNative/kRowMajor
//   native producer -> native quantize_w4a4; row_major producer -> cute quantize_w4a4

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "torch.h"
#include "runtime.h"
#include "native/zgemm.h"
#include "cute/fp4/gemm/gemm_w4a4.h"
#include "cute/fp4/quantize/quantize.h"

namespace svdq::ops {

inline Tensor get_tensor(std::optional<torch::Tensor> &tensor) {
  Tensor result = tensor.has_value() ? from_torch(tensor.value()) : Tensor{};
  if (result.valid()) {
    spdlog::trace("  {}", result.shape.str());
  } else {
    spdlog::trace("  <invalid>");
  }
  return result;
}

// CuTe-backed LoRA-down layout selector. Native runtime is always kNative.
inline LoraLayout parse_lora_layout(const std::string &lora_layout) {
  if (lora_layout == "native") {
    return LoraLayout::kNative;
  }
  if (lora_layout == "row_major") {
    return LoraLayout::kRowMajor;
  }
  throw std::runtime_error("lora_layout must be 'native' or 'row_major', got '" + lora_layout +
                           "'.");
}

inline RuntimeParams make_runtime_params(const std::string &runtime, bool fp4, float alpha,
                                         bool act_unsigned) {
  RuntimeParams params;
  if (runtime == "native") {
    params.kernel = RuntimeKernel::kNative;
  } else if (runtime == "cute") {
    params.kernel = RuntimeKernel::kCuTe;
  } else {
    throw std::runtime_error("runtime must be 'native' or 'cute', got '" + runtime + "'.");
  }
  params.fp4 = fp4;
  params.alpha = alpha;
  params.act_unsigned = act_unsigned;
  return params;
}

// Single public W4A4 GEMM entry. Python/native callers provide the full optional ABI once; this
// function normalizes the host parameters and selects the already-compiled kernel family. The
// fused arguments (qout/oscales/lora_down/lora_act_out/smooth_factor/fuse_silu) belong to the
// native kernel only, so the CuTe branch rejects them instead of silently dropping their meaning.
void gemm_w4a4(std::optional<torch::Tensor> act, std::optional<torch::Tensor> wgt,
               std::optional<torch::Tensor> out, std::optional<torch::Tensor> qout,
               std::optional<torch::Tensor> ascales, std::optional<torch::Tensor> wscales,
               std::optional<torch::Tensor> oscales, std::optional<torch::Tensor> lora_act_in,
               std::optional<torch::Tensor> lora_up, std::optional<torch::Tensor> lora_down,
               std::optional<torch::Tensor> lora_act_out, std::optional<torch::Tensor> bias,
               std::optional<torch::Tensor> smooth_factor, bool act_unsigned,
               std::vector<float> lora_scales, bool fuse_silu, bool fp4, float alpha,
               std::optional<torch::Tensor> wcscales, const std::string &runtime,
               const std::string &lora_layout) {
  const RuntimeParams params = make_runtime_params(runtime, fp4, alpha, act_unsigned);
  const auto layout = parse_lora_layout(lora_layout);
  TorchOpContext ctx;

  if (params.kernel == RuntimeKernel::kCuTe) {
    if (!params.fp4) {
      throw std::runtime_error("runtime='cute' implements the NVFP4 path only; pass fp4=True.");
    }
    if (qout.has_value() || oscales.has_value() || lora_down.has_value() ||
        lora_act_out.has_value() || smooth_factor.has_value() || fuse_silu) {
      throw std::runtime_error(
        "runtime='cute' does not accept the fused native arguments "
        "(qout/oscales/lora_down/lora_act_out/smooth_factor/fuse_silu).");
    }
    svdq::cute_kernels::gemm_w4a4(get_tensor(act), get_tensor(wgt), get_tensor(out),
                                  get_tensor(ascales), get_tensor(wscales), get_tensor(lora_act_in),
                                  get_tensor(lora_up), get_tensor(bias), get_tensor(wcscales),
                                  std::move(lora_scales), params.alpha, layout);
    return;
  }

  if (layout != LoraLayout::kNative) {
    throw std::runtime_error("runtime='native' only supports lora_layout='native'.");
  }
  // Fully qualified on purpose: this function shares its name with the native entry point.
  svdq::native_kernels::gemm_w4a4(
    get_tensor(act), get_tensor(wgt), get_tensor(out), get_tensor(qout), get_tensor(ascales),
    get_tensor(wscales), get_tensor(oscales), get_tensor(lora_act_in), get_tensor(lora_up),
    get_tensor(lora_down), get_tensor(lora_act_out), get_tensor(bias), get_tensor(smooth_factor),
    params.act_unsigned, std::move(lora_scales), fuse_silu, params.fp4, params.alpha,
    get_tensor(wcscales));
}

// Single public W4A4 activation/LoRA-down producer. LoraLayout determines both the output contract
// and the implementation: native layout -> native producer, row_major -> CuTe producer.
void quantize_w4a4(std::optional<torch::Tensor> input, std::optional<torch::Tensor> output,
                   std::optional<torch::Tensor> ascales, std::optional<torch::Tensor> lora_down,
                   std::optional<torch::Tensor> lora_act, std::optional<torch::Tensor> smooth,
                   bool fuse_glu, bool fp4, std::vector<float> lora_scales,
                   const std::string &lora_layout) {
  const auto layout = parse_lora_layout(lora_layout);
  TorchOpContext ctx;
  if (layout == LoraLayout::kRowMajor) {
    if (!fp4) {
      throw std::runtime_error("lora_layout='row_major' implements the NVFP4 producer only.");
    }
    const int64_t rank = lora_down && lora_down->dim() > 0 ? lora_down->size(-1) : 0;
    const int64_t K = input && input->dim() > 0 ? input->size(-1) : 0;
    if (input && input->dim() > 0 && K <= 0) {
      throw std::runtime_error("quantize_w4a4 requires a non-empty input channel dimension.");
    }
    const int64_t M = K > 0 ? input->numel() / K : 0;
    torch::Tensor lora_scratch_t;
    if (input && lora_down && input->dim() > 0 && lora_down->dim() > 0) {
      // Rank32 crosses to fused above 8K; rank64/128 use split-K throughout the measured matrix.
      const bool use_split_k = cute_kernels::is_specialized_rank(rank) && (rank != 32 || M <= 8192);
      if (use_split_k) {
        lora_scratch_t = at::empty({M, rank}, input->options().dtype(at::kFloat));
      }
    }
    svdq::Tensor lora_scratch;
    if (lora_scratch_t.defined()) {
      lora_scratch = from_torch(lora_scratch_t);
    }
    auto quantize = &cute_kernels::quantize_uni;
    if (rank == 32 && K > 3072) {
      // Keep the universal producer for rank32/K3072, where it remains faster at 2K/4K/8K.
      quantize = &cute_kernels::quantize_r32;
    } else if (rank == 64) {
      quantize = &cute_kernels::quantize_r64;
    } else if (rank == 128) {
      quantize = &cute_kernels::quantize_r128;
    }
    quantize(get_tensor(input), get_tensor(output), get_tensor(ascales), get_tensor(lora_down),
             get_tensor(lora_act), get_tensor(smooth), std::move(lora_scales), lora_scratch);
    return;
  }
  svdq::native_kernels::quantize_w4a4(get_tensor(input), get_tensor(output), get_tensor(ascales),
                                      get_tensor(lora_down), get_tensor(lora_act),
                                      get_tensor(smooth), fuse_glu, fp4);
}

}  // namespace svdq::ops
