// Copied from nunchaku/nunchaku/csrc/ops.h
#pragma once

// SVDQ source map
// ----------------
// Public host surface:
//   ops.h                  This file. Owns the only two public runtime dispatchers, `gemm_w4a4`
//                          and `quantize_w4a4`. It normalizes RuntimeParams/LoraLayout and selects
//                          native plain, native ext, or CuTe.
//   pybind.cpp              Binds only the two public dispatchers (and the logging utility).
//   runtime_params.h        Host-only RuntimeKernel/RuntimeParams (runtime/fp4/stage/alpha/
//                          act_unsigned); no device code depends on it.
//   torch.h / torch.cpp     Torch tensor adapters, TorchOpContext, CUDA stream and logging glue.
//   Tensor.h / common.h     Lightweight tensor descriptor, shape/stride helpers and common CUDA
//                          error/assert utilities shared by native and CuTe kernels.
//
// Native runtime:
//   native/zgemm.h          Native entry declarations:
//                          - gemm_w4a4: plain INT4 only, automatic stage=1 fast path;
//                          - gemm_w4a4_ext: full ABI for FP4, fused MLP, attention/QKV/qout,
//                            smooth-factor writeback and other extended outputs;
//                          - quantize_w4a4: native activation + LoRA-down producer;
//                          - quantize_w4a4_wo: internal native weight-only implementation, not
//                            exposed through pybind.
//   native/dispatch.h       Shared dtype/dispatch helpers for native template configuration.
//   native/int4/gemm_w4a4.{cu,cuh}
//                          The single plain native INT4 GEMM. It implements the normal runtime
//                          path and supports stage 1/2/3 with stage=1 as the default fast path.
//   native/int4/gemm_w4a4_impl.cuh / gemm_w4a4_launch.cuh
//                          Plain INT4 launch selector and specialization helpers.
//   native/int4/gemm_w4a4_i4_s{1,2,3}.cu
//                          Explicit BF16/FP16 instantiation for each plain INT4 stage.
//   native/int4/gemm_w4a4_ext_i4.cu
//                          Explicit INT4 instantiations of the full ext kernel.
//   native/fp4/gemm_w4a4_ext_f4.cu
//                          Explicit NVFP4 instantiations of the full ext kernel.
//   native/core/gemm_w4a4_ext.{cu,cuh}
//                          Full native ext GEMM implementation, including FP4 and fused
//                          MLP/attention/qout/smooth-output paths.
//   native/core/gemm_w4a4_ext_impl.cuh / gemm_w4a4_ext_launch.cuh
//                          Ext kernel body/launch wiring and explicit-instantiation helpers.
//   native/core/gemm_base.cuh, mma.cuh, lora.cuh, epilogues.cuh, gemm_utils.cuh, utils.cuh
//                          Shared native GEMM building blocks: tile/host config, MMA atoms, LoRA
//                          epilogue, fusion epilogues, quantization/vector utilities.
//
// CuTe runtime:
//   cute/fp4/gemm_w4a4.h   Public CuTe NVFP4 GEMM ABI and LoraLayout enum.
//   cute/fp4/gemm_w4a4.cuh / gemm_w4a4.cu
//                          SM120 CuTe NVFP4 TMA/MMA implementation for native and row-major D.
//   cute/fp4/quantize_w4a4.cu
//                          Unified W4A4QuantizeKernel producer: native activation pack + row-major
//                          BF16 LoRA-down D, launched when lora_layout='row_major'.
//
// Selection summary:
//   native + native layout + plain INT4 -> native gemm_w4a4 (stage=1 default)
//   native + FP4 or any ext/fused argument -> native gemm_w4a4_ext
//   cute + FP4 -> cute gemm_w4a4 with LoraLayout kNative/kRowMajor
//   native producer -> native quantize_w4a4; row_major producer -> cute quantize_w4a4

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "torch.h"
#include "runtime_params.h"
#include "native/zgemm.h"
#include "cute/fp4/gemm_w4a4.h"

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
inline cute_kernels::LoraLayout parse_lora_layout(const std::string &lora_layout) {
  if (lora_layout == "native") {
    return cute_kernels::LoraLayout::kNative;
  }
  if (lora_layout == "row_major") {
    return cute_kernels::LoraLayout::kRowMajor;
  }
  throw std::runtime_error("lora_layout must be 'native' or 'row_major', got '" + lora_layout +
                           "'.");
}

inline RuntimeParams make_runtime_params(const std::string &runtime, bool fp4, int stage,
                                         float alpha, bool act_unsigned) {
  RuntimeParams params;
  if (runtime == "native") {
    params.kernel = RuntimeKernel::kNative;
  } else if (runtime == "cute") {
    params.kernel = RuntimeKernel::kCuTe;
  } else {
    throw std::runtime_error("runtime must be 'native' or 'cute', got '" + runtime + "'.");
  }
  if (stage < 1 || stage > 3) {
    throw std::runtime_error("stage must be 1, 2, or 3.");
  }
  params.fp4 = fp4;
  params.stage = stage;
  params.alpha = alpha;
  params.act_unsigned = act_unsigned;
  return params;
}

// Extended-only arguments select the full native ext ABI. Keeping this check in one place prevents
// the plain INT4 path from silently dropping qout/oscales/attention/fused-MLP semantics.
inline bool has_extended_gemm_args(
  const std::optional<torch::Tensor> &qout, const std::optional<torch::Tensor> &oscales,
  const std::optional<torch::Tensor> &poolout, const std::optional<torch::Tensor> &lora_down,
  const std::optional<torch::Tensor> &lora_act_out, const std::optional<torch::Tensor> &norm_q,
  const std::optional<torch::Tensor> &norm_k, const std::optional<torch::Tensor> &rotary_emb,
  const std::optional<torch::Tensor> &smooth_factor, const std::optional<torch::Tensor> &out_vk,
  const std::optional<torch::Tensor> &out_linearattn, bool fuse_silu,
  const std::optional<torch::Tensor> &out_q, const std::optional<torch::Tensor> &out_k,
  const std::optional<torch::Tensor> &out_v, int attn_tokens) {
  return qout.has_value() || oscales.has_value() || poolout.has_value() || lora_down.has_value() ||
         lora_act_out.has_value() || norm_q.has_value() || norm_k.has_value() ||
         rotary_emb.has_value() || smooth_factor.has_value() || out_vk.has_value() ||
         out_linearattn.has_value() || fuse_silu || out_q.has_value() || out_k.has_value() ||
         out_v.has_value() || attn_tokens != 0;
}

// Single public W4A4 GEMM entry. Python/native callers provide the full optional ABI once; this
// function normalizes the host parameters and selects the already-compiled kernel family.
void gemm_w4a4(std::optional<torch::Tensor> act, std::optional<torch::Tensor> wgt,
               std::optional<torch::Tensor> out, std::optional<torch::Tensor> qout,
               std::optional<torch::Tensor> ascales, std::optional<torch::Tensor> wscales,
               std::optional<torch::Tensor> oscales, std::optional<torch::Tensor> poolout,
               std::optional<torch::Tensor> lora_act_in, std::optional<torch::Tensor> lora_up,
               std::optional<torch::Tensor> lora_down, std::optional<torch::Tensor> lora_act_out,
               std::optional<torch::Tensor> norm_q, std::optional<torch::Tensor> norm_k,
               std::optional<torch::Tensor> rotary_emb, std::optional<torch::Tensor> bias,
               std::optional<torch::Tensor> smooth_factor, std::optional<torch::Tensor> out_vk,
               std::optional<torch::Tensor> out_linearattn, bool act_unsigned,
               std::vector<float> lora_scales, bool fuse_silu, bool fp4, float alpha,
               std::optional<torch::Tensor> wcscales, std::optional<torch::Tensor> out_q,
               std::optional<torch::Tensor> out_k, std::optional<torch::Tensor> out_v,
               int attn_tokens, const std::string &runtime, const std::string &lora_layout,
               int stage = 1) {
  const RuntimeParams params = make_runtime_params(runtime, fp4, stage, alpha, act_unsigned);
  const auto layout = parse_lora_layout(lora_layout);
  const bool extended = has_extended_gemm_args(
    qout, oscales, poolout, lora_down, lora_act_out, norm_q, norm_k, rotary_emb, smooth_factor,
    out_vk, out_linearattn, fuse_silu, out_q, out_k, out_v, attn_tokens);
  TorchOpContext ctx;

  if (params.kernel == RuntimeKernel::kCuTe) {
    if (!params.fp4) {
      throw std::runtime_error("runtime='cute' implements the NVFP4 path only; pass fp4=True.");
    }
    if (extended) {
      throw std::runtime_error(
        "runtime='cute' does not accept extended native arguments (qout/oscales/fused outputs).");
    }
    cute_kernels::gemm_w4a4(get_tensor(act), get_tensor(wgt), get_tensor(out), get_tensor(ascales),
                            get_tensor(wscales), get_tensor(lora_act_in), get_tensor(lora_up),
                            get_tensor(bias), get_tensor(wcscales), std::move(lora_scales),
                            params.alpha, layout);
    return;
  }

  if (layout != cute_kernels::LoraLayout::kNative) {
    throw std::runtime_error("runtime='native' only supports lora_layout='native'.");
  }
  if (params.fp4 || extended) {
    // Full ext ABI; still internal so public callers only see one dispatcher.
    native_kernels::gemm_w4a4_ext(
      get_tensor(act), get_tensor(wgt), get_tensor(out), get_tensor(qout), get_tensor(ascales),
      get_tensor(wscales), get_tensor(oscales), get_tensor(poolout), get_tensor(lora_act_in),
      get_tensor(lora_up), get_tensor(lora_down), get_tensor(lora_act_out), get_tensor(norm_q),
      get_tensor(norm_k), get_tensor(rotary_emb), get_tensor(bias), get_tensor(smooth_factor),
      get_tensor(out_vk), get_tensor(out_linearattn), params.act_unsigned, std::move(lora_scales),
      fuse_silu, params.fp4, params.alpha, get_tensor(wcscales), get_tensor(out_q),
      get_tensor(out_k), get_tensor(out_v), attn_tokens);
    return;
  }

  // Plain INT4 path: native gemm_w4a4 is the only implementation, with stage=1 as the default.
  native_kernels::gemm_w4a4(get_tensor(act), get_tensor(wgt), get_tensor(out), get_tensor(ascales),
                            get_tensor(wscales), get_tensor(lora_act_in), get_tensor(lora_up),
                            get_tensor(bias), params.act_unsigned, params.alpha,
                            get_tensor(wcscales), params.stage);
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
  if (layout == cute_kernels::LoraLayout::kRowMajor) {
    if (!fp4) {
      throw std::runtime_error("lora_layout='row_major' implements the NVFP4 producer only.");
    }
    cute_kernels::quantize_w4a4(get_tensor(input), get_tensor(output), get_tensor(ascales),
                                get_tensor(lora_down), get_tensor(lora_act), get_tensor(smooth),
                                std::move(lora_scales));
    return;
  }
  native_kernels::quantize_w4a4(get_tensor(input), get_tensor(output), get_tensor(ascales),
                                get_tensor(lora_down), get_tensor(lora_act), get_tensor(smooth),
                                fuse_glu, fp4);
}

}  // namespace svdq::ops
