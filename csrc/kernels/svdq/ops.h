// Copied from nunchaku/nunchaku/csrc/ops.h
#pragma once

#include <optional>
#include <stdexcept>
#include <vector>

#include "torch.h"
#include "zgemm.h"

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

inline void gemm_w4a4(std::optional<torch::Tensor> act, std::optional<torch::Tensor> wgt,
                      std::optional<torch::Tensor> out, std::optional<torch::Tensor> qout,
                      std::optional<torch::Tensor> ascales, std::optional<torch::Tensor> wscales,
                      std::optional<torch::Tensor> oscales, std::optional<torch::Tensor> poolout,
                      std::optional<torch::Tensor> lora_act_in,
                      std::optional<torch::Tensor> lora_up, std::optional<torch::Tensor> lora_down,
                      std::optional<torch::Tensor> lora_act_out,
                      std::optional<torch::Tensor> norm_q, std::optional<torch::Tensor> norm_k,
                      std::optional<torch::Tensor> rotary_emb, std::optional<torch::Tensor> bias,
                      std::optional<torch::Tensor> smooth_factor,
                      std::optional<torch::Tensor> out_vk,
                      std::optional<torch::Tensor> out_linearattn, bool act_unsigned,
                      std::vector<float> lora_scales, bool fuse_silu, bool fp4, float alpha,
                      std::optional<torch::Tensor> wcscales, std::optional<torch::Tensor> out_q,
                      std::optional<torch::Tensor> out_k, std::optional<torch::Tensor> out_v,
                      int attn_tokens) {
  TorchOpContext ctx;
  spdlog::trace("running gemm_w4a4:");
  svdq::kernels::gemm_w4a4(
    get_tensor(act), get_tensor(wgt), get_tensor(out), get_tensor(qout), get_tensor(ascales),
    get_tensor(wscales), get_tensor(oscales), get_tensor(poolout), get_tensor(lora_act_in),
    get_tensor(lora_up), get_tensor(lora_down), get_tensor(lora_act_out), get_tensor(norm_q),
    get_tensor(norm_k), get_tensor(rotary_emb), get_tensor(bias), get_tensor(smooth_factor),
    get_tensor(out_vk), get_tensor(out_linearattn), act_unsigned, lora_scales, fuse_silu, fp4,
    alpha, get_tensor(wcscales), get_tensor(out_q), get_tensor(out_k), get_tensor(out_v),
    attn_tokens);
}

inline void gemm_w4a4_v2(std::optional<torch::Tensor> act, std::optional<torch::Tensor> wgt,
                         std::optional<torch::Tensor> out,
                         std::optional<torch::Tensor> ascales,
                         std::optional<torch::Tensor> wscales,
                         std::optional<torch::Tensor> lora_act_in,
                         std::optional<torch::Tensor> lora_up,
                         std::optional<torch::Tensor> bias, bool fp4, float alpha,
                         std::optional<torch::Tensor> wcscales, bool act_unsigned,
                         int stage = 1) {
  if (fp4) {
    throw std::runtime_error(
      "svdq_gemm_w4a4_v2 currently supports the INT4 runtime path only.");
  }

  // Ada-class GPUs such as L20 keep this plain v2 path register-bound. `stage=1`
  // is the public default because it preserves the best occupancy for the hot
  // runtime shapes, while higher stage counts remain available for explicit
  // profiling and follow-up experiments.

  TorchOpContext ctx;
  spdlog::trace("running gemm_w4a4_v2:");
  svdq::kernels::gemm_w4a4_v2(
    get_tensor(act), get_tensor(wgt), get_tensor(out), get_tensor(ascales), get_tensor(wscales),
    get_tensor(lora_act_in), get_tensor(lora_up), get_tensor(bias), act_unsigned, alpha,
    get_tensor(wcscales), stage);
}

inline void quantize_w4a4_act_fuse_lora(std::optional<torch::Tensor> input,
                                        std::optional<torch::Tensor> output,
                                        std::optional<torch::Tensor> oscales,
                                        std::optional<torch::Tensor> lora_down,
                                        std::optional<torch::Tensor> lora_act_out,
                                        std::optional<torch::Tensor> smooth, bool fuse_glu,
                                        bool fp4) {
  TorchOpContext ctx;
  spdlog::trace("running quantize_w4a4_act_fuse_lora:");
  svdq::kernels::quantize_w4a4_act_fuse_lora(
    get_tensor(input), get_tensor(output), get_tensor(oscales), get_tensor(lora_down),
    get_tensor(lora_act_out), get_tensor(smooth), fuse_glu, fp4);
}

inline void quantize_w4a4_wgt(torch::Tensor input, torch::Tensor output, torch::Tensor oscales) {
  TorchOpContext ctx;
  spdlog::trace("running quantize_w4a4_wgt:");
  svdq::kernels::quantize_w4a4_wgt(from_torch(input), from_torch(output), from_torch(oscales));
}

inline void set_faster_i2f_mode(const std::string &mode) {
  svdq::kernels::set_faster_i2f_mode(mode);
}

}  // namespace svdq::ops
