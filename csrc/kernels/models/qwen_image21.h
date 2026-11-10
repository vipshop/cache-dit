#pragma once

#include <ATen/ATen.h>

#include <optional>

namespace pybind11 {
class module_;
}  // namespace pybind11

namespace cache_dit::models {

// Per-head RMSNorm followed by interleaved-pair complex RoPE.
// Mirrors diffusers' `RMSNorm(dim_head)` + `apply_rotary_emb_qwen(use_real=False)` chain
// with its three separate bf16 roundings (after the norm, after the channel scale, after
// the rotation), so the result is bit-identical to the eager expression.
// `out` may be any dense bf16 buffer of the same shape as `x` (e.g. the target tail of a
// persistent KV buffer); its strides are honoured.
// `variant` selects the FMA contraction pattern of the rotation (see the probe in
// `fused_ops`); 0 = no contraction, 1 = fuse the real product, 2 = fuse the imaginary one.
void qwen_image21_qk_norm_rope(at::Tensor x, at::Tensor weight, at::Tensor cos, at::Tensor sin,
                               double eps, int64_t variant, at::Tensor out);

// LayerNorm(elementwise_affine=False) followed by the `1 + scale` modulation and the
// `tanh(gate)` row-select of one transformer block. `params` holds `scale | gate` for each
// batch row plus the trailing `t = 0` row; `target_token_mask` (optional) selects between
// them per token, exactly like `_select_modulation_rows`.
void qwen_image21_ln_modulate(at::Tensor x, at::Tensor params,
                              const std::optional<at::Tensor>& target_token_mask, double eps,
                              at::Tensor x_mod, at::Tensor gate_tanh);

// Same epilogue as `qwen_image21_ln_modulate` but on an already normalized input, for the
// tier where the device's LayerNorm reduction cannot be reproduced bit-exactly.
void qwen_image21_modulate_affine(at::Tensor x_normed, at::Tensor params,
                                  const std::optional<at::Tensor>& target_token_mask,
                                  at::Tensor x_mod, at::Tensor gate_tanh);

// y = hidden_states + bf16(gate * update), out of place, two bf16 roundings.
void qwen_image21_gated_residual(at::Tensor hidden_states, at::Tensor gate, at::Tensor update,
                                 at::Tensor out);

// y = bf16(silu_fp32(gate)) * up, one bf16 rounding before the product.
void qwen_image21_silu_mul(at::Tensor gate, at::Tensor up, at::Tensor out);

// Hooks the four Qwen-Image 2.1 operators onto the extension's `ops` submodule.
void register_qwen_image21_ops(pybind11::module_& ops);

}  // namespace cache_dit::models
