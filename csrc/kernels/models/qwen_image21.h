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
// `variant` selects the FMA contraction pattern of the rotation; 3 (the classic `__mulsc3`
// pairing) is what `fused_ops.ROTATION_VARIANT` ships.
void qwen_image21_qk_norm_rope(at::Tensor x, at::Tensor weight, at::Tensor cos, at::Tensor sin,
                               double eps, int64_t variant, at::Tensor out);

// `x_mod = layer_norm(x) * one_plus[row]`, where `row` is selected per token.
// `one_plus` holds the already folded `bf16(1 + scale)` rows `(B (+ 1), D)`: the modulation
// is constant across tokens, so `fused_ops` prepares it once instead of letting the kernel
// recompute it - and materialise a `(B, S, D)` gate - for every token.
void qwen_image21_ln_modulate(at::Tensor x, at::Tensor one_plus,
                              const std::optional<at::Tensor>& target_token_mask, double eps,
                              at::Tensor x_mod);

// `y = hidden_states + bf16(gate_rows[row] * update)`, out of place.
// `gate_rows` holds the already `tanh`-ed gate `(B (+ 1), D)` rows, so this reads 2 rows
// instead of a full `(B, S, D)` tensor.
void qwen_image21_gated_residual(at::Tensor hidden_states, at::Tensor gate_rows,
                                 const std::optional<at::Tensor>& target_token_mask,
                                 at::Tensor update, at::Tensor out, int64_t seq_len);

// `y = bf16(silu_fp32(gate)) * up`, one bf16 rounding before the product.
void qwen_image21_silu_mul(at::Tensor gate, at::Tensor up, at::Tensor out);

// Hooks the Qwen-Image 2.1 operators onto the extension's `ops` submodule.
void register_qwen_image21_ops(pybind11::module_& ops);

}  // namespace cache_dit::models
