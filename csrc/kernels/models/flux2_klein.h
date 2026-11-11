#pragma once

#include <ATen/ATen.h>

namespace pybind11 {
class module_;
}  // namespace pybind11

namespace cache_dit::models {

// Per-head RMSNorm followed by GPT-J (interleaved-pair) RoPE: one read and one write.
//
// Mirrors diffusers' `torch.nn.RMSNorm(head_dim)` + `apply_rotary_emb(use_real=True,
// use_real_unbind_dim=-1, sequence_dim=1)` chain. `torch.nn.RMSNorm` accumulates in fp32 and
// rounds once, after the affine (`xDiT .../fused_qk_rope_flydsl.py:726-757`), while
// `apply_rotary_emb` upcasts to fp32 for the rotation and rounds once at the end
// (`diffusers/src/diffusers/models/embeddings.py:986-997`). Both roundings are replayed, so the
// result follows the same schedule; the only remaining difference is the reduction tree of the
// RMS, which shows up as roughly 0.5% of elements off by one bf16 ulp (measured worst rel-L2
// 2.8e-6 against the eager expression).
//
// `cos`/`sin` are the fp32 `[rope_rows, head_dim]` tables; `rope_row_offset` selects where
// this stream starts inside them, which is what lets the joined variant share one
// `torch.cat([text, image])` table. `out` is a dense `[S_total, H, D]` bf16 buffer and
// `out_token_offset` selects the token range this stream owns.
void flux2_klein_qk_norm_rope(at::Tensor x, at::Tensor weight, at::Tensor cos, at::Tensor sin,
                              int64_t rope_row_offset, double eps, at::Tensor out,
                              int64_t out_token_offset);

// The joined variant: normalises and rotates both streams in one call and writes them into one
// buffer, so `torch.cat([encoder_q, q], dim=1)` never materialises a second copy. Text tokens
// come first, exactly like the eager cat.
void flux2_klein_qk_norm_rope_join(at::Tensor image, at::Tensor image_weight, at::Tensor text,
                                   at::Tensor text_weight, at::Tensor cos, at::Tensor sin,
                                   double eps, at::Tensor out);

// `out = bf16(bf16(layer_norm(x)) * one_plus[row]) + shift[row]`.
// `one_plus`/`shift` are the folded `(1 + scale)` and `shift` rows, prepared once per forward,
// so no per-token modulation tensor is materialised.
void flux2_klein_ln_modulate(at::Tensor x, at::Tensor one_plus, at::Tensor shift, double eps,
                             at::Tensor out);

// The block's residual add and the normalisation that follows it, in one pass over `hidden`:
//   `hidden_out = bf16(hidden + bf16(gate[row] * delta))`
//   `mod_out    = bf16(bf16(layer_norm(hidden_out)) * one_plus[row]) + shift[row]`
// The eager chain reads `hidden` three times (add, statistics, apply); this reads it once.
void flux2_klein_residual_ln_modulate(at::Tensor hidden, at::Tensor gate, at::Tensor delta,
                                      at::Tensor one_plus, at::Tensor shift, double eps,
                                      at::Tensor hidden_out, at::Tensor mod_out);

// `out = bf16(hidden + bf16(gate[row] * update))`, out of place.
void flux2_klein_gated_residual(at::Tensor hidden, at::Tensor gate, at::Tensor update,
                                at::Tensor out);

// `out = bf16(silu_fp32(gate)) * up` over the two halves of `gate_up`'s last dimension, i.e.
// `F.silu(x[..., :D]) * x[..., D:]`'s rounding schedule.
void flux2_klein_silu_mul(at::Tensor gate_up, at::Tensor out);

// Hooks the FLUX.2-Klein operators onto the extension's `ops` submodule.
void register_flux2_klein_ops(pybind11::module_& ops);

}  // namespace cache_dit::models
