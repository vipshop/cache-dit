#pragma once

#include <ATen/ATen.h>

#include <optional>

namespace pybind11 {
class module_;
}  // namespace pybind11

namespace cache_dit::models {

// Per-head RMSNorm followed by interleaved-pair RoPE, on the fp32 chain diffusers uses.
//
// Upstream (`Flux3AttnProcessor.__call__` + `apply_rotary_emb(use_real=True,
// use_real_unbind_dim=-1, sequence_dim=1)`) keeps everything in fp32 - `norm_q`/`norm_k` are
// listed in `_keep_in_fp32_modules`, so the weight is fp32 and the tensor is upcast once - and
// rounds to bf16 exactly once, on the final `.to(value.dtype)`. This kernel reproduces that:
// fp32 throughout, `__fmul_rn`/`__fsub_rn`/`__fadd_rn` so nvcc cannot contract the rotation,
// one `__float2bfloat16_rn` per output channel.
//
// `cos` and `sin` are the model's full-width fp32 `(S, head_dim)` tables (adjacent entries are
// duplicated by `repeat_interleave`, which the kernel simply reads).
void flux3_klein_qk_norm_rope(at::Tensor x, at::Tensor weight, at::Tensor cos, at::Tensor sin,
                              double eps, at::Tensor out);

// `out = (1 + scale) * layer_norm(x) + shift`, all in fp32, one bf16 store.
//
// `scale`/`shift` may be per token `(B, S, D)` or per sample `(B, 1, D)`: their strides are
// honoured, so a broadcast row costs one row of reads instead of a `(B, S, D)` tensor.
void flux3_klein_ln_modulate(at::Tensor x, at::Tensor scale, at::Tensor shift, double eps,
                             at::Tensor out);

// Same, for the joint block's two streams, written straight into one `[text, content]` buffer.
void flux3_klein_ln_modulate_cat(at::Tensor hidden_states, at::Tensor encoder_hidden_states,
                                 at::Tensor scale, at::Tensor shift, at::Tensor context_scale,
                                 at::Tensor context_shift, double eps, at::Tensor out);

// `out = hidden + gate * (attn + ff)`; the branch sum never reaches memory.
void flux3_klein_gated_residual(at::Tensor hidden_states, at::Tensor gate, at::Tensor attn_output,
                                at::Tensor ff_output, at::Tensor out);

// The joint block variant: rows before `prefix_len` belong to the text stream, the rest to the
// content stream, each with its own gate.
void flux3_klein_gated_residual_split(at::Tensor hidden_states, at::Tensor encoder_hidden_states,
                                      at::Tensor gate, at::Tensor context_gate,
                                      at::Tensor attn_output, at::Tensor ff_output,
                                      int64_t prefix_len, at::Tensor out_hidden,
                                      at::Tensor out_encoder);

// Hooks the FLUX.3 Klein operators onto the extension's `ops` submodule.
void register_flux3_klein_ops(pybind11::module_& ops);

}  // namespace cache_dit::models
