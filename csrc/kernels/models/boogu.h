#pragma once

#include <ATen/ATen.h>

#include <optional>

namespace pybind11 {
class module_;
}  // namespace pybind11

namespace cache_dit::models {

// y = rms_norm(x) * weight * (1 + scale), one bf16 rounding at the store.
// `scale` is a per-channel [N] vector (the call sites broadcast a [B, N]
// modulation term with B == 1). `zero_centered_weight` applies the
// (weight + 1) convention used by zero-centered RMSNorm variants.
void boogu_scaled_rms_norm(at::Tensor x, at::Tensor weight, double eps,
                           const std::optional<at::Tensor>& scale, bool zero_centered_weight,
                           at::Tensor out);

// y = residual + tanh(gate) * rms_norm(x); `gate` is a per-channel [N] vector.
void boogu_gated_rms_norm(at::Tensor x, at::Tensor weight, double eps, at::Tensor gate,
                          at::Tensor residual, bool zero_centered_weight, at::Tensor out);

// Per-head RMSNorm followed by interleaved-pair complex RoPE, fused so the
// rotated value is rounded to bf16 exactly once.
void boogu_fused_qk_norm_rope(at::Tensor x, at::Tensor freqs_cis, at::Tensor weight, double eps,
                              at::Tensor out);

// Hooks the three Boogu operators onto the extension's `ops` submodule.
void register_boogu_ops(pybind11::module_& ops);

}  // namespace cache_dit::models
