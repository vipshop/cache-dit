#pragma once

#include <ATen/ATen.h>

namespace pybind11 {
class module_;
}  // namespace pybind11

namespace cache_dit::attn {

void fused_merge_attn_states(at::Tensor prev_out, at::Tensor prev_lse, at::Tensor suff_out,
                             at::Tensor suff_lse, at::Tensor out, at::Tensor lse);

// Hooks the attention-state merge operator onto the extension's `ops` submodule.
void register_attn_ops(pybind11::module_& ops);

}  // namespace cache_dit::attn
