#pragma once

#include <ATen/ATen.h>

namespace cache_dit::attn {

void fused_merge_attn_states(
  at::Tensor prev_out,
  at::Tensor prev_lse,
  at::Tensor suff_out,
  at::Tensor suff_lse,
  at::Tensor out,
  at::Tensor lse);

} // namespace cache_dit::attn
