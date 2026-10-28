#pragma once

#include <ATen/ATen.h>

namespace cache_dit::comm {

void fp8_comm_per_token_quant(at::Tensor x, at::Tensor y);
void fp8_comm_per_token_dequant(at::Tensor x, at::Tensor y);
void fp8_comm_qkv_permute_quant(at::Tensor x, at::Tensor y);
void fp8_comm_qkv_permute_dequant(at::Tensor quant_x, at::Tensor x);
void fused_merge_attn_states(
  at::Tensor prev_out,
  at::Tensor prev_lse,
  at::Tensor suff_out,
  at::Tensor suff_lse,
  at::Tensor out,
  at::Tensor lse);

} // namespace cache_dit::comm
