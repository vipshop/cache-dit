#pragma once

#include <ATen/ATen.h>

namespace cache_dit::comm {

void fp8_comm_per_token_quant(at::Tensor x, at::Tensor y);
void fp8_comm_per_token_dequant(at::Tensor x, at::Tensor y);
void fp8_comm_qkv_permute_quant(at::Tensor x, at::Tensor y);
void fp8_comm_qkv_permute_dequant(at::Tensor quant_x, at::Tensor x);

} // namespace cache_dit::comm
