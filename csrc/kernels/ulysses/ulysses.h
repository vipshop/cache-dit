#pragma once

#include <ATen/ATen.h>

namespace pybind11 {
class module_;
}  // namespace pybind11

namespace cache_dit::ulysses {

void ulysses_fp8_per_token_quant(at::Tensor x, at::Tensor y, int64_t packed_head_size);
void ulysses_fp8_per_token_dequant(at::Tensor x, at::Tensor y, int64_t head_size);
void ulysses_fp8_qkv_permute_quant(at::Tensor x, at::Tensor y, int64_t packed_head_size);
void ulysses_fp8_qkv_permute_dequant(at::Tensor quant_x, at::Tensor x, int64_t head_size);
void ulysses_fp8_o_permute_quant(at::Tensor x, at::Tensor y, int64_t partitions,
                                 int64_t packed_head_size);
void ulysses_fp8_o_permute_dequant(at::Tensor quant_x, at::Tensor x, int64_t head_size);

// Hooks the communication operators onto the extension's `ops` submodule.
void register_ulysses_ops(pybind11::module_& ops);

}  // namespace cache_dit::ulysses
