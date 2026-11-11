#pragma once

#include <ATen/ATen.h>

namespace pybind11 {
class module_;
}  // namespace pybind11

namespace cache_dit::comm {

void fp8_comm_per_token_quant(at::Tensor x, at::Tensor y);
void fp8_comm_per_token_dequant(at::Tensor x, at::Tensor y);
void fp8_comm_qkv_permute_quant(at::Tensor x, at::Tensor y);
void fp8_comm_qkv_permute_dequant(at::Tensor quant_x, at::Tensor x);

// Hooks the communication operators onto the extension's `ops` submodule.
void register_comm_ops(pybind11::module_& ops);

}  // namespace cache_dit::comm
