#include <torch/extension.h>

#include "comm_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  auto ops = m.def_submodule("ops");
  ops.def("fp8_comm_per_token_quant", &cache_dit::comm::fp8_comm_per_token_quant);
  ops.def("fp8_comm_per_token_dequant", &cache_dit::comm::fp8_comm_per_token_dequant);
  ops.def("fp8_comm_qkv_permute_quant", &cache_dit::comm::fp8_comm_qkv_permute_quant);
  ops.def("fp8_comm_qkv_permute_dequant", &cache_dit::comm::fp8_comm_qkv_permute_dequant);
}
