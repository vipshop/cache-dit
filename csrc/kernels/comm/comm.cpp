#include <torch/extension.h>

#include "comm.h"

namespace cache_dit::comm {

void register_comm_ops(pybind11::module_& ops) {
  ops.def("fp8_comm_per_token_quant", &fp8_comm_per_token_quant);
  ops.def("fp8_comm_per_token_dequant", &fp8_comm_per_token_dequant);
  ops.def("fp8_comm_qkv_permute_quant", &fp8_comm_qkv_permute_quant);
  ops.def("fp8_comm_qkv_permute_dequant", &fp8_comm_qkv_permute_dequant);
}

}  // namespace cache_dit::comm
