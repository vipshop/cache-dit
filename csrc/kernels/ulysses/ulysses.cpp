#include <torch/extension.h>

#include "ulysses.h"

namespace cache_dit::ulysses {

void register_ulysses_ops(pybind11::module_& ops) {
  ops.def("ulysses_fp8_per_token_quant", &ulysses_fp8_per_token_quant);
  ops.def("ulysses_fp8_per_token_dequant", &ulysses_fp8_per_token_dequant);
  ops.def("ulysses_fp8_qkv_permute_quant", &ulysses_fp8_qkv_permute_quant);
  ops.def("ulysses_fp8_qkv_permute_dequant", &ulysses_fp8_qkv_permute_dequant);
  ops.def("ulysses_fp8_o_permute_quant", &ulysses_fp8_o_permute_quant);
  ops.def("ulysses_fp8_o_permute_dequant", &ulysses_fp8_o_permute_dequant);
}

}  // namespace cache_dit::ulysses
