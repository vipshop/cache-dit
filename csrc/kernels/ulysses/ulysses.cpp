#include <torch/extension.h>

#include "ulysses.h"

namespace cache_dit::ulysses {

void register_ulysses_ops(pybind11::module_& ops) {
  ops.def("ulysses_fp8_per_token_quant", &ulysses_fp8_per_token_quant, py::arg("x"), py::arg("y"),
          py::arg("packed_head_size"), py::arg("group_size") = 0);
  ops.def("ulysses_fp8_per_token_dequant", &ulysses_fp8_per_token_dequant, py::arg("x"),
          py::arg("y"), py::arg("head_size"), py::arg("group_size") = 0);
  ops.def("ulysses_fp8_qkv_permute_quant", &ulysses_fp8_qkv_permute_quant, py::arg("x"),
          py::arg("y"), py::arg("packed_head_size"), py::arg("group_size") = 0);
  ops.def("ulysses_fp8_qkv_permute_dequant", &ulysses_fp8_qkv_permute_dequant, py::arg("quant_x"),
          py::arg("x"), py::arg("head_size"), py::arg("group_size") = 0);
  ops.def("ulysses_fp8_o_permute_quant", &ulysses_fp8_o_permute_quant, py::arg("x"), py::arg("y"),
          py::arg("partitions"), py::arg("packed_head_size"), py::arg("group_size") = 0);
  ops.def("ulysses_fp8_o_permute_dequant", &ulysses_fp8_o_permute_dequant, py::arg("quant_x"),
          py::arg("x"), py::arg("head_size"), py::arg("group_size") = 0);
}

}  // namespace cache_dit::ulysses
