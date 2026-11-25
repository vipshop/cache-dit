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
  ops.def("ulysses_fp6_per_token_quant", &ulysses_fp6_per_token_quant, py::arg("x"), py::arg("y"),
          py::arg("packed_head_size"), py::arg("group_size") = 32);
  ops.def("ulysses_fp6_per_token_dequant", &ulysses_fp6_per_token_dequant, py::arg("x"),
          py::arg("y"), py::arg("head_size"), py::arg("group_size"));
  ops.def("ulysses_fp6_qkv_permute_quant", &ulysses_fp6_qkv_permute_quant, py::arg("x"),
          py::arg("y"), py::arg("packed_head_size"), py::arg("group_size") = 32);
  ops.def("ulysses_fp6_qkv_permute_dequant", &ulysses_fp6_qkv_permute_dequant, py::arg("quant_x"),
          py::arg("x"), py::arg("head_size"), py::arg("group_size"));
  ops.def("ulysses_fp6_o_permute_quant", &ulysses_fp6_o_permute_quant, py::arg("x"), py::arg("y"),
          py::arg("partitions"), py::arg("packed_head_size"), py::arg("group_size") = 32);
  ops.def("ulysses_fp6_o_permute_dequant", &ulysses_fp6_o_permute_dequant, py::arg("quant_x"),
          py::arg("x"), py::arg("head_size"), py::arg("group_size"));
  ops.def("ulysses_fp4_per_token_quant", &ulysses_fp4_per_token_quant, py::arg("x"), py::arg("y"),
          py::arg("packed_head_size"), py::arg("group_size") = 16);
  ops.def("ulysses_fp4_per_token_dequant", &ulysses_fp4_per_token_dequant, py::arg("x"),
          py::arg("y"), py::arg("head_size"), py::arg("group_size"));
  ops.def("ulysses_fp4_qkv_permute_quant", &ulysses_fp4_qkv_permute_quant, py::arg("x"),
          py::arg("y"), py::arg("packed_head_size"), py::arg("group_size") = 16);
  ops.def("ulysses_fp4_qkv_permute_dequant", &ulysses_fp4_qkv_permute_dequant, py::arg("quant_x"),
          py::arg("x"), py::arg("head_size"), py::arg("group_size"));
  ops.def("ulysses_fp4_o_permute_quant", &ulysses_fp4_o_permute_quant, py::arg("x"), py::arg("y"),
          py::arg("partitions"), py::arg("packed_head_size"), py::arg("group_size") = 16);
  ops.def("ulysses_fp4_o_permute_dequant", &ulysses_fp4_o_permute_dequant, py::arg("quant_x"),
          py::arg("x"), py::arg("head_size"), py::arg("group_size"));
  ops.def("ulysses_fN_fused_qkv_quant", &ulysses_fN_fused_qkv_quant, py::arg("q"), py::arg("k"),
          py::arg("v"), py::arg("y"), py::arg("partitions"), py::arg("q_heads"), py::arg("k_heads"),
          py::arg("v_heads"), py::arg("q_width"), py::arg("k_width"), py::arg("v_width"),
          py::arg("chunk_row_bytes"), py::arg("group_size"), py::arg("bits"),
          py::arg("quantize_k"));
  ops.def("ulysses_fN_fused_qkv_dequant", &ulysses_fN_fused_qkv_dequant, py::arg("packed"),
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("q_heads"), py::arg("k_heads"),
          py::arg("v_heads"), py::arg("head_size"), py::arg("q_width"), py::arg("k_width"),
          py::arg("v_width"), py::arg("chunk_row_bytes"), py::arg("group_size"), py::arg("bits"),
          py::arg("quantize_k"));
}

}  // namespace cache_dit::ulysses
