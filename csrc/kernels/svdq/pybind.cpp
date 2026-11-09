// Copied from nunchaku/nunchaku/csrc/pybind.cpp
#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ops.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def_submodule("ops")
    .def("gemm_w4a4", &svdq::ops::gemm_w4a4, py::arg("act"), py::arg("wgt"), py::arg("out"),
         py::arg("qout"), py::arg("ascales"), py::arg("wscales"), py::arg("oscales"),
         py::arg("poolout"), py::arg("lora_act_in"), py::arg("lora_up"), py::arg("lora_down"),
         py::arg("lora_act_out"), py::arg("norm_q"), py::arg("norm_k"), py::arg("rotary_emb"),
         py::arg("bias"), py::arg("smooth_factor"), py::arg("out_vk"), py::arg("out_linearattn"),
         py::arg("act_unsigned"), py::arg("lora_scales"), py::arg("fuse_silu"), py::arg("fp4"),
         py::arg("alpha"), py::arg("wcscales"), py::arg("out_q"), py::arg("out_k"),
         py::arg("out_v"), py::arg("attn_tokens"), py::arg("runtime"),
         py::arg("lora_layout") = "native", py::arg("stage") = 1)
    .def("quantize_w4a4", &svdq::ops::quantize_w4a4, py::arg("input"), py::arg("output"),
         py::arg("ascales"), py::arg("lora_down"), py::arg("lora_act"), py::arg("smooth"),
         py::arg("fuse_glu"), py::arg("fp4"), py::arg("lora_scales"),
         py::arg("lora_layout") = "native");

  module.def_submodule("utils").def("set_log_level", [](const std::string &level) {
    spdlog::set_level(spdlog::level::from_str(level));
  });
}
