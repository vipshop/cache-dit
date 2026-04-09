// Copied from nunchaku/nunchaku/csrc/pybind.cpp
#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ops.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def_submodule("ops")
    .def("gemm_w4a4", &svdq::ops::gemm_w4a4)
    .def("gemm_w4a4_v2",
         &svdq::ops::gemm_w4a4_v2,
         py::arg("act"),
         py::arg("wgt"),
         py::arg("out"),
         py::arg("ascales"),
         py::arg("wscales"),
         py::arg("lora_act_in"),
         py::arg("lora_up"),
         py::arg("bias"),
         py::arg("fp4"),
         py::arg("alpha"),
         py::arg("wcscales"),
         py::arg("act_unsigned"),
         py::arg("stage") = 2)
    .def("quantize_w4a4_act_fuse_lora", &svdq::ops::quantize_w4a4_act_fuse_lora)
    .def("quantize_w4a4_wgt", &svdq::ops::quantize_w4a4_wgt);

  module.def_submodule("utils")
    .def("set_log_level",
         [](const std::string &level) { spdlog::set_level(spdlog::level::from_str(level)); })
    .def("set_faster_i2f_mode", &svdq::ops::set_faster_i2f_mode);
}
