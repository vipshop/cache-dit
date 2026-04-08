// Copied from nunchaku/nunchaku/csrc/pybind.cpp
#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ops.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def_submodule("ops")
    .def("gemm_w4a4", &svdq::ops::gemm_w4a4)
    .def("quantize_w4a4_act_fuse_lora", &svdq::ops::quantize_w4a4_act_fuse_lora)
    .def("quantize_w4a4_wgt", &svdq::ops::quantize_w4a4_wgt);

  module.def_submodule("utils")
    .def("set_log_level",
         [](const std::string &level) { spdlog::set_level(spdlog::level::from_str(level)); })
    .def("set_faster_i2f_mode", &svdq::ops::set_faster_i2f_mode);
}
