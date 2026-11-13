#include <torch/extension.h>

#include "fp4.h"

namespace cache_dit::fp4 {

void register_fp4_ops(pybind11::module_& ops) {
  ops.def("fp4_quantize", &fp4_quantize);
  ops.def("fp4_gemm", &fp4_gemm);
}

}  // namespace cache_dit::fp4
