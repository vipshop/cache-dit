#include <torch/extension.h>

#include "boogu.h"

namespace cache_dit::models {

void register_boogu_ops(pybind11::module_& ops) {
  ops.def("boogu_scaled_rms_norm", &boogu_scaled_rms_norm);
  ops.def("boogu_gated_rms_norm", &boogu_gated_rms_norm);
  ops.def("boogu_fused_qk_norm_rope", &boogu_fused_qk_norm_rope);
}

}  // namespace cache_dit::models
