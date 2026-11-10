#include <torch/extension.h>

#include "qwen_image21.h"

namespace cache_dit::models {

void register_qwen_image21_ops(pybind11::module_& ops) {
  ops.def("qwen_image21_qk_norm_rope", &qwen_image21_qk_norm_rope);
  ops.def("qwen_image21_ln_modulate", &qwen_image21_ln_modulate);
  ops.def("qwen_image21_modulate_affine", &qwen_image21_modulate_affine);
  ops.def("qwen_image21_gated_residual", &qwen_image21_gated_residual);
  ops.def("qwen_image21_silu_mul", &qwen_image21_silu_mul);
}

}  // namespace cache_dit::models
