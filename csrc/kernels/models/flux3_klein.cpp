#include <torch/extension.h>

#include "flux3_klein.h"

namespace cache_dit::models {

void register_flux3_klein_ops(pybind11::module_& ops) {
  ops.def("flux3_klein_qk_norm_rope", &flux3_klein_qk_norm_rope);
  ops.def("flux3_klein_ln_modulate", &flux3_klein_ln_modulate);
  ops.def("flux3_klein_ln_modulate_cat", &flux3_klein_ln_modulate_cat);
  ops.def("flux3_klein_gated_residual", &flux3_klein_gated_residual);
  ops.def("flux3_klein_gated_residual_split", &flux3_klein_gated_residual_split);
}

}  // namespace cache_dit::models
