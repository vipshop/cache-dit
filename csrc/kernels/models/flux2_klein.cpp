#include <torch/extension.h>

#include "flux2_klein.h"

namespace cache_dit::models {

void register_flux2_klein_ops(pybind11::module_& ops) {
  ops.def("flux2_klein_qk_norm_rope", &flux2_klein_qk_norm_rope);
  ops.def("flux2_klein_qk_norm_rope_join", &flux2_klein_qk_norm_rope_join);
  ops.def("flux2_klein_ln_modulate", &flux2_klein_ln_modulate);
  ops.def("flux2_klein_residual_ln_modulate", &flux2_klein_residual_ln_modulate);
  ops.def("flux2_klein_gated_residual", &flux2_klein_gated_residual);
  ops.def("flux2_klein_silu_mul", &flux2_klein_silu_mul);
}

}  // namespace cache_dit::models
