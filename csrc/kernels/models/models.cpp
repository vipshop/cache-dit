// Registrar of the model-specific operator family inside `cache_dit._C_ops`.
//
// The extension itself is defined once in `csrc/kernels/ops.cpp`. This TU only delegates
// to one registrar per model, so adding a model means adding its `<model>.cu` kernels plus
// a `<model>.cpp` registrar and calling it here.

#include <torch/extension.h>

#include "models.h"
#include "boogu.h"
#include "flux2_klein.h"
#include "flux3_klein.h"
#include "qwen_image21.h"

namespace cache_dit::models {

void register_models_ops(pybind11::module_& ops) {
  register_boogu_ops(ops);
  register_qwen_image21_ops(ops);
  register_flux2_klein_ops(ops);
  register_flux3_klein_ops(ops);
}

}  // namespace cache_dit::models
