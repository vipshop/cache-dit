// Entry point of the model-specific operator family (`cache_dit._C_models`).
//
// A pybind module can only be defined once, so this TU owns the extension and
// delegates to one registrar per model. Adding a model means adding its
// `<model>.cu` kernels plus a `<model>.cpp` registrar and calling it here.

#include <torch/extension.h>

#include "boogu.h"
#include "qwen_image21.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  auto ops = m.def_submodule("ops");
  cache_dit::models::register_boogu_ops(ops);
  cache_dit::models::register_qwen_image21_ops(ops);
}
