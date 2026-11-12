// Entry point of the single `cache_dit._C_ops` extension.
//
// A pybind module can only be defined once, so every kernel family exports a
// `register_<family>_ops(py::module_&)` hook and this TU hooks all of them onto the shared
// `ops` submodule. Adding a family means adding its registrar here plus its sources to
// `_required_cuda_sources()` in setup.py, not another extension.
//
// Quoted includes carry the `<family>/` prefix so the family header is unambiguous from this
// directory.

#include <torch/extension.h>

#include "attn/attn.h"
#include "fp8/fp8.h"
#include "ulysses/ulysses.h"
#include "models/models.h"
#include "svdq/svdq.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  auto ops = module.def_submodule("ops");
  cache_dit::attn::register_attn_ops(ops);
  cache_dit::fp8::register_fp8_ops(ops);
  cache_dit::ulysses::register_ulysses_ops(ops);
  cache_dit::models::register_models_ops(ops);
  cache_dit::svdq::register_svdq_ops(ops);
}
