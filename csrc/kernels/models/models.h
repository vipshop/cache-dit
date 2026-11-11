#pragma once

namespace pybind11 {
class module_;
}  // namespace pybind11

namespace cache_dit::models {

// Hooks every model-specific operator family onto the extension's `ops` submodule.
// Each model contributes its own `register_<model>_ops(pybind11::module_&)`; this
// family-level registrar is the only name the extension entry point knows about.
void register_models_ops(pybind11::module_& ops);

}  // namespace cache_dit::models
