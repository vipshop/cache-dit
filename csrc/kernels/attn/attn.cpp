#include <torch/extension.h>

#include "attn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  auto ops = m.def_submodule("ops");
  ops.def("fused_merge_attn_states", &cache_dit::attn::fused_merge_attn_states);
}
