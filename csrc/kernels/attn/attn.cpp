#include <torch/extension.h>

#include "attn.h"

namespace cache_dit::attn {

void register_attn_ops(pybind11::module_& ops) {
  ops.def("fused_merge_attn_states", &fused_merge_attn_states);
}

}  // namespace cache_dit::attn
