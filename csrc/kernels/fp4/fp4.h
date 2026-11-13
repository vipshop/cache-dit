#pragma once

#include <ATen/ATen.h>

#include <optional>

namespace pybind11 {
class module_;
}  // namespace pybind11

namespace cache_dit::fp4 {

void fp4_quantize(const at::Tensor& input, at::Tensor& qdata, at::Tensor& block_scales,
                  at::Tensor& global_scales, bool level1);
void fp4_gemm(const at::Tensor& a, const at::Tensor& b, const at::Tensor& a_scales,
              const at::Tensor& b_scales, const at::Tensor& a_global_scales,
              const at::Tensor& b_global_scales, const std::optional<at::Tensor>& bias,
              at::Tensor& out, bool use_ws);
void register_fp4_ops(pybind11::module_& ops);

}  // namespace cache_dit::fp4
