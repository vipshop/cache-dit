#pragma once

#include <ATen/ATen.h>

#include <optional>

namespace pybind11 {
class module_;
}  // namespace pybind11

namespace cache_dit::fp8 {

// Scale granularity shared by the quantizer and GEMM entry points.
constexpr int64_t kPerRow = 0;
constexpr int64_t kPerTensor = 1;
constexpr int64_t kPerBlock = 2;

void fp8_quantize(const at::Tensor& input, at::Tensor& output, at::Tensor& scale, int64_t mode,
                  bool is_weight);
void fp8_gemm(const at::Tensor& a, const at::Tensor& b, const at::Tensor& a_scale,
              const at::Tensor& b_scale, const std::optional<at::Tensor>& bias, at::Tensor& out,
              int64_t mode, bool use_ws);

// Hooks the FP8 quantize/GEMM operators onto the extension's `ops` submodule.
void register_fp8_ops(pybind11::module_& ops);

}  // namespace cache_dit::fp8
