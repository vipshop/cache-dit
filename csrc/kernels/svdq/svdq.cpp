// Adapted from nunchaku/nunchaku/csrc/pybind.cpp
//
// Definitions of the two public SVDQ dispatchers plus their registrar. The extension itself is
// defined in `csrc/kernels/ops.cpp`, so this TU must not define a `PYBIND11_MODULE` of its own.
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "svdq.h"
#include "runtime.h"
#include "native/zgemm.h"
#include "cute/fp4/gemm/gemm_w4a4.h"
#include "cute/fp4/quantize/quantize.h"

namespace py = pybind11;

namespace cache_dit::svdq {

// The host-side routing types still live in the global `svdq` namespace next to the kernel
// implementations; pull them in so this file reads like the other `cache_dit::<family>` layers.
using ::svdq::LoraLayout;
using ::svdq::RuntimeKernel;
using ::svdq::RuntimeParams;

namespace {

Tensor get_tensor(std::optional<torch::Tensor> &tensor) {
  Tensor result = tensor.has_value() ? from_torch(tensor.value()) : Tensor{};
  if (result.valid()) {
    spdlog::trace("  {}", result.shape.str());
  } else {
    spdlog::trace("  <invalid>");
  }
  return result;
}

// CuTe-backed LoRA-down layout selector. Native runtime is always kNative.
LoraLayout parse_lora_layout(const std::string &lora_layout) {
  if (lora_layout == "native") {
    return LoraLayout::kNative;
  }
  if (lora_layout == "row_major") {
    return LoraLayout::kRowMajor;
  }
  throw std::runtime_error("lora_layout must be 'native' or 'row_major', got '" + lora_layout +
                           "'.");
}

RuntimeParams make_runtime_params(const std::string &runtime, bool fp4, float alpha,
                                  bool act_unsigned) {
  RuntimeParams params;
  if (runtime == "native") {
    params.kernel = RuntimeKernel::kNative;
  } else if (runtime == "cute") {
    params.kernel = RuntimeKernel::kCuTe;
  } else {
    throw std::runtime_error("runtime must be 'native' or 'cute', got '" + runtime + "'.");
  }
  params.fp4 = fp4;
  params.alpha = alpha;
  params.act_unsigned = act_unsigned;
  return params;
}

}  // namespace

void gemm_w4a4(std::optional<torch::Tensor> act, std::optional<torch::Tensor> wgt,
               std::optional<torch::Tensor> out, std::optional<torch::Tensor> qout,
               std::optional<torch::Tensor> ascales, std::optional<torch::Tensor> wscales,
               std::optional<torch::Tensor> oscales, std::optional<torch::Tensor> lora_act_in,
               std::optional<torch::Tensor> lora_up, std::optional<torch::Tensor> lora_down,
               std::optional<torch::Tensor> lora_act_out, std::optional<torch::Tensor> bias,
               std::optional<torch::Tensor> smooth_factor, bool act_unsigned,
               std::vector<float> lora_scales, bool fuse_silu, bool fp4, float alpha,
               std::optional<torch::Tensor> wcscales, const std::string &runtime,
               const std::string &lora_layout) {
  const RuntimeParams params = make_runtime_params(runtime, fp4, alpha, act_unsigned);
  const auto layout = parse_lora_layout(lora_layout);
  TorchOpContext ctx;

  if (params.kernel == RuntimeKernel::kCuTe) {
    if (!params.fp4) {
      throw std::runtime_error("runtime='cute' implements the NVFP4 path only; pass fp4=True.");
    }
    if (qout.has_value() || oscales.has_value() || lora_down.has_value() ||
        lora_act_out.has_value() || smooth_factor.has_value() || fuse_silu) {
      throw std::runtime_error(
        "runtime='cute' does not accept the fused native arguments "
        "(qout/oscales/lora_down/lora_act_out/smooth_factor/fuse_silu).");
    }
    // `::svdq::` is required: inside `cache_dit::svdq` the leftmost `svdq` would otherwise
    // resolve to this namespace, which does not contain the kernel namespaces.
    ::svdq::cute_kernels::gemm_w4a4(
      get_tensor(act), get_tensor(wgt), get_tensor(out), get_tensor(ascales), get_tensor(wscales),
      get_tensor(lora_act_in), get_tensor(lora_up), get_tensor(bias), get_tensor(wcscales),
      std::move(lora_scales), params.alpha, layout);
    return;
  }

  if (layout != LoraLayout::kNative) {
    throw std::runtime_error("runtime='native' only supports lora_layout='native'.");
  }
  // Fully qualified on purpose: this function shares its name with the native entry point.
  ::svdq::native_kernels::gemm_w4a4(
    get_tensor(act), get_tensor(wgt), get_tensor(out), get_tensor(qout), get_tensor(ascales),
    get_tensor(wscales), get_tensor(oscales), get_tensor(lora_act_in), get_tensor(lora_up),
    get_tensor(lora_down), get_tensor(lora_act_out), get_tensor(bias), get_tensor(smooth_factor),
    params.act_unsigned, std::move(lora_scales), fuse_silu, params.fp4, params.alpha,
    get_tensor(wcscales));
}

void quantize_w4a4(std::optional<torch::Tensor> input, std::optional<torch::Tensor> output,
                   std::optional<torch::Tensor> ascales, std::optional<torch::Tensor> lora_down,
                   std::optional<torch::Tensor> lora_act, std::optional<torch::Tensor> smooth,
                   bool fuse_glu, bool fp4, std::vector<float> lora_scales,
                   const std::string &lora_layout) {
  const auto layout = parse_lora_layout(lora_layout);
  TorchOpContext ctx;
  if (layout == LoraLayout::kRowMajor) {
    if (!fp4) {
      throw std::runtime_error("lora_layout='row_major' implements the NVFP4 producer only.");
    }
    const int64_t rank = lora_down && lora_down->dim() > 0 ? lora_down->size(-1) : 0;
    const int64_t K = input && input->dim() > 0 ? input->size(-1) : 0;
    if (input && input->dim() > 0 && K <= 0) {
      throw std::runtime_error("quantize_w4a4 requires a non-empty input channel dimension.");
    }
    const int64_t M = K > 0 ? input->numel() / K : 0;
    torch::Tensor lora_scratch_t;
    if (input && lora_down && input->dim() > 0 && lora_down->dim() > 0) {
      // Small rank64 producers use the fused schedule to avoid scratch/finalize overhead and
      // launch more CTAs. Long-K keeps split-K for its additional K-dimension parallelism.
      const bool use_small_fused = rank == 64 && M <= 512 && K <= 3072;
      const bool use_split_k = ::svdq::cute_kernels::is_specialized_rank(rank) &&
                               (rank != 32 || M <= 8192) && !use_small_fused;
      if (use_split_k) {
        lora_scratch_t = at::empty({M, rank}, input->options().dtype(at::kFloat));
      }
    }
    ::Tensor lora_scratch;
    if (lora_scratch_t.defined()) {
      lora_scratch = from_torch(lora_scratch_t);
    }
    auto quantize = &::svdq::cute_kernels::quantize_uni;
    if (rank == 32 && K > 3072) {
      // Keep the universal producer for rank32/K3072, where it remains faster at 2K/4K/8K.
      quantize = &::svdq::cute_kernels::quantize_r32;
    } else if (rank == 64) {
      quantize = &::svdq::cute_kernels::quantize_r64;
    } else if (rank == 128) {
      quantize = &::svdq::cute_kernels::quantize_r128;
    }
    quantize(get_tensor(input), get_tensor(output), get_tensor(ascales), get_tensor(lora_down),
             get_tensor(lora_act), get_tensor(smooth), std::move(lora_scales), lora_scratch);
    return;
  }
  ::svdq::native_kernels::quantize_w4a4(get_tensor(input), get_tensor(output), get_tensor(ascales),
                                        get_tensor(lora_down), get_tensor(lora_act),
                                        get_tensor(smooth), fuse_glu, fp4);
}

void register_svdq_ops(py::module_ &ops) {
  ops
    .def("gemm_w4a4", &gemm_w4a4, py::arg("act"), py::arg("wgt"), py::arg("out"), py::arg("qout"),
         py::arg("ascales"), py::arg("wscales"), py::arg("oscales"), py::arg("lora_act_in"),
         py::arg("lora_up"), py::arg("lora_down"), py::arg("lora_act_out"), py::arg("bias"),
         py::arg("smooth_factor"), py::arg("act_unsigned"), py::arg("lora_scales"),
         py::arg("fuse_silu"), py::arg("fp4"), py::arg("alpha"), py::arg("wcscales"),
         py::arg("runtime"), py::arg("lora_layout") = "native")
    .def("quantize_w4a4", &quantize_w4a4, py::arg("input"), py::arg("output"), py::arg("ascales"),
         py::arg("lora_down"), py::arg("lora_act"), py::arg("smooth"), py::arg("fuse_glu"),
         py::arg("fp4"), py::arg("lora_scales"), py::arg("lora_layout") = "native");
}

}  // namespace cache_dit::svdq
