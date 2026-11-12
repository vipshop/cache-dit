#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

#include "fp8.h"
#include "gemm.h"

namespace cache_dit::fp8 {
namespace {

void check_runtime_device(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_cuda(), "FP8 operators require CUDA tensors.");
  TORCH_CHECK(tensor.get_device() >= 0, "FP8 operators require a concrete CUDA device.");
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(tensor.get_device());
  TORCH_CHECK(properties->major >= 10,
              "cache-dit FP8 kernels require sm100 or newer; this device is sm", properties->major,
              properties->minor, ". Use --quantize-backend torchao on Ada or Hopper.");
}

void check_gemm_inputs(const at::Tensor& a, const at::Tensor& b, const at::Tensor& a_scale,
                       const at::Tensor& b_scale, const std::optional<at::Tensor>& bias,
                       const at::Tensor& out, int64_t mode) {
  check_runtime_device(a);
  TORCH_CHECK(a.scalar_type() == at::kFloat8_e4m3fn && b.scalar_type() == at::kFloat8_e4m3fn,
              "fp8_gemm expects float8_e4m3fn A and B.");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && out.dim() == 2,
              "fp8_gemm expects 2D A, B and output.");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && out.is_contiguous(),
              "fp8_gemm expects contiguous A, B and output.");
  TORCH_CHECK(a.device() == b.device() && a.device() == out.device() &&
                a.device() == a_scale.device() && a.device() == b_scale.device(),
              "fp8_gemm expects every tensor on one device.");
  TORCH_CHECK(a_scale.scalar_type() == at::kFloat && b_scale.scalar_type() == at::kFloat,
              "fp8_gemm expects float32 scales.");
  TORCH_CHECK(a_scale.is_contiguous() && b_scale.is_contiguous(),
              "fp8_gemm expects contiguous scales.");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "fp8_gemm output must be bfloat16.");

  const int64_t M = a.size(0);
  const int64_t K = a.size(1);
  const int64_t N = b.size(0);
  TORCH_CHECK(b.size(1) == K, "fp8_gemm inner dimensions do not match.");
  TORCH_CHECK(out.size(0) == M && out.size(1) == N, "fp8_gemm output shape must be [M, N].");
  TORCH_CHECK(M > 0 && N > 0 && K > 0, "fp8_gemm does not accept empty tensors.");
  TORCH_CHECK(K % 16 == 0, "fp8_gemm requires K to be a multiple of 16.");
  TORCH_CHECK(N % 8 == 0, "fp8_gemm requires N to be a multiple of 8.");
  TORCH_CHECK(M <= INT32_MAX && N <= INT32_MAX && K <= INT32_MAX,
              "fp8_gemm dimensions must fit in int32.");
  const auto require_aligned = [](const void* pointer, const char* name) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(pointer) % 16 == 0,
                "fp8_gemm requires 16-byte aligned ", name, ".");
  };
  require_aligned(a.data_ptr(), "A");
  require_aligned(b.data_ptr(), "B");
  require_aligned(out.data_ptr(), "output");
  if (bias.has_value()) {
    require_aligned(bias->data_ptr(), "bias");
  }

  const int64_t num_k_blocks = (K + 128 - 1) / 128;
  if (mode == kPerRow) {
    TORCH_CHECK(a_scale.numel() == M && b_scale.numel() == N,
                "per-row fp8_gemm expects A scale [M] and B scale [N].");
  } else if (mode == kPerTensor) {
    TORCH_CHECK(a_scale.numel() == 1 && b_scale.numel() == 1,
                "per-tensor fp8_gemm expects one A scale and one B scale.");
  } else if (mode == kPerBlock) {
    TORCH_CHECK(K % 128 == 0, "per-block fp8_gemm requires K to be a multiple of 128.");
    TORCH_CHECK(a_scale.numel() == M * num_k_blocks,
                "per-block fp8_gemm expects A scale [M, K / 128].");
    TORCH_CHECK(b_scale.numel() == ((N + 127) / 128) * num_k_blocks,
                "per-block fp8_gemm expects B scale [ceil(N / 128), K / 128].");
  } else {
    TORCH_CHECK(false, "fp8_gemm mode must be 0 (per_row), 1 (per_tensor), or 2 (per_block).");
  }

  if (bias.has_value()) {
    const at::Tensor& value = bias.value();
    TORCH_CHECK(value.is_cuda() && value.device() == a.device(),
                "fp8_gemm bias must be on the input device.");
    TORCH_CHECK(value.scalar_type() == at::kBFloat16, "fp8_gemm bias must be bfloat16.");
    TORCH_CHECK(value.is_contiguous(), "fp8_gemm bias must be contiguous.");
    TORCH_CHECK(value.dim() == 1 && value.size(0) == N, "fp8_gemm bias must have shape [N].");
  }
}

}  // namespace

void fp8_gemm(const at::Tensor& a, const at::Tensor& b, const at::Tensor& a_scale,
              const at::Tensor& b_scale, const std::optional<at::Tensor>& bias, at::Tensor& out,
              int64_t mode, bool use_ws) {
  const c10::cuda::CUDAGuard device_guard(a.device());
  check_gemm_inputs(a, b, a_scale, b_scale, bias, out, mode);
  const int M = static_cast<int>(a.size(0));
  const int N = static_cast<int>(b.size(0));
  const int K = static_cast<int>(a.size(1));
  const int num_k_blocks = (K + 127) / 128;
  const void* a_ptr = a.data_ptr<at::Float8_e4m3fn>();
  const void* b_ptr = b.data_ptr<at::Float8_e4m3fn>();
  void* out_ptr = out.data_ptr<at::BFloat16>();
  const void* bias_ptr = bias.has_value() ? bias->data_ptr<at::BFloat16>() : nullptr;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (mode == kPerBlock) {
    launch_fp8_gemm_per_block(a_ptr, b_ptr, a_scale.data_ptr<float>(), b_scale.data_ptr<float>(),
                              bias_ptr, out_ptr, M, N, K, num_k_blocks, use_ws, stream);
  } else if (mode == kPerTensor) {
    launch_fp8_gemm_per_tensor(a_ptr, b_ptr, a_scale.data_ptr<float>(), b_scale.data_ptr<float>(),
                               bias_ptr, out_ptr, M, N, K, use_ws, stream);
  } else {
    launch_fp8_gemm_per_row(a_ptr, b_ptr, a_scale.data_ptr<float>(), b_scale.data_ptr<float>(),
                            bias_ptr, out_ptr, M, N, K, use_ws, stream);
  }
}

void register_fp8_ops(pybind11::module_& ops) {
  ops.def("fp8_quantize", &fp8_quantize);
  ops.def("fp8_gemm", &fp8_gemm);
}

}  // namespace cache_dit::fp8
