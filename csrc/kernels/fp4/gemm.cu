#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

#include "fp4.h"
#include "gemm.cuh"
#include "collective_gemm.cuh"

namespace cache_dit::fp4 {
namespace {

void check_runtime_device(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_cuda(), "FP4 operators require CUDA tensors.");
  TORCH_CHECK(tensor.get_device() >= 0, "FP4 operators require a concrete CUDA device.");
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(tensor.get_device());
  TORCH_CHECK(properties->major == 12,
              "cache-dit native NVFP4 kernels require sm120; this device is sm", properties->major,
              properties->minor, ". Use --quantize-backend torchao on this device.");
}

void check_gemm_inputs(const at::Tensor& a, const at::Tensor& b, const at::Tensor& a_scales,
                       const at::Tensor& b_scales, const at::Tensor& a_global_scales,
                       const at::Tensor& b_global_scales, const std::optional<at::Tensor>& bias,
                       const at::Tensor& out) {
  check_runtime_device(a);
  TORCH_CHECK(a.scalar_type() == at::kFloat4_e2m1fn_x2 && b.scalar_type() == at::kFloat4_e2m1fn_x2,
              "fp4_gemm expects float4_e2m1fn_x2 A and B.");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && out.dim() == 2,
              "fp4_gemm expects 2D A, B and output.");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && out.is_contiguous(),
              "fp4_gemm expects contiguous A, B and output.");
  TORCH_CHECK(a.device() == b.device() && a.device() == out.device() &&
                a.device() == a_scales.device() && a.device() == b_scales.device() &&
                a.device() == a_global_scales.device() && a.device() == b_global_scales.device(),
              "fp4_gemm expects every tensor on one device.");
  TORCH_CHECK(
    a_scales.scalar_type() == at::kFloat8_e4m3fn && b_scales.scalar_type() == at::kFloat8_e4m3fn,
    "fp4_gemm block scales must be float8_e4m3fn.");
  TORCH_CHECK(
    a_global_scales.scalar_type() == at::kFloat && b_global_scales.scalar_type() == at::kFloat,
    "fp4_gemm global scales must be float32.");
  TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous() &&
                a_global_scales.is_contiguous() && b_global_scales.is_contiguous(),
              "fp4_gemm expects contiguous scales.");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "fp4_gemm output must be bfloat16.");

  const int64_t M = a.size(0);
  const int64_t N = b.size(0);
  const int64_t K = a.size(1) * 2;
  TORCH_CHECK(b.size(1) * 2 == K, "fp4_gemm inner dimensions do not match.");
  TORCH_CHECK(out.size(0) == M && out.size(1) == N, "fp4_gemm output shape must be [M, N].");
  TORCH_CHECK(M > 0 && N > 0 && K > 0, "fp4_gemm does not accept empty operands.");
  TORCH_CHECK(K % 64 == 0, "fp4_gemm requires logical K to be a multiple of 64.");
  TORCH_CHECK(N % 8 == 0, "fp4_gemm requires N to be a multiple of 8.");
  TORCH_CHECK(M <= INT32_MAX && N <= INT32_MAX && K <= INT32_MAX,
              "fp4_gemm dimensions must fit in int32.");
  TORCH_CHECK(a_scales.numel() == static_cast<int64_t>(sf_buffer_bytes(M, K)) &&
                b_scales.numel() == static_cast<int64_t>(sf_buffer_bytes(N, K)),
              "fp4_gemm block scale buffers do not match the operand shapes.");
  TORCH_CHECK(a_global_scales.dim() == 1 && b_global_scales.dim() == 1,
              "fp4_gemm global scales must be 1D.");
  TORCH_CHECK(a_global_scales.numel() == 0 || a_global_scales.numel() == M,
              "fp4_gemm A global scales must be empty or [M].");
  TORCH_CHECK(b_global_scales.numel() == 0 || b_global_scales.numel() == N,
              "fp4_gemm B global scales must be empty or [N].");

  const auto require_aligned = [](const void* pointer, const char* name) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(pointer) % 16 == 0,
                "fp4_gemm requires 16-byte aligned ", name, ".");
  };
  require_aligned(a.data_ptr(), "A");
  require_aligned(b.data_ptr(), "B");
  require_aligned(a_scales.data_ptr(), "A block scales");
  require_aligned(b_scales.data_ptr(), "B block scales");
  require_aligned(out.data_ptr(), "output");
  if (bias.has_value()) {
    const at::Tensor& value = bias.value();
    TORCH_CHECK(value.is_cuda() && value.device() == a.device(),
                "fp4_gemm bias must be on the input device.");
    TORCH_CHECK(value.scalar_type() == at::kBFloat16, "fp4_gemm bias must be bfloat16.");
    TORCH_CHECK(value.is_contiguous() && value.dim() == 1 && value.size(0) == N,
                "fp4_gemm bias must be contiguous with shape [N].");
    require_aligned(value.data_ptr(), "bias");
  }
}

template <bool kLevel1A, bool kLevel1B, bool kWS, typename Traits>
void launch_tile(const cutlass::float_e2m1_t* a, const cutlass::float_e2m1_t* b,
                 const cutlass::float_ue4m3_t* sfa, const cutlass::float_ue4m3_t* sfb,
                 const float* sa, const float* sb, const cutlass::bfloat16_t* bias,
                 cutlass::bfloat16_t* out, int M, int N, int K, cudaStream_t stream) {
  fp4_gemm_tma_fwd<kLevel1A, kLevel1B, kWS, Traits>(a, b, sfa, sfb, sa, sb, bias, out, M, N, K,
                                                    stream);
}

template <bool kLevel1A, bool kLevel1B, bool kWS>
void launch_shape(const cutlass::float_e2m1_t* a, const cutlass::float_e2m1_t* b,
                  const cutlass::float_ue4m3_t* sfa, const cutlass::float_ue4m3_t* sfb,
                  const float* sa, const float* sb, const cutlass::bfloat16_t* bias,
                  cutlass::bfloat16_t* out, int M, int N, int K, cudaStream_t stream) {
  if (fp4_gemm_use_wide_tile(M, N, K)) {
    launch_tile<kLevel1A, kLevel1B, kWS, Fp4GemmTraits<256, 6>>(a, b, sfa, sfb, sa, sb, bias, out,
                                                                M, N, K, stream);
  } else if (K % 128 == 0) {
    launch_tile<kLevel1A, kLevel1B, kWS, Fp4GemmTraits<128, 4, 128>>(a, b, sfa, sfb, sa, sb, bias,
                                                                     out, M, N, K, stream);
  } else {
    launch_tile<kLevel1A, kLevel1B, kWS, Fp4GemmTraits<>>(a, b, sfa, sfb, sa, sb, bias, out, M, N,
                                                          K, stream);
  }
}

template <bool kLevel1A, bool kLevel1B>
void launch_ws(const cutlass::float_e2m1_t* a, const cutlass::float_e2m1_t* b,
               const cutlass::float_ue4m3_t* sfa, const cutlass::float_ue4m3_t* sfb,
               const float* sa, const float* sb, const cutlass::bfloat16_t* bias,
               cutlass::bfloat16_t* out, int M, int N, int K, bool use_ws, cudaStream_t stream) {
  if (use_ws) {
    launch_shape<kLevel1A, kLevel1B, true>(a, b, sfa, sfb, sa, sb, bias, out, M, N, K, stream);
  } else {
    launch_shape<kLevel1A, kLevel1B, false>(a, b, sfa, sfb, sa, sb, bias, out, M, N, K, stream);
  }
}

}  // namespace

void fp4_gemm(const at::Tensor& a, const at::Tensor& b, const at::Tensor& a_scales,
              const at::Tensor& b_scales, const at::Tensor& a_global_scales,
              const at::Tensor& b_global_scales, const std::optional<at::Tensor>& bias,
              at::Tensor& out, bool use_ws) {
  const c10::cuda::CUDAGuard device_guard(a.device());
  check_gemm_inputs(a, b, a_scales, b_scales, a_global_scales, b_global_scales, bias, out);
  const int M = static_cast<int>(a.size(0));
  const int N = static_cast<int>(b.size(0));
  const int K = static_cast<int>(a.size(1) * 2);
  const bool level1_a = a_global_scales.numel() != 0;
  const bool level1_b = b_global_scales.numel() != 0;

  const auto* a_ptr = reinterpret_cast<const cutlass::float_e2m1_t*>(a.data_ptr());
  const auto* b_ptr = reinterpret_cast<const cutlass::float_e2m1_t*>(b.data_ptr());
  const auto* sfa =
    reinterpret_cast<const cutlass::float_ue4m3_t*>(a_scales.data_ptr<at::Float8_e4m3fn>());
  const auto* sfb =
    reinterpret_cast<const cutlass::float_ue4m3_t*>(b_scales.data_ptr<at::Float8_e4m3fn>());
  const float* sa = level1_a ? a_global_scales.data_ptr<float>() : nullptr;
  const float* sb = level1_b ? b_global_scales.data_ptr<float>() : nullptr;
  const auto* bias_ptr =
    bias.has_value() ? reinterpret_cast<const cutlass::bfloat16_t*>(bias->data_ptr<at::BFloat16>())
                     : nullptr;
  auto* out_ptr = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (K % 256 == 0 && N % 128 == 0) {
    if (level1_a || level1_b) {
      fp4_gemm_cutlass(a, b, a_scales, b_scales, a_global_scales, b_global_scales, bias, out,
                       stream);
    } else {
      fp4_gemm_cutlass_simple(a, b, a_scales, b_scales, bias, out, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  if (level1_a && level1_b) {
    launch_ws<true, true>(a_ptr, b_ptr, sfa, sfb, sa, sb, bias_ptr, out_ptr, M, N, K, use_ws,
                          stream);
  } else if (level1_a) {
    launch_ws<true, false>(a_ptr, b_ptr, sfa, sfb, sa, sb, bias_ptr, out_ptr, M, N, K, use_ws,
                           stream);
  } else if (level1_b) {
    launch_ws<false, true>(a_ptr, b_ptr, sfa, sfb, sa, sb, bias_ptr, out_ptr, M, N, K, use_ws,
                           stream);
  } else {
    launch_ws<false, false>(a_ptr, b_ptr, sfa, sfb, sa, sb, bias_ptr, out_ptr, M, N, K, use_ws,
                            stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::fp4
