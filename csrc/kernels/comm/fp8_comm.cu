#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <cstring>

#include "comm_ops.h"

namespace cache_dit::comm {
namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;
constexpr float kFp8Max = 448.0f;

__device__ float warp_absmax(float value) {
  #pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return __shfl_sync(0xffffffff, value, 0);
}

constexpr int kPackSize = 8;

__device__ uint8_t to_fp8(float value) {
  const float clipped = fminf(fmaxf(value, -kFp8Max), kFp8Max);
  const __half half_value = __float2half_rn(clipped);
  return __nv_cvt_halfraw_to_fp8(
    static_cast<__half_raw>(half_value), __NV_SATFINITE, __NV_E4M3);
}

__device__ float from_fp8(uint8_t value) {
  const __half_raw raw = __nv_cvt_fp8_to_halfraw(value, __NV_E4M3);
  return __half2float(static_cast<__half>(raw));
}

__device__ bool is_aligned_16(const void* pointer) {
  return reinterpret_cast<uintptr_t>(pointer) % 16 == 0;
}

__device__ void load_bf16_pack(const __nv_bfloat16* input, __nv_bfloat16* values) {
  const uint4 pack = *reinterpret_cast<const uint4*>(input);
  std::memcpy(values, &pack, sizeof(pack));
}

__device__ void store_bf16_pack(__nv_bfloat16* output, const __nv_bfloat16* values) {
  uint4 pack;
  std::memcpy(&pack, values, sizeof(pack));
  *reinterpret_cast<uint4*>(output) = pack;
}

__device__ void load_fp8_pack(const uint8_t* input, uint8_t* values) {
  if (is_aligned_16(input)) {
    const uint4 pack = *reinterpret_cast<const uint4*>(input);
    std::memcpy(values, &pack, sizeof(pack));
  } else if (reinterpret_cast<uintptr_t>(input) % 4 == 0) {
    #pragma unroll
    for (int i = 0; i < kPackSize / 4; ++i) {
      const uint32_t pack = *reinterpret_cast<const uint32_t*>(input + i * 4);
      std::memcpy(values + i * 4, &pack, sizeof(pack));
    }
  } else if (reinterpret_cast<uintptr_t>(input) % 2 == 0) {
    #pragma unroll
    for (int i = 0; i < kPackSize / 2; ++i) {
      const uint16_t pack = *reinterpret_cast<const uint16_t*>(input + i * 2);
      std::memcpy(values + i * 2, &pack, sizeof(pack));
    }
  } else {
    #pragma unroll
    for (int i = 0; i < kPackSize; ++i) {
      values[i] = input[i];
    }
  }
}

__device__ void store_fp8_pack(uint8_t* output, const uint8_t* values) {
  if (is_aligned_16(output)) {
    uint4 pack;
    std::memcpy(&pack, values, sizeof(pack));
    *reinterpret_cast<uint4*>(output) = pack;
  } else if (reinterpret_cast<uintptr_t>(output) % 4 == 0) {
    #pragma unroll
    for (int i = 0; i < kPackSize / 4; ++i) {
      uint32_t pack;
      std::memcpy(&pack, values + i * 4, sizeof(pack));
      *reinterpret_cast<uint32_t*>(output + i * 4) = pack;
    }
  } else if (reinterpret_cast<uintptr_t>(output) % 2 == 0) {
    #pragma unroll
    for (int i = 0; i < kPackSize / 2; ++i) {
      uint16_t pack;
      std::memcpy(&pack, values + i * 2, sizeof(pack));
      *reinterpret_cast<uint16_t*>(output + i * 2) = pack;
    }
  } else {
    #pragma unroll
    for (int i = 0; i < kPackSize; ++i) {
      output[i] = values[i];
    }
  }
}

__device__ float row_absmax(const __nv_bfloat16* input, int64_t head_size, int lane) {
  float absmax = 0.0f;
  if (head_size % kPackSize == 0 && is_aligned_16(input)) {
    for (int64_t d = lane * kPackSize; d < head_size; d += kWarpSize * kPackSize) {
      __nv_bfloat16 values[kPackSize];
      load_bf16_pack(input + d, values);
      #pragma unroll
      for (int i = 0; i < kPackSize; ++i) {
        absmax = fmaxf(absmax, fabsf(__bfloat162float(values[i])));
      }
    }
  } else {
    for (int64_t d = lane; d < head_size; d += kWarpSize) {
      absmax = fmaxf(absmax, fabsf(__bfloat162float(input[d])));
    }
  }
  return warp_absmax(absmax);
}

__device__ void quantize_row(
  const __nv_bfloat16* input,
  uint8_t* output,
  int64_t head_size,
  int lane,
  float inv_scale) {
  if (head_size % kPackSize == 0 && is_aligned_16(input)) {
    for (int64_t d = lane * kPackSize; d < head_size; d += kWarpSize * kPackSize) {
      __nv_bfloat16 values[kPackSize];
      uint8_t quantized[kPackSize];
      load_bf16_pack(input + d, values);
      #pragma unroll
      for (int i = 0; i < kPackSize; ++i) {
        quantized[i] = to_fp8(__bfloat162float(values[i]) * inv_scale);
      }
      store_fp8_pack(output + d, quantized);
    }
  } else {
    for (int64_t d = lane; d < head_size; d += kWarpSize) {
      output[d] = to_fp8(__bfloat162float(input[d]) * inv_scale);
    }
  }
}

__device__ void dequantize_row(
  const uint8_t* input,
  __nv_bfloat16* output,
  int64_t head_size,
  int lane,
  float scale) {
  if (head_size % kPackSize == 0 && is_aligned_16(output)) {
    for (int64_t d = lane * kPackSize; d < head_size; d += kWarpSize * kPackSize) {
      uint8_t quantized[kPackSize];
      __nv_bfloat16 values[kPackSize];
      load_fp8_pack(input + d, quantized);
      #pragma unroll
      for (int i = 0; i < kPackSize; ++i) {
        values[i] = __float2bfloat16_rn(from_fp8(quantized[i]) * scale);
      }
      store_bf16_pack(output + d, values);
    }
  } else {
    for (int64_t d = lane; d < head_size; d += kWarpSize) {
      output[d] = __float2bfloat16_rn(from_fp8(input[d]) * scale);
    }
  }
}

__global__ void per_token_quant_kernel(
  const __nv_bfloat16* x,
  uint8_t* y,
  int64_t rows,
  int64_t head_size) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;

  const __nv_bfloat16* input = x + row * head_size;
  uint8_t* output = y + row * (head_size + 2);
  const float absmax = row_absmax(input, head_size, lane);
  const float scale = fmaxf(absmax / kFp8Max, 1e-4f);
  quantize_row(input, output, head_size, lane, 1.0f / scale);
  if (lane == 0) {
    const __nv_bfloat16 stored_scale = __float2bfloat16_rn(scale);
    std::memcpy(output + head_size, &stored_scale, sizeof(stored_scale));
  }
}

__global__ void per_token_dequant_kernel(
  const uint8_t* x,
  __nv_bfloat16* y,
  int64_t rows,
  int64_t head_size) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;

  const uint8_t* input = x + row * (head_size + 2);
  __nv_bfloat16* output = y + row * head_size;
  __nv_bfloat16 stored_scale;
  std::memcpy(&stored_scale, input + head_size, sizeof(stored_scale));
  dequantize_row(input, output, head_size, lane, __bfloat162float(stored_scale));
}

__global__ void qkv_permute_quant_kernel(
  const __nv_bfloat16* x,
  uint8_t* y,
  int64_t batches,
  int64_t seq_len,
  int64_t num_heads,
  int64_t head_size,
  int64_t stride_b,
  int64_t stride_s,
  int64_t stride_p,
  int64_t stride_n,
  int64_t rows) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;

  const int64_t n = row % num_heads;
  int64_t index = row / num_heads;
  const int64_t b = index % batches;
  index /= batches;
  const int64_t s = index % seq_len;
  const int64_t p = index / seq_len;
  const __nv_bfloat16* input =
    x + b * stride_b + s * stride_s + p * stride_p + n * stride_n;
  uint8_t* output = y + row * (head_size + 4);

  const float absmax = row_absmax(input, head_size, lane);
  const float scale = fmaxf(absmax / kFp8Max, 1e-6f);
  quantize_row(input, output, head_size, lane, 1.0f / scale);
  if (lane == 0) {
    std::memcpy(output + head_size, &scale, sizeof(scale));
  }
}

__global__ void qkv_permute_dequant_kernel(
  const uint8_t* quant_x,
  __nv_bfloat16* x,
  int64_t batches,
  int64_t seq_len,
  int64_t num_heads,
  int64_t head_size,
  int64_t stride_s,
  int64_t stride_b,
  int64_t stride_n,
  int64_t rows) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;

  const int64_t n = row % num_heads;
  int64_t index = row / num_heads;
  const int64_t b = index % batches;
  const int64_t s = index / batches;
  const uint8_t* input = quant_x + s * stride_s + b * stride_b + n * stride_n;
  __nv_bfloat16* output = x + ((b * seq_len + s) * num_heads + n) * head_size;
  float scale;
  std::memcpy(&scale, input + head_size, sizeof(scale));
  dequantize_row(input, output, head_size, lane, scale);
}

void check_fp8_pair(const at::Tensor& x, const at::Tensor& y) {
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), "communication operators require CUDA tensors");
  TORCH_CHECK(x.device() == y.device(), "input and output must be on the same CUDA device");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "expected BF16 input");
  TORCH_CHECK(y.scalar_type() == at::kFloat8_e4m3fn, "expected FP8 E4M3 output");
}

} // namespace

void fp8_comm_per_token_quant(at::Tensor x, at::Tensor y) {
  check_fp8_pair(x, y);
  TORCH_CHECK(x.dim() == 2 && y.dim() == 2, "per-token quant expects 2D tensors");
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "per-token quant expects contiguous tensors");
  const int64_t rows = x.size(0);
  const int64_t head_size = x.size(1);
  TORCH_CHECK(head_size > 0 && head_size % 2 == 0, "per-token head size must be positive and even");
  TORCH_CHECK(y.size(0) == rows && y.size(1) == head_size + 2, "invalid quantized output shape");

  if (rows == 0) return;

  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  per_token_quant_kernel<<<grid, kThreads, 0, stream>>>(
    reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
    reinterpret_cast<uint8_t*>(y.data_ptr()),
    rows,
    head_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fp8_comm_per_token_dequant(at::Tensor x, at::Tensor y) {
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), "communication operators require CUDA tensors");
  TORCH_CHECK(x.device() == y.device(), "input and output must be on the same CUDA device");
  TORCH_CHECK(x.scalar_type() == at::kFloat8_e4m3fn, "expected FP8 E4M3 input");
  TORCH_CHECK(y.scalar_type() == at::kBFloat16, "expected BF16 output");
  TORCH_CHECK(x.dim() == 2 && y.dim() == 2, "per-token dequant expects 2D tensors");
  TORCH_CHECK(
    x.is_contiguous() && y.is_contiguous(), "per-token dequant expects contiguous tensors");
  const int64_t rows = x.size(0);
  const int64_t head_size = x.size(1) - 2;
  TORCH_CHECK(head_size > 0 && head_size % 2 == 0, "per-token head size must be positive and even");
  TORCH_CHECK(y.size(0) == rows && y.size(1) == head_size, "invalid dequantized output shape");

  if (rows == 0) return;

  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  per_token_dequant_kernel<<<grid, kThreads, 0, stream>>>(
    reinterpret_cast<const uint8_t*>(x.data_ptr()),
    reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
    rows,
    head_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fp8_comm_qkv_permute_quant(at::Tensor x, at::Tensor y) {
  check_fp8_pair(x, y);
  TORCH_CHECK(x.dim() == 5 && y.dim() == 5, "QKV quant expects 5D tensors");
  TORCH_CHECK(y.is_contiguous(), "QKV quantized output must be contiguous");
  TORCH_CHECK(x.stride(4) == 1, "QKV head dimension must be contiguous");
  const int64_t batches = x.size(0);
  const int64_t seq_len = x.size(1);
  const int64_t partitions = x.size(2);
  const int64_t num_heads = x.size(3);
  const int64_t head_size = x.size(4);
  TORCH_CHECK(
    head_size > 0 && head_size % 4 == 0, "QKV head size must be a positive multiple of 4");
  TORCH_CHECK(
    y.size(0) == partitions && y.size(1) == seq_len && y.size(2) == batches &&
    y.size(3) == num_heads && y.size(4) == head_size + 4,
    "invalid QKV quantized output shape");
  const int64_t rows = partitions * seq_len * batches * num_heads;

  if (rows == 0) return;

  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  qkv_permute_quant_kernel<<<grid, kThreads, 0, stream>>>(
    reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
    reinterpret_cast<uint8_t*>(y.data_ptr()),
    batches,
    seq_len,
    num_heads,
    head_size,
    x.stride(0),
    x.stride(1),
    x.stride(2),
    x.stride(3),
    rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fp8_comm_qkv_permute_dequant(at::Tensor quant_x, at::Tensor x) {
  TORCH_CHECK(quant_x.is_cuda() && x.is_cuda(), "communication operators require CUDA tensors");
  TORCH_CHECK(quant_x.device() == x.device(), "input and output must be on the same CUDA device");
  TORCH_CHECK(quant_x.scalar_type() == at::kFloat8_e4m3fn, "expected FP8 E4M3 input");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "expected BF16 output");
  TORCH_CHECK(quant_x.dim() == 4 && x.dim() == 4, "QKV dequant expects 4D tensors");
  TORCH_CHECK(x.is_contiguous(), "QKV dequantized output must be contiguous");
  TORCH_CHECK(quant_x.stride(3) == 1, "QKV packed head dimension must be contiguous");
  const int64_t seq_len = quant_x.size(0);
  const int64_t batches = quant_x.size(1);
  const int64_t num_heads = quant_x.size(2);
  const int64_t head_size = quant_x.size(3) - 4;
  TORCH_CHECK(
    head_size > 0 && head_size % 4 == 0, "QKV head size must be a positive multiple of 4");
  TORCH_CHECK(
    x.size(0) == batches && x.size(1) == seq_len && x.size(2) == num_heads &&
    x.size(3) == head_size,
    "invalid QKV dequantized output shape");
  const int64_t rows = seq_len * batches * num_heads;

  if (rows == 0) return;

  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  qkv_permute_dequant_kernel<<<grid, kThreads, 0, stream>>>(
    reinterpret_cast<const uint8_t*>(quant_x.data_ptr()),
    reinterpret_cast<__nv_bfloat16*>(x.data_ptr()),
    batches,
    seq_len,
    num_heads,
    head_size,
    quant_x.stride(0),
    quant_x.stride(1),
    quant_x.stride(2),
    rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace cache_dit::comm
