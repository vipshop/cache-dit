#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <limits>

#include "comm_ops.h"

namespace cache_dit::comm {
namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;

template <typename T>
__device__ float to_float(T value);
template <typename T>
__device__ T from_float(float value);

template <>
__device__ float to_float<float>(float value) {
  return value;
}

template <>
__device__ float to_float<__half>(__half value) {
  return __half2float(value);
}

template <>
__device__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <>
__device__ float from_float<float>(float value) {
  return value;
}

template <>
__device__ __half from_float<__half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

__device__ void compute_merge_weights(
  float p_lse,
  float s_lse,
  float& p_scale,
  float& s_scale,
  float& merged_lse) {
  const float neg_inf = -std::numeric_limits<float>::infinity();
  if (isnan(p_lse) || isnan(s_lse)) {
    p_scale = std::numeric_limits<float>::quiet_NaN();
    s_scale = p_scale;
    merged_lse = p_scale;
    return;
  }
  p_lse = isinf(p_lse) ? neg_inf : p_lse;
  s_lse = isinf(s_lse) ? neg_inf : s_lse;
  const float max_lse = fmaxf(p_lse, s_lse);
  const bool both_empty = isinf(max_lse) && max_lse < 0.0f;
  const float p_se = both_empty ? 0.0f : expf(p_lse - max_lse);
  const float s_se = both_empty ? 0.0f : expf(s_lse - max_lse);
  const float denom = p_se + s_se;
  p_scale = both_empty ? 1.0f : p_se / denom;
  s_scale = both_empty ? 0.0f : s_se / denom;
  merged_lse = both_empty ? neg_inf : max_lse + logf(denom);
}

template <typename scalar_t, typename lse_t>
__global__ void merge_attn_states_packed_kernel(
  const scalar_t* prev_out,
  const lse_t* prev_lse,
  const scalar_t* suff_out,
  const lse_t* suff_lse,
  scalar_t* out,
  lse_t* lse,
  int64_t rows,
  int64_t head_size) {
  constexpr int pack_size = 16 / sizeof(scalar_t);
  const int64_t packs_per_row = head_size / pack_size;
  const int64_t index = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (index >= rows * packs_per_row) return;
  const int64_t row = index / packs_per_row;
  const int64_t pack_index = index % packs_per_row;

  float p_scale;
  float s_scale;
  float merged_lse;
  compute_merge_weights(
    to_float(prev_lse[row]), to_float(suff_lse[row]), p_scale, s_scale, merged_lse);

  const int64_t offset = row * head_size;
  const uint4 prev_pack = reinterpret_cast<const uint4*>(prev_out + offset)[pack_index];
  const uint4 suff_pack = reinterpret_cast<const uint4*>(suff_out + offset)[pack_index];
  uint4 out_pack;
  #pragma unroll
  for (int i = 0; i < pack_size; ++i) {
    const float prev_value = to_float(reinterpret_cast<const scalar_t*>(&prev_pack)[i]);
    const float suff_value = to_float(reinterpret_cast<const scalar_t*>(&suff_pack)[i]);
    reinterpret_cast<scalar_t*>(&out_pack)[i] =
      from_float<scalar_t>(prev_value * p_scale + suff_value * s_scale);
  }
  reinterpret_cast<uint4*>(out + offset)[pack_index] = out_pack;
  if (pack_index == 0) {
    lse[row] = from_float<lse_t>(merged_lse);
  }
}

template <typename scalar_t, typename lse_t>
__global__ void merge_attn_states_kernel(
  const scalar_t* prev_out,
  const lse_t* prev_lse,
  const scalar_t* suff_out,
  const lse_t* suff_lse,
  scalar_t* out,
  lse_t* lse,
  int64_t rows,
  int64_t head_size) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;

  float p_scale;
  float s_scale;
  float merged_lse;
  compute_merge_weights(
    to_float(prev_lse[row]), to_float(suff_lse[row]), p_scale, s_scale, merged_lse);

  const int64_t offset = row * head_size;
  for (int64_t d = lane; d < head_size; d += kWarpSize) {
    const float merged =
      to_float(prev_out[offset + d]) * p_scale + to_float(suff_out[offset + d]) * s_scale;
    out[offset + d] = from_float<scalar_t>(merged);
  }
  if (lane == 0) {
    lse[row] = from_float<lse_t>(merged_lse);
  }
}

template <typename scalar_t, typename lse_t>
void launch_merge(
  const at::Tensor& prev_out,
  const at::Tensor& prev_lse,
  const at::Tensor& suff_out,
  const at::Tensor& suff_lse,
  at::Tensor& out,
  at::Tensor& lse,
  int64_t rows,
  int64_t head_size,
  cudaStream_t stream) {
  if (rows == 0) return;

  constexpr int pack_size = 16 / sizeof(scalar_t);
  const bool aligned =
    reinterpret_cast<uintptr_t>(prev_out.data_ptr()) % 16 == 0 &&
    reinterpret_cast<uintptr_t>(suff_out.data_ptr()) % 16 == 0 &&
    reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0;
  if (head_size % pack_size == 0 && aligned) {
    const int64_t total_threads = rows * (head_size / pack_size);
    merge_attn_states_packed_kernel<scalar_t, lse_t>
      <<<dim3((total_threads + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        reinterpret_cast<const scalar_t*>(prev_out.data_ptr()),
        reinterpret_cast<const lse_t*>(prev_lse.data_ptr()),
        reinterpret_cast<const scalar_t*>(suff_out.data_ptr()),
        reinterpret_cast<const lse_t*>(suff_lse.data_ptr()),
        reinterpret_cast<scalar_t*>(out.data_ptr()),
        reinterpret_cast<lse_t*>(lse.data_ptr()),
        rows,
        head_size);
  } else {
    merge_attn_states_kernel<scalar_t, lse_t>
      <<<dim3((rows + kWarpsPerBlock - 1) / kWarpsPerBlock), kThreads, 0, stream>>>(
        reinterpret_cast<const scalar_t*>(prev_out.data_ptr()),
        reinterpret_cast<const lse_t*>(prev_lse.data_ptr()),
        reinterpret_cast<const scalar_t*>(suff_out.data_ptr()),
        reinterpret_cast<const lse_t*>(suff_lse.data_ptr()),
        reinterpret_cast<scalar_t*>(out.data_ptr()),
        reinterpret_cast<lse_t*>(lse.data_ptr()),
        rows,
        head_size);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void dispatch_lse(
  const at::Tensor& prev_out,
  const at::Tensor& prev_lse,
  const at::Tensor& suff_out,
  const at::Tensor& suff_lse,
  at::Tensor& out,
  at::Tensor& lse,
  int64_t rows,
  int64_t head_size,
  cudaStream_t stream) {
  if (lse.scalar_type() == at::kFloat) {
    launch_merge<scalar_t, float>(
      prev_out, prev_lse, suff_out, suff_lse, out, lse, rows, head_size, stream);
  } else if (lse.scalar_type() == at::kHalf) {
    launch_merge<scalar_t, __half>(
      prev_out, prev_lse, suff_out, suff_lse, out, lse, rows, head_size, stream);
  } else if (lse.scalar_type() == at::kBFloat16) {
    launch_merge<scalar_t, __nv_bfloat16>(
      prev_out, prev_lse, suff_out, suff_lse, out, lse, rows, head_size, stream);
  } else {
    TORCH_CHECK(false, "unsupported LSE dtype: ", lse.scalar_type());
  }
}

} // namespace

void fused_merge_attn_states(
  at::Tensor prev_out,
  at::Tensor prev_lse,
  at::Tensor suff_out,
  at::Tensor suff_lse,
  at::Tensor out,
  at::Tensor lse) {
  TORCH_CHECK(
    prev_out.is_cuda() && prev_lse.is_cuda() && suff_out.is_cuda() && suff_lse.is_cuda() &&
    out.is_cuda() && lse.is_cuda(),
    "merge_attn_states requires CUDA tensors");
  TORCH_CHECK(
    prev_out.device() == out.device() && prev_lse.device() == out.device() &&
    suff_out.device() == out.device() && suff_lse.device() == out.device() &&
    lse.device() == out.device(),
    "all merge_attn_states tensors must be on the same CUDA device");
  TORCH_CHECK(
    prev_out.is_contiguous() && prev_lse.is_contiguous() && suff_out.is_contiguous() &&
    suff_lse.is_contiguous() && out.is_contiguous() && lse.is_contiguous(),
    "merge_attn_states expects contiguous tensors");
  TORCH_CHECK(
    prev_out.dim() == 3 && suff_out.dim() == 3 && out.dim() == 3 &&
    prev_lse.dim() == 2 && suff_lse.dim() == 2 && lse.dim() == 2,
    "merge_attn_states expects flattened 3D outputs and 2D LSE tensors");
  TORCH_CHECK(
    prev_out.sizes() == suff_out.sizes() && suff_out.sizes() == out.sizes() &&
    prev_lse.sizes() == suff_lse.sizes() && suff_lse.sizes() == lse.sizes(),
    "merge_attn_states input and output shapes must match");
  TORCH_CHECK(
    prev_out.scalar_type() == suff_out.scalar_type() && suff_out.scalar_type() == out.scalar_type(),
    "attention output tensors must share dtype");
  TORCH_CHECK(
    prev_lse.scalar_type() == suff_lse.scalar_type() && suff_lse.scalar_type() == lse.scalar_type(),
    "LSE tensors must share dtype");
  const int64_t rows = out.size(0) * out.size(1);
  const int64_t head_size = out.size(2);
  TORCH_CHECK(prev_lse.numel() == rows, "invalid LSE shape");
  TORCH_CHECK(head_size > 0, "head size must be positive");

  const c10::cuda::CUDAGuard guard(out.device());
  const auto stream = at::cuda::getCurrentCUDAStream(out.device().index());
  if (out.scalar_type() == at::kFloat) {
    dispatch_lse<float>(
      prev_out, prev_lse, suff_out, suff_lse, out, lse, rows, head_size, stream);
  } else if (out.scalar_type() == at::kHalf) {
    dispatch_lse<__half>(
      prev_out, prev_lse, suff_out, suff_lse, out, lse, rows, head_size, stream);
  } else if (out.scalar_type() == at::kBFloat16) {
    dispatch_lse<__nv_bfloat16>(
      prev_out, prev_lse, suff_out, suff_lse, out, lse, rows, head_size, stream);
  } else {
    TORCH_CHECK(false, "unsupported attention output dtype: ", out.scalar_type());
  }
}

} // namespace cache_dit::comm
