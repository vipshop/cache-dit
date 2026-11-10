// Fused RMSNorm with a per-channel affine epilogue:
//   y = rms_norm(x) * weight * (1 + scale)
//
// One CTA per row; the row stays in registers between the statistics pass and
// the epilogue, and `(1 + scale)` is folded into the same store, so the result
// carries a single bf16 rounding instead of two. `scale` is a per-channel [N]
// vector because the call sites broadcast a [B, N] modulation term with B == 1.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include "boogu.h"

namespace cache_dit::models {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ void unpack8(uint4 raw, float* f) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 t = __bfloat1622float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

__device__ __forceinline__ uint4 pack8(const float* f) {
  uint4 out;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    h[i] = __float22bfloat162_rn(make_float2(f[2 * i], f[2 * i + 1]));
  }
  return out;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, o);
  }
  __shared__ float warp_sums[kThreads / 32];
  if (lane == 0) {
    warp_sums[wid] = v;
  }
  __syncthreads();
  __shared__ float total;
  if (threadIdx.x == 0) {
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < kThreads / 32; ++i) {
      t += warp_sums[i];
    }
    total = t;
  }
  __syncthreads();
  return total;
}

// kUnits = ceil((N/8) / kThreads) vec8 units per thread.
template <int kUnits, bool kHasScale>
__global__ void boogu_scaled_rms_norm_kernel(const uint4* __restrict__ x,
                                             const __nv_bfloat16* __restrict__ weight,
                                             const __nv_bfloat16* __restrict__ scale,
                                             uint4* __restrict__ out, int nvec, int n, float eps,
                                             bool zero_centered_weight) {
  const int v0 = blockIdx.x * nvec;
  float vals[kUnits][8];
  float ss = 0.f;
#pragma unroll
  for (int u = 0; u < kUnits; ++u) {
    const int v = threadIdx.x + u * kThreads;
    if (v < nvec) {
      unpack8(x[v0 + v], vals[u]);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        ss += vals[u][j] * vals[u][j];
      }
    }
  }
  const float rstd = 1.0f / sqrtf(block_reduce_sum(ss) / n + eps);
#pragma unroll
  for (int u = 0; u < kUnits; ++u) {
    const int v = threadIdx.x + u * kThreads;
    if (v < nvec) {
      float wv[8], y[8];
      unpack8(*reinterpret_cast<const uint4*>(weight + v * 8), wv);
      if constexpr (kHasScale) {
        float sv[8];
        unpack8(*reinterpret_cast<const uint4*>(scale + v * 8), sv);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float w = zero_centered_weight ? wv[j] + 1.0f : wv[j];
          y[j] = vals[u][j] * rstd * w * (1.0f + sv[j]);
        }
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float w = zero_centered_weight ? wv[j] + 1.0f : wv[j];
          y[j] = vals[u][j] * rstd * w;
        }
      }
      out[v0 + v] = pack8(y);
    }
  }
}

template <int kUnits>
void launch_scaled(const uint4* x8, const __nv_bfloat16* w, const __nv_bfloat16* s, uint4* o8,
                   int64_t rows, int nvec, int n, float eps, bool zero_centered, bool has_scale,
                   cudaStream_t stream) {
  if (has_scale) {
    boogu_scaled_rms_norm_kernel<kUnits, true>
      <<<rows, kThreads, 0, stream>>>(x8, w, s, o8, nvec, n, eps, zero_centered);
  } else {
    boogu_scaled_rms_norm_kernel<kUnits, false>
      <<<rows, kThreads, 0, stream>>>(x8, w, s, o8, nvec, n, eps, zero_centered);
  }
}

bool is_aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0; }

}  // namespace

void boogu_scaled_rms_norm(at::Tensor x, at::Tensor weight, double eps,
                           const std::optional<at::Tensor>& scale, bool zero_centered_weight,
                           at::Tensor out) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(),
              "boogu_scaled_rms_norm: x must be contiguous CUDA bf16");
  TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == at::kBFloat16,
              "boogu_scaled_rms_norm: weight must be contiguous CUDA bf16");
  const int64_t n = x.size(-1);
  const int64_t rows = n == 0 ? 0 : x.numel() / n;
  const int64_t nvec = n / 8;
  TORCH_CHECK(n > 0 && n % 8 == 0 && nvec <= (int64_t)kThreads * 4,
              "boogu_scaled_rms_norm: N must be a multiple of 8 and <= ", kThreads * 4 * 8);
  TORCH_CHECK(weight.numel() == n, "boogu_scaled_rms_norm: weight must have N elements");
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == at::kBFloat16 &&
                out.sizes() == x.sizes(),
              "boogu_scaled_rms_norm: out must be a contiguous CUDA bf16 buffer matching x");
  const bool has_scale = scale.has_value() && scale->defined();
  if (has_scale) {
    TORCH_CHECK(scale->is_cuda() && scale->is_contiguous() &&
                  scale->scalar_type() == at::kBFloat16 && scale->numel() == n,
                "boogu_scaled_rms_norm: scale must be contiguous CUDA bf16 with N elements");
  }
  TORCH_CHECK(is_aligned16(x.const_data_ptr()) && is_aligned16(weight.const_data_ptr()),
              "boogu_scaled_rms_norm: x/weight must be 16B-aligned");
  if (has_scale) {
    TORCH_CHECK(is_aligned16(scale->const_data_ptr()),
                "boogu_scaled_rms_norm: scale must be 16B-aligned");
  }

  if (rows == 0) {
    return;
  }
  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int units = (int)((nvec + kThreads - 1) / kThreads);
  const uint4* x8 = reinterpret_cast<const uint4*>(x.const_data_ptr());
  uint4* o8 = reinterpret_cast<uint4*>(out.data_ptr());
  const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(weight.const_data_ptr());
  const __nv_bfloat16* s =
    has_scale ? reinterpret_cast<const __nv_bfloat16*>(scale->const_data_ptr()) : nullptr;
  switch (units) {
    case 1:
      launch_scaled<1>(x8, w, s, o8, rows, (int)nvec, (int)n, (float)eps, zero_centered_weight,
                       has_scale, stream.stream());
      break;
    case 2:
      launch_scaled<2>(x8, w, s, o8, rows, (int)nvec, (int)n, (float)eps, zero_centered_weight,
                       has_scale, stream.stream());
      break;
    case 3:
      launch_scaled<3>(x8, w, s, o8, rows, (int)nvec, (int)n, (float)eps, zero_centered_weight,
                       has_scale, stream.stream());
      break;
    case 4:
      launch_scaled<4>(x8, w, s, o8, rows, (int)nvec, (int)n, (float)eps, zero_centered_weight,
                       has_scale, stream.stream());
      break;
    default:
      // Unreachable while nvec <= kThreads * 4; a wider bound must add cases
      // here, otherwise the tail units would be silently skipped.
      TORCH_CHECK(false, "boogu_scaled_rms_norm: unsupported units ", units);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::models
