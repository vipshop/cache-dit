// Fused post-norm gated residual RMSNorm:
//   y = residual + tanh(gate) * rms_norm(x)
//
// One CTA per row; the row stays in registers between the statistics pass and
// the epilogue, so `x` is read once and the result carries a single bf16
// rounding at the store.

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
template <int kUnits>
__global__ void boogu_gated_rms_norm_kernel(const uint4* __restrict__ x,
                                            const __nv_bfloat16* __restrict__ weight,
                                            const __nv_bfloat16* __restrict__ gate,
                                            const uint4* __restrict__ residual,
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
      float wv[8], gv[8], rv[8], y[8];
      unpack8(*reinterpret_cast<const uint4*>(weight + v * 8), wv);
      unpack8(*reinterpret_cast<const uint4*>(gate + v * 8), gv);
      unpack8(residual[v0 + v], rv);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float w = zero_centered_weight ? wv[j] + 1.0f : wv[j];
        y[j] = rv[j] + tanhf(gv[j]) * (vals[u][j] * rstd * w);
      }
      out[v0 + v] = pack8(y);
    }
  }
}

bool is_aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0; }

}  // namespace

void boogu_gated_rms_norm(at::Tensor x, at::Tensor weight, double eps, at::Tensor gate,
                          at::Tensor residual, bool zero_centered_weight, at::Tensor out) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(),
              "boogu_gated_rms_norm: x must be contiguous CUDA bf16");
  TORCH_CHECK(residual.sizes() == x.sizes() && residual.is_cuda() && residual.is_contiguous() &&
                residual.scalar_type() == at::kBFloat16,
              "boogu_gated_rms_norm: residual must match x (contiguous CUDA bf16)");
  TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == at::kBFloat16,
              "boogu_gated_rms_norm: weight must be contiguous CUDA bf16");
  TORCH_CHECK(gate.is_cuda() && gate.is_contiguous() && gate.scalar_type() == at::kBFloat16,
              "boogu_gated_rms_norm: gate must be contiguous CUDA bf16");
  const int64_t n = x.size(-1);
  const int64_t rows = n == 0 ? 0 : x.numel() / n;
  const int64_t nvec = n / 8;
  TORCH_CHECK(n > 0 && n % 8 == 0 && nvec <= (int64_t)kThreads * 4,
              "boogu_gated_rms_norm: N must be a multiple of 8 and <= ", kThreads * 4 * 8);
  TORCH_CHECK(weight.numel() == n && gate.numel() == n,
              "boogu_gated_rms_norm: weight/gate must have N elements");
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == at::kBFloat16 &&
                out.sizes() == x.sizes(),
              "boogu_gated_rms_norm: out must be a contiguous CUDA bf16 buffer matching x");
  TORCH_CHECK(is_aligned16(x.const_data_ptr()) && is_aligned16(residual.const_data_ptr()) &&
                is_aligned16(weight.const_data_ptr()) && is_aligned16(gate.const_data_ptr()),
              "boogu_gated_rms_norm: all inputs must be 16B-aligned (contiguous views with "
              "odd storage_offset are unsupported)");

  if (rows == 0) {
    return;
  }
  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int units = (int)((nvec + kThreads - 1) / kThreads);
  const uint4* x8 = reinterpret_cast<const uint4*>(x.const_data_ptr());
  const uint4* r8 = reinterpret_cast<const uint4*>(residual.const_data_ptr());
  uint4* o8 = reinterpret_cast<uint4*>(out.data_ptr());
  const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(weight.const_data_ptr());
  const __nv_bfloat16* g = reinterpret_cast<const __nv_bfloat16*>(gate.const_data_ptr());
  switch (units) {
    case 1:
      boogu_gated_rms_norm_kernel<1><<<rows, kThreads, 0, stream>>>(
        x8, w, g, r8, o8, (int)nvec, (int)n, (float)eps, zero_centered_weight);
      break;
    case 2:
      boogu_gated_rms_norm_kernel<2><<<rows, kThreads, 0, stream>>>(
        x8, w, g, r8, o8, (int)nvec, (int)n, (float)eps, zero_centered_weight);
      break;
    case 3:
      boogu_gated_rms_norm_kernel<3><<<rows, kThreads, 0, stream>>>(
        x8, w, g, r8, o8, (int)nvec, (int)n, (float)eps, zero_centered_weight);
      break;
    default:
      boogu_gated_rms_norm_kernel<4><<<rows, kThreads, 0, stream>>>(
        x8, w, g, r8, o8, (int)nvec, (int)n, (float)eps, zero_centered_weight);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::models
