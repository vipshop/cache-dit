// Fused per-head RMSNorm + interleaved-pair RoPE.
//
// Semantics: fp32 RMS statistics over the head channel, per-channel weight,
// interleaved (re, im) complex rotation, and a single bf16 rounding at the
// store. `freqs_cis` is read directly as (re, im) pairs and narrowed to fp32;
// both complex64 and complex128 are accepted because Boogu's RoPE builder emits
// float64 cos/sin on every non-MPS platform.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include "boogu.h"

namespace cache_dit::models {
namespace {

constexpr int kWarpsPerBlock = 4;

__device__ __forceinline__ float2 pair_to_float2(const float2& v) { return v; }

__device__ __forceinline__ float2 pair_to_float2(const double2& v) {
  return make_float2(__double2float_rn(v.x), __double2float_rn(v.y));
}

// One warp per row (row = ((b * S + s) * H + h)). Two passes over the row's
// bf16 pairs; the row (240B at D=120) stays L1-resident between passes.
template <typename FreqT>
__global__ void boogu_qk_norm_rope_kernel(const __nv_bfloat162* __restrict__ x,
                                          const FreqT* __restrict__ freqs,
                                          const __nv_bfloat162* __restrict__ weight,
                                          __nv_bfloat162* __restrict__ out, int rows, int s_len,
                                          int h_heads, int dh, float eps) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kWarpsPerBlock + warp;
  if (row >= rows) {
    return;
  }
  const int sh = s_len * h_heads;
  const int b = row / sh;
  const int s = (row % sh) / h_heads;
  const FreqT* fb = freqs + (size_t)(b * s_len + s) * dh;
  const __nv_bfloat162* xr = x + (size_t)row * dh;

  float ss = 0.f;
  for (int f = lane; f < dh; f += 32) {
    const float2 e = __bfloat1622float2(xr[f]);
    ss += e.x * e.x + e.y * e.y;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    ss += __shfl_xor_sync(0xffffffffu, ss, o);
  }
  const float rstd = 1.0f / sqrtf(ss / (2 * dh) + eps);

  for (int f = lane; f < dh; f += 32) {
    const float2 e = __bfloat1622float2(xr[f]);
    const float2 wv = __bfloat1622float2(weight[f]);
    const float xn_e = e.x * rstd * wv.x;
    const float xn_o = e.y * rstd * wv.y;
    const float2 cs = pair_to_float2(fb[f]);
    __nv_bfloat162 o2;
    o2.x = __float2bfloat16_rn(xn_e * cs.x - xn_o * cs.y);
    o2.y = __float2bfloat16_rn(xn_o * cs.x + xn_e * cs.y);
    out[(size_t)row * dh + f] = o2;
  }
}

bool is_aligned(const void* p, uintptr_t bytes) {
  return (reinterpret_cast<uintptr_t>(p) & (bytes - 1)) == 0;
}

}  // namespace

void boogu_fused_qk_norm_rope(at::Tensor x, at::Tensor freqs_cis, at::Tensor weight, double eps,
                              at::Tensor out) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 4,
              "boogu_fused_qk_norm_rope: x must be contiguous CUDA bf16 [B,S,H,D]");
  const int64_t b = x.size(0);
  const int64_t s = x.size(1);
  const int64_t h = x.size(2);
  const int64_t d = x.size(3);
  TORCH_CHECK(d > 0 && d % 2 == 0, "boogu_fused_qk_norm_rope: D must be even");
  const bool freqs_f32 = freqs_cis.scalar_type() == at::kComplexFloat;
  TORCH_CHECK(freqs_cis.is_cuda() && freqs_cis.is_contiguous() &&
                (freqs_f32 || freqs_cis.scalar_type() == at::kComplexDouble) &&
                freqs_cis.dim() == 3 && freqs_cis.size(0) == b && freqs_cis.size(1) == s &&
                freqs_cis.size(2) == d / 2,
              "boogu_fused_qk_norm_rope: freqs_cis must be contiguous complex64/complex128 "
              "[B,S,D/2]");
  TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == at::kBFloat16 &&
                weight.numel() == d,
              "boogu_fused_qk_norm_rope: weight must be contiguous CUDA bf16 [D]");
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == at::kBFloat16 &&
                out.sizes() == x.sizes(),
              "boogu_fused_qk_norm_rope: out must be a contiguous CUDA bf16 buffer matching x");
  TORCH_CHECK(is_aligned(x.const_data_ptr(), 4) && is_aligned(weight.const_data_ptr(), 4) &&
                is_aligned(freqs_cis.const_data_ptr(), freqs_f32 ? 8 : 16),
              "boogu_fused_qk_norm_rope: inputs must be 4B-aligned; freqs_cis needs 8B "
              "(complex64, even element offset) or 16B (complex128) so its (re, im) pairs "
              "can be read directly");

  const int64_t rows = x.numel() / d;
  if (rows == 0) {
    return;
  }
  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t dh = d / 2;
  const int64_t grid = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  if (freqs_f32) {
    boogu_qk_norm_rope_kernel<<<grid, kWarpsPerBlock * 32, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat162*>(x.const_data_ptr()),
      reinterpret_cast<const float2*>(freqs_cis.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat162*>(weight.const_data_ptr()),
      reinterpret_cast<__nv_bfloat162*>(out.data_ptr()), (int)rows, (int)s, (int)h, (int)dh,
      (float)eps);
  } else {
    boogu_qk_norm_rope_kernel<<<grid, kWarpsPerBlock * 32, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat162*>(x.const_data_ptr()),
      reinterpret_cast<const double2*>(freqs_cis.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat162*>(weight.const_data_ptr()),
      reinterpret_cast<__nv_bfloat162*>(out.data_ptr()), (int)rows, (int)s, (int)h, (int)dh,
      (float)eps);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::models
