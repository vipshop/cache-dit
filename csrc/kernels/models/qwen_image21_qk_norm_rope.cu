// Qwen-Image 2.1 q/k RMSNorm + interleaved-pair complex RoPE.
//
// Numerics are pinned to the eager expression
//   variance = x.float().pow(2).mean(-1, keepdim=True)
//   n        = (x * rsqrt(variance + eps)).to(weight.dtype)
//   o        = n * weight
//   out      = view_as_real(view_as_complex(o.float()) * freqs).flatten(3).type_as(x)
// which rounds to bf16 three times: after the norm, after the channel scale and after the
// rotation. The rotation itself is fp32 with explicit `__fmul_rn`/`__fadd_rn` so nvcc
// cannot contract it into an FMA that torch's complex multiply does not use.
//
// The mean over the 128 head channels reproduces ATen's reduction order for this shape:
// 32 lanes, lane `t` accumulating elements [4t, 4t + 4) in order, then a 32-lane
// butterfly tree (16/8/4/2/1), then `/ D`. This matches `torch.equal` on sm120 and is
// re-verified at runtime by `fused_ops` before the kernel is trusted.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include "qwen_image21.h"

namespace cache_dit::models {
namespace {

constexpr int kWarpsPerBlock = 4;
// One warp per row, 32 lanes x 4 channels = the whole 128-wide head.
constexpr int kPairs = 64;

__device__ __forceinline__ float warp_sum_32(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  }
  return v;
}

struct OutStrides {
  int64_t dim0;
  int64_t dim1;
  int64_t dim2;
};

// torch's complex64 multiply is compiled with default nvcc contraction, so which product
// gets folded into the FMA depends on the build. `fused_ops` probes the three candidates
// once per process and passes the matching id; a wrong choice costs one bf16 ulp.
template <int kVari>
__device__ __forceinline__ void rotate_pair(float re, float im, float c, float s,
                                            __nv_bfloat162* dst) {
  float out_re;
  float out_im;
  if (kVari == 0) {
    out_re = __fsub_rn(__fmul_rn(re, c), __fmul_rn(im, s));
    out_im = __fadd_rn(__fmul_rn(re, s), __fmul_rn(im, c));
  } else if (kVari == 1) {
    out_re = __fmaf_rn(re, c, -__fmul_rn(im, s));
    out_im = __fmaf_rn(re, s, __fmul_rn(im, c));
  } else if (kVari == 2) {
    out_re = __fmaf_rn(-im, s, __fmul_rn(re, c));
    out_im = __fmaf_rn(im, c, __fmul_rn(re, s));
  } else if (kVari == 3) {
    out_re = __fmaf_rn(re, c, -__fmul_rn(im, s));
    out_im = __fmaf_rn(im, c, __fmul_rn(re, s));
  } else {
    out_re = __fmaf_rn(-im, s, __fmul_rn(re, c));
    out_im = __fmaf_rn(re, s, __fmul_rn(im, c));
  }
  dst->x = __float2bfloat16_rn(out_re);
  dst->y = __float2bfloat16_rn(out_im);
}

// `out` is addressed through explicit strides so the kernel can write the target tail of a
// persistent [prefix | target] KV buffer without an intermediate copy.
template <int kVari>
__global__ void qwen_image21_qk_norm_rope_kernel(
  const __nv_bfloat162* __restrict__ x, const __nv_bfloat162* __restrict__ weight,
  const float* __restrict__ cos, const float* __restrict__ sin, __nv_bfloat162* __restrict__ out,
  OutStrides out_strides, int64_t rows, int seq_len, int heads, float eps) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t row = (int64_t)blockIdx.x * kWarpsPerBlock + warp;
  if (row >= rows) {
    return;
  }

  // Flat row order is ((b * S + s) * H + h), so the head is the fastest varying index.
  const int64_t head = row % heads;
  const int64_t pos = (row / heads) % seq_len;
  const int64_t batch = row / ((int64_t)heads * seq_len);

  // Lane `t` owns channels [4t, 4t + 4) = complex pairs {2t, 2t + 1}.
  const int pair_idx = lane << 1;
  const __nv_bfloat162* x_row = x + row * kPairs;
  const __nv_bfloat162 v0 = x_row[pair_idx];
  const __nv_bfloat162 v1 = x_row[pair_idx + 1];
  const float2 e0 = __bfloat1622float2(v0);
  const float2 e1 = __bfloat1622float2(v1);

  // Squares are fp32, then summed in ATen's order: left to right inside the lane.
  float ss = e0.x * e0.x;
  ss += e0.y * e0.y;
  ss += e1.x * e1.x;
  ss += e1.y * e1.y;
  ss = warp_sum_32(ss);
  const float rstd = rsqrtf(ss / (float)(kPairs * 2) + eps);

  const float2 wv0 = __bfloat1622float2(weight[pair_idx]);
  const float2 wv1 = __bfloat1622float2(weight[pair_idx + 1]);

  // Rounding 1: bf16(x * rstd). Rounding 2: bf16(n * weight).
  const float o0x = __bfloat162float(
    __float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(e0.x * rstd)) * wv0.x));
  const float o0y = __bfloat162float(
    __float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(e0.y * rstd)) * wv0.y));
  const float o1x = __bfloat162float(
    __float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(e1.x * rstd)) * wv1.x));
  const float o1y = __bfloat162float(
    __float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(e1.y * rstd)) * wv1.y));

  const float* cos_pos = cos + pos * kPairs + pair_idx;
  const float* sin_pos = sin + pos * kPairs + pair_idx;

  // Rounding 3: the rotation result.
  __nv_bfloat162 r0;
  rotate_pair<kVari>(o0x, o0y, cos_pos[0], sin_pos[0], &r0);
  __nv_bfloat162 r1;
  rotate_pair<kVari>(o1x, o1y, cos_pos[1], sin_pos[1], &r1);

  // `out_strides` are already expressed in bf16-pair units, so `pair_idx` indexes directly.
  const int64_t base =
    batch * out_strides.dim0 + pos * out_strides.dim1 + head * out_strides.dim2 + (int64_t)pair_idx;
  __nv_bfloat162* out_row = out + base;
  out_row[0] = r0;
  out_row[1] = r1;
}

bool is_aligned(const void* p, uintptr_t bytes) {
  return (reinterpret_cast<uintptr_t>(p) & (bytes - 1)) == 0;
}

}  // namespace

void qwen_image21_qk_norm_rope(at::Tensor x, at::Tensor weight, at::Tensor cos, at::Tensor sin,
                               double eps, int64_t variant, at::Tensor out) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() &&
                x.dim() == 4 && x.storage_offset() == 0,
              "qwen_image21_qk_norm_rope: x must be a contiguous CUDA bf16 [B,S,H,D] tensor");
  const int64_t batch = x.size(0);
  const int64_t seq_len = x.size(1);
  const int64_t heads = x.size(2);
  const int64_t head_dim = x.size(3);
  TORCH_CHECK(head_dim == 128, "qwen_image21_qk_norm_rope: only head_dim == 128 is supported, got ",
              head_dim);
  TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == at::kBFloat16 &&
                weight.numel() == head_dim && weight.storage_offset() == 0,
              "qwen_image21_qk_norm_rope: weight must be a contiguous CUDA bf16 [D] tensor");
  TORCH_CHECK(cos.is_cuda() && cos.is_contiguous() && cos.scalar_type() == at::kFloat &&
                cos.numel() == seq_len * (head_dim / 2) && cos.storage_offset() == 0,
              "qwen_image21_qk_norm_rope: cos must be a contiguous fp32 [S, D/2] tensor");
  TORCH_CHECK(sin.is_cuda() && sin.is_contiguous() && sin.scalar_type() == at::kFloat &&
                sin.numel() == seq_len * (head_dim / 2) && sin.storage_offset() == 0,
              "qwen_image21_qk_norm_rope: sin must be a contiguous fp32 [S, D/2] tensor");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.dim() == 4 &&
                out.sizes() == x.sizes() && out.stride(3) == 1,
              "qwen_image21_qk_norm_rope: out must be a dense CUDA bf16 [B,S,H,D] buffer of the "
              "same shape as x with unit innermost stride");
  TORCH_CHECK(is_aligned(x.const_data_ptr(), 4) && is_aligned(weight.const_data_ptr(), 4) &&
                is_aligned(cos.const_data_ptr(), 4) && is_aligned(sin.const_data_ptr(), 4) &&
                is_aligned(out.data_ptr(), 4),
              "qwen_image21_qk_norm_rope: inputs must be 4B-aligned");

  const int64_t rows = x.numel() / head_dim;
  if (rows == 0) {
    return;
  }

  // Pair-granular addressing: the outer strides have to be even, which holds for every
  // dense [B,S,H,D] buffer with D % 2 == 0.
  TORCH_CHECK(out.stride(0) % 2 == 0 && out.stride(1) % 2 == 0 && out.stride(2) % 2 == 0,
              "qwen_image21_qk_norm_rope: out outer strides must be even");

  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t grid = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const auto* x_ptr = reinterpret_cast<const __nv_bfloat162*>(x.const_data_ptr());
  const auto* w_ptr = reinterpret_cast<const __nv_bfloat162*>(weight.const_data_ptr());
  auto* out_ptr = reinterpret_cast<__nv_bfloat162*>(out.data_ptr());
  const OutStrides strides{out.stride(0) / 2, out.stride(1) / 2, out.stride(2) / 2};
  switch (variant) {
    case 1:
      qwen_image21_qk_norm_rope_kernel<1><<<grid, kWarpsPerBlock * 32, 0, stream>>>(
        x_ptr, w_ptr, cos.const_data_ptr<float>(), sin.const_data_ptr<float>(), out_ptr, strides,
        rows, (int)seq_len, (int)heads, (float)eps);
      break;
    case 2:
      qwen_image21_qk_norm_rope_kernel<2><<<grid, kWarpsPerBlock * 32, 0, stream>>>(
        x_ptr, w_ptr, cos.const_data_ptr<float>(), sin.const_data_ptr<float>(), out_ptr, strides,
        rows, (int)seq_len, (int)heads, (float)eps);
      break;
    case 3:
      qwen_image21_qk_norm_rope_kernel<3><<<grid, kWarpsPerBlock * 32, 0, stream>>>(
        x_ptr, w_ptr, cos.const_data_ptr<float>(), sin.const_data_ptr<float>(), out_ptr, strides,
        rows, (int)seq_len, (int)heads, (float)eps);
      break;
    case 4:
      qwen_image21_qk_norm_rope_kernel<4><<<grid, kWarpsPerBlock * 32, 0, stream>>>(
        x_ptr, w_ptr, cos.const_data_ptr<float>(), sin.const_data_ptr<float>(), out_ptr, strides,
        rows, (int)seq_len, (int)heads, (float)eps);
      break;
    default:
      qwen_image21_qk_norm_rope_kernel<0><<<grid, kWarpsPerBlock * 32, 0, stream>>>(
        x_ptr, w_ptr, cos.const_data_ptr<float>(), sin.const_data_ptr<float>(), out_ptr, strides,
        rows, (int)seq_len, (int)heads, (float)eps);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::models
