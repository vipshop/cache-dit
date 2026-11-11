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
// Layout. ATen reduces this shape with 32 leaves of 4 consecutive channels each, so the
// bit-exact mean has to come from that exact leaf tree. That does not force 8-byte accesses
// though: one lane loads 16 bytes = 8 consecutive channels, which is exactly two ATen
// leaves, and the tree is rebuilt with physical shuffle offsets 8/4/2/1 (leaf offsets
// 16/8/4/2) plus one final in-lane merge of the two leaves (leaf offset 1). A warp then
// covers two rows (lanes 0-15 and 16-31), so both the load, the weight, the cos/sin table
// and the store stay at 16 bytes per lane.
//
// `fused_ops.probe_rope_variant()` picks the FMA contraction pattern of the rotation once
// per process, verified with `torch.equal`.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "qwen_image21.h"

namespace cache_dit::models {
namespace {

constexpr int kWarpsPerBlock = 4;
constexpr int kLanesPerRow = 16;
constexpr int kPairsPerLane = 4;  // 4 complex pairs = 8 channels = 16 bytes
constexpr int kPairs = 64;

// torch's complex64 multiply is compiled with default nvcc contraction, so which product
// gets folded into the FMA depends on the build; a wrong choice costs one bf16 ulp.
template <int kVari>
__device__ __forceinline__ void rotate_pair(float re, float im, float c, float s,
                                            __nv_bfloat162* dst) {
  // `if constexpr`: only the selected contraction pattern is emitted, so the dead branches
  // can never influence instruction selection in the instantiation that actually ships.
  float out_re;
  float out_im;
  if constexpr (kVari == 0) {
    out_re = __fsub_rn(__fmul_rn(re, c), __fmul_rn(im, s));
    out_im = __fadd_rn(__fmul_rn(re, s), __fmul_rn(im, c));
  } else if constexpr (kVari == 1) {
    out_re = __fmaf_rn(re, c, -__fmul_rn(im, s));
    out_im = __fmaf_rn(re, s, __fmul_rn(im, c));
  } else if constexpr (kVari == 2) {
    out_re = __fmaf_rn(-im, s, __fmul_rn(re, c));
    out_im = __fmaf_rn(im, c, __fmul_rn(re, s));
  } else if constexpr (kVari == 3) {
    out_re = __fmaf_rn(re, c, -__fmul_rn(im, s));
    out_im = __fmaf_rn(im, c, __fmul_rn(re, s));
  } else {
    out_re = __fmaf_rn(-im, s, __fmul_rn(re, c));
    out_im = __fmaf_rn(re, s, __fmul_rn(im, c));
  }
  dst->x = __float2bfloat16_rn(out_re);
  dst->y = __float2bfloat16_rn(out_im);
}

struct OutStrides {
  int64_t dim0;
  int64_t dim1;
  int64_t dim2;
};

// `out` is addressed through explicit strides so the kernel can write the target tail of a
// persistent [prefix | target] KV buffer without an intermediate copy.
template <int kVari>
__global__ void qwen_image21_qk_norm_rope_kernel(
  const __nv_bfloat162* __restrict__ x, const __nv_bfloat162* __restrict__ weight,
  const float* __restrict__ cos, const float* __restrict__ sin, __nv_bfloat162* __restrict__ out,
  OutStrides out_strides, int64_t rows, int seq_len, int heads, float eps) {
  const int lane = threadIdx.x & 31;
  const int row_in_pair = lane >> 4;
  const int lid = lane & 15;
  const int warp = threadIdx.x >> 5;
  const int64_t row = ((int64_t)blockIdx.x * kWarpsPerBlock + warp) * 2 + row_in_pair;
  // A whole 16-lane group shares one row, so a group either runs entirely or not at all.
  const unsigned group_mask = (row_in_pair == 0) ? 0x0000FFFFu : 0xFFFF0000u;

  if (row >= rows) {
    // Both groups of a warp share the shuffle below, so this has to be uniform per group:
    // a whole 16-lane group maps to one row.
    return;
  }

  // Flat row order is ((b * S + s) * H + h), so the head is the fastest varying index.
  const int64_t head = row % heads;
  const int64_t pos = (row / heads) % seq_len;
  const __nv_bfloat162* x_row = x + row * kPairs;

  // 16 bytes: pairs [4*lid, 4*lid + 4), i.e. channels [8*lid, 8*lid + 8).
  uint4 xv = *reinterpret_cast<const uint4*>(x_row + (lid * kPairsPerLane));
  const __nv_bfloat162* xq = reinterpret_cast<const __nv_bfloat162*>(&xv);

  // Two ATen leaves per lane: channels [8*lid, 8*lid + 4) and [8*lid + 4, 8*lid + 8).
  const float2 a0 = __bfloat1622float2(xq[0]);
  const float2 a1 = __bfloat1622float2(xq[1]);
  const float2 b0 = __bfloat1622float2(xq[2]);
  const float2 b1 = __bfloat1622float2(xq[3]);
  // Squares are fp32, then summed in ATen's order: left to right inside the leaf.
  float ss_a = a0.x * a0.x;
  ss_a += a0.y * a0.y;
  ss_a += a1.x * a1.x;
  ss_a += a1.y * a1.y;
  float ss_b = b0.x * b0.x;
  ss_b += b0.y * b0.y;
  ss_b += b1.x * b1.x;
  ss_b += b1.y * b1.y;

  // Leaf tree: offsets 16/8/4/2 are physical 8/4/2/1 between lanes of the same row.
#pragma unroll
  for (int leaf_offset = 16; leaf_offset > 1; leaf_offset >>= 1) {
    const float oa = __shfl_down_sync(group_mask, ss_a, leaf_offset >> 1, kLanesPerRow);
    const float ob = __shfl_down_sync(group_mask, ss_b, leaf_offset >> 1, kLanesPerRow);
    if (lid < (leaf_offset >> 1)) {
      ss_a += oa;
      ss_b += ob;
    }
  }
  if (lid == 0) {
    ss_a += ss_b;  // leaf offset 1: the two leaves this lane holds
  }
  const float total = __shfl_sync(group_mask, ss_a, 0, kLanesPerRow);
  const float rstd = rsqrtf(total / (float)(kPairs * 2) + eps);

  const uint4 wv = *reinterpret_cast<const uint4*>(weight + lid * kPairsPerLane);
  const __nv_bfloat162* wq = reinterpret_cast<const __nv_bfloat162*>(&wv);
  const float4 c4 = *reinterpret_cast<const float4*>(cos + pos * kPairs + lid * kPairsPerLane);
  const float4 s4 = *reinterpret_cast<const float4*>(sin + pos * kPairs + lid * kPairsPerLane);
  const float cs[4] = {c4.x, c4.y, c4.z, c4.w};
  const float sn[4] = {s4.x, s4.y, s4.z, s4.w};

  uint4 rv;
  __nv_bfloat162* rq = reinterpret_cast<__nv_bfloat162*>(&rv);
#pragma unroll
  for (int j = 0; j < kPairsPerLane; ++j) {
    const float2 e = __bfloat1622float2(xq[j]);
    const float2 w = __bfloat1622float2(wq[j]);
    // Rounding 1: bf16(x * rstd). Rounding 2: bf16(n * weight).
    const float o_re = __bfloat162float(
      __float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(e.x * rstd)) * w.x));
    const float o_im = __bfloat162float(
      __float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(e.y * rstd)) * w.y));
    // Rounding 3: the rotation result.
    rotate_pair<kVari>(o_re, o_im, cs[j], sn[j], rq + j);
  }

  // `out_strides` are already expressed in bf16-pair units.
  const int64_t base = (row / ((int64_t)heads * seq_len)) * out_strides.dim0 +
                       pos * out_strides.dim1 + head * out_strides.dim2 +
                       (int64_t)lid * kPairsPerLane;
  *reinterpret_cast<uint4*>(out + base) = rv;
}

bool is_aligned(const void* p, uintptr_t bytes) {
  return (reinterpret_cast<uintptr_t>(p) & (bytes - 1)) == 0;
}

}  // namespace

void qwen_image21_qk_norm_rope(at::Tensor x, at::Tensor weight, at::Tensor cos, at::Tensor sin,
                               double eps, int64_t variant, at::Tensor out) {
  // A non-zero storage offset is fine: `data_ptr()` already includes it and the kernel only
  // needs a dense layout plus aligned 16-byte vector loads.
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 4,
              "qwen_image21_qk_norm_rope: x must be a contiguous CUDA bf16 [B,S,H,D] tensor");
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
  TORCH_CHECK(is_aligned(x.const_data_ptr(), 16) && is_aligned(weight.const_data_ptr(), 16) &&
                is_aligned(cos.const_data_ptr(), 16) && is_aligned(sin.const_data_ptr(), 16) &&
                is_aligned(out.data_ptr(), 16),
              "qwen_image21_qk_norm_rope: buffers must be 16-byte aligned for the uint4 path");

  // The store is a 16-byte vector, so every outer stride has to keep it aligned.
  TORCH_CHECK(out.stride(0) % 8 == 0 && out.stride(1) % 8 == 0 && out.stride(2) % 8 == 0,
              "qwen_image21_qk_norm_rope: out outer strides must be multiples of 8 elements");

  const int64_t rows = x.numel() / head_dim;
  if (rows == 0) {
    return;
  }

  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  // Two rows per warp.
  const int64_t warp_count = (rows + 1) / 2;
  const int64_t grid = (warp_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const auto* x_ptr = reinterpret_cast<const __nv_bfloat162*>(x.const_data_ptr());
  const auto* w_ptr = reinterpret_cast<const __nv_bfloat162*>(weight.const_data_ptr());
  auto* out_ptr = reinterpret_cast<__nv_bfloat162*>(out.data_ptr());
  const OutStrides strides{out.stride(0) / 2, out.stride(1) / 2, out.stride(2) / 2};
#define CACHE_DIT_QWEN21_ROPE_LAUNCH(VARI)                                                    \
  qwen_image21_qk_norm_rope_kernel<VARI><<<grid, kWarpsPerBlock * 32, 0, stream>>>(           \
    x_ptr, w_ptr, cos.const_data_ptr<float>(), sin.const_data_ptr<float>(), out_ptr, strides, \
    rows, (int)seq_len, (int)heads, (float)eps)
  switch (variant) {
    case 1:
      CACHE_DIT_QWEN21_ROPE_LAUNCH(1);
      break;
    case 2:
      CACHE_DIT_QWEN21_ROPE_LAUNCH(2);
      break;
    case 3:
      CACHE_DIT_QWEN21_ROPE_LAUNCH(3);
      break;
    case 4:
      CACHE_DIT_QWEN21_ROPE_LAUNCH(4);
      break;
    default:
      CACHE_DIT_QWEN21_ROPE_LAUNCH(0);
      break;
  }
#undef CACHE_DIT_QWEN21_ROPE_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::models
