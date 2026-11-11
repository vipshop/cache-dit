// FLUX.3 Klein q/k RMSNorm + interleaved-pair RoPE.
//
// Numerics are pinned to the chain the compiled step runs:
//   n = rms_norm_fp32(x.float(), (128,), weight_fp32, eps)   # no bf16 rounding inside
//   o = n * cos + rotate(n) * sin                             # interleaved pairs, fp32
//   out = bf16(o)                                             # the only rounding
// `__fmul_rn` / `__fsub_rn` / `__fadd_rn` keep nvcc from contracting the rotation into an FMA,
// which would drop one rounding and make the result depend on the build.
//
// Layout. A 128-channel row is 16 lanes of 8 channels (16 bytes) each, so the load, the weight,
// the cos/sin table and the store are each one 128-bit transaction per lane, and four warps per
// block cover eight rows. This is the op that ncu flagged: inductor's fused version of the same
// chain runs 64 threads with two elements per thread and reaches 71.8% of DRAM peak (L2 hit
// 46.5%), the only elementwise kernel in the compiled step with headroom.
//
// The sum of squares follows ATen's `Reduce.cuh` leaf tree, because the mean is the only place
// where a different summation order would show up: the `vectorized` path sums four adjacent
// channels per leaf sequentially and then merges leaves in descending order. One lane holds two
// such leaves (channels 8*lid .. 8*lid+7), so the merge is: left-to-right inside each leaf, add
// the two leaves in-lane, then `shfl_down` over the 16 lanes with physical offsets 8/4/2/1.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "flux3_klein.h"

namespace cache_dit::models {
namespace {

constexpr int kHeadDim = 128;
constexpr int kWarpsPerBlock = 4;
constexpr int kLanesPerRow = 16;
constexpr int kPairsPerLane = 4;  // 4 bf16 pairs = 8 channels = 16 bytes
constexpr int kPairs = kHeadDim / 2;

__device__ __forceinline__ float leaf_sum(float a, float b, float c, float d) {
  // ATen's vectorized leaf: four adjacent channels, summed left to right.
  return ((a * a + b * b) + c * c) + d * d;
}

__global__ void flux3_klein_qk_norm_rope_kernel(const __nv_bfloat162* __restrict__ x,
                                                const float* __restrict__ weight,
                                                const float* __restrict__ cos_table,
                                                const float* __restrict__ sin_table,
                                                __nv_bfloat162* __restrict__ out, int64_t rows,
                                                int seq_len, int heads, float eps) {
  const int lane = threadIdx.x & 31;
  const int row_in_pair = lane >> 4;
  const int lid = lane & 15;
  const int warp = threadIdx.x >> 5;
  const int64_t row = (int64_t(blockIdx.x) * kWarpsPerBlock + warp) * 2 + row_in_pair;
  // A whole 16-lane group shares one row, so a group either runs entirely or not at all.
  const unsigned group_mask = (row_in_pair == 0) ? 0x0000FFFFu : 0xFFFF0000u;

  if (row >= rows) {
    return;
  }

  // Flat row order is ((b * S + s) * H + h), so the head is the fastest varying index.
  const int64_t head = row % heads;
  const int64_t pos = (row / heads) % seq_len;
  const __nv_bfloat162* x_row = x + row * kPairs;

  const uint4 xv = *reinterpret_cast<const uint4*>(x_row + lid * kPairsPerLane);
  const __nv_bfloat162* xq = reinterpret_cast<const __nv_bfloat162*>(&xv);

  const float2 e0 = __bfloat1622float2(xq[0]);
  const float2 e1 = __bfloat1622float2(xq[1]);
  const float2 e2 = __bfloat1622float2(xq[2]);
  const float2 e3 = __bfloat1622float2(xq[3]);
  float local = leaf_sum(e0.x, e0.y, e1.x, e1.y);
  local += leaf_sum(e2.x, e2.y, e3.x, e3.y);

#pragma unroll
  for (int offset = kLanesPerRow / 2; offset > 0; offset >>= 1) {
    const float other = __shfl_down_sync(group_mask, local, offset, kLanesPerRow);
    if (lid < offset) {
      local += other;
    }
  }
  const float total = __shfl_sync(group_mask, local, 0, kLanesPerRow);
  const float rstd = rsqrtf(total / float(kHeadDim) + eps);

  const float4 w0 = *reinterpret_cast<const float4*>(weight + lid * 8);
  const float4 w1 = *reinterpret_cast<const float4*>(weight + lid * 8 + 4);
  const float4 c0 = *reinterpret_cast<const float4*>(cos_table + pos * kHeadDim + lid * 8);
  const float4 c1 = *reinterpret_cast<const float4*>(cos_table + pos * kHeadDim + lid * 8 + 4);
  const float4 s0 = *reinterpret_cast<const float4*>(sin_table + pos * kHeadDim + lid * 8);
  const float4 s1 = *reinterpret_cast<const float4*>(sin_table + pos * kHeadDim + lid * 8 + 4);
  const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
  const float cv[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
  const float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};

  uint4 rv;
  __nv_bfloat162* rq = reinterpret_cast<__nv_bfloat162*>(&rv);
#pragma unroll
  for (int j = 0; j < kPairsPerLane; ++j) {
    const float2 e = __bfloat1622float2(xq[j]);
    // Normalize, scale by the fp32 weight, rotate the (real, imag) pair: fp32 throughout.
    const float re = __fmul_rn(__fmul_rn(e.x, rstd), wv[2 * j]);
    const float im = __fmul_rn(__fmul_rn(e.y, rstd), wv[2 * j + 1]);
    const float rotated_re = __fsub_rn(__fmul_rn(re, cv[2 * j]), __fmul_rn(im, sv[2 * j]));
    const float rotated_im = __fadd_rn(__fmul_rn(re, sv[2 * j + 1]), __fmul_rn(im, cv[2 * j + 1]));
    rq[j].x = __float2bfloat16_rn(rotated_re);
    rq[j].y = __float2bfloat16_rn(rotated_im);
  }

  *reinterpret_cast<uint4*>(out + row * kPairs + lid * kPairsPerLane) = rv;
}

bool is_aligned(const void* pointer, uintptr_t bytes) {
  return (reinterpret_cast<uintptr_t>(pointer) & (bytes - 1)) == 0;
}

}  // namespace

void flux3_klein_qk_norm_rope(at::Tensor x, at::Tensor weight, at::Tensor cos, at::Tensor sin,
                              double eps, at::Tensor out) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 4,
              "flux3_klein_qk_norm_rope: x must be a contiguous CUDA bf16 [B,S,H,D] tensor");
  const int64_t seq_len = x.size(1);
  const int64_t heads = x.size(2);
  const int64_t head_dim = x.size(3);
  TORCH_CHECK(head_dim == kHeadDim,
              "flux3_klein_qk_norm_rope: only head_dim == 128 is supported, "
              "got ",
              head_dim);
  TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == at::kFloat &&
                weight.numel() == head_dim && weight.storage_offset() == 0,
              "flux3_klein_qk_norm_rope: weight must be a contiguous CUDA fp32 [D] tensor");
  TORCH_CHECK(cos.is_cuda() && cos.is_contiguous() && cos.scalar_type() == at::kFloat &&
                cos.numel() == seq_len * head_dim && cos.storage_offset() == 0,
              "flux3_klein_qk_norm_rope: cos must be a contiguous fp32 [S, D] tensor");
  TORCH_CHECK(sin.is_cuda() && sin.is_contiguous() && sin.scalar_type() == at::kFloat &&
                sin.numel() == seq_len * head_dim && sin.storage_offset() == 0,
              "flux3_klein_qk_norm_rope: sin must be a contiguous fp32 [S, D] tensor");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.is_contiguous() &&
                out.sizes() == x.sizes(),
              "flux3_klein_qk_norm_rope: out must be a contiguous CUDA bf16 buffer with the shape "
              "of x");
  TORCH_CHECK(is_aligned(x.const_data_ptr(), 16) && is_aligned(weight.const_data_ptr(), 16) &&
                is_aligned(cos.const_data_ptr(), 16) && is_aligned(sin.const_data_ptr(), 16) &&
                is_aligned(out.data_ptr(), 16),
              "flux3_klein_qk_norm_rope: buffers must be 16-byte aligned for the uint4 path");

  const int64_t rows = x.numel() / head_dim;
  if (rows == 0) {
    return;
  }

  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  // Two rows per warp.
  const int64_t warp_count = (rows + 1) / 2;
  const int64_t grid = (warp_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
  flux3_klein_qk_norm_rope_kernel<<<grid, kWarpsPerBlock * 32, 0, stream>>>(
    reinterpret_cast<const __nv_bfloat162*>(x.const_data_ptr()), weight.const_data_ptr<float>(),
    cos.const_data_ptr<float>(), sin.const_data_ptr<float>(),
    reinterpret_cast<__nv_bfloat162*>(out.data_ptr()), rows, int(seq_len), int(heads), float(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::models
