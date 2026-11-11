// FLUX.2-Klein block modulation, with the preceding residual add folded in.
//
// Eager reference (`diffusers/models/transformers/transformer_flux2.py:855-856`, `:929-934`,
// `:947-951`, `:957-961`):
//   out    = (1 + scale) * layer_norm(x) + shift
//   hidden = hidden + gate * delta                        # when the block has a residual
// `LayerNorm` there is `elementwise_affine=False`, and every arithmetic step is a bf16 tensor
// op, i.e. it rounds to bf16 once per step.
//
// Why a kernel: ATen's layer norm reads `x` twice (a statistics pass and an apply pass) and
// inductor keeps that shape, so the chain costs three full reads plus one write per call. A
// 3072-channel bf16 row is 6 KiB, so one warp can hold a whole row while it reduces and the
// chain becomes one read plus one write. The residual variant additionally removes the read
// that the separate `hidden + gate * delta` pass performs, which is where most of the traffic
// is: `_QwenImage21` measured the same shape of win.
//
// Numerics: the modulation chain is reproduced exactly - `__hmul2`/`__hadd2` round to bf16
// once per op, like the eager tensor ops (`__hfma2` would drop a rounding). The normalisation
// is fp32 with one bf16 rounding, like ATen's epilogue. Only the reduction tree is not
// bit-identical: with the same row the modulation is `torch.equal`, and the residual
// deviation is the reduction's, measured at rel-L2 1.4e-5 (one bf16 ulp) for both entry
// points, so the operators are gated at 1e-5 and by end-to-end PSNR.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "flux2_klein.h"
#include "flux2_klein_common.cuh"

namespace cache_dit::models {
namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kLanesPerRow = 32;
constexpr int kChannelsPerLane = 4;  // 8 bytes per lane per 128-channel group
constexpr int kChannelsPerGroup = kLanesPerRow * kChannelsPerLane;
constexpr int kMaxGroups = 32;  // D up to 4096, held in registers

// One warp owns one row. The row is kept in registers across the two passes so the input is
// read from memory exactly once.
template <bool kFuseResidual>
__global__ void flux2_klein_ln_modulate_kernel(
  const __nv_bfloat162* __restrict__ x, const __nv_bfloat162* __restrict__ gate,
  const __nv_bfloat162* __restrict__ delta, const __nv_bfloat162* __restrict__ one_plus,
  const __nv_bfloat162* __restrict__ shift, __nv_bfloat162* __restrict__ out,
  __nv_bfloat162* __restrict__ hidden_out, int64_t rows, int seq_len, int groups, int dim,
  int modulation_rows, float eps) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t row = (int64_t)blockIdx.x * kWarpsPerBlock + warp;
  if (row >= rows) {
    return;
  }

  const int64_t row_base = row * (dim >> 1);
  const int64_t mod_base = (modulation_rows == 1 ? 0 : (row / seq_len) * (dim >> 1));

  uint2 held[kMaxGroups];
#pragma unroll
  for (int g = 0; g < kMaxGroups; ++g) {
    if (g < groups) {
      held[g] = *reinterpret_cast<const uint2*>(x + row_base +
                                                (int64_t)g * (kChannelsPerGroup >> 1) + lane * 2);
    }
  }

  if constexpr (kFuseResidual) {
    __nv_bfloat162* hidden_row = hidden_out + row_base;
#pragma unroll
    for (int g = 0; g < kMaxGroups; ++g) {
      if (g < groups) {
        const int64_t offset = (int64_t)g * (kChannelsPerGroup >> 1) + lane * 2;
        const uint2 gv = *reinterpret_cast<const uint2*>(gate + mod_base + offset);
        const uint2 dv = *reinterpret_cast<const uint2*>(delta + row_base + offset);
        const __nv_bfloat162* gq = reinterpret_cast<const __nv_bfloat162*>(&gv);
        const __nv_bfloat162* dq = reinterpret_cast<const __nv_bfloat162*>(&dv);
        __nv_bfloat162* hq = reinterpret_cast<__nv_bfloat162*>(&held[g]);
        // Roundings: bf16 product, then bf16 add - the eager `hidden + gate * delta`. The
        // product has to be materialised as bf16 first, or nvcc contracts the pair into one
        // bf16x2 FMA and drops a rounding.
        hq[0] = bf16_mul_add_rounded(gq[0], dq[0], hq[0]);
        hq[1] = bf16_mul_add_rounded(gq[1], dq[1], hq[1]);
        // The rounded value is what the normalisation must see, so this store is part of the
        // contract, not a convenience.
        *reinterpret_cast<uint2*>(hidden_row + offset) = held[g];
      }
    }
  }

  float sum = 0.0f;
  float sum_sq = 0.0f;
#pragma unroll
  for (int g = 0; g < kMaxGroups; ++g) {
    if (g < groups) {
      const __nv_bfloat162* q = reinterpret_cast<const __nv_bfloat162*>(&held[g]);
      const float2 a = __bfloat1622float2(q[0]);
      const float2 b = __bfloat1622float2(q[1]);
      sum += a.x;
      sum += a.y;
      sum += b.x;
      sum += b.y;
      sum_sq += a.x * a.x;
      sum_sq += a.y * a.y;
      sum_sq += b.x * b.x;
      sum_sq += b.y * b.y;
    }
  }
  // Down-tree broadcast: every lane ends with the whole row's totals, so `mean` and `var` are
  // single-valued (a butterfly would leave each lane with a different partial sum).
#pragma unroll
  for (int offset = kLanesPerRow / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset, kLanesPerRow);
    sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, offset, kLanesPerRow);
  }
  sum = __shfl_sync(0xffffffffu, sum, 0, kLanesPerRow);
  sum_sq = __shfl_sync(0xffffffffu, sum_sq, 0, kLanesPerRow);

  const float inv = 1.0f / (float)dim;
  const float mean = sum * inv;
  const float variance = fmaxf(sum_sq * inv - mean * mean, 0.0f);
  const float rstd = rsqrtf(variance + eps);

  __nv_bfloat162* out_row = out + row_base;
#pragma unroll
  for (int g = 0; g < kMaxGroups; ++g) {
    if (g < groups) {
      const int64_t offset = (int64_t)g * (kChannelsPerGroup >> 1) + lane * 2;
      const uint2 sv = *reinterpret_cast<const uint2*>(one_plus + mod_base + offset);
      const uint2 hv = *reinterpret_cast<const uint2*>(shift + mod_base + offset);
      const __nv_bfloat162* sq = reinterpret_cast<const __nv_bfloat162*>(&sv);
      const __nv_bfloat162* hq = reinterpret_cast<const __nv_bfloat162*>(&hv);
      const __nv_bfloat162* xq = reinterpret_cast<const __nv_bfloat162*>(&held[g]);
      uint2 rv;
      __nv_bfloat162* rq = reinterpret_cast<__nv_bfloat162*>(&rv);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const float2 e = __bfloat1622float2(xq[j]);
        // Rounding 1: the normalised value, as ATen's bf16 epilogue writes it.
        const __nv_bfloat162 normed = {__float2bfloat16_rn((e.x - mean) * rstd),
                                       __float2bfloat16_rn((e.y - mean) * rstd)};
        // Roundings 2 and 3: the bf16 modulate multiply and the bf16 shift add.
        rq[j] = bf16_mul_add_rounded(normed, sq[j], hq[j]);
      }
      *reinterpret_cast<uint2*>(out_row + offset) = rv;
    }
  }
}

struct Common {
  int64_t rows;
  int seq_len;
  int groups;
  int dim;
  int modulation_rows;
};

void check_bf16(const at::Tensor& t, const char* name, const char* op) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous(), op, ": ", name,
              " must be a contiguous CUDA bf16 tensor");
}

Common check_common(const at::Tensor& x, const at::Tensor& one_plus, const at::Tensor& shift,
                    const char* op) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 3,
              op, ": x must be a contiguous CUDA bf16 [B,S,D] tensor");
  const int64_t dim = x.size(2);
  TORCH_CHECK(dim % kChannelsPerGroup == 0 && dim > 0, op, ": D must be a multiple of ",
              kChannelsPerGroup, ", got ", dim);
  TORCH_CHECK(dim <= kMaxGroups * kChannelsPerGroup, op, ": D must not exceed ",
              kMaxGroups * kChannelsPerGroup, ", got ", dim);
  check_bf16(one_plus, "one_plus", op);
  check_bf16(shift, "shift", op);
  // A non-zero storage offset is fine: the modulation rows are chunks of one narrow tensor and
  // `data_ptr()` already includes the offset.
  TORCH_CHECK(one_plus.dim() == 2 && one_plus.size(1) == dim, op,
              ": one_plus must be a contiguous bf16 [R, D] tensor");
  TORCH_CHECK(shift.sizes() == one_plus.sizes(), op,
              ": shift must have the same [R, D] shape as one_plus");
  const int64_t modulation_rows = one_plus.size(0);
  TORCH_CHECK(modulation_rows == 1 || modulation_rows == x.size(0), op,
              ": one_plus rows must be 1 or B, got ", modulation_rows, " for B=", x.size(0));
  TORCH_CHECK((reinterpret_cast<uintptr_t>(x.const_data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(one_plus.const_data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(shift.const_data_ptr()) & 15) == 0,
              op, ": buffers must be 16-byte aligned for the vector path");
  return Common{x.size(0) * x.size(1), (int)x.size(1), (int)(dim / kChannelsPerGroup), (int)dim,
                (int)modulation_rows};
}

}  // namespace

void flux2_klein_ln_modulate(at::Tensor x, at::Tensor one_plus, at::Tensor shift, double eps,
                             at::Tensor out) {
  const Common c = check_common(x, one_plus, shift, "flux2_klein_ln_modulate");
  check_bf16(out, "out", "flux2_klein_ln_modulate");
  TORCH_CHECK(out.sizes() == x.sizes(), "flux2_klein_ln_modulate: out must match x");
  if (c.rows == 0) {
    return;
  }
  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t grid = (c.rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  flux2_klein_ln_modulate_kernel<false><<<grid, kWarpsPerBlock * 32, 0, stream>>>(
    reinterpret_cast<const __nv_bfloat162*>(x.const_data_ptr()), nullptr, nullptr,
    reinterpret_cast<const __nv_bfloat162*>(one_plus.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(shift.const_data_ptr()),
    reinterpret_cast<__nv_bfloat162*>(out.data_ptr()), nullptr, c.rows, c.seq_len, c.groups, c.dim,
    c.modulation_rows, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void flux2_klein_residual_ln_modulate(at::Tensor hidden, at::Tensor gate, at::Tensor delta,
                                      at::Tensor one_plus, at::Tensor shift, double eps,
                                      at::Tensor hidden_out, at::Tensor mod_out) {
  const char* op = "flux2_klein_residual_ln_modulate";
  const Common c = check_common(hidden, one_plus, shift, op);
  check_bf16(gate, "gate", op);
  check_bf16(delta, "delta", op);
  check_bf16(hidden_out, "hidden_out", op);
  check_bf16(mod_out, "mod_out", op);
  TORCH_CHECK(delta.sizes() == hidden.sizes(), op, ": delta must match hidden");
  TORCH_CHECK(gate.sizes() == one_plus.sizes(), op,
              ": gate must have the same [R, D] shape as one_plus");
  TORCH_CHECK(hidden_out.sizes() == hidden.sizes() && mod_out.sizes() == hidden.sizes(), op,
              ": outputs must match hidden");
  if (c.rows == 0) {
    return;
  }
  const c10::cuda::CUDAGuard guard(hidden.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t grid = (c.rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  flux2_klein_ln_modulate_kernel<true><<<grid, kWarpsPerBlock * 32, 0, stream>>>(
    reinterpret_cast<const __nv_bfloat162*>(hidden.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(gate.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(delta.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(one_plus.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(shift.const_data_ptr()),
    reinterpret_cast<__nv_bfloat162*>(mod_out.data_ptr()),
    reinterpret_cast<__nv_bfloat162*>(hidden_out.data_ptr()), c.rows, c.seq_len, c.groups, c.dim,
    c.modulation_rows, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::models
