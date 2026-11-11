// Qwen-Image 2.1 block modulation.
//
// Two entry points share one epilogue:
//   qwen_image21_ln_modulate      LayerNorm(elementwise_affine=False) + `1 + scale` + `tanh(gate)`
//   qwen_image21_modulate_affine  the same epilogue on an already normalized input
//
// The second exists because a bit-exact LayerNorm has to reproduce ATen's reduction tree;
// if the runtime self-check in `fused_ops` finds that it does not (another GPU or another
// torch build), the normalization stays in ATen and only the memory-bound epilogue is
// fused, which is still exact and still removes the two materialized `(B, S, D)` tensors
// that `_select_modulation_rows` would otherwise produce.
//
// Rounding points, matching `_modulate` exactly:
//   one_plus = bf16(1 + scale)      scale / gate rows are selected per token by the mask
//   x_mod    = bf16(x_norm * one_plus)
//   gate_out = bf16(tanh_f32(gate))
//
// Layout: one lane owns 16 bytes = 8 consecutive channels of every 128-channel group, and a
// warp covers two rows (lanes 0-15 and 16-31, each with its own shuffle mask). Both
// outputs, both modulation halves and the input are therefore 16 bytes per lane.
//
// Scope: batch 1. The row index is derived from `(warp, half)` and the modulation row from
// `(batch, mask)`, which is validated for a single sample; `fused_ops` keeps the eager
// expression for `B > 1` and the kernel rejects it instead of producing a wrong row mapping.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <optional>

#include "qwen_image21.h"

namespace cache_dit::models {
namespace {

// Eight warps per block: the Welford pass is a ~256-step serial chain per lane, so more
// resident warps per block is what hides it (four warps left the SM at ~30% occupancy).
constexpr int kWarpsPerBlock = 8;
// 16 lanes per row x 4 complex pairs = 8 channels per lane = 16 bytes per access.
constexpr int kLanesPerRow = 16;
constexpr int kPairsPerLane = 4;
// Largest running count a lane can reach: 8 elements per 128-channel group, i.e. D / 16.
constexpr int kMaxRecip = 256;

struct Welford {
  float mean;
  float sigma2;
  float count;
};

// `inv_count` is the precomputed reciprocal of the running count. `__frcp_rn(x)` is the
// IEEE correctly-rounded reciprocal, i.e. the same value as `1.0f / x` under
// `-prec-div=true`, so the numerics are unchanged; it just moves the division out of the
// dependency chain (the counts are 1, 2, 3, ... and are shared by every lane).
__device__ __forceinline__ Welford welford_online(float val, const Welford& curr, float inv_count) {
  const float count = curr.count + 1.0f;
  const float delta = val - curr.mean;
  const float new_mean = curr.mean + delta * inv_count;
  const float new_sigma2 = curr.sigma2 + delta * (val - new_mean);
  return Welford{new_mean, new_sigma2, count};
}

__device__ __forceinline__ Welford welford_combine(const Welford& a, const Welford& b) {
  const float count = a.count + b.count;
  if (count <= 0.0f) {
    return Welford{0.0f, 0.0f, 0.0f};
  }
  // ATen's `cuWelfordCombine` multiplies by `coef = 1 / count` instead of dividing, which
  // is a different fp32 rounding for non power-of-two counts.
  const float coef = 1.0f / count;
  const float delta = b.mean - a.mean;
  const float new_mean = a.mean + delta * b.count * coef;
  const float new_sigma2 = a.sigma2 + b.sigma2 + delta * delta * a.count * b.count * coef;
  return Welford{new_mean, new_sigma2, count};
}

__device__ __forceinline__ int select_param_row(int64_t batch, int64_t pos, const bool* mask,
                                                int param_rows) {
  if (mask == nullptr) {
    return (int)batch;
  }
  // Rows [0, B) come from the real timestep; the trailing row is the t = 0 row used by the
  // text and condition-image tokens.
  return mask[pos] ? (int)batch : param_rows - 1;
}

// `if constexpr` keeps the unused branch out of the emitted code instead of relying on the
// compiler to fold a constant condition.
template <bool kFuseNorm>
__global__ void qwen_image21_modulate_kernel(const __nv_bfloat162* __restrict__ x,
                                             const __nv_bfloat162* __restrict__ params,
                                             const bool* __restrict__ mask,
                                             __nv_bfloat162* __restrict__ x_mod,
                                             __nv_bfloat162* __restrict__ gate_tanh, int64_t rows,
                                             int seq_len, int dim, int param_rows, float eps) {
  const int lane = threadIdx.x & 31;
  const int row_in_pair = lane >> 4;
  const int lid = lane & 15;
  const int warp = threadIdx.x >> 5;
  const int64_t row = ((int64_t)blockIdx.x * kWarpsPerBlock + warp) * 2 + row_in_pair;
  // A whole 16-lane group shares one row, so the group is uniformly active or not.
  const unsigned group_mask = (row_in_pair == 0) ? 0x0000FFFFu : 0xFFFF0000u;

  // Reciprocals of 1..kMaxRecip, built once per block and before any early exit so the
  // barrier stays uniform. Every lane walks the same count sequence, so this turns the
  // per-element correctly-rounded division into a shared-memory hit.
  __shared__ float s_recip[kMaxRecip];
  if constexpr (kFuseNorm) {
    for (int i = threadIdx.x; i < kMaxRecip; i += blockDim.x) {
      s_recip[i] = __frcp_rn((float)(i + 1));
    }
    __syncthreads();
  }

  if (row >= rows) {
    return;
  }

  const int64_t pos = row % seq_len;
  const int64_t batch = row / seq_len;
  const int pairs = dim >> 1;
  const int groups = dim >> 7;  // 128 channels per 16-lane step
  const __nv_bfloat162* x_row = x + row * pairs;

  float mean = 0.0f;
  float rstd = 1.0f;
  if constexpr (kFuseNorm) {
    // One leaf per 8 consecutive channels, which is ATen's vectorised bf16 read; the tree
    // then runs over 16 leaves with offsets 8/4/2/1 between lanes of the same row.
    Welford wd{0.0f, 0.0f, 0.0f};
    int count = 0;
    for (int g = 0; g < groups; ++g) {
      const int base = (g << 6) + (lid << 2);
      // One 16-byte load for the whole leaf: four scalar pair loads here were what kept this
      // pass at half bandwidth.
      const uint4 xv = *reinterpret_cast<const uint4*>(x_row + base);
      const __nv_bfloat162* xq = reinterpret_cast<const __nv_bfloat162*>(&xv);
      const float2 a0 = __bfloat1622float2(xq[0]);
      const float2 a1 = __bfloat1622float2(xq[1]);
      const float2 b0 = __bfloat1622float2(xq[2]);
      const float2 b1 = __bfloat1622float2(xq[3]);
      wd = welford_online(a0.x, wd, s_recip[count++]);
      wd = welford_online(a0.y, wd, s_recip[count++]);
      wd = welford_online(a1.x, wd, s_recip[count++]);
      wd = welford_online(a1.y, wd, s_recip[count++]);
      wd = welford_online(b0.x, wd, s_recip[count++]);
      wd = welford_online(b0.y, wd, s_recip[count++]);
      wd = welford_online(b1.x, wd, s_recip[count++]);
      wd = welford_online(b1.y, wd, s_recip[count++]);
    }
    // Welford's merge is neither commutative nor associative in fp32, so this has to be a
    // down-tree broadcast from lane 0 rather than a butterfly.
#pragma unroll
    for (int offset = kLanesPerRow / 2; offset > 0; offset >>= 1) {
      Welford other;
      other.mean = __shfl_down_sync(group_mask, wd.mean, offset, kLanesPerRow);
      other.sigma2 = __shfl_down_sync(group_mask, wd.sigma2, offset, kLanesPerRow);
      other.count = __shfl_down_sync(group_mask, wd.count, offset, kLanesPerRow);
      if (lid < offset) {
        wd = welford_combine(wd, other);
      }
    }
    mean = __shfl_sync(group_mask, wd.mean, 0, kLanesPerRow);
    const float sigma2 = __shfl_sync(group_mask, wd.sigma2, 0, kLanesPerRow);
    rstd = rsqrtf(sigma2 / (float)dim + eps);
  }

  const int param_row = select_param_row(batch, pos, mask, param_rows);
  // One modulation row packs `scale | gate`, i.e. `dim` bf16 pairs holding 2 * dim channels.
  const __nv_bfloat162* params_row = params + (int64_t)param_row * dim;
  const __nv_bfloat162* scale_row = params_row;
  const __nv_bfloat162* gate_row = params_row + pairs;
  __nv_bfloat162* out_row = x_mod + row * pairs;
  __nv_bfloat162* gate_out_row = gate_tanh + row * pairs;
  const __nv_bfloat162 one = __float2bfloat162_rn(1.0f);

  for (int g = 0; g < groups; ++g) {
    const int base = (g << 6) + (lid << 2);
    const uint4 xv = *reinterpret_cast<const uint4*>(x_row + base);
    const uint4 sv = *reinterpret_cast<const uint4*>(scale_row + base);
    const uint4 gv = *reinterpret_cast<const uint4*>(gate_row + base);
    const __nv_bfloat162* xq = reinterpret_cast<const __nv_bfloat162*>(&xv);
    const __nv_bfloat162* sq = reinterpret_cast<const __nv_bfloat162*>(&sv);
    const __nv_bfloat162* gq = reinterpret_cast<const __nv_bfloat162*>(&gv);
    uint4 rv;
    uint4 rgate;
    __nv_bfloat162* rq = reinterpret_cast<__nv_bfloat162*>(&rv);
    __nv_bfloat162* rgq = reinterpret_cast<__nv_bfloat162*>(&rgate);
#pragma unroll
    for (int j = 0; j < kPairsPerLane; ++j) {
      // Normalisation is fp32 with one bf16 rounding, exactly like ATen's layer-norm
      // epilogue; the modulation is then two native bf16 roundings (`1 + scale`, then the
      // product), so `__hfma2` is not usable here.
      __nv_bfloat162 normed;
      if constexpr (kFuseNorm) {
        const float2 xf = __bfloat1622float2(xq[j]);
        normed.x = __float2bfloat16_rn((xf.x - mean) * rstd);
        normed.y = __float2bfloat16_rn((xf.y - mean) * rstd);
      } else {
        normed = xq[j];
      }
      rq[j] = __hmul2(normed, __hadd2(sq[j], one));

      // tanh only exists in fp32 here: it has to match torch's fp32 `tanh` bit for bit.
      const float2 gvf = __bfloat1622float2(gq[j]);
      rgq[j].x = __float2bfloat16_rn(tanhf(gvf.x));
      rgq[j].y = __float2bfloat16_rn(tanhf(gvf.y));
    }
    *reinterpret_cast<uint4*>(out_row + base) = rv;
    *reinterpret_cast<uint4*>(gate_out_row + base) = rgate;
  }
}

struct Common {
  int64_t rows;
  int64_t seq_len;
  int64_t dim;
  int64_t param_rows;
  const bool* mask;
};

Common check_and_common(const at::Tensor& x, const at::Tensor& params,
                        const std::optional<at::Tensor>& mask, at::Tensor& x_mod,
                        at::Tensor& gate_tanh, const char* op) {
  // A non-zero storage offset is fine: `data_ptr()` already includes it and the kernel only
  // needs a dense layout plus aligned 16-byte vector loads.
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 3,
              op, ": x must be a contiguous CUDA bf16 [B,S,D] tensor");
  const int64_t dim = x.size(2);
  TORCH_CHECK(dim % 128 == 0, op, ": D must be a multiple of 128, got ", dim);
  TORCH_CHECK(dim <= kMaxRecip * 16, op, ": D must not exceed ", kMaxRecip * 16,
              " so the reciprocal table covers every running count");
  TORCH_CHECK(params.is_cuda() && params.scalar_type() == at::kBFloat16 && params.is_contiguous() &&
                params.dim() == 2 && params.size(1) == 2 * dim && params.storage_offset() == 0,
              op, ": params must be a contiguous CUDA bf16 [R, 2*D] tensor");
  TORCH_CHECK(x_mod.is_cuda() && x_mod.scalar_type() == at::kBFloat16 && x_mod.sizes() == x.sizes(),
              op, ": x_mod must be a bf16 [B,S,D] buffer");
  TORCH_CHECK(gate_tanh.is_cuda() && gate_tanh.scalar_type() == at::kBFloat16 &&
                gate_tanh.sizes() == x.sizes(),
              op, ": gate_tanh must be a bf16 [B,S,D] buffer");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(x.const_data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(params.const_data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(x_mod.data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(gate_tanh.data_ptr()) & 15) == 0,
              op, ": buffers must be 16-byte aligned for the uint4 path");
  TORCH_CHECK(x.size(0) == 1, op,
              ": only batch 1 is supported by the 16-byte path; `fused_ops` keeps the eager "
              "expression for larger batches instead of mis-mapping modulation rows");
  const int64_t param_rows = params.size(0);
  const int64_t batch = x.size(0);
  TORCH_CHECK(param_rows == batch || param_rows == batch + 1, op,
              ": params rows must be B or B+1 (got ", param_rows, " for B=", batch, ")");

  const bool* mask_ptr = nullptr;
  if (mask.has_value()) {
    const at::Tensor& m = mask.value();
    TORCH_CHECK(m.is_cuda() && m.scalar_type() == at::kBool && m.is_contiguous() &&
                  m.numel() == x.size(1) && m.storage_offset() == 0,
                op, ": target_token_mask must be a contiguous CUDA bool [S] tensor");
    TORCH_CHECK(param_rows == batch + 1, op, ": a target_token_mask requires B+1 modulation rows");
    mask_ptr = m.const_data_ptr<bool>();
  }
  return Common{x.size(0) * x.size(1), x.size(1), dim, param_rows, mask_ptr};
}

}  // namespace

void qwen_image21_ln_modulate(at::Tensor x, at::Tensor params,
                              const std::optional<at::Tensor>& target_token_mask, double eps,
                              at::Tensor x_mod, at::Tensor gate_tanh) {
  Common c =
    check_and_common(x, params, target_token_mask, x_mod, gate_tanh, "qwen_image21_ln_modulate");
  if (c.rows == 0) {
    return;
  }
  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t warp_count = (c.rows + 1) / 2;  // two rows per warp
  const int64_t grid = (warp_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
  // One warp per two rows, so the block is sized in warps - not in row lanes.
  qwen_image21_modulate_kernel<true><<<grid, kWarpsPerBlock * 32, 0, stream>>>(
    reinterpret_cast<const __nv_bfloat162*>(x.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(params.const_data_ptr()), c.mask,
    reinterpret_cast<__nv_bfloat162*>(x_mod.data_ptr()),
    reinterpret_cast<__nv_bfloat162*>(gate_tanh.data_ptr()), c.rows, (int)c.seq_len, (int)c.dim,
    (int)c.param_rows, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qwen_image21_modulate_affine(at::Tensor x_normed, at::Tensor params,
                                  const std::optional<at::Tensor>& target_token_mask,
                                  at::Tensor x_mod, at::Tensor gate_tanh) {
  Common c = check_and_common(x_normed, params, target_token_mask, x_mod, gate_tanh,
                              "qwen_image21_modulate_affine");
  if (c.rows == 0) {
    return;
  }
  const c10::cuda::CUDAGuard guard(x_normed.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t warp_count = (c.rows + 1) / 2;
  const int64_t grid = (warp_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
  qwen_image21_modulate_kernel<false><<<grid, kWarpsPerBlock * 32, 0, stream>>>(
    reinterpret_cast<const __nv_bfloat162*>(x_normed.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(params.const_data_ptr()), c.mask,
    reinterpret_cast<__nv_bfloat162*>(x_mod.data_ptr()),
    reinterpret_cast<__nv_bfloat162*>(gate_tanh.data_ptr()), c.rows, (int)c.seq_len, (int)c.dim,
    (int)c.param_rows, 0.0f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::models
