// Qwen-Image 2.1 block modulation: `x_norm * one_plus` with a per-token row select.
//
// `one_plus` (the `1 + scale` row) is constant across tokens: it only depends on the sample
// and on whether the token is a target-image or a `t = 0` token. The eager chain evaluated
// `bf16(1 + scale)` and `tanh(gate)` once per token - 4096x more often than needed - and
// materialised a second `(B, S, D)` tensor for the gate.
//
// Folding that out is the point of this operator: `fused_ops` prepares the two `(R, D)` rows
// in torch (`1 + scale` and `gate.tanh()`, both elementwise and therefore exact) and the
// kernel only normalises and writes `x_mod`.
//
// Entry point: `qwen_image21_ln_modulate`, i.e. LayerNorm(elementwise_affine=False) followed
// by the row-selected modulation. Measured against the eager expression it lands at rel-L2
// 7.9e-06 at D=4096 (a single bf16 ulp), so there is exactly one implementation and no runtime
// selection between alternatives.
//
// Layout: one lane owns 16 bytes = 8 consecutive channels of every 128-channel group, and a
// warp covers two rows (lanes 0-15 and 16-31, each with its own shuffle mask).
//
// Scope: batch 1 (the shape this model always runs); `fused_ops` keeps the eager expression
// for larger batches and the kernel rejects them instead of mis-mapping rows.

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
constexpr int kLanesPerRow = 16;
constexpr int kPairsPerLane = 4;
// Largest running count a lane can reach: 8 elements per 128-channel group, i.e. D / 16.
constexpr int kMaxRecip = 256;

struct Welford {
  float mean;
  float sigma2;
  float count;
};

// `inv_count` is the precomputed reciprocal of the running count. `__frcp_rn(x)` is the IEEE
// correctly-rounded reciprocal, i.e. the same value as `1.0f / x` under `-prec-div=true`.
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
  // ATen's `cuWelfordCombine` multiplies by `coef = 1 / count` instead of dividing, which is
  // a different fp32 rounding for non power-of-two counts.
  const float coef = 1.0f / count;
  const float delta = b.mean - a.mean;
  const float new_mean = a.mean + delta * b.count * coef;
  const float new_sigma2 = a.sigma2 + b.sigma2 + delta * delta * a.count * b.count * coef;
  return Welford{new_mean, new_sigma2, count};
}

__device__ __forceinline__ int select_row(int64_t pos, const bool* mask, int rows) {
  if (mask == nullptr) {
    return 0;  // batch 1 only: the sample row is row 0
  }
  // Rows [0, B) come from the real timestep; the trailing row is the t = 0 row used by the
  // text and condition-image tokens.
  return mask[pos] ? 0 : rows - 1;
}

template <bool kFuseNorm>
__global__ void qwen_image21_modulate_kernel(const __nv_bfloat162* __restrict__ x,
                                             const __nv_bfloat162* __restrict__ one_plus,
                                             const bool* __restrict__ mask,
                                             __nv_bfloat162* __restrict__ x_mod, int64_t rows,
                                             int seq_len, int dim, int one_plus_rows, float eps) {
  const int lane = threadIdx.x & 31;
  const int row_in_pair = lane >> 4;
  const int lid = lane & 15;
  const int warp = threadIdx.x >> 5;
  const int64_t row = ((int64_t)blockIdx.x * kWarpsPerBlock + warp) * 2 + row_in_pair;
  // A whole 16-lane group shares one row, so the group is uniformly active or not.
  const unsigned group_mask = (row_in_pair == 0) ? 0x0000FFFFu : 0xFFFF0000u;

  // Reciprocals of 1..kMaxRecip, built once per block and before any early exit so the
  // barrier stays uniform.
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

  const __nv_bfloat162* one_plus_row =
    one_plus + (int64_t)select_row(pos, mask, one_plus_rows) * pairs;
  __nv_bfloat162* out_row = x_mod + row * pairs;

  for (int g = 0; g < groups; ++g) {
    const int base = (g << 6) + (lid << 2);
    const uint4 xv = *reinterpret_cast<const uint4*>(x_row + base);
    const uint4 sv = *reinterpret_cast<const uint4*>(one_plus_row + base);
    const __nv_bfloat162* xq = reinterpret_cast<const __nv_bfloat162*>(&xv);
    const __nv_bfloat162* sq = reinterpret_cast<const __nv_bfloat162*>(&sv);
    uint4 rv;
    __nv_bfloat162* rq = reinterpret_cast<__nv_bfloat162*>(&rv);
#pragma unroll
    for (int j = 0; j < kPairsPerLane; ++j) {
      // Normalisation is fp32 with one bf16 rounding, exactly like ATen's layer-norm
      // epilogue; the modulation is then one native bf16 rounding, so `__hfma2` is not
      // usable here.
      __nv_bfloat162 normed;
      if constexpr (kFuseNorm) {
        const float2 xf = __bfloat1622float2(xq[j]);
        normed.x = __float2bfloat16_rn((xf.x - mean) * rstd);
        normed.y = __float2bfloat16_rn((xf.y - mean) * rstd);
      } else {
        normed = xq[j];
      }
      rq[j] = __hmul2(normed, sq[j]);
    }
    *reinterpret_cast<uint4*>(out_row + base) = rv;
  }
}

struct Common {
  int64_t rows;
  int64_t seq_len;
  int64_t dim;
  int64_t one_plus_rows;
  const bool* mask;
};

Common check_and_common(const at::Tensor& x, const at::Tensor& one_plus,
                        const std::optional<at::Tensor>& mask, at::Tensor& x_mod, const char* op) {
  // A non-zero storage offset is fine: `data_ptr()` already includes it and the kernel only
  // needs a dense layout plus aligned 16-byte vector loads.
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 3,
              op, ": x must be a contiguous CUDA bf16 [B,S,D] tensor");
  const int64_t dim = x.size(2);
  TORCH_CHECK(dim % 128 == 0, op, ": D must be a multiple of 128, got ", dim);
  TORCH_CHECK(dim <= kMaxRecip * 16, op, ": D must not exceed ", kMaxRecip * 16,
              " so the reciprocal table covers every running count");
  TORCH_CHECK(one_plus.is_cuda() && one_plus.scalar_type() == at::kBFloat16 &&
                one_plus.is_contiguous() && one_plus.dim() == 2 && one_plus.size(1) == dim &&
                one_plus.storage_offset() == 0,
              op, ": one_plus must be a contiguous CUDA bf16 [R, D] tensor");
  TORCH_CHECK(x_mod.is_cuda() && x_mod.scalar_type() == at::kBFloat16 && x_mod.sizes() == x.sizes(),
              op, ": x_mod must be a bf16 [B,S,D] buffer");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(x.const_data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(one_plus.const_data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(x_mod.data_ptr()) & 15) == 0,
              op, ": buffers must be 16-byte aligned for the uint4 path");
  TORCH_CHECK(x.size(0) == 1, op,
              ": only batch 1 is supported by the 16-byte path; `fused_ops` keeps the eager "
              "expression for larger batches instead of mis-mapping modulation rows");
  const int64_t one_plus_rows = one_plus.size(0);
  TORCH_CHECK(one_plus_rows == 1 || one_plus_rows == 2, op,
              ": one_plus rows must be B or B+1 with B=1 (got ", one_plus_rows, ")");

  const bool* mask_ptr = nullptr;
  if (mask.has_value()) {
    const at::Tensor& m = mask.value();
    TORCH_CHECK(
      m.is_cuda() && m.scalar_type() == at::kBool && m.is_contiguous() && m.numel() == x.size(1),
      op, ": target_token_mask must be a contiguous CUDA bool [S] tensor");
    TORCH_CHECK(one_plus_rows == 2, op, ": a target_token_mask requires B+1 rows");
    mask_ptr = m.const_data_ptr<bool>();
  }
  return Common{x.size(0) * x.size(1), x.size(1), dim, one_plus_rows, mask_ptr};
}

}  // namespace

void qwen_image21_ln_modulate(at::Tensor x, at::Tensor one_plus,
                              const std::optional<at::Tensor>& target_token_mask, double eps,
                              at::Tensor x_mod) {
  Common c = check_and_common(x, one_plus, target_token_mask, x_mod, "qwen_image21_ln_modulate");
  if (c.rows == 0) {
    return;
  }
  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t warp_count = (c.rows + 1) / 2;  // two rows per warp
  const int64_t grid = (warp_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
  qwen_image21_modulate_kernel<true><<<grid, kWarpsPerBlock * 32, 0, stream>>>(
    reinterpret_cast<const __nv_bfloat162*>(x.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(one_plus.const_data_ptr()), c.mask,
    reinterpret_cast<__nv_bfloat162*>(x_mod.data_ptr()), c.rows, (int)c.seq_len, (int)c.dim,
    (int)c.one_plus_rows, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::models
