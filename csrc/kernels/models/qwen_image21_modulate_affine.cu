// Qwen-Image 2.1 block modulation.
//
// Two entry points share one epilogue:
//   qwen_image21_ln_modulate      LayerNorm(elementwise_affine=False) + `1 + scale` + `tanh(gate)`
//   qwen_image21_modulate_affine  the same epilogue on an already normalized input
//
// The second exists because a bit-exact LayerNorm has to reproduce ATen's Welford tree; if
// the runtime self-check in `fused_ops` finds that it does not (another GPU or another
// torch build), the normalization stays in ATen and only the memory-bound epilogue is
// fused, which is still exact and still removes the two materialized `(B, S, D)` tensors
// that `_select_modulation_rows` would otherwise produce.
//
// Rounding points, matching `_modulate` exactly:
//   one_plus = bf16(1 + scale)      scale / gate rows are selected per token by the mask
//   x_mod    = bf16(x_norm * one_plus)
//   gate_out = bf16(tanh_f32(gate))
//
// One warp per row, 4 channels per lane, 16-byte vectorised loads across the warp.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <optional>

#include "qwen_image21.h"

namespace cache_dit::models {
namespace {

constexpr int kWarpsPerBlock = 4;
constexpr int kLanesPerRow = 32;

struct Welford {
  float mean;
  float sigma2;
  float count;
};

__device__ __forceinline__ Welford welford_online(float val, const Welford& curr) {
  const float count = curr.count + 1.0f;
  const float delta = val - curr.mean;
  const float new_mean = curr.mean + delta * (1.0f / count);
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

// Welford's merge is neither commutative nor associative in fp32, so a butterfly would
// leave every lane with a slightly different mean. This mirrors ATen's `warp_reduce`
// (shfl_down tree, then broadcast from lane 0) so the whole row shares one value.
__device__ __forceinline__ Welford warp_welford_32(Welford v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    Welford other;
    other.mean = __shfl_down_sync(0xffffffffu, v.mean, offset);
    other.sigma2 = __shfl_down_sync(0xffffffffu, v.sigma2, offset);
    other.count = __shfl_down_sync(0xffffffffu, v.count, offset);
    if ((threadIdx.x & 31) < offset) {
      v = welford_combine(v, other);
    }
  }
  Welford out;
  out.mean = __shfl_sync(0xffffffffu, v.mean, 0);
  out.sigma2 = __shfl_sync(0xffffffffu, v.sigma2, 0);
  out.count = __shfl_sync(0xffffffffu, v.count, 0);
  return out;
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

template <bool kFuseNorm>
__global__ void qwen_image21_modulate_kernel(const __nv_bfloat162* __restrict__ x,
                                             const __nv_bfloat162* __restrict__ params,
                                             const bool* __restrict__ mask,
                                             __nv_bfloat162* __restrict__ x_mod,
                                             __nv_bfloat162* __restrict__ gate_tanh, int64_t rows,
                                             int seq_len, int dim, int param_rows, float eps) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t row = (int64_t)blockIdx.x * kWarpsPerBlock + warp;
  if (row >= rows) {
    return;
  }

  const int64_t pos = row % seq_len;
  const int64_t batch = row / seq_len;
  const int pairs = dim >> 1;
  const int groups = dim >> 7;  // 128 channels per warp-wide step
  const __nv_bfloat162* x_row = x + row * pairs;

  float mean = 0.0f;
  float rstd = 1.0f;
  if (kFuseNorm) {
    Welford wd{0.0f, 0.0f, 0.0f};
    for (int g = 0; g < groups; ++g) {
      const int base = (g << 6) + (lane << 1);
      const float2 a = __bfloat1622float2(x_row[base]);
      const float2 b = __bfloat1622float2(x_row[base + 1]);
      wd = welford_online(a.x, wd);
      wd = welford_online(a.y, wd);
      wd = welford_online(b.x, wd);
      wd = welford_online(b.y, wd);
    }
    wd = warp_welford_32(wd);
    mean = wd.mean;
    rstd = rsqrtf(wd.sigma2 / (float)dim + eps);
  }

  const int param_row = select_param_row(batch, pos, mask, param_rows);
  // One modulation row packs `scale | gate`, i.e. `dim` bf16 pairs holding 2 * dim channels.
  const __nv_bfloat162* params_row = params + (int64_t)param_row * dim;
  const __nv_bfloat162* scale_row = params_row;
  const __nv_bfloat162* gate_row = params_row + pairs;
  __nv_bfloat162* out_row = x_mod + row * pairs;
  __nv_bfloat162* gate_out_row = gate_tanh + row * pairs;

  for (int g = 0; g < groups; ++g) {
    const int base = (g << 6) + (lane << 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int idx = base + i;
      const __nv_bfloat162 xv = x_row[idx];
      const __nv_bfloat162 sv = scale_row[idx];
      const __nv_bfloat162 gv = gate_row[idx];

      // Normalisation is fp32 with one bf16 rounding, exactly like ATen's layer-norm
      // epilogue; the modulation itself is then two native bf16 roundings
      // (`1 + scale`, then the product), so `__hfma2` is not usable here.
      __nv_bfloat162 normed;
      if (kFuseNorm) {
        const float2 xf = __bfloat1622float2(xv);
        normed.x = __float2bfloat16_rn((xf.x - mean) * rstd);
        normed.y = __float2bfloat16_rn((xf.y - mean) * rstd);
      } else {
        normed = xv;
      }
      out_row[idx] = __hmul2(normed, __hadd2(sv, __float2bfloat162_rn(1.0f)));

      // tanh only exists in fp32 here: it has to match torch's fp32 `tanh` bit for bit.
      const float2 gvf = __bfloat1622float2(gv);
      __nv_bfloat162 gatev;
      gatev.x = __float2bfloat16_rn(tanhf(gvf.x));
      gatev.y = __float2bfloat16_rn(tanhf(gvf.y));
      gate_out_row[idx] = gatev;
    }
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
  // needs a dense layout plus 4-byte aligned pair loads.
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 3,
              op, ": x must be a contiguous CUDA bf16 [B,S,D] tensor");
  const int64_t dim = x.size(2);
  TORCH_CHECK(dim % 128 == 0, op, ": D must be a multiple of 128, got ", dim);
  TORCH_CHECK(params.is_cuda() && params.scalar_type() == at::kBFloat16 && params.is_contiguous() &&
                params.dim() == 2 && params.size(1) == 2 * dim && params.storage_offset() == 0,
              op, ": params must be a contiguous CUDA bf16 [R, 2*D] tensor");
  TORCH_CHECK(x_mod.is_cuda() && x_mod.scalar_type() == at::kBFloat16 && x_mod.sizes() == x.sizes(),
              op, ": x_mod must be a bf16 [B,S,D] buffer");
  TORCH_CHECK(gate_tanh.is_cuda() && gate_tanh.scalar_type() == at::kBFloat16 &&
                gate_tanh.sizes() == x.sizes(),
              op, ": gate_tanh must be a bf16 [B,S,D] buffer");
  const int64_t param_rows = params.size(0);
  int64_t batch = x.size(0);
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
  const int64_t grid = (c.rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  qwen_image21_modulate_kernel<true><<<grid, kWarpsPerBlock * kLanesPerRow, 0, stream>>>(
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
  const int64_t grid = (c.rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  qwen_image21_modulate_kernel<false><<<grid, kWarpsPerBlock * kLanesPerRow, 0, stream>>>(
    reinterpret_cast<const __nv_bfloat162*>(x_normed.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(params.const_data_ptr()), c.mask,
    reinterpret_cast<__nv_bfloat162*>(x_mod.data_ptr()),
    reinterpret_cast<__nv_bfloat162*>(gate_tanh.data_ptr()), c.rows, (int)c.seq_len, (int)c.dim,
    (int)c.param_rows, 0.0f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::models
