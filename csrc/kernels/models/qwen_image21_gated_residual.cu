// Qwen-Image 2.1 gated residual: y = hidden_states + bf16(gate * update).
//
// The gate is constant across tokens (it only depends on the sample and on whether the token
// is a target-image or a `t = 0` token), so `fused_ops` passes the already `tanh`-ed `(R, D)`
// rows and the kernel selects one per token. That drops a full `(B, S, D)` read per call -
// 32 MiB on the 1024x1024 shape, twice per block.
//
// The eager expression rounds twice - the product is bf16 before the add - so the kernel uses
// `__hmul2` then `__hadd2`, i.e. two native bf16 roundings. `__hfma2` would drop the first
// one and change the result by a bf16 ulp.
//
// Vectorised: one thread consumes 8 bf16 (one 16-byte transaction per input/output).
//
// Scope: batch 1; `fused_ops` keeps the eager expression for larger batches.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <optional>

#include "qwen_image21.h"

namespace cache_dit::models {
namespace {

constexpr int kElemsPerThread = 8;

bool is_aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

// `d_vec` walks the row, `row` walks the tokens; the gate row is picked per token, so the
// whole `(B, S, D)` gate never has to exist in memory.
__global__ void qwen_image21_gated_residual_kernel(
  const uint4* __restrict__ hidden, const uint4* __restrict__ gate_rows,
  const bool* __restrict__ mask, const uint4* __restrict__ update, uint4* __restrict__ out,
  int64_t vecs, int seq_len, int row_shift, int d_mask, int gate_row_last) {
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < vecs;
       idx += (int64_t)gridDim.x * blockDim.x) {
    const int d_vec = (int)(idx & d_mask);
    const int64_t row = idx >> row_shift;
    const int64_t pos = row % seq_len;
    int gate_row = 0;
    if (mask != nullptr && !mask[pos]) {
      gate_row = gate_row_last;
    }

    const uint4 hv = hidden[idx];
    const uint4 gv = gate_rows[(int64_t)gate_row * ((int64_t)d_mask + 1) + d_vec];
    const uint4 uv = update[idx];
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&hv);
    const __nv_bfloat162* g = reinterpret_cast<const __nv_bfloat162*>(&gv);
    const __nv_bfloat162* u = reinterpret_cast<const __nv_bfloat162*>(&uv);
    uint4 result;
    __nv_bfloat162* r = reinterpret_cast<__nv_bfloat162*>(&result);
#pragma unroll
    for (int i = 0; i < kElemsPerThread / 2; ++i) {
      // Two native bf16 roundings: mul rounds to bf16, add rounds to bf16.
      r[i] = __hadd2(h[i], __hmul2(g[i], u[i]));
    }
    out[idx] = result;
  }
}

}  // namespace

void qwen_image21_gated_residual(at::Tensor hidden_states, at::Tensor gate_rows,
                                 const std::optional<at::Tensor>& target_token_mask,
                                 at::Tensor update, at::Tensor out, int64_t seq_len) {
  // A non-zero storage offset is fine (a contiguous slice is a valid input); what the
  // 16-byte vector loads need is pointer alignment.
  TORCH_CHECK(hidden_states.is_cuda() && hidden_states.scalar_type() == at::kBFloat16 &&
                hidden_states.is_contiguous() && hidden_states.dim() == 3,
              "qwen_image21_gated_residual: hidden_states must be contiguous CUDA bf16 [B,S,D]");
  TORCH_CHECK(
    gate_rows.is_cuda() && gate_rows.scalar_type() == at::kBFloat16 && gate_rows.is_contiguous() &&
      gate_rows.dim() == 2 && gate_rows.size(1) == hidden_states.size(2) &&
      gate_rows.storage_offset() == 0,
    "qwen_image21_gated_residual: gate_rows must be a contiguous CUDA bf16 [R, D] tensor");
  TORCH_CHECK(
    update.is_cuda() && update.scalar_type() == at::kBFloat16 && update.is_contiguous() &&
      update.sizes() == hidden_states.sizes(),
    "qwen_image21_gated_residual: update must be contiguous CUDA bf16 with the same shape");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.is_contiguous() &&
                out.sizes() == hidden_states.sizes(),
              "qwen_image21_gated_residual: out must be contiguous CUDA bf16 with the same shape");
  TORCH_CHECK(
    out.data_ptr() != hidden_states.data_ptr(),
    "qwen_image21_gated_residual: out must not alias hidden_states; cache-dit derives the "
    "block residual as (output - input)");
  TORCH_CHECK(is_aligned16(hidden_states.const_data_ptr()) &&
                is_aligned16(gate_rows.const_data_ptr()) && is_aligned16(update.const_data_ptr()) &&
                is_aligned16(out.data_ptr()),
              "qwen_image21_gated_residual: buffers must be 16-byte aligned for the uint4 path");
  TORCH_CHECK(hidden_states.size(0) == 1,
              "qwen_image21_gated_residual: only batch 1 is supported; `fused_ops` keeps the eager "
              "expression for larger batches");

  const int64_t dim = hidden_states.size(2);
  const int64_t vecs_per_row = dim / kElemsPerThread;
  TORCH_CHECK((vecs_per_row & (vecs_per_row - 1)) == 0,
              "qwen_image21_gated_residual: D/8 must be a power of two, got ", dim);
  TORCH_CHECK(seq_len == hidden_states.size(1), "qwen_image21_gated_residual: seq_len mismatch");

  int row_shift = 0;
  while ((int64_t(1) << row_shift) < vecs_per_row) {
    ++row_shift;
  }

  const bool* mask_ptr = nullptr;
  int gate_row_last = 0;
  if (target_token_mask.has_value()) {
    const at::Tensor& m = target_token_mask.value();
    TORCH_CHECK(
      m.is_cuda() && m.scalar_type() == at::kBool && m.is_contiguous() && m.numel() == seq_len,
      "qwen_image21_gated_residual: target_token_mask must be a contiguous CUDA bool [S]");
    TORCH_CHECK(gate_rows.size(0) == 2,
                "qwen_image21_gated_residual: a target_token_mask requires two gate rows");
    mask_ptr = m.const_data_ptr<bool>();
    gate_row_last = 1;
  }

  const int64_t numel = hidden_states.numel();
  TORCH_CHECK(numel % kElemsPerThread == 0,
              "qwen_image21_gated_residual: element count must be a multiple of 8");
  const int64_t vecs = numel / kElemsPerThread;
  if (vecs == 0) {
    return;
  }
  const int threads = 256;
  const int blocks = (int)std::min<int64_t>((vecs + threads - 1) / threads, 65535);

  const c10::cuda::CUDAGuard guard(hidden_states.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  qwen_image21_gated_residual_kernel<<<blocks, threads, 0, stream>>>(
    reinterpret_cast<const uint4*>(hidden_states.const_data_ptr()),
    reinterpret_cast<const uint4*>(gate_rows.const_data_ptr()), mask_ptr,
    reinterpret_cast<const uint4*>(update.const_data_ptr()),
    reinterpret_cast<uint4*>(out.data_ptr()), vecs, (int)seq_len, row_shift, (int)vecs_per_row - 1,
    gate_row_last);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::models
