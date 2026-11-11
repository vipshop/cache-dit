// FLUX.2-Klein SwiGLU: `out = bf16(silu_fp32(gate)) * up`, over the two halves of one tensor.
//
// Eager reference (`diffusers/models/transformers/transformer_flux2.py:295-298`):
//   half = x.shape[-1] // 2
//   x = F.silu(x[..., :half]) * x[..., half:]
// `F.silu` on bf16 computes in fp32 and rounds once; the following multiply is a bf16 product
// and rounds again. Both roundings are replayed, but `sigmoid` goes through `expf`, which the
// extension's global `--use_fast_math` turns into the approximate SFU form, so this operator is
// accepted at rel-L2 <= 1e-2 instead of `torch.equal`.
//
// The source is read through its own row stride: the real caller hands over the MLP half of a
// fused QKV+MLP projection (`to_qkv_mlp_proj(...).split(...)`), which is packed in `(2D)` but
// whose row stride spans the whole fused width, so no `contiguous()` copy is needed.
//
// This operator exists for completeness only. NCU measured inductor's fused SwiGLU+cat at
// 90.5% DRAM throughput with the traffic of the theoretical minimum (198 MiB in + 113 MiB out
// per single block), so there is nothing to save and it ships disabled by default.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "flux2_klein.h"

namespace cache_dit::models {
namespace {

constexpr int kThreads = 256;
constexpr int kChannelsPerChunk = 8;  // one uint4 = 8 bf16 channels

__global__ void flux2_klein_silu_mul_kernel(const __nv_bfloat162* __restrict__ gate_up,
                                            __nv_bfloat162* __restrict__ out, int64_t in_row_pairs,
                                            int half_pairs, int chunks_per_half) {
  const int64_t row = blockIdx.x;
  const __nv_bfloat162* gate_row = gate_up + row * in_row_pairs;
  const __nv_bfloat162* up_row = gate_row + half_pairs;
  __nv_bfloat162* out_row = out + row * half_pairs;
  for (int chunk = threadIdx.x; chunk < chunks_per_half; chunk += kThreads) {
    const uint4 gv = *reinterpret_cast<const uint4*>(gate_row + chunk * 4);
    const uint4 uv = *reinterpret_cast<const uint4*>(up_row + chunk * 4);
    const __nv_bfloat162* gq = reinterpret_cast<const __nv_bfloat162*>(&gv);
    const __nv_bfloat162* uq = reinterpret_cast<const __nv_bfloat162*>(&uv);
    uint4 rv;
    __nv_bfloat162* rq = reinterpret_cast<__nv_bfloat162*>(&rv);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 g = __bfloat1622float2(gq[j]);
      // Rounding 1: fp32 sigmoid, bf16 round - the same place `F.silu` rounds.
      const __nv_bfloat162 s = {__float2bfloat16_rn(g.x / (1.0f + __expf(-g.x))),
                                __float2bfloat16_rn(g.y / (1.0f + __expf(-g.y)))};
      // Rounding 2: the bf16 product.
      rq[j] = __hmul2(s, uq[j]);
    }
    *reinterpret_cast<uint4*>(out_row + chunk * 4) = rv;
  }
}

}  // namespace

void flux2_klein_silu_mul(at::Tensor gate_up, at::Tensor out) {
  const char* op = "flux2_klein_silu_mul";
  TORCH_CHECK(gate_up.is_cuda() && gate_up.scalar_type() == at::kBFloat16 && gate_up.dim() == 3, op,
              ": gate_up must be a CUDA bf16 [B,S,2D] tensor");
  TORCH_CHECK(gate_up.size(0) == 1, op, ": only batch 1 is supported, got ", gate_up.size(0));
  TORCH_CHECK(gate_up.stride(2) == 1 && gate_up.stride(1) % 8 == 0, op,
              ": gate_up must be dense along the last dimension with a 16-byte aligned row "
              "stride, got ",
              gate_up.stride(1), "/", gate_up.stride(2));
  const int64_t packed = gate_up.size(2);
  TORCH_CHECK(packed % (2 * kChannelsPerChunk) == 0, op,
              ": the last dimension must be an even multiple of ", kChannelsPerChunk, ", got ",
              packed);
  TORCH_CHECK(gate_up.stride(1) >= packed, op, ": gate_up rows must not overlap, got row stride ",
              gate_up.stride(1), " for a packed width of ", packed);
  const int64_t half = packed / 2;
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.is_contiguous() &&
                out.sizes() == at::IntArrayRef({gate_up.size(0), gate_up.size(1), half}),
              op, ": out must be a contiguous bf16 [B,S,D] buffer matching half of gate_up");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(gate_up.const_data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(out.data_ptr()) & 15) == 0,
              op, ": buffers must be 16-byte aligned for the uint4 path");
  const int64_t rows = gate_up.size(0) * gate_up.size(1);
  if (rows == 0) {
    return;
  }
  const c10::cuda::CUDAGuard guard(gate_up.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  flux2_klein_silu_mul_kernel<<<rows, kThreads, 0, stream>>>(
    reinterpret_cast<const __nv_bfloat162*>(gate_up.const_data_ptr()),
    reinterpret_cast<__nv_bfloat162*>(out.data_ptr()), gate_up.stride(1) / 2, (int)(half / 2),
    (int)(half / kChannelsPerChunk));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::models
