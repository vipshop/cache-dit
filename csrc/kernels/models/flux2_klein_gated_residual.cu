// FLUX.2-Klein gated residual: `out = bf16(hidden + bf16(gate * update))`.
//
// Eager reference (`diffusers/models/transformers/transformer_flux2.py:947-948`, `:954`,
// `:957-958`, `:964`): two bf16 tensor ops, each rounding once. Both roundings are replayed
// with `__hmul2`/`__hadd2`, so the operator is bit-exact (verified with `torch.equal`; note
// `__hfma2` cannot be used, it drops the product's rounding).
//
// Scope: batch 1 with a single modulation row, which is the shape these blocks always run;
// `fused_ops` keeps the eager expression for anything else instead of mis-indexing rows.
// The operator only removes one launch and one intermediate on the residual that is *not*
// followed by a normalisation (those go through `flux2_klein_residual_ln_modulate`), and
// inductor already fuses this chain, so it ships disabled by default.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "flux2_klein.h"
#include "flux2_klein_common.cuh"

namespace cache_dit::models {
namespace {

constexpr int kThreads = 256;
constexpr int kChannelsPerChunk = 8;  // one uint4 = 8 bf16 channels

// One block owns one row: the gate vector is indexed by channel inside the row, so no modulo
// and no cross-row address arithmetic is needed.
__global__ void flux2_klein_gated_residual_kernel(const __nv_bfloat162* __restrict__ hidden,
                                                  const __nv_bfloat162* __restrict__ gate,
                                                  const __nv_bfloat162* __restrict__ update,
                                                  __nv_bfloat162* __restrict__ out,
                                                  int chunks_per_row) {
  const int64_t row = blockIdx.x;
  const __nv_bfloat162* hidden_row = hidden + row * ((int64_t)chunks_per_row * 4);
  const __nv_bfloat162* update_row = update + row * ((int64_t)chunks_per_row * 4);
  __nv_bfloat162* out_row = out + row * ((int64_t)chunks_per_row * 4);
  for (int chunk = threadIdx.x; chunk < chunks_per_row; chunk += kThreads) {
    const __nv_bfloat162* gate_chunk = gate + chunk * 4;
    const uint4 hv = *reinterpret_cast<const uint4*>(hidden_row + chunk * 4);
    const uint4 uv = *reinterpret_cast<const uint4*>(update_row + chunk * 4);
    const __nv_bfloat162* hq = reinterpret_cast<const __nv_bfloat162*>(&hv);
    const __nv_bfloat162* uq = reinterpret_cast<const __nv_bfloat162*>(&uv);
    uint4 rv;
    __nv_bfloat162* rq = reinterpret_cast<__nv_bfloat162*>(&rv);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      rq[j] = cache_dit::models::bf16_mul_add_rounded(gate_chunk[j], uq[j], hq[j]);
    }
    *reinterpret_cast<uint4*>(out_row + chunk * 4) = rv;
  }
}

}  // namespace

void flux2_klein_gated_residual(at::Tensor hidden, at::Tensor gate, at::Tensor update,
                                at::Tensor out) {
  const char* op = "flux2_klein_gated_residual";
  TORCH_CHECK(hidden.is_cuda() && hidden.scalar_type() == at::kBFloat16 && hidden.is_contiguous() &&
                hidden.dim() == 3 && hidden.size(0) == 1,
              op, ": hidden must be a contiguous CUDA bf16 [1,S,D] tensor");
  // A non-zero storage offset is fine: the row is a chunk of the narrow modulation tensor and
  // `data_ptr()` already includes the offset.
  TORCH_CHECK(gate.is_cuda() && gate.scalar_type() == at::kBFloat16 && gate.is_contiguous() &&
                gate.numel() == hidden.size(2),
              op, ": gate must be a contiguous CUDA bf16 [D] row");
  TORCH_CHECK(update.is_cuda() && update.scalar_type() == at::kBFloat16 && update.is_contiguous() &&
                update.sizes() == hidden.sizes(),
              op, ": update must be a contiguous CUDA bf16 tensor shaped like hidden");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.is_contiguous() &&
                out.sizes() == hidden.sizes(),
              op, ": out must be a contiguous bf16 buffer shaped like hidden");
  const int64_t dim = hidden.size(2);
  TORCH_CHECK(dim % kChannelsPerChunk == 0, op, ": D must be a multiple of ", kChannelsPerChunk,
              ", got ", dim);
  TORCH_CHECK((reinterpret_cast<uintptr_t>(hidden.const_data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(gate.const_data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(update.const_data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(out.data_ptr()) & 15) == 0,
              op, ": buffers must be 16-byte aligned for the uint4 path");
  const int64_t rows = hidden.size(1);
  if (rows == 0) {
    return;
  }
  const c10::cuda::CUDAGuard guard(hidden.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  flux2_klein_gated_residual_kernel<<<rows, kThreads, 0, stream>>>(
    reinterpret_cast<const __nv_bfloat162*>(hidden.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(gate.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(update.const_data_ptr()),
    reinterpret_cast<__nv_bfloat162*>(out.data_ptr()), (int)(dim / kChannelsPerChunk));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::models
