// Qwen-Image 2.1 gated residual: y = hidden_states + bf16(gate * update).
//
// The eager expression rounds twice — the product is bf16 before the add — so the kernel
// uses explicit `__fmul_rn`/`__fadd_rn` and a bf16 round trip in between. `__fmul_rn`
// followed by `__fadd_rn` also keeps nvcc from contracting the pair into one FMA, which
// would drop a rounding point and change the result by a bf16 ulp.
//
// Vectorised: one thread consumes 8 bf16 (one 16-byte transaction per input/output).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>

#include "qwen_image21.h"

namespace cache_dit::models {
namespace {

constexpr int kElemsPerThread = 8;

__global__ void qwen_image21_gated_residual_kernel(const uint4* __restrict__ hidden,
                                                   const uint4* __restrict__ gate,
                                                   const uint4* __restrict__ update,
                                                   uint4* __restrict__ out, int64_t vecs) {
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < vecs;
       idx += (int64_t)gridDim.x * blockDim.x) {
    const uint4 hv = hidden[idx];
    const uint4 gv = gate[idx];
    const uint4 uv = update[idx];
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&hv);
    const __nv_bfloat162* g = reinterpret_cast<const __nv_bfloat162*>(&gv);
    const __nv_bfloat162* u = reinterpret_cast<const __nv_bfloat162*>(&uv);
    uint4 result;
    __nv_bfloat162* r = reinterpret_cast<__nv_bfloat162*>(&result);
#pragma unroll
    for (int i = 0; i < kElemsPerThread / 2; ++i) {
      const float2 hf = __bfloat1622float2(h[i]);
      const float2 gf = __bfloat1622float2(g[i]);
      const float2 uf = __bfloat1622float2(u[i]);
      // Rounding 1: bf16(gate * update).  Rounding 2: bf16(hidden + that).
      const float px = __bfloat162float(__float2bfloat16_rn(__fmul_rn(gf.x, uf.x)));
      const float py = __bfloat162float(__float2bfloat16_rn(__fmul_rn(gf.y, uf.y)));
      r[i].x = __float2bfloat16_rn(__fadd_rn(hf.x, px));
      r[i].y = __float2bfloat16_rn(__fadd_rn(hf.y, py));
    }
    out[idx] = result;
  }
}

}  // namespace

void qwen_image21_gated_residual(at::Tensor hidden_states, at::Tensor gate, at::Tensor update,
                                 at::Tensor out) {
  TORCH_CHECK(hidden_states.is_cuda() && hidden_states.scalar_type() == at::kBFloat16 &&
                hidden_states.is_contiguous() && hidden_states.storage_offset() == 0,
              "qwen_image21_gated_residual: hidden_states must be contiguous CUDA bf16");
  TORCH_CHECK(gate.is_cuda() && gate.scalar_type() == at::kBFloat16 && gate.is_contiguous() &&
                gate.storage_offset() == 0 && gate.sizes() == hidden_states.sizes(),
              "qwen_image21_gated_residual: gate must be contiguous CUDA bf16 with the same shape");
  TORCH_CHECK(
    update.is_cuda() && update.scalar_type() == at::kBFloat16 && update.is_contiguous() &&
      update.storage_offset() == 0 && update.sizes() == hidden_states.sizes(),
    "qwen_image21_gated_residual: update must be contiguous CUDA bf16 with the same shape");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.is_contiguous() &&
                out.storage_offset() == 0 && out.sizes() == hidden_states.sizes(),
              "qwen_image21_gated_residual: out must be contiguous CUDA bf16 with the same shape");
  TORCH_CHECK(
    out.data_ptr() != hidden_states.data_ptr(),
    "qwen_image21_gated_residual: out must not alias hidden_states; cache-dit derives the "
    "block residual as (output - input)");
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
    reinterpret_cast<const uint4*>(gate.const_data_ptr()),
    reinterpret_cast<const uint4*>(update.const_data_ptr()),
    reinterpret_cast<uint4*>(out.data_ptr()), vecs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::models
