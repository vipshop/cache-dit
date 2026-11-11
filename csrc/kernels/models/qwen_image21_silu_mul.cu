// Qwen-Image 2.1 SwiGLU gate: y = bf16(silu_fp32(gate)) * up.
//
// ATen evaluates `silu` in fp32 as `x / (1 + exp(-x))` and stores bf16, then the product is
// a bf16 multiply. The kernel keeps the division (a reciprocal multiply would differ by an
// ulp) and the intermediate bf16 rounding.
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

bool is_aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

__global__ void qwen_image21_silu_mul_kernel(const uint4* __restrict__ gate,
                                             const uint4* __restrict__ up, uint4* __restrict__ out,
                                             int64_t vecs) {
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < vecs;
       idx += (int64_t)gridDim.x * blockDim.x) {
    const uint4 gv = gate[idx];
    const uint4 uv = up[idx];
    const __nv_bfloat162* g = reinterpret_cast<const __nv_bfloat162*>(&gv);
    const __nv_bfloat162* u = reinterpret_cast<const __nv_bfloat162*>(&uv);
    uint4 result;
    __nv_bfloat162* r = reinterpret_cast<__nv_bfloat162*>(&result);
#pragma unroll
    for (int i = 0; i < kElemsPerThread / 2; ++i) {
      // silu stays in fp32 with a real division (ATen's `silu` divides, and a reciprocal
      // multiply differs by an ulp), then the product is one native bf16 rounding.
      const float2 gf = __bfloat1622float2(g[i]);
      __nv_bfloat162 silu;
      silu.x = __float2bfloat16_rn(gf.x / (1.0f + expf(-gf.x)));
      silu.y = __float2bfloat16_rn(gf.y / (1.0f + expf(-gf.y)));
      r[i] = __hmul2(silu, u[i]);
    }
    out[idx] = result;
  }
}

}  // namespace

void qwen_image21_silu_mul(at::Tensor gate, at::Tensor up, at::Tensor out) {
  // A non-zero storage offset is fine (a contiguous slice is a valid input); what the
  // 16-byte vector loads need is pointer alignment.
  TORCH_CHECK(gate.is_cuda() && gate.scalar_type() == at::kBFloat16 && gate.is_contiguous(),
              "qwen_image21_silu_mul: gate must be contiguous CUDA bf16");
  TORCH_CHECK(up.is_cuda() && up.scalar_type() == at::kBFloat16 && up.is_contiguous() &&
                up.sizes() == gate.sizes(),
              "qwen_image21_silu_mul: up must be contiguous CUDA bf16 with the same shape");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.is_contiguous() &&
                out.sizes() == gate.sizes(),
              "qwen_image21_silu_mul: out must be contiguous CUDA bf16 with the same shape");
  TORCH_CHECK(is_aligned16(gate.const_data_ptr()) && is_aligned16(up.const_data_ptr()) &&
                is_aligned16(out.data_ptr()),
              "qwen_image21_silu_mul: buffers must be 16-byte aligned for the uint4 path");
  const int64_t numel = gate.numel();
  TORCH_CHECK(numel % kElemsPerThread == 0,
              "qwen_image21_silu_mul: element count must be a multiple of 8");

  const int64_t vecs = numel / kElemsPerThread;
  if (vecs == 0) {
    return;
  }
  const int threads = 256;
  const int blocks = (int)std::min<int64_t>((vecs + threads - 1) / threads, 65535);

  const c10::cuda::CUDAGuard guard(gate.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  qwen_image21_silu_mul_kernel<<<blocks, threads, 0, stream>>>(
    reinterpret_cast<const uint4*>(gate.const_data_ptr()),
    reinterpret_cast<const uint4*>(up.const_data_ptr()), reinterpret_cast<uint4*>(out.data_ptr()),
    vecs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::models
