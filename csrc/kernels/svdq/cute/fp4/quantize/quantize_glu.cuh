#pragma once

#include <cuda_bf16.h>

namespace svdq::cute_kernels {

// Shared gated-linear-unit math for the CuTe activation producers.
//
// Both producer implementations (`quantize_impl.cuh` for the rank-specialized TUs and
// `quantize_uni.cu` for the general one) load a value half and, when the caller asks for GLU
// fusion, a gate half; the two halves are multiplied in BF16 before quantization and before the
// LoRA-down reduction. Keeping the pair math in one header guarantees that the two TUs round
// identically instead of drifting apart over time.

// BF16 sigmoid through the approximate exponential, the same form the CuTe fused GEMM epilogue uses
// for its SiLU branch. It deliberately does not reproduce the native producer's `rcp.approx`
// variant: the two runtimes are compared with tolerances, not bitwise.
static __device__ __forceinline__ float sigmoid_approx_bf16(float x) {
  float e;
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(-x * 1.4426950408889634f));
  return 1.0f / (1.0f + e);
}

// One BF16 pair of the gated product `value * silu(gate)`.
//
// `silu(gate)` is rounded to BF16 before the product, because that is the arithmetic the model
// itself performs: both operands of a SwiGLU block are BF16 activations, so torch computes
// `bf16(value * bf16(silu(gate)))`. Rounding once at the end instead would make the fused chain
// differ from the `silu(gate) * value` it replaces by more than the FP4 quantization noise on a
// measurable share of the codes.
static __device__ __forceinline__ __nv_bfloat162 glu_pair(__nv_bfloat162 value,
                                                          __nv_bfloat162 gate) {
  const float2 v = __bfloat1622float2(value);
  const float2 g = __bfloat1622float2(gate);
  const float2 s = __bfloat1622float2(__float22bfloat162_rn(
    make_float2(g.x * sigmoid_approx_bf16(g.x), g.y * sigmoid_approx_bf16(g.y))));
  return __float22bfloat162_rn(make_float2(v.x * s.x, v.y * s.y));
}

}  // namespace svdq::cute_kernels
