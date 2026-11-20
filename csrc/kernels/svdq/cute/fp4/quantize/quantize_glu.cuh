#pragma once

#include <cuda_bf16.h>

#include "gate_activation.h"

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

// Evaluate the selected gate activation and round it to BF16 before the product. This is the
// arithmetic the model performs with two BF16 activation operands: `bf16(value * bf16(act(gate)))`.
// SiLU keeps the producer's existing approximate sigmoid; the new GELU variants use the same
// formulas as torch and rely on CUDA's full-precision tanhf/erff rather than an approximation.
static __device__ __forceinline__ float gate_activation_bf16(float x, GateAct act) {
  float activated;
  if (act == GateAct::kSiLU) {
    activated = x * sigmoid_approx_bf16(x);
  } else if (act == GateAct::kGeluTanh) {
    activated = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
  } else {
    activated = 0.5f * x * (1.0f + erff(0.7071067812f * x));
  }
  return __bfloat162float(__float2bfloat16_rn(activated));
}

// One BF16 pair of the gated product `value * act(gate)`.
static __device__ __forceinline__ __nv_bfloat162 glu_pair(__nv_bfloat162 value, __nv_bfloat162 gate,
                                                          GateAct act) {
  const float2 v = __bfloat1622float2(value);
  const float2 g = __bfloat1622float2(gate);
  return __float22bfloat162_rn(
    make_float2(v.x * gate_activation_bf16(g.x, act), v.y * gate_activation_bf16(g.y, act)));
}

}  // namespace svdq::cute_kernels
