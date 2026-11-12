#pragma once

#include <cuda_bf16.h>

namespace cache_dit::models {

// The eager chains these kernels replace perform *two* bf16 roundings: the product
// `gate * delta` is rounded, and the sum `hidden + (...)` is rounded again. Writing the pair as
// `__hadd2(a, __hmul2(b, c))` does not reproduce that: nvcc contracts it into a bf16x2 FMA
// (`fma.rn.bf16x2`), which rounds once and differs by one bf16 ulp on ~23% of the elements
// (measured). A rounding conversion is a barrier for that contraction, so the product is
// materialised as a bf16 value first; a bf16 product is exact in fp32, so the value is the same
// the eager kernel computes.
__device__ __forceinline__ __nv_bfloat162 bf16_mul_rounded(__nv_bfloat162 a, __nv_bfloat162 b) {
  const float2 af = __bfloat1622float2(a);
  const float2 bf = __bfloat1622float2(b);
  __nv_bfloat162 product;
  product.x = __float2bfloat16_rn(af.x * bf.x);
  product.y = __float2bfloat16_rn(af.y * bf.y);
  return product;
}

// `c + rn(a * b)`, with the product rounded before the add, i.e. exactly `c + a * b` in bf16
// tensor arithmetic.
__device__ __forceinline__ __nv_bfloat162 bf16_mul_add_rounded(__nv_bfloat162 a, __nv_bfloat162 b,
                                                               __nv_bfloat162 c) {
  return __hadd2(bf16_mul_rounded(a, b), c);
}

// `rn(b + rn(a * c))` for the shift add of a modulation chain.
__device__ __forceinline__ __nv_bfloat162 bf16_mul_add_then_add_rounded(__nv_bfloat162 a,
                                                                        __nv_bfloat162 b,
                                                                        __nv_bfloat162 c,
                                                                        __nv_bfloat162 d) {
  return __hadd2(bf16_mul_add_rounded(a, b, c), d);
}

}  // namespace cache_dit::models
