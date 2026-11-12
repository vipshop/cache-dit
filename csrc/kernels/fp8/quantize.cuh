#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace cache_dit::fp8 {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;
constexpr int kScaleBlock = 128;
constexpr float kE4M3Max = 448.0f;

union Vec4BF16 {
  uint2 raw;
  __nv_bfloat16 elem[4];
};

union Vec8BF16 {
  uint4 raw;
  __nv_bfloat16 elem[8];
};

static_assert(sizeof(Vec4BF16) == 8, "Vec4BF16 must be 64-bit");
static_assert(sizeof(Vec8BF16) == 16, "Vec8BF16 must be 128-bit");

__device__ __forceinline__ unsigned short cvt_f2_to_e4m3x2(float lo, float hi) {
  return __nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ void store_e4m3x8(__nv_fp8_e4m3* output, const float* values,
                                             float scale) {
  uint2 packed;
  auto* halves = reinterpret_cast<unsigned short*>(&packed);
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    halves[i / 2] = cvt_f2_to_e4m3x2(values[i] / scale, values[i + 1] / scale);
  }
  *reinterpret_cast<uint2*>(output) = packed;
}

}  // namespace cache_dit::fp8
