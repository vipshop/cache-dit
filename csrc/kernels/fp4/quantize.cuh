#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cute/tensor.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/numeric_types.h>

#include <cstddef>
#include <cstdint>

namespace cache_dit::fp4 {

constexpr float kE2M1Max = 6.0f;
constexpr float kUe4m3Max = 448.0f;
constexpr float kFp4ScaleMax = kE2M1Max * kUe4m3Max;
constexpr int kSFVec = 16;
constexpr int kRowBlock = 128;
constexpr int kSFVecPerBlk = 4;

union Vec8BF16 {
  uint4 raw;
  __nv_bfloat16 elem[8];
};
static_assert(sizeof(Vec8BF16) == 16, "Vec8BF16 must be 128-bit");

__device__ __forceinline__ unsigned char cvt_f2_to_e2m1x2(float hi, float lo) {
  unsigned int tmp;
  asm volatile(
    "{\n"
    ".reg .b8 byte0;\n"
    ".reg .b8 byte1;\n"
    ".reg .b8 byte2;\n"
    ".reg .b8 byte3;\n"
    "cvt.rn.satfinite.e2m1x2.f32 byte0, %1, %2;\n"
    "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
    "}\n"
    : "=r"(tmp)
    : "f"(hi), "f"(lo));
  return static_cast<unsigned char>(tmp & 0xffu);
}

__device__ __forceinline__ unsigned char cvt_f2_to_ue4m3(float x) {
  return static_cast<unsigned char>(
    __nv_cvt_float2_to_fp8x2(make_float2(x, 0.0f), __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float cvt_ue4m3_to_f32(unsigned char scale) {
  return __half2float(__nv_cvt_fp8_to_halfraw(scale, __NV_E4M3));
}

__host__ __device__ constexpr int sf_byte_offset(int mn, int s, int nb_k) {
  return (mn / kRowBlock) * nb_k * 512 + (s / kSFVecPerBlk) * 512 + (mn % 32) * 16 +
         ((mn / 32) % 4) * 4 + (s % kSFVecPerBlk);
}

__host__ __device__ constexpr size_t sf_buffer_bytes(int mn, int k) {
  return static_cast<size_t>((mn + kRowBlock - 1) / kRowBlock) * (k / 64) * 512;
}

CUTE_HOST_DEVICE auto sf_gmem_layout(int mn, int k) {
  using Config = cutlass::detail::Sm1xxBlockScaledConfig<kSFVec>;
  return cute::tile_to_shape(typename Config::SfAtom{}, cute::make_shape(mn, k),
                             cute::Step<cute::_2, cute::_1>{});
}

}  // namespace cache_dit::fp4
