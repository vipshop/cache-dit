#pragma once

#include "fp8_common.cuh"

#if __has_include(<cuda_fp4.h>)
#include <cuda_fp4.h>
#define CACHE_DIT_ULYSSES_HAS_FP4 1
#else
#define CACHE_DIT_ULYSSES_HAS_FP4 0
#endif

namespace cache_dit::ulysses {
namespace {

#if CACHE_DIT_ULYSSES_HAS_FP4

constexpr float kFp4Max = 6.0f;
constexpr int kFp4FastPack = 16;
constexpr int kFp4ScalarPack = 2;
constexpr int kFp4GroupSize = 16;

__host__ __device__ int64_t fp4_payload_bytes(int64_t head_size) { return (head_size + 1) / 2; }

int64_t fp4_packed_min_width(int64_t head_size, int64_t group_size, int64_t scale_bytes) {
  return fp4_payload_bytes(head_size) + fp8_group_count(head_size, group_size) * scale_bytes;
}

void check_fp4_group_size(int64_t group_size) {
  TORCH_CHECK(group_size == kFp4GroupSize, "ulysses FP4 communication supports only group_size 16");
}

void check_fp4_device(const at::Tensor& x) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 10,
              "FP4 E2M1 communication requires sm100+ with native conversion support");
}

void check_fp4_pair(const at::Tensor& x, const at::Tensor& y) {
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), "communication operators require CUDA tensors");
  check_fp4_device(x);
  TORCH_CHECK(x.device() == y.device(), "input and output must be on the same CUDA device");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "expected BF16 input");
  TORCH_CHECK(y.scalar_type() == at::kByte, "expected packed uint8 FP4 output");
}

__device__ uint8_t to_fp4x2(float first, float second) {
  return __nv_cvt_float2_to_fp4x2({first, second}, __NV_E2M1, cudaRoundNearest);
}

__device__ float2 from_fp4x2(uint8_t packed) {
  const __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2(packed, __NV_E2M1);
  return __half22float2(static_cast<__half2>(raw));
}

// A group of 16 values is one lane of work in the fast path. Keeping the whole group in one lane
// removes the scale reduction/broadcast entirely; four independent groups form a D=64 row and eight
// groups form a D=128 row. Multirow kernels pack four such D=128 rows into one physical warp.
template <bool kBf16Scale>
__device__ void quantize_row_groups_fp4(const __nv_bfloat16* input, uint8_t* output,
                                        int64_t head_size, int64_t packed_head_size, int lane,
                                        int group_size, float epsilon,
                                        int lanes_per_row = kWarpSize) {
  const int64_t payload_bytes = fp4_payload_bytes(head_size);
  const int64_t groups = (head_size + group_size - 1) / group_size;
  if (head_size % kFp4FastPack == 0 && is_aligned_16(input) &&
      reinterpret_cast<uintptr_t>(output) % 8 == 0) {
    constexpr int lanes_per_group = kFp4GroupSize / kFp4FastPack;
    const int groups_per_iteration = lanes_per_row / lanes_per_group;
    for (int64_t first_group = 0; first_group < groups; first_group += groups_per_iteration) {
      const int64_t group = first_group + lane;
      const int64_t d = group * kFp4GroupSize;
      const bool active = group < groups && d < head_size;
      __nv_bfloat16 values[kFp4FastPack];
      float absmax = 0.0f;
      if (active) {
        load_bf16_pack(input + d, values);
        load_bf16_pack(input + d + kPackSize, values + kPackSize);
#pragma unroll
        for (int i = 0; i < kFp4FastPack; ++i) {
          absmax = fmaxf(absmax, fabsf(__bfloat162float(values[i])));
        }
      }
      const float scale = fmaxf(__fdiv_rn(absmax, kFp4Max), epsilon);
      const float inv_scale = __fdiv_rn(1.0f, scale);
      if (active) {
        uint32_t packed_lo = 0;
        uint32_t packed_hi = 0;
#pragma unroll
        for (int i = 0; i < kFp4FastPack; i += 2) {
          const uint32_t packed = to_fp4x2(__fmul_rn(__bfloat162float(values[i]), inv_scale),
                                           __fmul_rn(__bfloat162float(values[i + 1]), inv_scale));
          if (i < 8) {
            packed_lo |= packed << (i * 4);
          } else {
            packed_hi |= packed << ((i - 8) * 4);
          }
        }
        // Eight scalar STG.U8 stores per group create severe write-sector amplification. The row
        // guard above makes this uint2 store legal and coalesced across adjacent groups.
        *reinterpret_cast<uint2*>(output + d / 2) = make_uint2(packed_lo, packed_hi);
        write_group_scale<kBf16Scale>(output, payload_bytes, group, scale);
      }
    }
  } else {
    constexpr int lanes_per_group = kFp4GroupSize / kFp4ScalarPack;
    const int groups_per_iteration = lanes_per_row / lanes_per_group;
    for (int64_t first_group = 0; first_group < groups; first_group += groups_per_iteration) {
      const int lane_in_group = lane % lanes_per_group;
      const int64_t group = first_group + lane / lanes_per_group;
      const int64_t d = group * kFp4GroupSize + lane_in_group * kFp4ScalarPack;
      const bool active = group < groups && d < head_size;
      float first = 0.0f;
      float second = 0.0f;
      if (active) {
        first = __bfloat162float(input[d]);
        second = d + 1 < head_size ? __bfloat162float(input[d + 1]) : 0.0f;
      }
      const float absmax =
        segmented_absmax(active ? fmaxf(fabsf(first), fabsf(second)) : 0.0f, lanes_per_group);
      const float scale = fmaxf(__fdiv_rn(absmax, kFp4Max), epsilon);
      const float inv_scale = __fdiv_rn(1.0f, scale);
      if (active) {
        output[d / 2] = to_fp4x2(__fmul_rn(first, inv_scale), __fmul_rn(second, inv_scale));
      }
      if (group < groups && lane_in_group == 0) {
        write_group_scale<kBf16Scale>(output, payload_bytes, group, scale);
      }
    }
  }
  constexpr int scale_bytes = kBf16Scale ? 2 : 4;
  for (int64_t d = payload_bytes + groups * scale_bytes + lane; d < packed_head_size;
       d += lanes_per_row) {
    output[d] = 0;
  }
}

template <bool kBf16Scale>
__device__ void dequantize_row_groups_fp4(const uint8_t* input, __nv_bfloat16* output,
                                          int64_t head_size, int lane, int group_size,
                                          int lanes_per_row = kWarpSize) {
  const int shuffle_lane = threadIdx.x % kWarpSize;
  const int64_t payload_bytes = fp4_payload_bytes(head_size);
  const int64_t groups = (head_size + group_size - 1) / group_size;
  if (head_size % kFp4FastPack == 0 && is_aligned_16(output)) {
    constexpr int lanes_per_group = kFp4GroupSize / kFp4FastPack;
    const int groups_per_iteration = lanes_per_row / lanes_per_group;
    for (int64_t first_group = 0; first_group < groups; first_group += groups_per_iteration) {
      const int64_t group = first_group + lane;
      const int64_t d = group * kFp4GroupSize;
      const bool active = group < groups && d < head_size;
      if (active) {
        const float scale = read_group_scale<kBf16Scale>(input, payload_bytes, group);
        __nv_bfloat16 values[kFp4FastPack];
#pragma unroll
        for (int i = 0; i < kFp4FastPack; i += 2) {
          const float2 converted = from_fp4x2(input[d / 2 + i / 2]);
          values[i] = __float2bfloat16_rn(__fmul_rn(converted.x, scale));
          values[i + 1] = __float2bfloat16_rn(__fmul_rn(converted.y, scale));
        }
        store_bf16_pack(output + d, values);
        store_bf16_pack(output + d + kPackSize, values + kPackSize);
      }
    }
  } else {
    constexpr int lanes_per_group = kFp4GroupSize / kFp4ScalarPack;
    const int groups_per_iteration = lanes_per_row / lanes_per_group;
    for (int64_t first_group = 0; first_group < groups; first_group += groups_per_iteration) {
      const int lane_in_group = lane % lanes_per_group;
      const int64_t group = first_group + lane / lanes_per_group;
      const int64_t d = group * kFp4GroupSize + lane_in_group * kFp4ScalarPack;
      const bool active = group < groups && d < head_size;
      float scale = 0.0f;
      if (group < groups && lane_in_group == 0) {
        scale = read_group_scale<kBf16Scale>(input, payload_bytes, group);
      }
      scale = __shfl_sync(0xffffffff, scale, shuffle_lane - lane_in_group);
      if (active) {
        const float2 converted = from_fp4x2(input[d / 2]);
        output[d] = __float2bfloat16_rn(__fmul_rn(converted.x, scale));
        if (d + 1 < head_size) {
          output[d + 1] = __float2bfloat16_rn(__fmul_rn(converted.y, scale));
        }
      }
    }
  }
}

#endif

}  // namespace
}  // namespace cache_dit::ulysses
