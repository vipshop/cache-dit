#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <cstring>

#include "ulysses.h"

namespace cache_dit::ulysses {
namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;
constexpr float kFp8Max = 448.0f;

__device__ float warp_absmax(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return __shfl_sync(0xffffffff, value, 0);
}

constexpr int kPackSize = 8;

__device__ uint8_t to_fp8(float value) {
  const float clipped = fminf(fmaxf(value, -kFp8Max), kFp8Max);
  const __half half_value = __float2half_rn(clipped);
  return __nv_cvt_halfraw_to_fp8(static_cast<__half_raw>(half_value), __NV_SATFINITE, __NV_E4M3);
}

__device__ float from_fp8(uint8_t value) {
  const __half_raw raw = __nv_cvt_fp8_to_halfraw(value, __NV_E4M3);
  return __half2float(static_cast<__half>(raw));
}

__device__ bool is_aligned_16(const void* pointer) {
  return reinterpret_cast<uintptr_t>(pointer) % 16 == 0;
}

__device__ void load_bf16_pack(const __nv_bfloat16* input, __nv_bfloat16* values) {
  const uint4 pack = *reinterpret_cast<const uint4*>(input);
  std::memcpy(values, &pack, sizeof(pack));
}

__device__ void store_bf16_pack(__nv_bfloat16* output, const __nv_bfloat16* values) {
  uint4 pack;
  std::memcpy(&pack, values, sizeof(pack));
  *reinterpret_cast<uint4*>(output) = pack;
}

__device__ void load_fp8_pack(const uint8_t* input, uint8_t* values) {
  if (is_aligned_16(input)) {
    const uint4 pack = *reinterpret_cast<const uint4*>(input);
    std::memcpy(values, &pack, sizeof(pack));
  } else if (reinterpret_cast<uintptr_t>(input) % 4 == 0) {
#pragma unroll
    for (int i = 0; i < kPackSize / 4; ++i) {
      const uint32_t pack = *reinterpret_cast<const uint32_t*>(input + i * 4);
      std::memcpy(values + i * 4, &pack, sizeof(pack));
    }
  } else if (reinterpret_cast<uintptr_t>(input) % 2 == 0) {
#pragma unroll
    for (int i = 0; i < kPackSize / 2; ++i) {
      const uint16_t pack = *reinterpret_cast<const uint16_t*>(input + i * 2);
      std::memcpy(values + i * 2, &pack, sizeof(pack));
    }
  } else {
#pragma unroll
    for (int i = 0; i < kPackSize; ++i) {
      values[i] = input[i];
    }
  }
}

__device__ void store_fp8_pack(uint8_t* output, const uint8_t* values) {
  if (is_aligned_16(output)) {
    uint4 pack;
    std::memcpy(&pack, values, sizeof(pack));
    *reinterpret_cast<uint4*>(output) = pack;
  } else if (reinterpret_cast<uintptr_t>(output) % 4 == 0) {
#pragma unroll
    for (int i = 0; i < kPackSize / 4; ++i) {
      uint32_t pack;
      std::memcpy(&pack, values + i * 4, sizeof(pack));
      *reinterpret_cast<uint32_t*>(output + i * 4) = pack;
    }
  } else if (reinterpret_cast<uintptr_t>(output) % 2 == 0) {
#pragma unroll
    for (int i = 0; i < kPackSize / 2; ++i) {
      uint16_t pack;
      std::memcpy(&pack, values + i * 2, sizeof(pack));
      *reinterpret_cast<uint16_t*>(output + i * 2) = pack;
    }
  } else {
#pragma unroll
    for (int i = 0; i < kPackSize; ++i) {
      output[i] = values[i];
    }
  }
}

__device__ float row_absmax(const __nv_bfloat16* input, int64_t head_size, int lane) {
  float absmax = 0.0f;
  if (head_size % kPackSize == 0 && is_aligned_16(input)) {
    for (int64_t d = lane * kPackSize; d < head_size; d += kWarpSize * kPackSize) {
      __nv_bfloat16 values[kPackSize];
      load_bf16_pack(input + d, values);
#pragma unroll
      for (int i = 0; i < kPackSize; ++i) {
        absmax = fmaxf(absmax, fabsf(__bfloat162float(values[i])));
      }
    }
  } else {
    for (int64_t d = lane; d < head_size; d += kWarpSize) {
      absmax = fmaxf(absmax, fabsf(__bfloat162float(input[d])));
    }
  }
  return warp_absmax(absmax);
}

__device__ void quantize_row(const __nv_bfloat16* input, uint8_t* output, int64_t head_size,
                             int lane, float inv_scale) {
  if (head_size % kPackSize == 0 && is_aligned_16(input)) {
    for (int64_t d = lane * kPackSize; d < head_size; d += kWarpSize * kPackSize) {
      __nv_bfloat16 values[kPackSize];
      uint8_t quantized[kPackSize];
      load_bf16_pack(input + d, values);
#pragma unroll
      for (int i = 0; i < kPackSize; ++i) {
        quantized[i] = to_fp8(__bfloat162float(values[i]) * inv_scale);
      }
      store_fp8_pack(output + d, quantized);
    }
  } else {
    for (int64_t d = lane; d < head_size; d += kWarpSize) {
      output[d] = to_fp8(__bfloat162float(input[d]) * inv_scale);
    }
  }
}

__device__ void dequantize_row(const uint8_t* input, __nv_bfloat16* output, int64_t head_size,
                               int lane, float scale) {
  if (head_size % kPackSize == 0 && is_aligned_16(output)) {
    for (int64_t d = lane * kPackSize; d < head_size; d += kWarpSize * kPackSize) {
      uint8_t quantized[kPackSize];
      __nv_bfloat16 values[kPackSize];
      load_fp8_pack(input + d, quantized);
#pragma unroll
      for (int i = 0; i < kPackSize; ++i) {
        values[i] = __float2bfloat16_rn(from_fp8(quantized[i]) * scale);
      }
      store_bf16_pack(output + d, values);
    }
  } else {
    for (int64_t d = lane; d < head_size; d += kWarpSize) {
      output[d] = __float2bfloat16_rn(from_fp8(input[d]) * scale);
    }
  }
}

int64_t fp8_group_count(int64_t head_size, int64_t group_size) {
  return group_size == 0 ? 1 : (head_size + group_size - 1) / group_size;
}

int64_t fp8_packed_min_width(int64_t head_size, int64_t group_size, int64_t scale_bytes) {
  return head_size + fp8_group_count(head_size, group_size) * scale_bytes;
}

void check_fp8_group_size(int64_t group_size) {
  TORCH_CHECK(group_size == 0 || group_size == 16 || group_size == 32,
              "ulysses FP8 communication supports group_size 0 (per-token), 16, or 32");
}

__device__ float segmented_absmax(float value, int lanes_per_group) {
  for (int offset = 1; offset < lanes_per_group; offset *= 2) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
  }
  return value;
}

template <bool kBf16Scale>
__device__ void write_group_scale(uint8_t* input, int64_t head_size, int64_t group, float scale) {
  uint8_t* output = input + head_size + group * (kBf16Scale ? 2 : 4);
  if constexpr (kBf16Scale) {
    const __nv_bfloat16 stored_scale = __float2bfloat16_rn(scale);
    std::memcpy(output, &stored_scale, sizeof(stored_scale));
  } else {
    std::memcpy(output, &scale, sizeof(scale));
  }
}

template <bool kBf16Scale>
__device__ float read_group_scale(const uint8_t* input, int64_t head_size, int64_t group) {
  const uint8_t* scale_ptr = input + head_size + group * (kBf16Scale ? 2 : 4);
  if constexpr (kBf16Scale) {
    __nv_bfloat16 scale;
    std::memcpy(&scale, scale_ptr, sizeof(scale));
    return __bfloat162float(scale);
  } else {
    float scale;
    std::memcpy(&scale, scale_ptr, sizeof(scale));
    return scale;
  }
}

template <bool kBf16Scale>
__device__ void quantize_row_groups(const __nv_bfloat16* input, uint8_t* output, int64_t head_size,
                                    int64_t packed_head_size, int lane, int group_size,
                                    float epsilon) {
  const int64_t groups = (head_size + group_size - 1) / group_size;
  if (head_size % kPackSize == 0 && is_aligned_16(input)) {
    // A warp handles 256 elements per iteration. Since 256 % group_size == 0 for 16/32,
    // no group crosses an iteration boundary and 2/4 lanes can reduce it independently.
    const int lanes_per_group = group_size / kPackSize;
    const int groups_per_iteration = kWarpSize / lanes_per_group;
    for (int64_t first_group = 0; first_group < groups; first_group += groups_per_iteration) {
      const int lane_in_group = lane % lanes_per_group;
      const int64_t group = first_group + lane / lanes_per_group;
      const int64_t d = group * group_size + lane_in_group * kPackSize;
      const bool active = group < groups && d < head_size;
      __nv_bfloat16 values[kPackSize];
      float absmax = 0.0f;
      if (active) {
        load_bf16_pack(input + d, values);
#pragma unroll
        for (int i = 0; i < kPackSize; ++i) {
          absmax = fmaxf(absmax, fabsf(__bfloat162float(values[i])));
        }
      }
      absmax = segmented_absmax(absmax, lanes_per_group);
      const float scale = fmaxf(absmax / kFp8Max, epsilon);
      const float inv_scale = 1.0f / scale;
      if (active) {
        uint8_t quantized[kPackSize];
#pragma unroll
        for (int i = 0; i < kPackSize; ++i) {
          quantized[i] = to_fp8(__bfloat162float(values[i]) * inv_scale);
        }
        store_fp8_pack(output + d, quantized);
      }
      if (group < groups && lane_in_group == 0) {
        write_group_scale<kBf16Scale>(output, head_size, group, scale);
      }
    }
  } else {
    const int groups_per_iteration = kWarpSize / group_size;
    for (int64_t first_group = 0; first_group < groups; first_group += groups_per_iteration) {
      const int lane_in_group = lane % group_size;
      const int64_t group = first_group + lane / group_size;
      const int64_t d = group * group_size + lane_in_group;
      const bool active = group < groups && d < head_size;
      float value = 0.0f;
      if (active) {
        value = __bfloat162float(input[d]);
      }
      const float absmax = segmented_absmax(active ? fabsf(value) : 0.0f, group_size);
      const float scale = fmaxf(absmax / kFp8Max, epsilon);
      if (active) {
        output[d] = to_fp8(value * (1.0f / scale));
      }
      if (group < groups && lane_in_group == 0) {
        write_group_scale<kBf16Scale>(output, head_size, group, scale);
      }
    }
  }
  constexpr int scale_bytes = kBf16Scale ? 2 : 4;
  for (int64_t d = head_size + groups * scale_bytes + lane; d < packed_head_size; d += kWarpSize) {
    output[d] = 0;
  }
}

template <bool kBf16Scale>
__device__ void dequantize_row_groups(const uint8_t* input, __nv_bfloat16* output,
                                      int64_t head_size, int lane, int group_size) {
  const int64_t groups = (head_size + group_size - 1) / group_size;
  if (head_size % kPackSize == 0 && is_aligned_16(output)) {
    const int lanes_per_group = group_size / kPackSize;
    const int groups_per_iteration = kWarpSize / lanes_per_group;
    for (int64_t first_group = 0; first_group < groups; first_group += groups_per_iteration) {
      const int lane_in_group = lane % lanes_per_group;
      const int64_t group = first_group + lane / lanes_per_group;
      const int64_t d = group * group_size + lane_in_group * kPackSize;
      const bool active = group < groups && d < head_size;
      float scale = 0.0f;
      if (group < groups && lane_in_group == 0) {
        scale = read_group_scale<kBf16Scale>(input, head_size, group);
      }
      scale = __shfl_sync(0xffffffff, scale, lane - lane_in_group);
      if (active) {
        uint8_t quantized[kPackSize];
        __nv_bfloat16 values[kPackSize];
        load_fp8_pack(input + d, quantized);
#pragma unroll
        for (int i = 0; i < kPackSize; ++i) {
          values[i] = __float2bfloat16_rn(from_fp8(quantized[i]) * scale);
        }
        store_bf16_pack(output + d, values);
      }
    }
  } else {
    const int groups_per_iteration = kWarpSize / group_size;
    for (int64_t first_group = 0; first_group < groups; first_group += groups_per_iteration) {
      const int lane_in_group = lane % group_size;
      const int64_t group = first_group + lane / group_size;
      const int64_t d = group * group_size + lane_in_group;
      const bool active = group < groups && d < head_size;
      float scale = 0.0f;
      if (group < groups && lane_in_group == 0) {
        scale = read_group_scale<kBf16Scale>(input, head_size, group);
      }
      scale = __shfl_sync(0xffffffff, scale, lane - lane_in_group);
      if (active) {
        output[d] = __float2bfloat16_rn(from_fp8(input[d]) * scale);
      }
    }
  }
}

void check_fp8_pair(const at::Tensor& x, const at::Tensor& y) {
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), "communication operators require CUDA tensors");
  TORCH_CHECK(x.device() == y.device(), "input and output must be on the same CUDA device");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "expected BF16 input");
  TORCH_CHECK(y.scalar_type() == at::kFloat8_e4m3fn, "expected FP8 E4M3 output");
}

}  // namespace
}  // namespace cache_dit::ulysses
