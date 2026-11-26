#pragma once

#include "fp8_common.cuh"

#if __has_include(<cuda_fp6.h>)
#include <cuda_fp6.h>
#define CACHE_DIT_ULYSSES_HAS_FP6 1
#else
#define CACHE_DIT_ULYSSES_HAS_FP6 0
#endif

namespace cache_dit::ulysses {
namespace {

#if CACHE_DIT_ULYSSES_HAS_FP6

constexpr float kFp6Max = 7.5f;
constexpr int kFp6FastPack = 16;
constexpr int kFp6ScalarPack = 4;

__host__ __device__ int64_t fp6_payload_bytes(int64_t head_size) {
  const int64_t bitstream_bytes = ((head_size + 3) / 4) * 3;
  return (bitstream_bytes + 3) / 4 * 4;
}

int64_t fp6_packed_min_width(int64_t head_size, int64_t group_size, int64_t scale_bytes) {
  return fp6_payload_bytes(head_size) + fp8_group_count(head_size, group_size) * scale_bytes;
}

void check_fp6_group_size(int64_t group_size) {
  TORCH_CHECK(group_size == 0 || group_size == 16 || group_size == 32,
              "ulysses FP6 communication supports group_size 0 (per-token), 16, or 32");
}

void check_fp6_device(const at::Tensor& x) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 10,
              "FP6 E2M3 communication requires sm100+ with native conversion support");
}

void check_fp6_pair(const at::Tensor& x, const at::Tensor& y) {
  check_fp6_device(x);
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), "communication operators require CUDA tensors");
  TORCH_CHECK(x.device() == y.device(), "input and output must be on the same CUDA device");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "expected BF16 input");
  TORCH_CHECK(y.scalar_type() == at::kByte, "expected packed uint8 FP6 output");
}

__device__ uint8_t to_fp6(float value) {
  return __nv_cvt_float_to_fp6(value, __NV_E2M3, cudaRoundNearest) & 0x3Fu;
}

__device__ float from_fp6(uint8_t value) {
  const __half_raw raw = __nv_cvt_fp6_to_halfraw(value & 0x3Fu, __NV_E2M3);
  return __half2float(static_cast<__half>(raw));
}

__device__ void store_fp6x4(uint8_t* output, const uint8_t* codes) {
  const uint8_t byte0 = codes[0] | (codes[1] << 6);
  const uint8_t byte1 = (codes[1] >> 2) | (codes[2] << 4);
  const uint8_t byte2 = (codes[2] >> 4) | (codes[3] << 2);
  if (reinterpret_cast<uintptr_t>(output) % 2 == 0) {
    *reinterpret_cast<uint16_t*>(output) =
      static_cast<uint16_t>(byte0) | (static_cast<uint16_t>(byte1) << 8);
    output[2] = byte2;
  } else {
    output[0] = byte0;
    output[1] = byte1;
    output[2] = byte2;
  }
}

__device__ void load_fp6x4(const uint8_t* input, uint8_t* codes) {
  uint8_t bytes[3];
  if (reinterpret_cast<uintptr_t>(input) % 2 == 0) {
    const uint16_t first = *reinterpret_cast<const uint16_t*>(input);
    bytes[0] = first & 0xff;
    bytes[1] = first >> 8;
  } else {
    bytes[0] = input[0];
    bytes[1] = input[1];
  }
  bytes[2] = input[2];
  codes[0] = bytes[0] & 0x3f;
  codes[1] = ((bytes[0] >> 6) | (bytes[1] << 2)) & 0x3f;
  codes[2] = ((bytes[1] >> 4) | (bytes[2] << 4)) & 0x3f;
  codes[3] = (bytes[2] >> 2) & 0x3f;
}

__device__ void store_fp6x16(uint8_t* output, const uint8_t* codes) {
  if (reinterpret_cast<uintptr_t>(output) % 4 != 0) {
#pragma unroll
    for (int quarter = 0; quarter < 4; ++quarter) {
      store_fp6x4(output + quarter * 3, codes + quarter * 4);
    }
    return;
  }
  uint32_t packed[4];
#pragma unroll
  for (int quarter = 0; quarter < 4; ++quarter) {
    const uint8_t* c = codes + quarter * 4;
    packed[quarter] =
      (c[0] & 0x3f) | ((c[1] & 0x3f) << 6) | ((c[2] & 0x3f) << 12) | ((c[3] & 0x3f) << 18);
  }
  // Sixteen FP6 codes occupy 12 bytes, so three u32 stores exactly cover one group's payload.
  // Wider per-thread stores are not automatically better for this 12-byte-strided lane layout.
  // A uint2+u32 experiment increased registers 56 -> 64, global-store sectors 319488 -> 466944,
  // requests 43008 -> 55296, and nsys 4.927 -> 4.992 us. Keep these exact-width u32 stores unless
  // a different lane/packing layout demonstrates better NCU coalescing and end-to-end latency.
  uint32_t* words = reinterpret_cast<uint32_t*>(output);
  words[0] = packed[0] | ((packed[1] & 0xff) << 24);
  words[1] = (packed[1] >> 8) | ((packed[2] & 0xffff) << 16);
  words[2] = (packed[2] >> 16) | (packed[3] << 8);
}

__device__ void load_fp6x16(const uint8_t* input, uint8_t* codes) {
  if (reinterpret_cast<uintptr_t>(input) % 4 != 0) {
#pragma unroll
    for (int quarter = 0; quarter < 4; ++quarter) {
      load_fp6x4(input + quarter * 3, codes + quarter * 4);
    }
    return;
  }
  const uint32_t* words = reinterpret_cast<const uint32_t*>(input);
  const uint32_t word0 = words[0];
  const uint32_t word1 = words[1];
  const uint32_t word2 = words[2];
  const uint32_t packed[4] = {
    word0 & 0xffffff,
    (word0 >> 24) | ((word1 & 0xffff) << 8),
    (word1 >> 16) | ((word2 & 0xff) << 16),
    word2 >> 8,
  };
#pragma unroll
  for (int quarter = 0; quarter < 4; ++quarter) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      codes[quarter * 4 + i] = (packed[quarter] >> (i * 6)) & 0x3f;
    }
  }
}

template <bool kBf16Scale>
__device__ void quantize_row_fp6(const __nv_bfloat16* input, uint8_t* output, int64_t head_size,
                                 int64_t packed_head_size, int lane, float scale) {
  const int64_t payload_bytes = fp6_payload_bytes(head_size);
  const float inv_scale = __fdiv_rn(1.0f, scale);
  if (head_size % kFp6FastPack == 0 && is_aligned_16(input)) {
    for (int64_t d = lane * kFp6FastPack; d < head_size; d += kWarpSize * kFp6FastPack) {
      __nv_bfloat16 values[kFp6FastPack];
      uint8_t codes[kFp6FastPack];
      load_bf16_pack(input + d, values);
      load_bf16_pack(input + d + kPackSize, values + kPackSize);
#pragma unroll
      for (int i = 0; i < kFp6FastPack; ++i) {
        codes[i] = to_fp6(__fmul_rn(__bfloat162float(values[i]), inv_scale));
      }
      store_fp6x16(output + (d / 4) * 3, codes);
    }
  } else {
    for (int64_t d = lane * kFp6ScalarPack; d < head_size; d += kWarpSize * kFp6ScalarPack) {
      uint8_t codes[kFp6ScalarPack] = {0, 0, 0, 0};
#pragma unroll
      for (int i = 0; i < kFp6ScalarPack; ++i) {
        if (d + i < head_size) {
          codes[i] = to_fp6(__fmul_rn(__bfloat162float(input[d + i]), inv_scale));
        }
      }
      store_fp6x4(output + (d / 4) * 3, codes);
    }
  }
  const int64_t bitstream_bytes = ((head_size + 3) / 4) * 3;
  for (int64_t d = bitstream_bytes + lane; d < payload_bytes; d += kWarpSize) {
    output[d] = 0;
  }
  if (lane == 0) {
    write_group_scale<kBf16Scale>(output, payload_bytes, 0, scale);
  }
  for (int64_t d = payload_bytes + (kBf16Scale ? 2 : 4) + lane; d < packed_head_size;
       d += kWarpSize) {
    output[d] = 0;
  }
}

template <bool kBf16Scale>
__device__ void quantize_row_per_token_fp6(const __nv_bfloat16* input, uint8_t* output,
                                           int64_t head_size, int64_t packed_head_size, int lane,
                                           float epsilon, int lanes_per_row = kWarpSize) {
  if (head_size > kWarpSize * kFp6FastPack || head_size % kFp6FastPack != 0 ||
      !is_aligned_16(input)) {
    const float absmax = row_absmax(input, head_size, lane);
    const float scale = fmaxf(__fdiv_rn(absmax, kFp6Max), epsilon);
    quantize_row_fp6<kBf16Scale>(input, output, head_size, packed_head_size, lane, scale);
    return;
  }

  const int64_t payload_bytes = fp6_payload_bytes(head_size);
  const int64_t d = lane * kFp6FastPack;
  const bool active = d < head_size;
  __nv_bfloat16 values[kFp6FastPack];
  float absmax = 0.0f;
  if (active) {
    load_bf16_pack(input + d, values);
    load_bf16_pack(input + d + kPackSize, values + kPackSize);
#pragma unroll
    for (int i = 0; i < kFp6FastPack; ++i) {
      absmax = fmaxf(absmax, fabsf(__bfloat162float(values[i])));
    }
  }
  absmax = segmented_absmax(absmax, lanes_per_row);
  const float scale = fmaxf(__fdiv_rn(absmax, kFp6Max), epsilon);
  const float inv_scale = __fdiv_rn(1.0f, scale);
  if (active) {
    uint8_t codes[kFp6FastPack];
#pragma unroll
    for (int i = 0; i < kFp6FastPack; ++i) {
      codes[i] = to_fp6(__fmul_rn(__bfloat162float(values[i]), inv_scale));
    }
    store_fp6x16(output + (d / 4) * 3, codes);
  }
  const int64_t bitstream_bytes = ((head_size + 3) / 4) * 3;
  for (int64_t padding = bitstream_bytes + lane; padding < payload_bytes;
       padding += lanes_per_row) {
    output[padding] = 0;
  }
  if (lane == 0) {
    write_group_scale<kBf16Scale>(output, payload_bytes, 0, scale);
  }
  for (int64_t padding = payload_bytes + (kBf16Scale ? 2 : 4) + lane; padding < packed_head_size;
       padding += lanes_per_row) {
    output[padding] = 0;
  }
}

template <bool kBf16Scale>
__device__ void quantize_row_groups_fp6(const __nv_bfloat16* input, uint8_t* output,
                                        int64_t head_size, int64_t packed_head_size, int lane,
                                        int group_size, float epsilon,
                                        int lanes_per_row = kWarpSize) {
  const int64_t payload_bytes = fp6_payload_bytes(head_size);
  const int64_t groups = (head_size + group_size - 1) / group_size;
  if (head_size % kFp6FastPack == 0 && is_aligned_16(input)) {
    const int lanes_per_group = group_size / kFp6FastPack;
    const int groups_per_iteration = lanes_per_row / lanes_per_group;
    for (int64_t first_group = 0; first_group < groups; first_group += groups_per_iteration) {
      const int lane_in_group = lane % lanes_per_group;
      const int64_t group = first_group + lane / lanes_per_group;
      const int64_t d = group * group_size + lane_in_group * kFp6FastPack;
      const bool active = group < groups && d < head_size;
      __nv_bfloat16 values[kFp6FastPack];
      float absmax = 0.0f;
      if (active) {
        load_bf16_pack(input + d, values);
        load_bf16_pack(input + d + kPackSize, values + kPackSize);
#pragma unroll
        for (int i = 0; i < kFp6FastPack; ++i) {
          absmax = fmaxf(absmax, fabsf(__bfloat162float(values[i])));
        }
      }
      absmax = segmented_absmax(absmax, lanes_per_group);
      const float scale = fmaxf(__fdiv_rn(absmax, kFp6Max), epsilon);
      const float inv_scale = __fdiv_rn(1.0f, scale);
      if (active) {
        uint8_t codes[kFp6FastPack];
#pragma unroll
        for (int i = 0; i < kFp6FastPack; ++i) {
          codes[i] = to_fp6(__fmul_rn(__bfloat162float(values[i]), inv_scale));
        }
        store_fp6x16(output + (d / 4) * 3, codes);
      }
      if (group < groups && lane_in_group == 0) {
        write_group_scale<kBf16Scale>(output, payload_bytes, group, scale);
      }
    }
  } else {
    const int lanes_per_group = group_size / kFp6ScalarPack;
    const int groups_per_iteration = lanes_per_row / lanes_per_group;
    for (int64_t first_group = 0; first_group < groups; first_group += groups_per_iteration) {
      const int lane_in_group = lane % lanes_per_group;
      const int64_t group = first_group + lane / lanes_per_group;
      const int64_t d = group * group_size + lane_in_group * kFp6ScalarPack;
      const bool active = group < groups && d < head_size;
      uint8_t codes[kFp6ScalarPack] = {0, 0, 0, 0};
      float absmax = 0.0f;
      float values[kFp6ScalarPack] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int i = 0; i < kFp6ScalarPack; ++i) {
        if (active && d + i < head_size) {
          values[i] = __bfloat162float(input[d + i]);
          absmax = fmaxf(absmax, fabsf(values[i]));
        }
      }
      absmax = segmented_absmax(absmax, lanes_per_group);
      const float scale = fmaxf(__fdiv_rn(absmax, kFp6Max), epsilon);
      const float inv_scale = __fdiv_rn(1.0f, scale);
      if (active) {
#pragma unroll
        for (int i = 0; i < kFp6ScalarPack; ++i) {
          if (d + i < head_size) {
            codes[i] = to_fp6(__fmul_rn(values[i], inv_scale));
          }
        }
        store_fp6x4(output + (d / 4) * 3, codes);
      }
      if (group < groups && lane_in_group == 0) {
        write_group_scale<kBf16Scale>(output, payload_bytes, group, scale);
      }
    }
  }
  const int64_t bitstream_bytes = ((head_size + 3) / 4) * 3;
  for (int64_t d = bitstream_bytes + lane; d < payload_bytes; d += lanes_per_row) {
    output[d] = 0;
  }
  constexpr int scale_bytes = kBf16Scale ? 2 : 4;
  for (int64_t d = payload_bytes + groups * scale_bytes + lane; d < packed_head_size;
       d += lanes_per_row) {
    output[d] = 0;
  }
}

template <bool kBf16Scale>
__device__ void dequantize_row_fp6(const uint8_t* input, __nv_bfloat16* output, int64_t head_size,
                                   int lane, float scale, int lanes_per_row = kWarpSize) {
  if (head_size % kFp6FastPack == 0 && is_aligned_16(output)) {
    for (int64_t d = lane * kFp6FastPack; d < head_size; d += lanes_per_row * kFp6FastPack) {
      uint8_t codes[kFp6FastPack];
      __nv_bfloat16 values[kFp6FastPack];
      load_fp6x16(input + (d / 4) * 3, codes);
#pragma unroll
      for (int i = 0; i < kFp6FastPack; ++i) {
        values[i] = __float2bfloat16_rn(__fmul_rn(from_fp6(codes[i]), scale));
      }
      store_bf16_pack(output + d, values);
      store_bf16_pack(output + d + kPackSize, values + kPackSize);
    }
  } else {
    for (int64_t d = lane * kFp6ScalarPack; d < head_size; d += lanes_per_row * kFp6ScalarPack) {
      uint8_t codes[kFp6ScalarPack];
      load_fp6x4(input + (d / 4) * 3, codes);
#pragma unroll
      for (int i = 0; i < kFp6ScalarPack; ++i) {
        if (d + i < head_size) {
          output[d + i] = __float2bfloat16_rn(__fmul_rn(from_fp6(codes[i]), scale));
        }
      }
    }
  }
}

template <bool kBf16Scale>
__device__ void dequantize_row_groups_fp6(const uint8_t* input, __nv_bfloat16* output,
                                          int64_t head_size, int lane, int group_size,
                                          int lanes_per_row = kWarpSize) {
  const int shuffle_lane = threadIdx.x % kWarpSize;
  const int64_t payload_bytes = fp6_payload_bytes(head_size);
  const int64_t groups = (head_size + group_size - 1) / group_size;
  if (head_size % kFp6FastPack == 0 && is_aligned_16(output)) {
    const int lanes_per_group = group_size / kFp6FastPack;
    const int groups_per_iteration = lanes_per_row / lanes_per_group;
    for (int64_t first_group = 0; first_group < groups; first_group += groups_per_iteration) {
      const int lane_in_group = lane % lanes_per_group;
      const int64_t group = first_group + lane / lanes_per_group;
      const int64_t d = group * group_size + lane_in_group * kFp6FastPack;
      const bool active = group < groups && d < head_size;
      float scale = 0.0f;
      if (group < groups && lane_in_group == 0) {
        scale = read_group_scale<kBf16Scale>(input, payload_bytes, group);
      }
      scale = __shfl_sync(0xffffffff, scale, shuffle_lane - lane_in_group);
      if (active) {
        uint8_t codes[kFp6FastPack];
        __nv_bfloat16 values[kFp6FastPack];
        load_fp6x16(input + (d / 4) * 3, codes);
#pragma unroll
        for (int i = 0; i < kFp6FastPack; ++i) {
          values[i] = __float2bfloat16_rn(__fmul_rn(from_fp6(codes[i]), scale));
        }
        store_bf16_pack(output + d, values);
        store_bf16_pack(output + d + kPackSize, values + kPackSize);
      }
    }
  } else {
    const int lanes_per_group = group_size / kFp6ScalarPack;
    const int groups_per_iteration = lanes_per_row / lanes_per_group;
    for (int64_t first_group = 0; first_group < groups; first_group += groups_per_iteration) {
      const int lane_in_group = lane % lanes_per_group;
      const int64_t group = first_group + lane / lanes_per_group;
      const int64_t d = group * group_size + lane_in_group * kFp6ScalarPack;
      const bool active = group < groups && d < head_size;
      float scale = 0.0f;
      if (group < groups && lane_in_group == 0) {
        scale = read_group_scale<kBf16Scale>(input, payload_bytes, group);
      }
      scale = __shfl_sync(0xffffffff, scale, shuffle_lane - lane_in_group);
      if (active) {
        uint8_t codes[kFp6ScalarPack];
        load_fp6x4(input + (d / 4) * 3, codes);
#pragma unroll
        for (int i = 0; i < kFp6ScalarPack; ++i) {
          if (d + i < head_size) {
            output[d + i] = __float2bfloat16_rn(__fmul_rn(from_fp6(codes[i]), scale));
          }
        }
      }
    }
  }
}

#endif

}  // namespace
}  // namespace cache_dit::ulysses
