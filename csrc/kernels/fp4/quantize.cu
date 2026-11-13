#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>

#include <cstdint>

#include "fp4.h"
#include "quantize.cuh"

namespace cache_dit::fp4 {
namespace {

__global__ void row_amax_fp4_kernel(const __nv_bfloat16* __restrict__ input,
                                    float* __restrict__ global_scales, int rows, int cols) {
  const int row = blockIdx.x * 8 + threadIdx.x / 32;
  if (row >= rows) {
    return;
  }
  const int lane = threadIdx.x % 32;
  const __nv_bfloat16* src = input + static_cast<size_t>(row) * cols;
  float amax = 0.0f;
  for (int col = lane * 8; col < cols; col += 32 * 8) {
    const Vec8BF16 value = *reinterpret_cast<const Vec8BF16*>(src + col);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      amax = fmaxf(amax, fabsf(__bfloat162float(value.elem[i])));
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
  }
  if (lane == 0) {
    global_scales[row] = __fdiv_rn(amax, kFp4ScaleMax);
  }
}

template <int kK, int kThreads>
__global__ void __launch_bounds__(kThreads, 1)
  quantize_fp4_fused_kernel(const __nv_bfloat16* __restrict__ input,
                            cutlass::float_e2m1_t* __restrict__ qdata,
                            cutlass::float_ue4m3_t* __restrict__ block_scales,
                            float* __restrict__ global_scales, int rows) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  constexpr int kChunks = kK / 64;
  constexpr int kGroups = kK / kSFVec;
  constexpr int kWarps = kThreads / 32;
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  extern __shared__ uint8_t fused_smem[];
  auto* x_cache = reinterpret_cast<Vec8BF16*>(fused_smem);
  auto* group_amax = reinterpret_cast<float*>(fused_smem + kK * sizeof(__nv_bfloat16));
  auto* warp_amax = group_amax + kGroups;
  auto* row_params = warp_amax + kWarps;
  const auto* input_row = input + static_cast<size_t>(row) * kK;
  auto* qdata_row = reinterpret_cast<uint8_t*>(qdata) + static_cast<size_t>(row) * (kK / 2);
  auto* scales_bytes = reinterpret_cast<uint8_t*>(block_scales);

  float thread_amax = 0.0f;
  for (int chunk = tid; chunk < kChunks; chunk += kThreads) {
    float local_amax[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int vec = 0; vec < 8; ++vec) {
      const int index = chunk * 8 + vec;
      const Vec8BF16 values = reinterpret_cast<const Vec8BF16*>(input_row)[index];
      x_cache[index] = values;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        local_amax[vec / 2] = fmaxf(local_amax[vec / 2], fabsf(__bfloat162float(values.elem[j])));
      }
    }
#pragma unroll
    for (int group = 0; group < 4; ++group) {
      group_amax[chunk * 4 + group] = local_amax[group];
      thread_amax = fmaxf(thread_amax, local_amax[group]);
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    thread_amax = fmaxf(thread_amax, __shfl_xor_sync(0xffffffff, thread_amax, offset));
  }
  if (lane == 0) {
    warp_amax[warp] = thread_amax;
  }
  __syncthreads();
  if (tid == 0) {
    float amax = warp_amax[0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) {
      amax = fmaxf(amax, warp_amax[w]);
    }
    const float global_scale = __fdiv_rn(amax, kFp4ScaleMax);
    const float inv_global = global_scale > 0.0f ? __fdiv_rn(1.0f, global_scale) : 0.0f;
    global_scales[row] = global_scale;
    row_params[0] = inv_global;
  }
  __syncthreads();
  const float inv_global = row_params[0];

  for (int chunk = tid; chunk < kChunks; chunk += kThreads) {
    uint4 packed0 = {};
    uint4 packed1 = {};
    auto* packed_words0 = reinterpret_cast<uint32_t*>(&packed0);
    auto* packed_words1 = reinterpret_cast<uint32_t*>(&packed1);
    uint32_t scale_word = 0;
#pragma unroll
    for (int group = 0; group < 4; ++group) {
      float values[kSFVec];
      float normalized_amax = 0.0f;
#pragma unroll
      for (int vec = 0; vec < 2; ++vec) {
        const Vec8BF16 input_values = x_cache[chunk * 8 + group * 2 + vec];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int index = vec * 8 + j;
          values[index] = __fmul_rn(__bfloat162float(input_values.elem[j]), inv_global);
          normalized_amax = fmaxf(normalized_amax, fabsf(values[index]));
        }
      }
      const unsigned char scale_byte = cvt_f2_to_ue4m3(__fdiv_rn(normalized_amax, kE2M1Max));
      const float block_fp32 = cvt_ue4m3_to_f32(scale_byte);
      const float inv_block = block_fp32 > 0.0f ? __fdiv_rn(1.0f, block_fp32) : 0.0f;
      uint32_t words[2] = {0u, 0u};
#pragma unroll
      for (int index = 0; index < kSFVec; ++index) {
        float v = __fmul_rn(values[index], inv_block);
        v = fminf(fmaxf(v, -kE2M1Max), kE2M1Max);
        words[index / 8] |= static_cast<uint32_t>(cvt_f2_to_e2m1x2(v, v) & 0xF)
                            << ((index % 8) * 4);
      }
      auto* destination = group < 2 ? packed_words0 + group * 2 : packed_words1 + (group - 2) * 2;
      destination[0] = words[0];
      destination[1] = words[1];
      scale_word |= static_cast<uint32_t>(scale_byte) << (group * 8);
    }
    reinterpret_cast<uint4*>(qdata_row)[chunk * 2] = packed0;
    reinterpret_cast<uint4*>(qdata_row)[chunk * 2 + 1] = packed1;
    *reinterpret_cast<uint32_t*>(scales_bytes + sf_byte_offset(row, chunk * 4, kChunks)) =
      scale_word;
  }
#endif
}

template <bool kLevel1>
__global__ void quantize_fp4_kernel(const __nv_bfloat16* __restrict__ input,
                                    cutlass::float_e2m1_t* __restrict__ qdata,
                                    cutlass::float_ue4m3_t* __restrict__ block_scales,
                                    const float* __restrict__ global_scales, int rows, int cols) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  const int groups_per_row = cols / kSFVec;
  const int group = blockIdx.x * blockDim.x + threadIdx.x;
  const int padded_rows = (rows + kRowBlock - 1) / kRowBlock * kRowBlock;
  if (group >= padded_rows * groups_per_row) {
    return;
  }
  const int row = group / groups_per_row;
  const int group_in_row = group % groups_per_row;
  const int col = group_in_row * kSFVec;
  const int nb_k = cols / 64;

  if (row >= rows) {
    block_scales[sf_byte_offset(row, group_in_row, nb_k)] = cutlass::float_ue4m3_t::bitcast(0);
    return;
  }

  float inv_global = 1.0f;
  if constexpr (kLevel1) {
    const float scale = global_scales[row];
    inv_global = scale > 0.0f ? __fdiv_rn(1.0f, scale) : 0.0f;
  }

  const __nv_bfloat16* src = input + static_cast<size_t>(row) * cols + col;
  const Vec8BF16 v0 = *reinterpret_cast<const Vec8BF16*>(src);
  const Vec8BF16 v1 = *reinterpret_cast<const Vec8BF16*>(src + 8);
  float values[kSFVec];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float a = __bfloat162float(v0.elem[i]) * inv_global;
    const float b = __bfloat162float(v1.elem[i]) * inv_global;
    values[i] = a;
    values[i + 8] = b;
    amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
  }

  const unsigned char scale_byte = cvt_f2_to_ue4m3(__fdiv_rn(amax, kE2M1Max));
  const float scale_value = cvt_ue4m3_to_f32(scale_byte);
  const float inv_block = scale_value > 0.0f ? __fdiv_rn(1.0f, scale_value) : 0.0f;
  block_scales[sf_byte_offset(row, group_in_row, nb_k)] =
    cutlass::float_ue4m3_t::bitcast(scale_byte);

  uint2 packed;
  auto* bytes = reinterpret_cast<unsigned char*>(&packed);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    bytes[i] = cvt_f2_to_e2m1x2(values[2 * i + 1] * inv_block, values[2 * i] * inv_block);
  }
  *reinterpret_cast<uint2*>(reinterpret_cast<unsigned char*>(qdata) +
                            static_cast<size_t>(row) * (cols / 2) + group_in_row * 8) = packed;
#else
  asm volatile("trap;");
#endif
}

void check_runtime_device(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_cuda(), "FP4 operators require CUDA tensors.");
  TORCH_CHECK(tensor.get_device() >= 0, "FP4 operators require a concrete CUDA device.");
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(tensor.get_device());
  TORCH_CHECK(properties->major == 12,
              "cache-dit native NVFP4 kernels require sm120; this device is sm", properties->major,
              properties->minor, ". Use --quantize-backend torchao on this device.");
}

void check_quantize_inputs(const at::Tensor& input, const at::Tensor& qdata,
                           const at::Tensor& block_scales, const at::Tensor& global_scales,
                           bool level1) {
  check_runtime_device(input);
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "fp4_quantize expects bfloat16 input.");
  TORCH_CHECK(input.dim() == 2, "fp4_quantize expects a 2D input.");
  TORCH_CHECK(input.is_contiguous(), "fp4_quantize expects contiguous input.");
  TORCH_CHECK(qdata.is_cuda() && block_scales.is_cuda() && global_scales.is_cuda(),
              "fp4_quantize outputs must be CUDA tensors.");
  TORCH_CHECK(qdata.device() == input.device() && block_scales.device() == input.device() &&
                global_scales.device() == input.device(),
              "fp4_quantize expects every tensor on one device.");
  TORCH_CHECK(qdata.scalar_type() == at::kFloat4_e2m1fn_x2,
              "fp4_quantize qdata must be float4_e2m1fn_x2.");
  TORCH_CHECK(block_scales.scalar_type() == at::kFloat8_e4m3fn,
              "fp4_quantize block scales must be float8_e4m3fn.");
  TORCH_CHECK(global_scales.scalar_type() == at::kFloat,
              "fp4_quantize global scales must be float32.");
  TORCH_CHECK(
    qdata.is_contiguous() && block_scales.is_contiguous() && global_scales.is_contiguous(),
    "fp4_quantize expects contiguous outputs.");

  const int64_t rows = input.size(0);
  const int64_t cols = input.size(1);
  TORCH_CHECK(rows > 0 && cols > 0, "fp4_quantize does not accept empty input.");
  TORCH_CHECK(cols % 64 == 0, "fp4_quantize requires K to be a multiple of 64.");
  TORCH_CHECK(rows <= INT32_MAX && cols <= INT32_MAX, "fp4_quantize dimensions must fit in int32.");
  TORCH_CHECK(qdata.dim() == 2 && qdata.size(0) == rows && qdata.size(1) == cols / 2,
              "fp4_quantize qdata shape must be [R, K / 2].");
  TORCH_CHECK(block_scales.numel() == static_cast<int64_t>(sf_buffer_bytes(rows, cols)),
              "fp4_quantize block scale buffer has an invalid size.");
  TORCH_CHECK(global_scales.dim() == 1 && global_scales.numel() == (level1 ? rows : 0),
              "fp4_quantize global scales must be [R] with level1, or empty without it.");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(qdata.data_ptr()) % 16 == 0,
              "fp4_quantize requires 16-byte aligned input and qdata.");
}

}  // namespace

void fp4_quantize(const at::Tensor& input, at::Tensor& qdata, at::Tensor& block_scales,
                  at::Tensor& global_scales, bool level1) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  check_quantize_inputs(input, qdata, block_scales, global_scales, level1);
  const int rows = static_cast<int>(input.size(0));
  const int cols = static_cast<int>(input.size(1));
  const int padded_rows = (rows + kRowBlock - 1) / kRowBlock * kRowBlock;
  const int64_t total_groups = static_cast<int64_t>(padded_rows) * (cols / kSFVec);
  TORCH_CHECK(total_groups <= INT32_MAX, "fp4_quantize padded group count exceeds int32.");

  const auto* src = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>());
  auto* packed = reinterpret_cast<cutlass::float_e2m1_t*>(qdata.data_ptr());
  auto* scales =
    reinterpret_cast<cutlass::float_ue4m3_t*>(block_scales.data_ptr<at::Float8_e4m3fn>());
  float* global = level1 ? global_scales.data_ptr<float>() : nullptr;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const bool use_fused = level1 && cols == 15360;
  if (use_fused) {
    cudaMemsetAsync(scales, 0, block_scales.numel(), stream);
    constexpr int kCols = 15360;
    constexpr int kThreads = 128;
    const int smem_bytes = kCols * 2 + (kCols / kSFVec) * 4 + 64;
    auto kernel = quantize_fp4_fused_kernel<kCols, kThreads>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    kernel<<<rows, kThreads, smem_bytes, stream>>>(src, packed, scales, global, rows);
  } else if (level1) {
    row_amax_fp4_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(src, global, rows, cols);
    quantize_fp4_kernel<true>
      <<<(total_groups + 255) / 256, 256, 0, stream>>>(src, packed, scales, global, rows, cols);
  } else {
    quantize_fp4_kernel<false>
      <<<(total_groups + 255) / 256, 256, 0, stream>>>(src, packed, scales, global, rows, cols);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::fp4
