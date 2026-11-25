#include "fp6_common.cuh"

namespace cache_dit::ulysses {

#if CACHE_DIT_ULYSSES_HAS_FP6
namespace {

__global__ void per_token_fp6_quant_kernel(const __nv_bfloat16* x, uint8_t* y, int64_t rows,
                                           int64_t head_size, int64_t packed_head_size) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;

  const __nv_bfloat16* input = x + row * head_size;
  uint8_t* output = y + row * packed_head_size;
  const float absmax = row_absmax(input, head_size, lane);
  const float scale = fmaxf(__fdiv_rn(absmax, kFp6Max), 1e-4f);
  quantize_row_fp6<true>(input, output, head_size, packed_head_size, lane, scale);
}

__global__ void per_token_fp6_group_quant_kernel(const __nv_bfloat16* x, uint8_t* y, int64_t rows,
                                                 int64_t head_size, int64_t packed_head_size,
                                                 int group_size) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;

  const __nv_bfloat16* input = x + row * head_size;
  uint8_t* output = y + row * packed_head_size;
  quantize_row_groups_fp6<true>(input, output, head_size, packed_head_size, lane, group_size,
                                1e-4f);
}

}  // namespace
#endif

void ulysses_fp6_per_token_quant(at::Tensor x, at::Tensor y, int64_t packed_head_size,
                                 int64_t group_size) {
#if CACHE_DIT_ULYSSES_HAS_FP6
  check_fp6_pair(x, y);
  check_fp6_group_size(group_size);
  TORCH_CHECK(x.dim() == 2 && y.dim() == 2, "per-token FP6 quant expects 2D tensors");
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous(),
              "per-token FP6 quant expects contiguous tensors");
  const int64_t rows = x.size(0);
  const int64_t head_size = x.size(1);
  TORCH_CHECK(head_size > 0 && head_size % 2 == 0,
              "per-token FP6 head size must be positive and even");
  TORCH_CHECK(packed_head_size % 4 == 0, "FP6 packed row width must be a multiple of 4 bytes");
  TORCH_CHECK(packed_head_size >= fp6_packed_min_width(head_size, group_size, 2),
              "per-token FP6 packed row leaves no room for scales");
  TORCH_CHECK(y.size(0) == rows && y.size(1) == packed_head_size, "invalid FP6 output shape");

  if (rows == 0) return;

  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  if (group_size == 0) {
    per_token_fp6_quant_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<uint8_t*>(y.data_ptr()), rows, head_size, packed_head_size);
  } else {
    per_token_fp6_group_quant_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<uint8_t*>(y.data_ptr()), rows, head_size, packed_head_size, group_size);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  TORCH_CHECK(false, "FP6 Ulysses operators require CUDA >= 12.8 with cuda_fp6.h");
#endif
}

}  // namespace cache_dit::ulysses
