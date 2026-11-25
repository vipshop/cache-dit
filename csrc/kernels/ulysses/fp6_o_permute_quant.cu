#include "fp6_common.cuh"

namespace cache_dit::ulysses {

#if CACHE_DIT_ULYSSES_HAS_FP6
namespace {

__global__ void o_permute_fp6_quant_kernel(const __nv_bfloat16* x, uint8_t* y, int64_t batches,
                                           int64_t local_seq_len, int64_t num_heads,
                                           int64_t head_size, int64_t packed_head_size,
                                           int64_t stride_b, int64_t stride_s, int64_t stride_n,
                                           int64_t rows) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;

  const int64_t s = row % local_seq_len;
  int64_t index = row / local_seq_len;
  const int64_t b = index % batches;
  index /= batches;
  const int64_t n = index % num_heads;
  const int64_t p = index / num_heads;
  const __nv_bfloat16* input = x + b * stride_b + (p * local_seq_len + s) * stride_s + n * stride_n;
  uint8_t* output = y + row * packed_head_size;
  const float absmax = row_absmax(input, head_size, lane);
  const float scale = fmaxf(__fdiv_rn(absmax, kFp6Max), 1e-4f);
  quantize_row_fp6<true>(input, output, head_size, packed_head_size, lane, scale);
}

__global__ void o_permute_fp6_group_quant_kernel(const __nv_bfloat16* x, uint8_t* y,
                                                 int64_t batches, int64_t local_seq_len,
                                                 int64_t num_heads, int64_t head_size,
                                                 int64_t packed_head_size, int64_t stride_b,
                                                 int64_t stride_s, int64_t stride_n, int64_t rows,
                                                 int group_size) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;

  const int64_t s = row % local_seq_len;
  int64_t index = row / local_seq_len;
  const int64_t b = index % batches;
  index /= batches;
  const int64_t n = index % num_heads;
  const int64_t p = index / num_heads;
  const __nv_bfloat16* input = x + b * stride_b + (p * local_seq_len + s) * stride_s + n * stride_n;
  uint8_t* output = y + row * packed_head_size;
  quantize_row_groups_fp6<true>(input, output, head_size, packed_head_size, lane, group_size,
                                1e-4f);
}

}  // namespace
#endif

void ulysses_fp6_o_permute_quant(at::Tensor x, at::Tensor y, int64_t partitions,
                                 int64_t packed_head_size, int64_t group_size) {
#if CACHE_DIT_ULYSSES_HAS_FP6
  check_fp6_pair(x, y);
  check_fp6_group_size(group_size);
  TORCH_CHECK(x.dim() == 4 && y.dim() == 5, "O FP6 quant expects 4D input and 5D output");
  TORCH_CHECK(y.is_contiguous(), "O FP6 output must be contiguous");
  TORCH_CHECK(x.stride(3) == 1, "O head dimension must be contiguous");
  TORCH_CHECK(partitions > 0, "O partitions must be positive");
  const int64_t batches = x.size(0);
  const int64_t seq_len = x.size(1);
  const int64_t num_heads = x.size(2);
  const int64_t head_size = x.size(3);
  TORCH_CHECK(head_size > 0 && head_size % 2 == 0, "O FP6 head size must be positive and even");
  TORCH_CHECK(seq_len % partitions == 0, "O sequence length must be divisible by partitions");
  const int64_t local_seq_len = seq_len / partitions;
  TORCH_CHECK(y.size(0) == partitions && y.size(1) == num_heads && y.size(2) == batches &&
                y.size(3) == local_seq_len && y.size(4) == packed_head_size,
              "invalid O FP6 output shape");
  TORCH_CHECK(packed_head_size % 4 == 0, "FP6 packed row width must be a multiple of 4 bytes");
  TORCH_CHECK(packed_head_size >= fp6_packed_min_width(head_size, group_size, 2),
              "O FP6 packed head size leaves no room for scales");
  TORCH_CHECK((num_heads * batches * local_seq_len * packed_head_size) % 16 == 0,
              "O FP6 packed rows must make each rank chunk a multiple of 16 bytes");
  const int64_t rows = partitions * num_heads * batches * local_seq_len;

  if (rows == 0) return;

  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  if (group_size == 0) {
    o_permute_fp6_quant_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<uint8_t*>(y.data_ptr()), batches, local_seq_len, num_heads, head_size,
      packed_head_size, x.stride(0), x.stride(1), x.stride(2), rows);
  } else {
    o_permute_fp6_group_quant_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<uint8_t*>(y.data_ptr()), batches, local_seq_len, num_heads, head_size,
      packed_head_size, x.stride(0), x.stride(1), x.stride(2), rows, group_size);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  TORCH_CHECK(false, "FP6 Ulysses operators require CUDA >= 12.8 with cuda_fp6.h");
#endif
}

}  // namespace cache_dit::ulysses
