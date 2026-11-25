#include "fp4_common.cuh"

namespace cache_dit::ulysses {

#if CACHE_DIT_ULYSSES_HAS_FP4
namespace {

template <int kLanesPerRow = kWarpSize>
__global__ void o_permute_fp4_group_dequant_kernel(const uint8_t* quant_x, __nv_bfloat16* x,
                                                   int64_t batches, int64_t seq_len,
                                                   int64_t num_heads, int64_t head_size,
                                                   int64_t stride_h, int64_t stride_b,
                                                   int64_t stride_s, int64_t rows, int group_size) {
  static_assert(kLanesPerRow == kWarpSize || kLanesPerRow == 8);
  constexpr int kRowsPerWarp = kWarpSize / kLanesPerRow;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int row_in_warp = lane / kLanesPerRow;
  const int row_lane = lane % kLanesPerRow;
  const int64_t row =
    (static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp) * kRowsPerWarp + row_in_warp;
  if (row >= rows) return;

  const int64_t n = row % num_heads;
  int64_t index = row / num_heads;
  const int64_t s = index % seq_len;
  const int64_t b = index / seq_len;
  const uint8_t* input = quant_x + n * stride_h + b * stride_b + s * stride_s;
  dequantize_row_groups_fp4<true>(input, x + row * head_size, head_size, row_lane, group_size,
                                  kLanesPerRow);
}

}  // namespace
#endif

void ulysses_fp4_o_permute_dequant(at::Tensor quant_x, at::Tensor x, int64_t head_size,
                                   int64_t group_size) {
#if CACHE_DIT_ULYSSES_HAS_FP4
  TORCH_CHECK(quant_x.is_cuda() && x.is_cuda(), "communication operators require CUDA tensors");
  TORCH_CHECK(quant_x.device() == x.device(), "input and output must be on the same CUDA device");
  TORCH_CHECK(quant_x.scalar_type() == at::kByte, "expected packed uint8 FP4 input");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "expected BF16 output");
  check_fp4_device(quant_x);
  check_fp4_group_size(group_size);
  TORCH_CHECK(quant_x.dim() == 4 && x.dim() == 4, "O FP4 dequant expects 4D tensors");
  TORCH_CHECK(x.is_contiguous(), "O FP4 dequantized output must be contiguous");
  TORCH_CHECK(quant_x.stride(3) == 1, "O packed head dimension must be contiguous");
  const int64_t num_heads = quant_x.size(0);
  const int64_t batches = quant_x.size(1);
  const int64_t seq_len = quant_x.size(2);
  const int64_t packed_head_size = quant_x.size(3);
  TORCH_CHECK(head_size > 0 && head_size % 2 == 0, "O FP4 head size must be positive and even");
  TORCH_CHECK(packed_head_size >= fp4_packed_min_width(head_size, group_size, 2),
              "O FP4 packed head size leaves no room for scales");
  TORCH_CHECK(x.size(0) == batches && x.size(1) == seq_len && x.size(2) == num_heads &&
                x.size(3) == head_size,
              "invalid O FP4 dequantized output shape");
  const int64_t rows = num_heads * batches * seq_len;
  if (rows == 0) return;

  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  const bool use_four_rows_per_warp = head_size == 128 && rows >= 2048 &&
                                      rows % (4 * kWarpsPerBlock) == 0 &&
                                      reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0;
  if (use_four_rows_per_warp) {
    const dim3 multirow_grid(rows / (4 * kWarpsPerBlock));
    o_permute_fp4_group_dequant_kernel<8><<<multirow_grid, kThreads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(quant_x.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(x.data_ptr()), batches, seq_len, num_heads, head_size,
      quant_x.stride(0), quant_x.stride(1), quant_x.stride(2), rows, group_size);
  } else {
    o_permute_fp4_group_dequant_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(quant_x.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(x.data_ptr()), batches, seq_len, num_heads, head_size,
      quant_x.stride(0), quant_x.stride(1), quant_x.stride(2), rows, group_size);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  TORCH_CHECK(false, "FP4 Ulysses operators require CUDA >= 12.8 with cuda_fp4.h");
#endif
}

}  // namespace cache_dit::ulysses
