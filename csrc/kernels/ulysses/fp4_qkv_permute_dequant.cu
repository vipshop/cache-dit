#include "fp4_common.cuh"

namespace cache_dit::ulysses {

#if CACHE_DIT_ULYSSES_HAS_FP4
namespace {

__global__ void qkv_permute_fp4_group_dequant_kernel(const uint8_t* quant_x, __nv_bfloat16* x,
                                                     int64_t batches, int64_t seq_len,
                                                     int64_t num_heads, int64_t head_size,
                                                     int64_t stride_s, int64_t stride_b,
                                                     int64_t stride_n, int64_t rows,
                                                     int group_size) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;
  const int64_t n = row % num_heads;
  int64_t index = row / num_heads;
  const int64_t b = index % batches;
  const int64_t s = index / batches;
  const uint8_t* input = quant_x + s * stride_s + b * stride_b + n * stride_n;
  __nv_bfloat16* output = x + ((b * seq_len + s) * num_heads + n) * head_size;
  dequantize_row_groups_fp4<false>(input, output, head_size, lane, group_size);
}

}  // namespace
#endif

void ulysses_fp4_qkv_permute_dequant(at::Tensor quant_x, at::Tensor x, int64_t head_size,
                                     int64_t group_size) {
#if CACHE_DIT_ULYSSES_HAS_FP4
  TORCH_CHECK(quant_x.is_cuda() && x.is_cuda(), "communication operators require CUDA tensors");
  TORCH_CHECK(quant_x.device() == x.device(), "input and output must be on the same CUDA device");
  TORCH_CHECK(quant_x.scalar_type() == at::kByte, "expected packed uint8 FP4 input");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "expected BF16 output");
  check_fp4_device(quant_x);
  check_fp4_group_size(group_size);
  TORCH_CHECK(quant_x.dim() == 4 && x.dim() == 4, "QKV FP4 dequant expects 4D tensors");
  TORCH_CHECK(x.is_contiguous(), "QKV FP4 dequantized output must be contiguous");
  TORCH_CHECK(quant_x.stride(3) == 1, "QKV packed head dimension must be contiguous");
  const int64_t seq_len = quant_x.size(0);
  const int64_t batches = quant_x.size(1);
  const int64_t num_heads = quant_x.size(2);
  const int64_t packed_head_size = quant_x.size(3);
  TORCH_CHECK(head_size > 0 && head_size % 2 == 0, "QKV FP4 head size must be positive and even");
  TORCH_CHECK(packed_head_size >= fp4_packed_min_width(head_size, group_size, 4),
              "QKV FP4 packed head size leaves no room for scales");
  TORCH_CHECK(x.size(0) == batches && x.size(1) == seq_len && x.size(2) == num_heads &&
                x.size(3) == head_size,
              "invalid QKV FP4 dequantized output shape");
  const int64_t rows = seq_len * batches * num_heads;
  if (rows == 0) return;

  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  qkv_permute_fp4_group_dequant_kernel<<<grid, kThreads, 0, stream>>>(
    reinterpret_cast<const uint8_t*>(quant_x.data_ptr()),
    reinterpret_cast<__nv_bfloat16*>(x.data_ptr()), batches, seq_len, num_heads, head_size,
    quant_x.stride(0), quant_x.stride(1), quant_x.stride(2), rows, group_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  TORCH_CHECK(false, "FP4 Ulysses operators require CUDA >= 12.8 with cuda_fp4.h");
#endif
}

}  // namespace cache_dit::ulysses
