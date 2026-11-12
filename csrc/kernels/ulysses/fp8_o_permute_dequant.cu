#include "fp8_common.cuh"

namespace cache_dit::ulysses {
namespace {

__global__ void o_permute_dequant_kernel(const uint8_t* quant_x, __nv_bfloat16* x, int64_t batches,
                                         int64_t seq_len, int64_t num_heads, int64_t head_size,
                                         int64_t stride_h, int64_t stride_b, int64_t stride_s,
                                         int64_t rows) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;

  const int64_t n = row % num_heads;
  int64_t index = row / num_heads;
  const int64_t s = index % seq_len;
  const int64_t b = index / seq_len;
  const uint8_t* input = quant_x + n * stride_h + b * stride_b + s * stride_s;
  __nv_bfloat16* output = x + row * head_size;
  __nv_bfloat16 stored_scale;
  std::memcpy(&stored_scale, input + head_size, sizeof(stored_scale));
  dequantize_row(input, output, head_size, lane, __bfloat162float(stored_scale));
}

}  // namespace

void ulysses_fp8_o_permute_dequant(at::Tensor quant_x, at::Tensor x, int64_t head_size) {
  TORCH_CHECK(quant_x.is_cuda() && x.is_cuda(), "communication operators require CUDA tensors");
  TORCH_CHECK(quant_x.device() == x.device(), "input and output must be on the same CUDA device");
  TORCH_CHECK(quant_x.scalar_type() == at::kFloat8_e4m3fn, "expected FP8 E4M3 input");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "expected BF16 output");
  TORCH_CHECK(quant_x.dim() == 4 && x.dim() == 4, "O dequant expects 4D tensors");
  TORCH_CHECK(x.is_contiguous(), "O dequantized output must be contiguous");
  TORCH_CHECK(quant_x.stride(3) == 1, "O packed head dimension must be contiguous");
  const int64_t num_heads = quant_x.size(0);
  const int64_t batches = quant_x.size(1);
  const int64_t seq_len = quant_x.size(2);
  const int64_t packed_head_size = quant_x.size(3);
  TORCH_CHECK(head_size > 0 && head_size % 2 == 0, "O head size must be positive and even");
  TORCH_CHECK(packed_head_size >= head_size + 2, "O packed head size leaves no room for scale");
  TORCH_CHECK(x.size(0) == batches && x.size(1) == seq_len && x.size(2) == num_heads &&
                x.size(3) == head_size,
              "invalid O dequantized output shape");
  const int64_t rows = num_heads * batches * seq_len;

  if (rows == 0) return;

  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  o_permute_dequant_kernel<<<grid, kThreads, 0, stream>>>(
    reinterpret_cast<const uint8_t*>(quant_x.data_ptr()),
    reinterpret_cast<__nv_bfloat16*>(x.data_ptr()), batches, seq_len, num_heads, head_size,
    quant_x.stride(0), quant_x.stride(1), quant_x.stride(2), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::ulysses
