#include "fp8_common.cuh"

namespace cache_dit::ulysses {
namespace {

__global__ void o_permute_quant_kernel(const __nv_bfloat16* x, uint8_t* y, int64_t batches,
                                       int64_t local_seq_len, int64_t num_heads, int64_t head_size,
                                       int64_t packed_head_size, int64_t stride_b, int64_t stride_s,
                                       int64_t stride_n, int64_t rows) {
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
  // Keep the legacy O per-token scale epsilon (1e-4), not the QKV path's 1e-6.
  const float scale = fmaxf(absmax / kFp8Max, 1e-4f);
  quantize_row(input, output, head_size, lane, 1.0f / scale);
  if (lane == 0) {
    const __nv_bfloat16 stored_scale = __float2bfloat16_rn(scale);
    std::memcpy(output + head_size, &stored_scale, sizeof(stored_scale));
  }
  for (int64_t d = head_size + 2 + lane; d < packed_head_size; d += kWarpSize) {
    output[d] = 0;
  }
}

}  // namespace

void ulysses_fp8_o_permute_quant(at::Tensor x, at::Tensor y, int64_t partitions,
                                 int64_t packed_head_size) {
  check_fp8_pair(x, y);
  TORCH_CHECK(x.dim() == 4 && y.dim() == 5, "O quant expects 4D input and 5D output");
  TORCH_CHECK(y.is_contiguous(), "O quantized output must be contiguous");
  TORCH_CHECK(x.stride(3) == 1, "O head dimension must be contiguous");
  TORCH_CHECK(partitions > 0, "O partitions must be positive");
  const int64_t batches = x.size(0);
  const int64_t seq_len = x.size(1);
  const int64_t num_heads = x.size(2);
  const int64_t head_size = x.size(3);
  TORCH_CHECK(head_size > 0 && head_size % 2 == 0, "O head size must be positive and even");
  TORCH_CHECK(seq_len % partitions == 0, "O sequence length must be divisible by partitions");
  const int64_t local_seq_len = seq_len / partitions;
  TORCH_CHECK(y.size(0) == partitions && y.size(1) == num_heads && y.size(2) == batches &&
                y.size(3) == local_seq_len && y.size(4) == packed_head_size,
              "invalid O quantized output shape");
  TORCH_CHECK(packed_head_size >= head_size + 2, "O packed head size leaves no room for scale");
  TORCH_CHECK((num_heads * batches * local_seq_len * packed_head_size) % 16 == 0,
              "O packed rows must make each rank chunk a multiple of 16 bytes");
  const int64_t rows = partitions * num_heads * batches * local_seq_len;

  if (rows == 0) return;

  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  o_permute_quant_kernel<<<grid, kThreads, 0, stream>>>(
    reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<uint8_t*>(y.data_ptr()),
    batches, local_seq_len, num_heads, head_size, packed_head_size, x.stride(0), x.stride(1),
    x.stride(2), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::ulysses
