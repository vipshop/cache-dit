#include "fp8_common.cuh"

namespace cache_dit::ulysses {
namespace {

__global__ void qkv_permute_quant_kernel(const __nv_bfloat16* x, uint8_t* y, int64_t batches,
                                         int64_t seq_len, int64_t num_heads, int64_t head_size,
                                         int64_t stride_b, int64_t stride_s, int64_t stride_p,
                                         int64_t stride_n, int64_t packed_head_size, int64_t rows) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;

  const int64_t n = row % num_heads;
  int64_t index = row / num_heads;
  const int64_t b = index % batches;
  index /= batches;
  const int64_t s = index % seq_len;
  const int64_t p = index / seq_len;
  const __nv_bfloat16* input = x + b * stride_b + s * stride_s + p * stride_p + n * stride_n;
  uint8_t* output = y + row * packed_head_size;

  const float absmax = row_absmax(input, head_size, lane);
  const float scale = fmaxf(absmax / kFp8Max, 1e-6f);
  quantize_row(input, output, head_size, lane, 1.0f / scale);
  if (lane == 0) {
    std::memcpy(output + head_size, &scale, sizeof(scale));
  }
  for (int64_t d = head_size + 4 + lane; d < packed_head_size; d += kWarpSize) {
    output[d] = 0;
  }
}

}  // namespace

void ulysses_fp8_qkv_permute_quant(at::Tensor x, at::Tensor y, int64_t packed_head_size) {
  check_fp8_pair(x, y);
  TORCH_CHECK(x.dim() == 5 && y.dim() == 5, "QKV quant expects 5D tensors");
  TORCH_CHECK(y.is_contiguous(), "QKV quantized output must be contiguous");
  TORCH_CHECK(x.stride(4) == 1, "QKV head dimension must be contiguous");
  const int64_t batches = x.size(0);
  const int64_t seq_len = x.size(1);
  const int64_t partitions = x.size(2);
  const int64_t num_heads = x.size(3);
  const int64_t head_size = x.size(4);
  TORCH_CHECK(head_size > 0 && head_size % 4 == 0,
              "QKV head size must be a positive multiple of 4");
  TORCH_CHECK(packed_head_size >= head_size + 4, "QKV packed head size leaves no room for scale");
  TORCH_CHECK(y.size(0) == partitions && y.size(1) == seq_len && y.size(2) == batches &&
                y.size(3) == num_heads && y.size(4) == packed_head_size,
              "invalid QKV quantized output shape");
  const int64_t rows = partitions * seq_len * batches * num_heads;

  if (rows == 0) return;

  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  qkv_permute_quant_kernel<<<grid, kThreads, 0, stream>>>(
    reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<uint8_t*>(y.data_ptr()),
    batches, seq_len, num_heads, head_size, x.stride(0), x.stride(1), x.stride(2), x.stride(3),
    packed_head_size, rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::ulysses
