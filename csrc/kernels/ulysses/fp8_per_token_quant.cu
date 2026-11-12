#include "fp8_common.cuh"

namespace cache_dit::ulysses {
namespace {

__global__ void per_token_quant_kernel(const __nv_bfloat16* x, uint8_t* y, int64_t rows,
                                       int64_t head_size, int64_t packed_head_size) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;

  const __nv_bfloat16* input = x + row * head_size;
  uint8_t* output = y + row * packed_head_size;
  const float absmax = row_absmax(input, head_size, lane);
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

void ulysses_fp8_per_token_quant(at::Tensor x, at::Tensor y, int64_t packed_head_size) {
  check_fp8_pair(x, y);
  TORCH_CHECK(x.dim() == 2 && y.dim() == 2, "per-token quant expects 2D tensors");
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "per-token quant expects contiguous tensors");
  const int64_t rows = x.size(0);
  const int64_t head_size = x.size(1);
  TORCH_CHECK(head_size > 0 && head_size % 2 == 0, "per-token head size must be positive and even");
  TORCH_CHECK(packed_head_size >= head_size + 2, "per-token packed row leaves no room for scale");
  TORCH_CHECK(y.size(0) == rows && y.size(1) == packed_head_size, "invalid quantized output shape");

  if (rows == 0) return;

  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  per_token_quant_kernel<<<grid, kThreads, 0, stream>>>(
    reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<uint8_t*>(y.data_ptr()),
    rows, head_size, packed_head_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::ulysses
