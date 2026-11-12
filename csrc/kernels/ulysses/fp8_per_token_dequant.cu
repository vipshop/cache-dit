#include "fp8_common.cuh"

namespace cache_dit::ulysses {
namespace {

__global__ void per_token_dequant_kernel(const uint8_t* x, __nv_bfloat16* y, int64_t rows,
                                         int64_t head_size, int64_t packed_head_size) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;

  const uint8_t* input = x + row * packed_head_size;
  __nv_bfloat16* output = y + row * head_size;
  __nv_bfloat16 stored_scale;
  std::memcpy(&stored_scale, input + head_size, sizeof(stored_scale));
  dequantize_row(input, output, head_size, lane, __bfloat162float(stored_scale));
}

}  // namespace

void ulysses_fp8_per_token_dequant(at::Tensor x, at::Tensor y, int64_t head_size) {
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), "communication operators require CUDA tensors");
  TORCH_CHECK(x.device() == y.device(), "input and output must be on the same CUDA device");
  TORCH_CHECK(x.scalar_type() == at::kFloat8_e4m3fn, "expected FP8 E4M3 input");
  TORCH_CHECK(y.scalar_type() == at::kBFloat16, "expected BF16 output");
  TORCH_CHECK(x.dim() == 2 && y.dim() == 2, "per-token dequant expects 2D tensors");
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous(),
              "per-token dequant expects contiguous tensors");
  const int64_t rows = x.size(0);
  const int64_t packed_head_size = x.size(1);
  TORCH_CHECK(head_size > 0 && head_size % 2 == 0, "per-token head size must be positive and even");
  TORCH_CHECK(packed_head_size >= head_size + 2, "per-token packed row leaves no room for scale");
  TORCH_CHECK(y.size(0) == rows && y.size(1) == head_size, "invalid dequantized output shape");

  if (rows == 0) return;

  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  per_token_dequant_kernel<<<grid, kThreads, 0, stream>>>(
    reinterpret_cast<const uint8_t*>(x.data_ptr()), reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
    rows, head_size, packed_head_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::ulysses
