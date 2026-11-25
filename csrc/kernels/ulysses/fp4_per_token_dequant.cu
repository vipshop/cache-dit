#include "fp4_common.cuh"

namespace cache_dit::ulysses {

#if CACHE_DIT_ULYSSES_HAS_FP4
namespace {

__global__ void per_token_fp4_group_dequant_kernel(const uint8_t* x, __nv_bfloat16* y, int64_t rows,
                                                   int64_t head_size, int64_t packed_head_size,
                                                   int group_size) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;
  dequantize_row_groups_fp4<true>(x + row * packed_head_size, y + row * head_size, head_size, lane,
                                  group_size);
}

}  // namespace
#endif

void ulysses_fp4_per_token_dequant(at::Tensor x, at::Tensor y, int64_t head_size,
                                   int64_t group_size) {
#if CACHE_DIT_ULYSSES_HAS_FP4
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), "communication operators require CUDA tensors");
  TORCH_CHECK(x.device() == y.device(), "input and output must be on the same CUDA device");
  TORCH_CHECK(x.scalar_type() == at::kByte, "expected packed uint8 FP4 input");
  TORCH_CHECK(y.scalar_type() == at::kBFloat16, "expected BF16 output");
  check_fp4_device(x);
  check_fp4_group_size(group_size);
  TORCH_CHECK(x.dim() == 2 && y.dim() == 2, "per-token FP4 dequant expects 2D tensors");
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous(),
              "per-token FP4 dequant expects contiguous tensors");
  const int64_t rows = x.size(0);
  const int64_t packed_head_size = x.size(1);
  TORCH_CHECK(head_size > 0 && head_size % 2 == 0,
              "per-token FP4 head size must be positive and even");
  TORCH_CHECK(packed_head_size >= fp4_packed_min_width(head_size, group_size, 2),
              "per-token FP4 packed row leaves no room for scales");
  TORCH_CHECK(y.size(0) == rows && y.size(1) == head_size, "invalid FP4 dequantized output shape");
  if (rows == 0) return;

  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  per_token_fp4_group_dequant_kernel<<<grid, kThreads, 0, stream>>>(
    reinterpret_cast<const uint8_t*>(x.data_ptr()), reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
    rows, head_size, packed_head_size, group_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  TORCH_CHECK(false, "FP4 Ulysses operators require CUDA >= 12.8 with cuda_fp4.h");
#endif
}

}  // namespace cache_dit::ulysses
