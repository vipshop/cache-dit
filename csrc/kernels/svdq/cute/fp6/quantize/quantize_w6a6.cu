#include "quantize_w6a6.h"

#include <stdexcept>

#include "common.h"
#include "quantize_w6a6.cuh"

namespace svdq::cute_kernels {

// Rank-specialized entry points, one per translation unit so ranks compile in parallel.
void quantize_w6a6_rank32(void const *input, void const *smooth, void const *lora_down,
                          void *output, void *asf, void *lora_act, float *lora_scratch, int M,
                          int K, cudaStream_t stream);
void quantize_w6a6_rank64(void const *input, void const *smooth, void const *lora_down,
                          void *output, void *asf, void *lora_act, float *lora_scratch, int M,
                          int K, cudaStream_t stream);
void quantize_w6a6_rank128(void const *input, void const *smooth, void const *lora_down,
                           void *output, void *asf, void *lora_act, float *lora_scratch, int M,
                           int K, cudaStream_t stream);

int quantize_w6a6_k_splits(int M, int K) { return w6a6_quant_k_splits(M, K); }

void quantize_w6a6(Tensor input, Tensor output, Tensor asf, Tensor lora_down, Tensor lora_act,
                   Tensor smooth, Tensor lora_scratch) {
  auto require = [](bool condition, const char *message) {
    if (!condition) {
      throw std::runtime_error(spdlog::fmt_lib::format("quantize_w6a6: {}", message));
    }
  };

  require(input.valid() && input.ndims() == 2, "input must be a valid 2D tensor");
  require(output.valid() && output.ndims() == 2, "output must be a valid 2D tensor");
  require(asf.valid() && asf.ndims() == 2, "asf must be a valid 2D tensor");
  require(smooth.valid() && smooth.ndims() == 1, "smooth must be a valid 1D tensor");
  require(!lora_down.valid() || lora_down.ndims() == 2, "lora_down must be a 2D tensor");

  const int M = input.size(0);
  const int K = input.size(1);
  require(M > 0 && K > 0, "input must have non-empty M and K dimensions");
  const int rank = lora_down.valid() ? lora_down.size(0) : 0;

  require(input.dtype() == Tensor::BF16, "input must be BF16");
  require(smooth.dtype() == Tensor::BF16, "smooth must be BF16");
  require(output.dtype() == Tensor::INT8, "output must be the packed uint8 destination");
  require(asf.dtype() == Tensor::INT8, "asf must be uint8 UE8M0 codes");
  require(input.is_cuda() && output.is_cuda() && asf.is_cuda(),
          "input/output/asf must be CUDA tensors");
  require(input.is_contiguous() && output.is_contiguous() && smooth.is_contiguous() &&
            asf.is_contiguous(),
          "input/output/smooth/asf must be contiguous");
  require(output.size(0) == M && output.size(1) == K * 3 / 4, "output must have shape [M, K*3/4]");
  require(asf.size(0) == M && asf.size(1) == K / 32, "asf must have shape [M, K/32]");
  require(smooth.numel() == K, "smooth must have one entry per input channel");
  require(M % 128 == 0, "M must be a multiple of 128");
  require(K % 128 == 0, "K must be a multiple of 128");
  require(rank == 0 || rank == 32 || rank == 64 || rank == 128, "rank must be 0, 32, 64 or 128");
  if (rank > 0) {
    require(lora_act.valid(), "lora_act must be valid when lora_down is supplied");
    require(lora_down.dtype() == Tensor::BF16 && lora_act.dtype() == Tensor::BF16,
            "lora tensors must be BF16");
    require(lora_down.size(1) == K, "lora_down must have shape [rank, K]");
    require(lora_act.valid() && lora_act.size(0) == M && lora_act.size(1) == rank,
            "lora_act must have shape [M, rank]");
    require(lora_down.is_contiguous() && lora_act.is_contiguous(),
            "lora tensors must be contiguous");
  }

  auto *input_ptr = static_cast<void const *>(input.data_ptr());
  auto *smooth_ptr = static_cast<void const *>(smooth.data_ptr());
  auto *down_ptr = rank > 0 ? static_cast<void const *>(lora_down.data_ptr()) : nullptr;
  auto *act_ptr = rank > 0 ? lora_act.data_ptr() : nullptr;
  auto *scratch_ptr =
    lora_scratch.valid() ? static_cast<float *>(lora_scratch.data_ptr()) : nullptr;
  cudaStream_t stream = getCurrentCUDAStream();
  switch (rank) {
    case 0:
      launch_quantize_w6a6<0>(input_ptr, smooth_ptr, down_ptr, output.data_ptr(), asf.data_ptr(),
                              act_ptr, nullptr, M, K, stream);
      break;
    case 32:
      quantize_w6a6_rank32(input_ptr, smooth_ptr, down_ptr, output.data_ptr(), asf.data_ptr(),
                           act_ptr, scratch_ptr, M, K, stream);
      break;
    case 64:
      quantize_w6a6_rank64(input_ptr, smooth_ptr, down_ptr, output.data_ptr(), asf.data_ptr(),
                           act_ptr, scratch_ptr, M, K, stream);
      break;
    case 128:
      quantize_w6a6_rank128(input_ptr, smooth_ptr, down_ptr, output.data_ptr(), asf.data_ptr(),
                            act_ptr, scratch_ptr, M, K, stream);
      break;
  }
  checkCUDA(cudaGetLastError());
}

}  // namespace svdq::cute_kernels
