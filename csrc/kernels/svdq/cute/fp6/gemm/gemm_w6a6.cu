#include "gemm_w6a6.h"

#include <stdexcept>

#include "common.h"
#include "gemm_w6a6.cuh"

namespace svdq::cute_kernels {

// Host validation and the single SM120 instantiation intentionally share one TU. Split this file
// only when another architecture needs a genuinely different schedule.
void gemm_w6a6(Tensor act, Tensor wgt, Tensor out, Tensor ascales, Tensor wscales, Tensor asf,
               Tensor wsf, Tensor lora_act_in, Tensor lora_up, Tensor bias) {
  auto require = [](bool condition, const char *message) {
    if (!condition) {
      throw std::runtime_error(spdlog::fmt_lib::format("gemm_w6a6: {}", message));
    }
  };

  int device = 0;
  int major = 0;
  checkCUDA(cudaGetDevice(&device));
  checkCUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  require(major >= 12, "the W6A6 GEMM requires an sm120 or newer device");

  require(act.valid() && act.ndims() == 2, "act must be a valid 2D tensor");
  require(wgt.valid() && wgt.ndims() == 2, "wgt must be a valid packed 2D tensor");
  require(out.valid() && out.ndims() == 2, "out must be a valid 2D tensor");
  require(asf.valid() && asf.ndims() == 2, "asf must be a valid 2D tensor");
  require(wsf.valid() && wsf.ndims() == 2, "wsf must be a valid 2D tensor");
  require(lora_up.valid() == lora_act_in.valid(),
          "lora_act_in and lora_up must be supplied together");
  if (lora_up.valid()) {
    require(lora_up.ndims() == 2 && lora_act_in.ndims() == 2,
            "lora_act_in/lora_up must be 2D tensors");
  }

  const int M = act.size(0);
  require(act.size(1) % 3 == 0, "act must have shape [M, K*3/4]");
  const int K = act.size(1) / 3 * 4;
  const int N = wgt.size(0);
  const int rank = lora_up.valid() ? lora_up.size(1) : 0;

  require(act.dtype() == Tensor::INT8, "act must be uint8 packed E2M3 codes");
  require(wgt.dtype() == Tensor::INT8, "wgt must be uint8 packed E2M3 codes");
  require(out.dtype() == Tensor::BF16, "out must be BF16");
  require(act.is_cuda() && wgt.is_cuda() && out.is_cuda(), "act/wgt/out must be CUDA tensors");
  require(act.is_contiguous() && wgt.is_contiguous() && out.is_contiguous(),
          "act/wgt/out must be contiguous");
  require(wgt.size(1) == act.size(1), "wgt must have shape [N, K*3/4]");
  require(out.size(0) == M && out.size(1) == N, "out must have shape [M, N]");
  require(asf.dtype() == Tensor::INT8 && wsf.dtype() == Tensor::INT8,
          "asf/wsf must be uint8 UE8M0 codes");
  require(asf.is_cuda() && wsf.is_cuda(), "asf/wsf must be CUDA tensors");
  require(asf.is_contiguous() && wsf.is_contiguous(), "asf/wsf must be contiguous");
  require(asf.size(0) == M && asf.size(1) == K / 32, "asf must have shape [M, K/32]");
  require(wsf.size(0) == N && wsf.size(1) == K / 32, "wsf must have shape [N, K/32]");
  if (ascales.valid()) {
    require(ascales.dtype() == Tensor::FP32 && ascales.is_cuda() && ascales.is_contiguous(),
            "ascales must be a contiguous FP32 CUDA tensor");
    require(ascales.numel() == M, "ascales must have one entry per row");
  }
  if (wscales.valid()) {
    require(wscales.dtype() == Tensor::FP32 && wscales.is_cuda() && wscales.is_contiguous(),
            "wscales must be a contiguous FP32 CUDA tensor");
    require(wscales.numel() == N, "wscales must have one entry per output column");
  }
  require(M % 128 == 0 && N % 128 == 0 && K % 128 == 0, "M, N and K must all be multiples of 128");
  require(reinterpret_cast<uintptr_t>(act.data_ptr()) % 32 == 0,
          "packed act must be 32-byte aligned for 16U6 TMA");
  require(reinterpret_cast<uintptr_t>(wgt.data_ptr()) % 32 == 0,
          "packed wgt must be 32-byte aligned for 16U6 TMA");
  require(rank == 0 || rank == 32 || rank == 64 || rank == 128, "rank must be 0, 32, 64 or 128");
  if (rank > 0) {
    require(lora_act_in.size(0) == M && lora_act_in.size(1) == rank,
            "lora_act_in must have shape [M, R]");
    require(lora_up.size(0) == N, "lora_up must have shape [N, R]");
    require(lora_act_in.dtype() == Tensor::BF16 && lora_up.dtype() == Tensor::BF16,
            "lora tensors must be BF16");
    require(lora_act_in.is_contiguous() && lora_up.is_contiguous(),
            "lora tensors must be contiguous");
  }
  if (bias.valid()) {
    require(bias.dtype() == Tensor::BF16 && bias.numel() == N, "bias must be BF16 [N]");
    require(bias.is_contiguous(), "bias must be contiguous");
  }

  launch_gemm_w6a6<128, 2, 128>(
    act.data_ptr(), wgt.data_ptr(), out.data_ptr(),
    ascales.valid() ? static_cast<float const *>(ascales.data_ptr()) : nullptr,
    wscales.valid() ? static_cast<float const *>(wscales.data_ptr()) : nullptr, asf.data_ptr(),
    wsf.data_ptr(),
    rank > 0 ? static_cast<cutlass::bfloat16_t const *>(lora_act_in.data_ptr()) : nullptr,
    rank > 0 ? static_cast<cutlass::bfloat16_t const *>(lora_up.data_ptr()) : nullptr,
    bias.valid() ? static_cast<cutlass::bfloat16_t const *>(bias.data_ptr()) : nullptr, M, N, K,
    rank, getCurrentCUDAStream());
  checkCUDA(cudaGetLastError());
}

}  // namespace svdq::cute_kernels
