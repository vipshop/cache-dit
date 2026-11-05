#include "gemm_w4a4.h"

#include <stdexcept>

#include "common.h"
#include "gemm_w4a4.cuh"

namespace svdq::cute_kernels {

// Host entry for the SM120 CuTe path. Validation happens before any launch so unsupported dtypes
// or layouts fail instead of being misinterpreted by a TMA descriptor. Activation A/SFA keep the
// v1 256-row MMA-fragment pack and are loaded G2R; weights use standard row-major E2M1 codes plus
// CUTLASS 128x4 SF atoms and are loaded by TMA. The independent CuTe activation producer emits the
// v1 pack plus BF16 D directly, so there is no LoRA finalize or extra repack step.
//
// Two tile specializations are compiled:
//   BN=256, stages=3, BK=128: default when N/K alignment allows it; rank tiles have 32 columns.
//   BN=128, stages=4, BK=64:  fallback for narrower geometry; rank tiles have 16 columns.
// BK controls both the number of NVFP4 atoms per stage and the LoRA overlay width (BK/4), which
// is why rank divisibility is part of configuration selection. No runtime repack is performed.
void gemm_w4a4(::Tensor act, ::Tensor wgt, ::Tensor out, ::Tensor ascales, ::Tensor wscales,
                    ::Tensor lora_act_in, ::Tensor lora_up, ::Tensor bias, ::Tensor wcscales,
                    float alpha) {
  auto require = [](bool condition, const char *message) {
    if (!condition) {
      throw std::runtime_error(spdlog::fmt_lib::format("gemm_w4a4: {}", message));
    }
  };

  require(act.valid() && act.ndims() == 2, "act must be a valid 2D tensor");
  require(wgt.valid() && wgt.ndims() == 2, "wgt must be a valid 2D tensor");
  require(out.valid() && out.ndims() == 2, "out must be a valid 2D tensor");
  require(ascales.valid() && wscales.valid(), "ascales/wscales must be valid tensors");

  const int M = act.size(0);
  const int K = act.size(1) * 2;
  const int N = wgt.size(0);
  const int rank = lora_up.valid() ? lora_up.size(1) : 0;

  require(act.dtype() == ::Tensor::INT8 && wgt.dtype() == ::Tensor::INT8,
          "act/wgt must be packed int8/uint8 bytes");
  require(ascales.dtype() == ::Tensor::FP8_E4M3 && wscales.dtype() == ::Tensor::FP8_E4M3,
          "ascales/wscales must be FP8_E4M3");
  require(out.dtype() == ::Tensor::BF16, "out must be BF16; only the BF16 epilogue is wired up");
  require(act.is_cuda() && wgt.is_cuda() && out.is_cuda() && ascales.is_cuda() &&
            wscales.is_cuda(),
          "act/wgt/out/ascales/wscales must be CUDA tensors");
  require(act.is_contiguous() && wgt.is_contiguous() && out.is_contiguous() &&
            ascales.is_contiguous() && wscales.is_contiguous(),
          "act/wgt/out/ascales/wscales must be contiguous");

  require(wgt.size(1) == act.size(1), "wgt must have the same packed K extent as act");
  require(out.size(0) == M && out.size(1) == N, "out must be [M, N]");
  require(M % 256 == 0 && N % 128 == 0,
          "M must be a multiple of 256 for the v1 activation pack; N must be a multiple of 128");
  require(K % 64 == 0, "K must be a multiple of 64");
  require(ascales.numel() == sf_buffer_bytes(M, K),
          "ascales must be the v1 interleaved activation scale pack for [M, K]");
  require(wscales.numel() == sf_buffer_bytes(N, K),
          "wscales must be a flat 128x4 512B atom buffer for [N, K]");

  require(lora_act_in.valid() == lora_up.valid(), "lora_act_in and lora_up must agree");
  if (rank > 0) {
    require(rank % 16 == 0, "R must be a multiple of 16");
    require(lora_up.ndims() == 2 && lora_up.size(0) == N && lora_act_in.ndims() == 2 &&
              lora_act_in.size(0) == M,
            "LoRA tensors must be [N, R] and [M, R]");
    require(lora_up.dtype() == ::Tensor::BF16 && lora_act_in.dtype() == ::Tensor::BF16,
            "LoRA tensors must be BF16");
    require(lora_up.is_cuda() && lora_act_in.is_cuda() && lora_up.is_contiguous() &&
              lora_act_in.is_contiguous(),
            "LoRA tensors must be contiguous CUDA tensors");
  }
  if (bias.valid()) {
    require(bias.numel() == static_cast<size_t>(N) && bias.dtype() == ::Tensor::BF16 &&
              bias.is_cuda() && bias.is_contiguous(),
            "bias must be a contiguous BF16 [N] tensor");
  }
  if (wcscales.valid()) {
    require(wcscales.numel() == static_cast<size_t>(N) && wcscales.dtype() == ::Tensor::BF16 &&
              wcscales.is_cuda() && wcscales.is_contiguous(),
            "wcscales must be a contiguous BF16 [N] tensor");
  }

  const cudaDeviceProp *prop = getCurrentDeviceProperties();
  require(prop->major * 10 + prop->minor >= 120,
          "gemm_w4a4 requires an SM120 (or newer) device");

  using ElementSF = cutlass::float_ue4m3_t;
  using ElementLora = cutlass::bfloat16_t;

  if (N % 256 == 0 && K % 128 == 0) {
    require(rank % 32 == 0, "rank must be a multiple of 32 when BN=256 and BK=128");
    launch_gemm_w4a4<256, 3, 128>(
      act.data_ptr(), wgt.data_ptr(), out.data_ptr(),
      static_cast<ElementSF const *>(ascales.data_ptr()),
      static_cast<ElementSF const *>(wscales.data_ptr()),
      rank > 0 ? static_cast<ElementLora const *>(lora_act_in.data_ptr()) : nullptr,
      rank > 0 ? static_cast<ElementLora const *>(lora_up.data_ptr()) : nullptr,
      bias.valid() ? static_cast<ElementLora const *>(bias.data_ptr()) : nullptr,
      wcscales.valid() ? static_cast<ElementLora const *>(wcscales.data_ptr()) : nullptr, alpha, M,
      N, K, rank, getCurrentCUDAStream());
  } else {
    launch_gemm_w4a4<128, 4>(
      act.data_ptr(), wgt.data_ptr(), out.data_ptr(),
      static_cast<ElementSF const *>(ascales.data_ptr()),
      static_cast<ElementSF const *>(wscales.data_ptr()),
      rank > 0 ? static_cast<ElementLora const *>(lora_act_in.data_ptr()) : nullptr,
      rank > 0 ? static_cast<ElementLora const *>(lora_up.data_ptr()) : nullptr,
      bias.valid() ? static_cast<ElementLora const *>(bias.data_ptr()) : nullptr,
      wcscales.valid() ? static_cast<ElementLora const *>(wcscales.data_ptr()) : nullptr, alpha, M,
      N, K, rank, getCurrentCUDAStream());
  }
  checkCUDA(cudaGetLastError());
}

}  // namespace svdq::cute_kernels
