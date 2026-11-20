// Adapted from nunchaku/nunchaku/csrc/ops.h
#pragma once

// SVDQ source map
// ----------------
// Public host surface:
//   svdq.h                 This file. Declares the public W4A4 and W8A8 runtime dispatchers in
//                          `cache_dit::svdq` like the other operator families. The kernel
//                          namespaces (`svdq::native_kernels`, `svdq::cute_kernels`) stay global
//                          and are the implementation detail.
//   svdq.cpp                Their definitions plus the pybind registrar: it normalizes
//                          RuntimeParams/LoraLayout and selects native or CuTe.
//   runtime.h               Host-only RuntimeKernel/RuntimeParams (runtime/fp4/alpha/
//                          act_unsigned) plus the shared LoraLayout enum; no device kernel body
//                          depends on it.
//   torch.h / torch.cpp     Torch tensor adapters, TorchOpContext, CUDA stream and logging glue.
//   Tensor.h / common.h     Lightweight tensor descriptor, shape/stride helpers and common CUDA
//                          error/assert utilities shared by native and CuTe kernels.
//
// Native runtime:
//   native/zgemm.h          Native entry declarations: `gemm_w4a4` is the single W4A4 GEMM shared
//                          by INT4 and NVFP4, `quantize_w4a4` the activation + LoRA-down producer.
//   native/dispatch.h       Shared dtype/dispatch helpers for native template configuration.
//   native/gemm_w4a4.{cu,cuh}
//                          Native GEMM implementation, including FP4 and the fused MLP,
//                          quantized-output and smooth-output paths.
//   native/gemm_w4a4_impl.cuh / gemm_w4a4_launch.cuh
//                          Kernel body/launch wiring and explicit-instantiation helpers.
//   native/gemm_base.cuh, mma.cuh, lora.cuh, epilogues.cuh, gemm_utils.cuh, utils.cuh
//                          Shared native GEMM building blocks: tile/host config, MMA atoms, LoRA
//                          epilogue, fusion epilogues, quantization/vector utilities.
//   native/gemm_w4a4_int4.cu
//                          Explicit INT4 instantiations of the native GEMM.
//   native/gemm_w4a4_fp4.cu
//                          Explicit NVFP4 instantiations of the native GEMM.
//
// CuTe runtime:
//   cute/fp4/gemm/gemm_w4a4.{h,cuh,cu}
//                          SM120 CuTe NVFP4 TMA/MMA implementation for native and row-major D.
//   cute/fp4/quantize/quantize_uni.cu
//                          Universal producer for rank 0 and all non-specialized rank values.
//   cute/fp4/quantize/quantize_impl.cuh
//                          Shared TiledMMA/TiledCopy implementation for rank-specialized TUs.
//   cute/fp4/quantize/quantize_r{32,64,128}.cu
//                          Independent rank-specific fused and split-K instantiations.
//   cute/fp8/gemm/gemm_w8a8.{h,cuh,cu}
//                          SM120 E4M3 GEMM with a fused BF16 LoRA-up epilogue.
//   cute/fp8/quantize/quantize_w8a8.{h,cuh,cu}
//                          Fused per-token E4M3 activation and BF16 LoRA-down producer.
//   cute/fp8/quantize/quantize_w8a8_r{32,64,128}.cu
//                          Rank-specialized producer instantiations.
//
// Selection summary:
//   native + native layout -> native gemm_w4a4 (INT4 and NVFP4 share one kernel family)
//   cute + FP4 -> cute gemm_w4a4 with LoraLayout kNative/kRowMajor
//   native producer -> native quantize_w4a4; row_major producer -> cute quantize_w4a4

#include <optional>
#include <string>
#include <vector>

#include "torch.h"
#include "gate_activation.h"

namespace pybind11 {
class module_;
}  // namespace pybind11

namespace cache_dit::svdq {

// Single public W4A4 GEMM entry. Python/native callers provide the full optional ABI once; the
// definition normalizes the host parameters and selects the already-compiled kernel family. The
// fused arguments (qout/oscales/lora_down/lora_act_out/smooth_factor/fuse_silu) are supported by
// both families: the CuTe GEMM quantizes its activated tile into the same native pack and reduces
// the next layer's LoRA-down term, so a fused chain can stay entirely on either runtime. The
// row-major D layout needs an FP32 scratch for its cross-CTA reduction, which this layer allocates
// and hands to the CuTe entry as `lora_scratch`.
void gemm_w4a4(std::optional<torch::Tensor> act, std::optional<torch::Tensor> wgt,
               std::optional<torch::Tensor> out, std::optional<torch::Tensor> qout,
               std::optional<torch::Tensor> ascales, std::optional<torch::Tensor> wscales,
               std::optional<torch::Tensor> oscales, std::optional<torch::Tensor> lora_act_in,
               std::optional<torch::Tensor> lora_up, std::optional<torch::Tensor> lora_down,
               std::optional<torch::Tensor> lora_act_out, std::optional<torch::Tensor> bias,
               std::optional<torch::Tensor> smooth_factor, bool act_unsigned,
               std::vector<float> lora_scales, bool fuse_silu, bool fp4, float alpha,
               std::optional<torch::Tensor> wcscales, const std::string &runtime,
               const std::string &lora_layout);

// Single public W4A4 activation/LoRA-down producer. LoraLayout determines both the output contract
// and the implementation: native layout -> native producer, row_major -> CuTe producer.
void quantize_w4a4(std::optional<torch::Tensor> input, std::optional<torch::Tensor> output,
                   std::optional<torch::Tensor> ascales, std::optional<torch::Tensor> lora_down,
                   std::optional<torch::Tensor> lora_act, std::optional<torch::Tensor> smooth,
                   std::optional<torch::Tensor> gate, bool fuse_glu, bool fp4,
                   std::vector<float> lora_scales, const std::string &lora_layout,
                   const std::string &gate_act, bool gate_first);

// Single public W8A8 GEMM with a fused BF16 LoRA-up epilogue. Unlike `gemm_w4a4` there is only one
// implementation (CuTe SM120) and one scale contract: per-token activation scale,
// per-output-channel residual scale, both FP32. The low-rank branch is never launched as a separate
// GEMM.
void gemm_w8a8(std::optional<torch::Tensor> act, std::optional<torch::Tensor> wgt,
               std::optional<torch::Tensor> out, std::optional<torch::Tensor> ascales,
               std::optional<torch::Tensor> wscales, std::optional<torch::Tensor> lora_act_in,
               std::optional<torch::Tensor> lora_up, std::optional<torch::Tensor> bias);

// Single public W8A8 activation producer: per-token E4M3 quantization fused with the BF16
// LoRA-down projection, so the low-rank branch never launches a standalone GEMM either.
void quantize_w8a8(std::optional<torch::Tensor> input, std::optional<torch::Tensor> output,
                   std::optional<torch::Tensor> ascales, std::optional<torch::Tensor> lora_down,
                   std::optional<torch::Tensor> lora_act, std::optional<torch::Tensor> smooth);

// Hooks the public dispatchers onto the extension's `ops` submodule.
void register_svdq_ops(pybind11::module_ &ops);

}  // namespace cache_dit::svdq
