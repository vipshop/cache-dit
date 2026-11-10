#pragma once

namespace svdq {

// Host-side routing parameters shared by the public SVDQ operators. Only the already-compiled
// kernel family is described here; the device-side GEMM templates are picked before launch, so no
// kernel body depends on this header.
enum class RuntimeKernel {
  kNative,
  kCuTe,
};

struct RuntimeParams {
  RuntimeKernel kernel = RuntimeKernel::kNative;
  bool fp4 = false;
  float alpha = 1.0f;
  bool act_unsigned = false;
};

// How the LoRA-down term D is laid out in `lora_act_in`. Shared by the native dispatcher and the
// CuTe GEMM; the native runtime only accepts `kNative`.
//   kNative:   the native activation producer's packed FP32 MMA-fragment D. The CuTe kernel gathers
//              it gmem->register and applies `lora_scales[i]` to rank group i on the fly.
//   kRowMajor: plain row-major BF16 [M, R] D with the per-rank scales already folded in. The CuTe
//              kernel TMA-loads it next to the weight tiles.
enum class LoraLayout {
  kNative,
  kRowMajor,
};

}  // namespace svdq
