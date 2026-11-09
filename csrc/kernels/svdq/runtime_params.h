#pragma once

namespace svdq {

// Host-side routing parameters shared by the public SVDQ operators. Device code must not depend on
// this header; it only describes which already-compiled kernel family should be launched.
enum class RuntimeKernel {
  kNative,
  kCuTe,
};

struct RuntimeParams {
  RuntimeKernel kernel = RuntimeKernel::kNative;
  bool fp4 = false;
  int stage = 1;
  float alpha = 1.0f;
  bool act_unsigned = false;
};

}  // namespace svdq
