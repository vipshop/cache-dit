#pragma once

namespace svdq {

// Activation applied to the gate operand before the BF16 value/gate product.
// This small enum is shared by the host dispatchers and both producer families;
// it deliberately has no torch or CUTLASS dependency.
enum class GateAct : int {
  kSiLU = 0,
  kGeluTanh = 1,
  kGeluExact = 2,
};

}  // namespace svdq
