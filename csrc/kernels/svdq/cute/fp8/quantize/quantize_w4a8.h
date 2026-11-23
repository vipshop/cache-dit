#pragma once

#include "Tensor.h"

namespace svdq::cute_kernels {

// W4A8 uses the same E4M3 activation and BF16 LoRA-down producer as W8A8. This is an ABI alias,
// not a second quantization implementation; split-K scratch semantics are identical.
void quantize_w4a8(Tensor input, Tensor output, Tensor ascales, Tensor lora_down, Tensor lora_act,
                   Tensor smooth, Tensor lora_scratch);

}  // namespace svdq::cute_kernels
