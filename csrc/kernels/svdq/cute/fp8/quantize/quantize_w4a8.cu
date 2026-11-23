#include "quantize_w4a8.h"

#include "quantize_w8a8.h"

namespace svdq::cute_kernels {

void quantize_w4a8(Tensor input, Tensor output, Tensor ascales, Tensor lora_down, Tensor lora_act,
                   Tensor smooth, Tensor lora_scratch) {
  // Only the residual weight format differs in W4A8. Reuse the complete activation validation,
  // rank dispatch, split-K scheduling and fused LoRA-down implementation unchanged.
  quantize_w8a8(input, output, ascales, lora_down, lora_act, smooth, lora_scratch);
}

}  // namespace svdq::cute_kernels
