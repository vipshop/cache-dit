#include "quantize_impl.cuh"

namespace svdq::cute_kernels {

void quantize_r64(svdq::Tensor input, svdq::Tensor output, svdq::Tensor ascales,
                  svdq::Tensor lora_down, svdq::Tensor lora_act, svdq::Tensor smooth,
                  std::vector<float> lora_scales, svdq::Tensor lora_scratch) {
  quantize_cute<4>(input, output, ascales, lora_down, lora_act, smooth, std::move(lora_scales),
                   lora_scratch);
}

}  // namespace svdq::cute_kernels
