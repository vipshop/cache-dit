#include "zgemm.h"

#include <cstdlib>
#include <stdexcept>

#include "gemm_w4a4_v2_launch.cuh"

namespace svdq::kernels {

namespace {
// Some models may traced mix on Ada stays register-bound, so the useful tuning knob is CTA
// granularity. The hot shapes cluster into short prompt rows, 4096-row image rows, and longer
// mixed-token rows. Prompt-side small and medium-N projections prefer the original packed 256-row
// CTA, 4096-row image projections prefer 128 up to the 9K-width band before switching to 64 for
// the widest MLP projections, and the longer mixed-token rows split around the 12K-width band
// between 256 and 128.
static int get_v2_auto_block_m(int M, int N) {
  auto *prop = getCurrentDeviceProperties();
  if (prop->major != 8 || prop->minor != 9) {
    return 256;
  }

  if (M <= 1024) {
    return N <= 9216 ? 256 : 128;
  }
  if (M <= 4096) {
    return N <= 9216 ? 128 : 64;
  }
  return N <= 12288 ? 256 : 128;
}

static int get_v2_logical_block_m(int M, int N) {
  const char *value = std::getenv("CACHE_DIT_SVDQ_V2_BLOCK_M");
  if (value == nullptr || value[0] == '\0') {
    return get_v2_auto_block_m(M, N);
  }

  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0') {
    throw std::runtime_error(
      "CACHE_DIT_SVDQ_V2_BLOCK_M must be one of 64, 128, or 256.");
  }
  if (parsed != 64 && parsed != 128 && parsed != 256) {
    throw std::runtime_error(
      "CACHE_DIT_SVDQ_V2_BLOCK_M must be one of 64, 128, or 256.");
  }
  return static_cast<int>(parsed);
}

static int get_v2_num_stages(int stage) {
  if (stage < 1 || stage > 3) {
    throw std::runtime_error("svdq_gemm_w4a4_v2 stage must be one of 1, 2, or 3.");
  }
  return stage;
}

static gemm_w4a4_v2_launch_fn get_v2_launch_fn(Tensor::ScalarType dtype,
                                               int stage) {
  switch (stage) {
    case 1:
      if (dtype == Tensor::FP16) {
        return &launch_gemm_w4a4_v2_fp16_int4_stage1;
      }
      if (dtype == Tensor::BF16) {
        return &launch_gemm_w4a4_v2_bf16_int4_stage1;
      }
      throw std::runtime_error("svdq_gemm_w4a4_v2 expects FP16 or BF16 INT4 scales.");
    case 2:
      if (dtype == Tensor::FP16) {
        return &launch_gemm_w4a4_v2_fp16_int4_stage2;
      }
      if (dtype == Tensor::BF16) {
        return &launch_gemm_w4a4_v2_bf16_int4_stage2;
      }
      throw std::runtime_error("svdq_gemm_w4a4_v2 expects FP16 or BF16 INT4 scales.");
    case 3:
      if (dtype == Tensor::FP16) {
        return &launch_gemm_w4a4_v2_fp16_int4_stage3;
      }
      if (dtype == Tensor::BF16) {
        return &launch_gemm_w4a4_v2_bf16_int4_stage3;
      }
      throw std::runtime_error("svdq_gemm_w4a4_v2 expects FP16 or BF16 INT4 scales.");
    default:
      throw std::runtime_error("svdq_gemm_w4a4_v2 stage must be one of 1, 2, or 3.");
  }
}

}  // namespace

void gemm_w4a4_v2(Tensor act, Tensor wgt, Tensor out, Tensor ascales, Tensor wscales,
                  Tensor lora_act_in, Tensor lora_up, Tensor bias, bool act_unsigned,
                  float alpha, Tensor wcscales, int stage) {
  if (!act.valid() || !wgt.valid() || !out.valid() || !ascales.valid() || !wscales.valid()) {
    throw std::runtime_error(
      "svdq_gemm_w4a4_v2 requires act, wgt, out, ascales, and wscales tensors.");
  }
  if (alpha != 1.0f) {
    throw std::runtime_error(
      "svdq_gemm_w4a4_v2 only supports alpha=1.0 for the INT4 runtime path.");
  }

  const int M = act.numel() / act.shape[-1];
  const int N = wgt.shape[0];
  Tensor::ScalarType dtype = ascales.dtype();
  const int logical_block_m = get_v2_logical_block_m(M, N);
  const int num_stages = get_v2_num_stages(stage);
    const auto launch = get_v2_launch_fn(dtype, num_stages);
  launch(act, wgt, out, ascales, wscales, lora_act_in, lora_up, bias, act_unsigned, alpha,
      wcscales, logical_block_m);
}

}  // namespace svdq::kernels
