#pragma once

#include "Tensor.h"

namespace svdq::kernels {

using gemm_w4a4_v2_launch_fn = void (*)(Tensor act,
                                        Tensor wgt,
                                        Tensor out,
                                        Tensor ascales,
                                        Tensor wscales,
                                        Tensor lora_act_in,
                                        Tensor lora_up,
                                        Tensor bias,
                                        bool act_unsigned,
                                        float alpha,
                                        Tensor wcscales,
                                        int logical_block_m);

#define SVDQ_DECLARE_GEMM_W4A4_V2_LAUNCH(DTYPE_NAME, STAGE)              \
  void launch_gemm_w4a4_v2_##DTYPE_NAME##_int4_stage##STAGE(             \
    Tensor act,                                                          \
    Tensor wgt,                                                          \
    Tensor out,                                                          \
    Tensor ascales,                                                      \
    Tensor wscales,                                                      \
    Tensor lora_act_in,                                                  \
    Tensor lora_up,                                                      \
    Tensor bias,                                                         \
    bool act_unsigned,                                                   \
    float alpha,                                                         \
    Tensor wcscales,                                                     \
    int logical_block_m)

SVDQ_DECLARE_GEMM_W4A4_V2_LAUNCH(fp16, 1);
SVDQ_DECLARE_GEMM_W4A4_V2_LAUNCH(fp16, 2);
SVDQ_DECLARE_GEMM_W4A4_V2_LAUNCH(fp16, 3);

SVDQ_DECLARE_GEMM_W4A4_V2_LAUNCH(bf16, 1);
SVDQ_DECLARE_GEMM_W4A4_V2_LAUNCH(bf16, 2);
SVDQ_DECLARE_GEMM_W4A4_V2_LAUNCH(bf16, 3);

#undef SVDQ_DECLARE_GEMM_W4A4_V2_LAUNCH

}  // namespace svdq::kernels
