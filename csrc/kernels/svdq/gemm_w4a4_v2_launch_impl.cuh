#pragma once

#include "gemm_w4a4_v2.cuh"
#include "gemm_w4a4_v2_launch.cuh"

namespace svdq::kernels {

template <typename PackedConfig, int LogicalBlockM, int NumStages>
inline void launch_gemm_w4a4_v2_specialized(Tensor act,
                                            Tensor wgt,
                                            Tensor out,
                                            Tensor ascales,
                                            Tensor wscales,
                                            Tensor lora_act_in,
                                            Tensor lora_up,
                                            Tensor bias,
                                            bool act_unsigned,
                                            float alpha,
                                            Tensor wcscales) {
  using Config = GEMMConfig_W4A4_V2_Block<PackedConfig, LogicalBlockM>;
  GEMM_W4A4_V2_Launch<Config, NumStages>::gemm_w4a4(
    act,
    wgt,
    out,
    ascales,
    wscales,
    lora_act_in,
    lora_up,
    bias,
    act_unsigned,
    alpha,
    wcscales);
}

#define SVDQ_DEFINE_GEMM_W4A4_V2_LAUNCH(DTYPE_NAME, PACKED_CONFIG, STAGE)          \
  void launch_gemm_w4a4_v2_##DTYPE_NAME##_int4_stage##STAGE(                       \
    Tensor act,                                                                    \
    Tensor wgt,                                                                    \
    Tensor out,                                                                    \
    Tensor ascales,                                                                \
    Tensor wscales,                                                                \
    Tensor lora_act_in,                                                            \
    Tensor lora_up,                                                                \
    Tensor bias,                                                                   \
    bool act_unsigned,                                                             \
    float alpha,                                                                   \
    Tensor wcscales,                                                               \
    int logical_block_m) {                                                         \
    switch (logical_block_m) {                                                     \
      case 64:                                                                     \
        launch_gemm_w4a4_v2_specialized<PACKED_CONFIG, 64, STAGE>(                 \
          act,                                                                     \
          wgt,                                                                     \
          out,                                                                     \
          ascales,                                                                 \
          wscales,                                                                 \
          lora_act_in,                                                             \
          lora_up,                                                                 \
          bias,                                                                    \
          act_unsigned,                                                            \
          alpha,                                                                   \
          wcscales);                                                               \
        return;                                                                    \
      case 128:                                                                    \
        launch_gemm_w4a4_v2_specialized<PACKED_CONFIG, 128, STAGE>(                \
          act,                                                                     \
          wgt,                                                                     \
          out,                                                                     \
          ascales,                                                                 \
          wscales,                                                                 \
          lora_act_in,                                                             \
          lora_up,                                                                 \
          bias,                                                                    \
          act_unsigned,                                                            \
          alpha,                                                                   \
          wcscales);                                                               \
        return;                                                                    \
      case 256:                                                                    \
        launch_gemm_w4a4_v2_specialized<PACKED_CONFIG, 256, STAGE>(                \
          act,                                                                     \
          wgt,                                                                     \
          out,                                                                     \
          ascales,                                                                 \
          wscales,                                                                 \
          lora_act_in,                                                             \
          lora_up,                                                                 \
          bias,                                                                    \
          act_unsigned,                                                            \
          alpha,                                                                   \
          wcscales);                                                               \
        return;                                                                    \
      default:                                                                     \
        throw std::runtime_error("Unsupported svdq_gemm_w4a4_v2 BLOCK_M.");        \
    }                                                                              \
  }

}  // namespace svdq::kernels
