// Adapted from nunchaku/src/kernels/zgemm/gemm_w4a4_launch.cuh
#pragma once

#include "gemm_w4a4.cuh"
#include "epilogues.cuh"

namespace svdq::native_kernels {

template <typename Config, bool USE_FP4>
class GEMM_W4A4_Launch {
  using GEMM = GEMM_W4A4<Config>;
  using Epilogues = Epilogues<Config>;
  using Lora = Lora<Config>;

  using packed_act_t = typename GEMM::packed_act_t;
  using packed_wgt_t = typename GEMM::packed_wgt_t;
  using packed_ascale_t = typename GEMM::packed_ascale_t;
  using packed_wscale_t = typename GEMM::packed_wscale_t;
  using packed_amscale_t = typename GEMM::packed_amscale_t;
  using packed_wmscale_t = typename GEMM::packed_wmscale_t;
  using packed_fpsum_t = typename GEMM::packed_fpsum_t;
  using half_t = typename GEMM::half_t;

 public:
  static void gemm_w4a4(Tensor act,            // packed act [M, K / 2]
                        Tensor wgt,            // packed wgt [N, K / 2]
                        Tensor out,            // linear     [M, N]
                        Tensor qout,           // packed act [M, N / 2]
                        Tensor ascales,        // packed as  [K / 64, M]
                        Tensor wscales,        // packed ws  [K / 64, N]
                        Tensor oscales,        // packed as  [N / 64, M]
                        Tensor lora_act_in,    // packed lora_act [M, R]
                        Tensor lora_up,        // packed lora_wgt [N, R]
                        Tensor lora_down,      // packed lora_wgt [N, R]
                        Tensor lora_act_out,   // packed lora_act [M, R]
                        Tensor bias,           // linear     [N]
                        Tensor smooth_factor,  // packed ws  [N]
                        bool act_unsigned,
                        std::vector<float> lora_scales,  // [R / 16]
                        bool fuse_silu, bool fp4, float alpha, Tensor wcscales);

  static void quantize_w4a4(Tensor input, Tensor output, Tensor oscales, Tensor lora_down,
                            Tensor lora_act_out, Tensor smooth, bool fuse_glu, bool fp4);
};

};  // namespace svdq::native_kernels
