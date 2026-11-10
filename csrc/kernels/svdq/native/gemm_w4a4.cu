// Adapted from nunchaku/src/kernels/zgemm/gemm_w4a4.cu
#include "zgemm.h"
#include "gemm_w4a4_launch.cuh"

namespace svdq::native_kernels {

template <typename F>
static void invoke_launch(Tensor::ScalarType dtype, bool use_fp4, F &&launch) {
  dispatchBool(use_fp4, [&]<bool USE_FP4>() {
    if (dtype == Tensor::FP16) {
      launch.template operator()<GEMMConfig_W4A4_FP16, USE_FP4>();
    } else if (dtype == Tensor::BF16) {
      launch.template operator()<GEMMConfig_W4A4_BF16, USE_FP4>();
    } else {
      assert(false);
    }
  });
}

void gemm_w4a4(Tensor act,            // packed act [M, K / 2]
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
               bool fuse_silu, bool fp4, float alpha, Tensor wcscales) {
  Tensor::ScalarType dtype = Tensor::INVALID_SCALAR_TYPE;
  if (!fp4) {
    dtype = ascales.dtype();
  } else {
    for (auto tensor : {out, bias, lora_up, lora_down, wcscales}) {
      if (tensor.valid()) {
        assert(dtype == Tensor::INVALID_SCALAR_TYPE || dtype == tensor.dtype());
        dtype = tensor.dtype();
      }
    }
  }
  invoke_launch(dtype, fp4, [&]<typename Config, bool USE_FP4>() {
    GEMM_W4A4_Launch<Config, USE_FP4>::gemm_w4a4(
      act, wgt, out, qout, ascales, wscales, oscales, lora_act_in, lora_up, lora_down, lora_act_out,
      bias, smooth_factor, act_unsigned, lora_scales, fuse_silu, fp4, alpha, wcscales);
  });
}

void quantize_w4a4(Tensor input, Tensor output, Tensor oscales, Tensor lora_down,
                   Tensor lora_act_out, Tensor smooth, bool fuse_glu, bool fp4) {
  invoke_launch(input.dtype(), fp4, [&]<typename Config, bool USE_FP4>() {
    GEMM_W4A4_Launch<Config, USE_FP4>::quantize_w4a4(input, output, oscales, lora_down,
                                                     lora_act_out, smooth, fuse_glu, fp4);
  });
}

};  // namespace svdq::native_kernels
