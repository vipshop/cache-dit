// Adapted from nunchaku/src/kernels/zgemm/zgemm.h
#pragma once

#include "common.h"
#include "Tensor.h"

namespace svdq::native_kernels {

// The one native W4A4 GEMM shared by the INT4 and NVFP4 runtimes. Call sites must qualify it as
// `svdq::native_kernels::gemm_w4a4`: the public dispatcher `cache_dit::svdq::gemm_w4a4` forwards
// here, so an unqualified call inside that dispatcher would resolve to itself.
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
               Tensor smooth_factor,  // packed ws  [N], for quantization of the next layer
               bool act_unsigned,
               std::vector<float> lora_scales,  // [R / 16]
               bool fuse_silu, bool fp4, float alpha, Tensor wcscales);

void quantize_w4a4(Tensor input, Tensor output, Tensor oscales, Tensor lora_down,
                   Tensor lora_act_out, Tensor smooth = {}, bool fuse_glu = false,
                   bool fp4 = false);

};  // namespace svdq::native_kernels
