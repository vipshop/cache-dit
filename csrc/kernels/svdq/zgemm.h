// Copied from nunchaku/src/kernels/zgemm/zgemm.h
#pragma once

#include "common.h"
#include "Tensor.h"

namespace svdq::kernels {

void gemm_w4a4(Tensor act,             // packed act [M, K / 2]
               Tensor wgt,             // packed act [N, K / 2]
               Tensor out,             // linear     [M, N]
               Tensor qout,            // packed act [M, N / 2]
               Tensor ascales,         // packed as  [K / 64, M]
               Tensor wscales,         // packed ws  [K / 64, N]
               Tensor oscales,         // packed as  [N / 64, M]
               Tensor poolout,         // linear     [M / PoolSize, N]
               Tensor lora_act_in,     // packed lora_act [M, R]
               Tensor lora_up,         // packed lora_wgt [N, R]
               Tensor lora_down,       // packed lora_wgt [N, R]
               Tensor lora_act_out,    // packed lora_act [M, R]
               Tensor norm_q,          // linear     [HEAD_DIM]
               Tensor norm_k,          // linear     [HEAD_DIM]
               Tensor rotary_emb,      // linear     [M, HEAD_DIM / 2, 2, 2]
               Tensor bias,            // packed ws  [N]
               Tensor smooth_factor,   // packed ws  [N], for quantization of the next layer
               Tensor out_vk,          // linear     [B, num_heads, head_dim + 1, head_dim]
               Tensor out_linearattn,  // linear     [B, (M), N / 3]
               bool act_unsigned,
               std::vector<float> lora_scales,  // [R / 16]
               bool fuse_silu, bool fp4, float alpha, Tensor wcscales,
               Tensor out_q,  // packed attention [B, H, M, D]
               Tensor out_k,  // packed attention [B, H, M, D]
               Tensor out_v,  // packed attention [B, H, M, D]
               int attn_tokens);
void gemm_w4a4_v2(Tensor act, Tensor wgt, Tensor out, Tensor ascales, Tensor wscales,
                  Tensor lora_act_in, Tensor lora_up, Tensor bias, bool act_unsigned,
                  float alpha, Tensor wcscales, int stage = 2);
void linearattn_vk_mul_q(Tensor q, Tensor vk);

void quantize_w4a4_act_fuse_lora(Tensor input, Tensor output, Tensor oscales, Tensor lora_down,
                                 Tensor lora_act_out, Tensor smooth = {}, bool fuse_glu = false,
                                 bool fp4 = false);
void quantize_w4a4_act(Tensor input, Tensor output, Tensor oscales);
void quantize_w4a4_wgt(Tensor input, Tensor output, Tensor oscales);

// EXPERIMENTAL, for sm_75
void set_faster_i2f_mode(std::string mode);

};  // namespace svdq::kernels
