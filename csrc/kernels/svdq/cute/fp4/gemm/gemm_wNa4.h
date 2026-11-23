#pragma once

#include "gemm_w4a4.h"

// Packed W1/W2/W3 ABI.
//
// Two-stage NVFP4 remains the baseline: one BF16 per-output-channel scale
// `s_c`, one FP8_E4M3 per-16-input scale `s_g`, and R_hat = s_c * s_g * Q_b[q].
// W3 stores Q_b as a 2-bit magnitude plane plus a sign plane; W1/W2 store sign
// and zero/one magnitude bit. The optional W1/W2 third stage adds a uint8
// `wgt_micro` [N,ceil(K/16)] buffer containing four 2-bit exponents per 16-wide
// group. Each exponent selects a table of legal E2M1 magnitudes for four
// consecutive K elements. Because the table output is already E2M1, expansion
// only changes the B nibble and leaves the W4A4 MMA/epilogue untouched.
namespace svdq::cute_kernels {

void expand_wna4(svdq::Tensor wgt_lo, svdq::Tensor wgt_hi, svdq::Tensor out, int bits, int K,
                 svdq::Tensor wgt_micro, int inner_group);

void gemm_w1a4(svdq::Tensor act, svdq::Tensor wgt_lo, svdq::Tensor out, svdq::Tensor ascales,
               svdq::Tensor wscales, svdq::Tensor lora_act_in, svdq::Tensor lora_up,
               svdq::Tensor bias, svdq::Tensor wcscales, float alpha, int K, svdq::Tensor wgt_hi,
               svdq::Tensor wgt_micro, int inner_group);

void gemm_w2a4(svdq::Tensor act, svdq::Tensor wgt_lo, svdq::Tensor out, svdq::Tensor ascales,
               svdq::Tensor wscales, svdq::Tensor lora_act_in, svdq::Tensor lora_up,
               svdq::Tensor bias, svdq::Tensor wcscales, float alpha, int K, svdq::Tensor wgt_hi,
               svdq::Tensor wgt_micro, int inner_group);

void gemm_w3a4(svdq::Tensor act, svdq::Tensor wgt_lo, svdq::Tensor out, svdq::Tensor ascales,
               svdq::Tensor wscales, svdq::Tensor lora_act_in, svdq::Tensor lora_up,
               svdq::Tensor bias, svdq::Tensor wcscales, float alpha, int K, svdq::Tensor wgt_hi,
               svdq::Tensor wgt_micro, int inner_group);

}  // namespace svdq::cute_kernels
