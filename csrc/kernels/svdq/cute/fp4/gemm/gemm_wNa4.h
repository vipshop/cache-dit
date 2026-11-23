#pragma once

#include "gemm_w4a4.h"

namespace svdq::cute_kernels {

// Packed W1/W2/W3 weight GEMMs. The activation, LoRA and epilogue contracts are identical to
// gemm_w4a4; wgt_lo/wgt_hi are compact packed weights and are expanded on chip.
void expand_wna4(svdq::Tensor wgt_lo, svdq::Tensor wgt_hi, svdq::Tensor out, int bits, int K);

void gemm_w1a4(svdq::Tensor act, svdq::Tensor wgt_lo, svdq::Tensor out, svdq::Tensor ascales,
               svdq::Tensor wscales, svdq::Tensor lora_act_in, svdq::Tensor lora_up,
               svdq::Tensor bias, svdq::Tensor wcscales, float alpha, int K, svdq::Tensor wgt_hi);

void gemm_w2a4(svdq::Tensor act, svdq::Tensor wgt_lo, svdq::Tensor out, svdq::Tensor ascales,
               svdq::Tensor wscales, svdq::Tensor lora_act_in, svdq::Tensor lora_up,
               svdq::Tensor bias, svdq::Tensor wcscales, float alpha, int K, svdq::Tensor wgt_hi);

void gemm_w3a4(svdq::Tensor act, svdq::Tensor wgt_lo, svdq::Tensor out, svdq::Tensor ascales,
               svdq::Tensor wscales, svdq::Tensor lora_act_in, svdq::Tensor lora_up,
               svdq::Tensor bias, svdq::Tensor wcscales, float alpha, int K, svdq::Tensor wgt_hi);

}  // namespace svdq::cute_kernels
