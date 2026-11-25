#pragma once

#include <ATen/ATen.h>

namespace pybind11 {
class module_;
}  // namespace pybind11

namespace cache_dit::ulysses {

void ulysses_fp8_per_token_quant(at::Tensor x, at::Tensor y, int64_t packed_head_size,
                                 int64_t group_size = 0);
void ulysses_fp8_per_token_dequant(at::Tensor x, at::Tensor y, int64_t head_size,
                                   int64_t group_size = 0);
void ulysses_fp8_qkv_permute_quant(at::Tensor x, at::Tensor y, int64_t packed_head_size,
                                   int64_t group_size = 0);
void ulysses_fp8_qkv_permute_dequant(at::Tensor quant_x, at::Tensor x, int64_t head_size,
                                     int64_t group_size = 0);
void ulysses_fp8_o_permute_quant(at::Tensor x, at::Tensor y, int64_t partitions,
                                 int64_t packed_head_size, int64_t group_size = 0);
void ulysses_fp8_o_permute_dequant(at::Tensor quant_x, at::Tensor x, int64_t head_size,
                                   int64_t group_size = 0);

void ulysses_fp6_per_token_quant(at::Tensor x, at::Tensor y, int64_t packed_head_size,
                                 int64_t group_size = 32);
void ulysses_fp6_per_token_dequant(at::Tensor x, at::Tensor y, int64_t head_size,
                                   int64_t group_size);
void ulysses_fp6_qkv_permute_quant(at::Tensor x, at::Tensor y, int64_t packed_head_size,
                                   int64_t group_size = 32);
void ulysses_fp6_qkv_permute_dequant(at::Tensor quant_x, at::Tensor x, int64_t head_size,
                                     int64_t group_size);
void ulysses_fp6_o_permute_quant(at::Tensor x, at::Tensor y, int64_t partitions,
                                 int64_t packed_head_size, int64_t group_size = 32);
void ulysses_fp6_o_permute_dequant(at::Tensor quant_x, at::Tensor x, int64_t head_size,
                                   int64_t group_size);

void ulysses_fN_fused_qkv_quant(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor y,
                                int64_t partitions, int64_t q_heads, int64_t k_heads,
                                int64_t v_heads, int64_t q_width, int64_t k_width, int64_t v_width,
                                int64_t chunk_row_bytes, int64_t group_size, int64_t bits,
                                bool quantize_k);
void ulysses_fN_fused_qkv_dequant(at::Tensor packed, at::Tensor q, at::Tensor k, at::Tensor v,
                                  int64_t q_heads, int64_t k_heads, int64_t v_heads,
                                  int64_t head_size, int64_t q_width, int64_t k_width,
                                  int64_t v_width, int64_t chunk_row_bytes, int64_t group_size,
                                  int64_t bits, bool quantize_k);

// Hooks the communication operators onto the extension's `ops` submodule.
void register_ulysses_ops(pybind11::module_& ops);

}  // namespace cache_dit::ulysses
