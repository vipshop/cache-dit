# Currently we only have triton kernels for cache-dit, but we can re-implement
# some of them in CuteDSL for better performance if needed in the future. In
# that case, we can keep the same operator definitions in this file and just
# change the underlying implementations.

import torch
from typing import Tuple

from .triton_ops_registery import (
    triton_fp8_comm_per_token_quant,
    triton_fp8_comm_per_token_dequant,
    triton_fp8_comm_qkv_permute_quant,
    triton_fp8_comm_qkv_permute_dequant,
    triton_fused_merge_attn_states,
)


def fp8_comm_per_token_quant(x: torch.Tensor) -> torch.Tensor:
    return triton_fp8_comm_per_token_quant(x)


def fp8_comm_per_token_dequant(x: torch.Tensor) -> torch.Tensor:
    return triton_fp8_comm_per_token_dequant(x)


def fp8_comm_qkv_permute_quant(x: torch.Tensor) -> torch.Tensor:
    return triton_fp8_comm_qkv_permute_quant(x)


def fp8_comm_qkv_permute_dequant(quant_x: torch.Tensor) -> torch.Tensor:
    return triton_fp8_comm_qkv_permute_dequant(quant_x)


def fused_merge_attn_states(
    prev_out: torch.Tensor,
    prev_lse: torch.Tensor,
    suff_out: torch.Tensor,
    suff_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return triton_fused_merge_attn_states(
        prev_out,
        prev_lse,
        suff_out,
        suff_lse,
    )
