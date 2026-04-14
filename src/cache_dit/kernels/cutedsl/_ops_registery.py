import torch
from typing import Tuple

from ._float8_comm import fp8_comm_per_token_dequant as _cutedsl_fp8_comm_per_token_dequant
from ._float8_comm import fp8_comm_per_token_quant as _cutedsl_fp8_comm_per_token_quant
from ._float8_comm import fp8_comm_qkv_permute_dequant as _cutedsl_fp8_comm_qkv_permute_dequant
from ._float8_comm import fp8_comm_qkv_permute_quant as _cutedsl_fp8_comm_qkv_permute_quant
from ._merge_attn_states import fused_merge_attn_states as _cutedsl_fused_merge_attn_states

# FP8 related ops
torch.library.define(
  "cache_dit_cutedsl_ops::fp8_comm_per_token_quant",
  "(Tensor x) -> Tensor",
)
torch.library.define(
  "cache_dit_cutedsl_ops::fp8_comm_per_token_dequant",
  "(Tensor x) -> Tensor",
)
torch.library.define(
  "cache_dit_cutedsl_ops::fp8_comm_qkv_permute_quant",
  "(Tensor x) -> Tensor",
)
torch.library.define(
  "cache_dit_cutedsl_ops::fp8_comm_qkv_permute_dequant",
  "(Tensor quant_x) -> Tensor",
)

# Attention related ops
torch.library.define(
  "cache_dit_cutedsl_ops::fused_merge_attn_states",
  "(Tensor prev_out, Tensor prev_lse, Tensor suff_out, Tensor suff_lse) "
  "-> (Tensor out, Tensor lse)",
)


@torch.library.impl("cache_dit_cutedsl_ops::fp8_comm_per_token_quant", "CUDA")
def _fp8_comm_per_token_quant(x: torch.Tensor) -> torch.Tensor:
  return _cutedsl_fp8_comm_per_token_quant(x)


@torch.library.register_fake("cache_dit_cutedsl_ops::fp8_comm_per_token_quant")
def _fake_fp8_comm_per_token_quant(x: torch.Tensor) -> torch.Tensor:
  assert x.dtype == torch.bfloat16
  *shape, head_size = x.shape
  return x.new_empty((*shape, head_size + 2), dtype=torch.float8_e4m3fn)


@torch.library.impl("cache_dit_cutedsl_ops::fp8_comm_per_token_dequant", "CUDA")
def _fp8_comm_per_token_dequant(x: torch.Tensor) -> torch.Tensor:
  return _cutedsl_fp8_comm_per_token_dequant(x)


@torch.library.register_fake("cache_dit_cutedsl_ops::fp8_comm_per_token_dequant")
def _fake_fp8_comm_per_token_dequant(x: torch.Tensor) -> torch.Tensor:
  assert x.dtype == torch.float8_e4m3fn
  *shape, packed_head_size = x.shape
  return x.new_empty((*shape, packed_head_size - 2), dtype=torch.bfloat16)


@torch.library.impl("cache_dit_cutedsl_ops::fp8_comm_qkv_permute_quant", "CUDA")
def _fp8_comm_qkv_permute_quant(x: torch.Tensor) -> torch.Tensor:
  return _cutedsl_fp8_comm_qkv_permute_quant(x)


@torch.library.register_fake("cache_dit_cutedsl_ops::fp8_comm_qkv_permute_quant")
def _fake_fp8_comm_qkv_permute_quant(x: torch.Tensor) -> torch.Tensor:
  batch, seq_len, partitions, num_heads, head_size = x.shape
  return x.new_empty((partitions, seq_len, batch, num_heads, head_size + 4),
                     dtype=torch.float8_e4m3fn)


@torch.library.impl("cache_dit_cutedsl_ops::fp8_comm_qkv_permute_dequant", "CUDA")
def _fp8_comm_qkv_permute_dequant(quant_x: torch.Tensor) -> torch.Tensor:
  return _cutedsl_fp8_comm_qkv_permute_dequant(quant_x)


@torch.library.register_fake("cache_dit_cutedsl_ops::fp8_comm_qkv_permute_dequant")
def _fake_fp8_comm_qkv_permute_dequant(quant_x: torch.Tensor) -> torch.Tensor:
  seq_len, batch, num_heads, packed_head_size = quant_x.shape
  return quant_x.new_empty((batch, seq_len, num_heads, packed_head_size - 4), dtype=torch.bfloat16)


@torch.library.impl("cache_dit_cutedsl_ops::fused_merge_attn_states", "CUDA")
def _fused_merge_attn_states(
  prev_out: torch.Tensor,
  prev_lse: torch.Tensor,
  suff_out: torch.Tensor,
  suff_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  return _cutedsl_fused_merge_attn_states(
    prev_out=prev_out,
    prev_lse=prev_lse,
    suff_out=suff_out,
    suff_lse=suff_lse,
  )


@torch.library.register_fake("cache_dit_cutedsl_ops::fused_merge_attn_states")
def _fake_fused_merge_attn_states(
  prev_out: torch.Tensor,
  prev_lse: torch.Tensor,
  suff_out: torch.Tensor,
  suff_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  batch, seq_len, num_heads, head_size = suff_out.shape

  suff_out = suff_out.flatten(0, 1).contiguous().view(-1, head_size)
  suff_lse = suff_lse.flatten(0, 1).squeeze(-1).contiguous().view(-1)
  out = suff_out.new_empty(suff_out.shape, dtype=suff_out.dtype)
  lse = suff_lse.new_empty(suff_lse.shape, dtype=suff_lse.dtype)
  out = out.view(batch, seq_len, num_heads, head_size)
  lse = lse.view(batch, seq_len, num_heads, 1)
  return out, lse


def fused_merge_attn_states(
  prev_out: torch.Tensor,
  prev_lse: torch.Tensor,
  suff_out: torch.Tensor,
  suff_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Dispatch the CuTe DSL fused attention-state merge operator.

  :param prev_out: Previous attention output `[B, N, H, D]`.
  :param prev_lse: Previous attention LSE `[B, N, H, 1]`.
  :param suff_out: Suffix attention output `[B, N, H, D]`.
  :param suff_lse: Suffix attention LSE `[B, N, H, 1]`.
  :returns: Tuple `(out, lse)`.
  """

  return torch.ops.cache_dit_cutedsl_ops.fused_merge_attn_states(
    prev_out,
    prev_lse,
    suff_out,
    suff_lse,
  )


def fp8_comm_per_token_quant(x: torch.Tensor) -> torch.Tensor:
  return torch.ops.cache_dit_cutedsl_ops.fp8_comm_per_token_quant(x)


def fp8_comm_per_token_dequant(x: torch.Tensor) -> torch.Tensor:
  return torch.ops.cache_dit_cutedsl_ops.fp8_comm_per_token_dequant(x)


def fp8_comm_qkv_permute_quant(x: torch.Tensor) -> torch.Tensor:
  return torch.ops.cache_dit_cutedsl_ops.fp8_comm_qkv_permute_quant(x)


def fp8_comm_qkv_permute_dequant(quant_x: torch.Tensor) -> torch.Tensor:
  return torch.ops.cache_dit_cutedsl_ops.fp8_comm_qkv_permute_dequant(quant_x)


__all__ = [
  "fp8_comm_per_token_quant",
  "fp8_comm_per_token_dequant",
  "fp8_comm_qkv_permute_quant",
  "fp8_comm_qkv_permute_dequant",
  "fused_merge_attn_states",
]
