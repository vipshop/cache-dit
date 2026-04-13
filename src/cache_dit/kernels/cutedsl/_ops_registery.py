import torch
from typing import Tuple

from ._merge_attn_states import fused_merge_attn_states as _cutedsl_fused_merge_attn_states

# Attention related ops
torch.library.define(
  "cache_dit_cutedsl_ops::fused_merge_attn_states",
  "(Tensor prev_out, Tensor prev_lse, Tensor suff_out, Tensor suff_lse) "
  "-> (Tensor out, Tensor lse)",
)


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


__all__ = ["fused_merge_attn_states"]
