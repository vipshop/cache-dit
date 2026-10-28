import torch

from ._comm import _call_fused_merge_attn_states

torch.library.define(
  "cache_dit_cuda_ops::fused_merge_attn_states",
  "(Tensor prev_out, Tensor prev_lse, Tensor suff_out, Tensor suff_lse) "
  "-> (Tensor out, Tensor lse)",
)


@torch.library.impl("cache_dit_cuda_ops::fused_merge_attn_states", "CUDA")
def _merge_attn_states_impl(
  prev_out: torch.Tensor,
  prev_lse: torch.Tensor,
  suff_out: torch.Tensor,
  suff_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  batch, seq_len, num_heads, head_size = suff_out.shape
  # The extension consumes flattened attention states, so the batch and sequence
  # dimensions are merged before the call and restored afterwards.
  prev_out = prev_out.flatten(0, 1).contiguous()
  suff_out = suff_out.flatten(0, 1).contiguous()
  prev_lse = prev_lse.flatten(0, 1).squeeze(-1).contiguous()
  suff_lse = suff_lse.flatten(0, 1).squeeze(-1).contiguous()

  out = torch.empty_like(suff_out)
  lse = torch.empty_like(suff_lse)
  _call_fused_merge_attn_states(prev_out, prev_lse, suff_out, suff_lse, out, lse)
  return out.view(batch, seq_len, num_heads, head_size), lse.view(batch, seq_len, num_heads, 1)


@torch.library.register_fake("cache_dit_cuda_ops::fused_merge_attn_states")
def _fake_merge_attn_states(
  prev_out: torch.Tensor,
  prev_lse: torch.Tensor,
  suff_out: torch.Tensor,
  suff_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  del prev_out, prev_lse
  batch, seq_len, num_heads, head_size = suff_out.shape
  # Exactly replicate the view chain of the real implementation so the abstract
  # tensors keep the same shape and strides under torch.compile.
  suff_out = suff_out.flatten(0, 1).contiguous()
  suff_lse = suff_lse.flatten(0, 1).squeeze(-1).contiguous()
  out = suff_out.new_empty(suff_out.shape, dtype=suff_out.dtype)
  lse = suff_lse.new_empty(suff_lse.shape, dtype=suff_lse.dtype)
  return out.view(batch, seq_len, num_heads, head_size), lse.view(batch, seq_len, num_heads, 1)


def fused_merge_attn_states(
  prev_out: torch.Tensor,
  prev_lse: torch.Tensor,
  suff_out: torch.Tensor,
  suff_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Fuse the attention states of two consecutive attention chunks, e.g. Ring Attention.

  :param prev_out: Previous attention output shaped `[B, N, H, D]`.
  :param prev_lse: Previous log-sum-exp shaped `[B, N, H, 1]`.
  :param suff_out: Suffix attention output shaped `[B, N, H, D]`.
  :param suff_lse: Suffix log-sum-exp shaped `[B, N, H, 1]`.
  :returns: The fused attention output `[B, N, H, D]` and log-sum-exp `[B, N, H, 1]`.
  """

  return torch.ops.cache_dit_cuda_ops.fused_merge_attn_states(
    prev_out,
    prev_lse,
    suff_out,
    suff_lse,
  )


__all__ = [
  "fused_merge_attn_states",
]
