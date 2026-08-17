from typing import Optional

import torch

from ._distributed_primitives import _All2AllComm
from ._modeling_parallel import _ContextParallelConfig

__all__ = [
  "UlyssesAttention",
]


class UlyssesAttention(torch.autograd.Function):
  """Ulysses attention with cache-dit's async all-to-all kernels."""

  @staticmethod
  def forward(
    ctx: torch.autograd.function.FunctionCtx,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float],
    enable_gqa: bool,
    return_lse: bool,
    cp_gqa_strategy: Optional[str],
    forward_op,
    backward_op,
    _cp_config: Optional["_ContextParallelConfig"] = None,
  ):
    if _cp_config is None:
      raise ValueError("Context parallel config must be provided for Ulysses attention.")

    ctx.forward_op = forward_op
    ctx.backward_op = backward_op
    ctx._cp_config = _cp_config

    num_q_heads = query.shape[2]
    comm = _All2AllComm(_cp_config)

    if cp_gqa_strategy == "group_aligned_flash_varlen":
      if return_lse:
        raise ValueError("return_lse is not supported for group-aligned GQA Ulysses.")

      query, key, value, attn_mask, q_split_sizes, local_sequence_length = (
        comm.send_group_aligned_gqa_qkv(query, key, value, attn_mask))

      out = forward_op(
        ctx,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        True,
        return_lse,
        _save_ctx=False,
        _cp_config=_cp_config,
      )
      return comm.send_group_aligned_gqa_o(out, q_split_sizes, local_sequence_length)

    # Keep K/LSE on the non-fp8 path for better numerical stability.
    query_wait = comm.send_q(query)
    query = query_wait.wait()  # type: torch.Tensor

    local_seq_len = key.shape[1]
    if cp_gqa_strategy == "replicate_kv_sequence":
      key, value = comm.gather_replicated_kv_for_local_q(query, key, value, num_q_heads)
      enable_gqa = False
    else:
      key_wait = comm.send_k(key)
      value_wait = comm.send_v(value)
      key = key_wait.wait()  # type: torch.Tensor
      value = value_wait.wait()  # type: torch.Tensor

    if attn_mask is not None:
      # The mask must cover the gathered full-sequence K/V in original token
      # order. Callers either pass the full 2D [B, L] padding mask, or a rank
      # shard of it ([B, L/world]); gather shards back in rank order. Keep it
      # 2D so each backend op applies it its own way (view/unpad).
      if attn_mask.dim() != 2:
        raise ValueError("Ulysses attention expects a 2D [B, L] padding mask, "
                         f"got shape {tuple(attn_mask.shape)}.")
      if attn_mask.shape[1] != key.shape[1]:
        if attn_mask.shape[1] != local_seq_len:
          raise ValueError(f"Ulysses attention mask length {attn_mask.shape[1]} matches neither "
                           f"the local ({local_seq_len}) nor the global ({key.shape[1]}) sequence.")
        attn_mask = comm.all_gather_tensor_dim(attn_mask, dim=1)

    out = forward_op(
      ctx,
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      enable_gqa,
      return_lse,
      _save_ctx=False,
      _cp_config=_cp_config,
    )
    if return_lse:
      out, lse, *_ = out

    out_wait = comm.send_o(out)

    if return_lse:
      lse = lse.unsqueeze(-1)  # (B, S_Q_GLOBAL, H_LOCAL, D=1)
      lse_wait = comm.send_lse(lse)
      out = out_wait.wait()  # type: torch.Tensor
      lse = lse_wait.wait()  # type: torch.Tensor
      lse = lse.squeeze(-1).contiguous()  # (B, S_Q_LOCAL, H_GLOBAL)
    else:
      out = out_wait.wait()  # type: torch.Tensor
      lse = None

    return (out, lse) if return_lse else out

  @staticmethod
  def backward(
    ctx: torch.autograd.function.FunctionCtx,
    grad_out: torch.Tensor,
    *args,
  ):
    raise NotImplementedError(
      "Backward pass for Ulysses Attention in cache-dit is not implemented yet.")
