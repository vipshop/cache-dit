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
    forward_op,
    backward_op,
    _cp_config: Optional["_ContextParallelConfig"] = None,
  ):
    if _cp_config is None:
      raise ValueError("Context parallel config must be provided for Ulysses attention.")

    ctx.forward_op = forward_op
    ctx.backward_op = backward_op
    ctx._cp_config = _cp_config

    comm = _All2AllComm(_cp_config).init_meta(query)

    # Keep K/LSE on the non-fp8 path for better numerical stability.
    query_wait = comm.send_q(query)
    key_wait = comm.send_k(key)
    value_wait = comm.send_v(value)

    query = query_wait.wait()  # type: torch.Tensor
    key = key_wait.wait()  # type: torch.Tensor
    value = value_wait.wait()  # type: torch.Tensor
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
