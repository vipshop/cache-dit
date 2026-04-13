import torch
from typing import Optional
import torch.nn.functional as F

try:
  from ...kernels import fused_merge_attn_states

except ImportError:
  fused_merge_attn_states = None

from ._distributed_primitives import _RingP2PComm
from ._modeling_parallel import _ContextParallelConfig

__all__ = ["RingAttention"]


class RingAttention(torch.autograd.Function):

  @staticmethod
  def forward(
    ctx: torch.autograd.function.FunctionCtx,
    query: torch.Tensor,  # (B, S_LOCAL, H, D)
    key: torch.Tensor,  # (B, S_LOCAL, H, D)
    value: torch.Tensor,  # (B, S_LOCAL, H, D)
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
    ring_mesh = _cp_config._ring_mesh
    ring_group = ring_mesh.get_group()

    comm = _RingP2PComm(ring_group)

    prev_out = prev_lse = None

    ctx.forward_op = forward_op
    ctx.backward_op = backward_op
    ctx.q_shape = query.shape
    ctx.kv_shape = key.shape
    ctx._cp_config = _cp_config

    next_k, next_v = None, None

    for step in range(comm.world_size):
      if step + 1 != comm.world_size:
        next_k, next_v = comm.send_recv_kv(key, value)

      # [B, N, H, D], [B, N, H, 1]
      out, lse = forward_op(
        ctx,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        enable_gqa,
        True,  # return_lse
        _save_ctx=False,  # Only support forward pass here, no need to save ctx
        _cp_config=_cp_config,
      )

      # Refer to:
      # https://github.com/huggingface/diffusers/pull/12693#issuecomment-3627519544
      if lse.dim() == 3:
        lse = lse.unsqueeze(-1)  # type: torch.Tensor

      # Use _fused_merge_attn_states to combine the attention outputs and lses
      if prev_out is not None:
        # out = prev_out - F.sigmoid(lse - prev_lse) * (prev_out - out)
        # lse = prev_lse - F.logsigmoid(prev_lse - lse)
        if fused_merge_attn_states is not None:
          out, lse = fused_merge_attn_states(
            prev_out,
            prev_lse,
            out,
            lse,
          )
        else:
          if _cp_config.convert_to_fp32:
            out = out.to(torch.float32)
            lse = lse.to(torch.float32)

          out = prev_out - F.sigmoid(lse - prev_lse) * (prev_out - out)
          lse = prev_lse - F.logsigmoid(prev_lse - lse)

      prev_out, prev_lse = out, lse

      if step + 1 != comm.world_size:
        comm.wait()
        key, value = next_k, next_v

    out = out.to(query.dtype)
    lse = lse.squeeze(-1)  # (B, S_Q_GLOBAL, H_LOCAL)

    return (out, lse) if return_lse else out

  @staticmethod
  def backward(
    ctx: torch.autograd.function.FunctionCtx,
    grad_out: torch.Tensor,
    *args,
  ):
    raise NotImplementedError(
      "Backward pass is not implemented for _TemplatedRingBatchedP2PAttention.")
