from typing import Optional

import torch

from .register import (
  _AttnBackend,
  _AttnBackendRegistry,
  _context_parallel_attention,
  _ContextParallelConfig,
)


def _cudnn_attention_forward_op(
  ctx: torch.autograd.function.FunctionCtx,
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  _save_ctx: bool = True,
  _cp_config: Optional["_ContextParallelConfig"] = None,
):
  if return_lse:
    raise ValueError("cudnn attention via SDPA does not support return_lse=True")

  if _save_ctx:
    ctx.save_for_backward(query, key, value)
    ctx.attn_mask = attn_mask
    ctx.dropout_p = dropout_p
    ctx.is_causal = is_causal
    ctx.scale = scale
    ctx.enable_gqa = enable_gqa

  query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
  with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
    out = torch.nn.functional.scaled_dot_product_attention(
      query=query,
      key=key,
      value=value,
      attn_mask=attn_mask,
      dropout_p=dropout_p,
      is_causal=is_causal,
      scale=scale,
      enable_gqa=enable_gqa,
    )
  out = out.permute(0, 2, 1, 3)
  return out


def _cudnn_attention_backward_op(
  ctx: torch.autograd.function.FunctionCtx,
  grad_out: torch.Tensor,
  *args,
  **kwargs,
):
  raise NotImplementedError("Backward for cudnn attention via SDPA is not implemented yet.")


@_AttnBackendRegistry.register(
  _AttnBackend._SDPA_CUDNN,
  constraints=[],
  supports_context_parallel=True,
)
def _cudnn_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  _cp_config: Optional["_ContextParallelConfig"] = None,
) -> torch.Tensor:
  lse = None
  if _cp_config is None and not return_lse:
    query, key, value = (x.permute(0, 2, 1, 3).contiguous() for x in (query, key, value))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
      out = torch.nn.functional.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
      )
    out = out.permute(0, 2, 1, 3)
  else:
    out = _context_parallel_attention(
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      enable_gqa,
      return_lse,
      forward_op=_cudnn_attention_forward_op,
      backward_op=_cudnn_attention_backward_op,
      _cp_config=_cp_config,
    )
    if return_lse:
      out, lse = out

  return (out, lse) if return_lse else out
