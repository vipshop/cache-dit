from typing import Optional

import torch

from .register import (
  _AttnBackend,
  _AttnBackendRegistry,
  _context_parallel_attention,
  _ContextParallelConfig,
)


def _native_attention_forward_op(
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
  if _save_ctx:
    ctx.save_for_backward(query, key, value)
    ctx.attn_mask = attn_mask
    ctx.dropout_p = dropout_p
    ctx.is_causal = is_causal
    ctx.scale = scale
    ctx.enable_gqa = enable_gqa

  if return_lse:
    if attn_mask is not None:
      raise ValueError("`attn_mask` is not yet supported for native flash attention with lse.")
    out, lse = torch.ops.aten._scaled_dot_product_flash_attention(
      query.transpose(1, 2),
      key.transpose(1, 2),
      value.transpose(1, 2),
      dropout_p=dropout_p,
      is_causal=is_causal,
      scale=scale,
    )[:2]
    out = out.transpose(1, 2)
    lse = lse.transpose(1, 2)
    if lse.dim() == 3:
      lse = lse.unsqueeze(-1)
    return out, lse

  query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
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


def _native_attention_backward_op(
  ctx: torch.autograd.function.FunctionCtx,
  grad_out: torch.Tensor,
  *args,
  **kwargs,
):
  query, key, value = ctx.saved_tensors

  query.requires_grad_(True)
  key.requires_grad_(True)
  value.requires_grad_(True)

  query_t, key_t, value_t = (x.permute(0, 2, 1, 3) for x in (query, key, value))
  out = torch.nn.functional.scaled_dot_product_attention(
    query=query_t,
    key=key_t,
    value=value_t,
    attn_mask=ctx.attn_mask,
    dropout_p=ctx.dropout_p,
    is_causal=ctx.is_causal,
    scale=ctx.scale,
    enable_gqa=ctx.enable_gqa,
  )
  out = out.permute(0, 2, 1, 3)

  grad_out_t = grad_out.permute(0, 2, 1, 3)
  grad_query_t, grad_key_t, grad_value_t = torch.autograd.grad(
    outputs=out,
    inputs=[query_t, key_t, value_t],
    grad_outputs=grad_out_t,
    retain_graph=False,
  )

  grad_query = grad_query_t.permute(0, 2, 1, 3)
  grad_key = grad_key_t.permute(0, 2, 1, 3)
  grad_value = grad_value_t.permute(0, 2, 1, 3)

  return grad_query, grad_key, grad_value


@_AttnBackendRegistry.register(
  _AttnBackend.NATIVE,
  constraints=[],
  supports_context_parallel=True,
)
def _native_attention(
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
  if return_lse:
    raise ValueError("Native attention backend does not support setting `return_lse=True`.")
  if _cp_config is None:
    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
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
      forward_op=_native_attention_forward_op,
      backward_op=_native_attention_backward_op,
      _cp_config=_cp_config,
    )
  return out
