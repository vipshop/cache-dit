from typing import Optional

import torch

from .register import (
  _AttnBackend,
  _AttnBackendRegistry,
  _context_parallel_attention,
  _ContextParallelConfig,
)

try:
  from flash_attn import flash_attn_func
  from flash_attn.flash_attn_interface import (
    _wrapped_flash_attn_backward,
    _wrapped_flash_attn_forward,
  )
  _flash_attn_available = True
except Exception:
  flash_attn_func = None
  _wrapped_flash_attn_backward = None
  _wrapped_flash_attn_forward = None
  _flash_attn_available = False

try:
  from flash_attn_interface import flash_attn_func as flash_attn_3_func
  _flash_attn_3_available = True
except ImportError:
  flash_attn_3_func = None
  _flash_attn_3_available = False


def _flash_attention_forward_op(
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
  if attn_mask is not None:
    raise ValueError("`attn_mask` is not yet supported for flash-attn 2.")
  if enable_gqa:
    raise ValueError("`enable_gqa` is not yet supported for flash-attn 2.")
  if flash_attn_func is None or _wrapped_flash_attn_forward is None:
    raise RuntimeError("Flash attention backend is not available.")

  window_size = (-1, -1)
  softcap = 0.0
  alibi_slopes = None
  deterministic = False
  grad_enabled = any(x.requires_grad for x in (query, key, value))

  if scale is None:
    scale = query.shape[-1] ** (-0.5)

  if grad_enabled:
    dropout_p = dropout_p if dropout_p > 0 else 1e-30

  with torch.set_grad_enabled(grad_enabled):
    out, lse, _, rng_state = _wrapped_flash_attn_forward(
      query,
      key,
      value,
      dropout_p,
      scale,
      is_causal,
      window_size[0],
      window_size[1],
      softcap,
      alibi_slopes,
      return_lse,
    )
    lse = lse.permute(0, 2, 1)

  if _save_ctx:
    ctx.save_for_backward(query, key, value, out, lse, rng_state)
    ctx.dropout_p = dropout_p
    ctx.scale = scale
    ctx.is_causal = is_causal
    ctx.window_size = window_size
    ctx.softcap = softcap
    ctx.alibi_slopes = alibi_slopes
    ctx.deterministic = deterministic

  return (out, lse) if return_lse else out


def _flash_attention_backward_op(
  ctx: torch.autograd.function.FunctionCtx,
  grad_out: torch.Tensor,
  *args,
  **kwargs,
):
  if _wrapped_flash_attn_backward is None:
    raise RuntimeError("Flash attention backend is not available.")

  query, key, value, out, lse, rng_state = ctx.saved_tensors
  grad_query = torch.empty_like(query)
  grad_key = torch.empty_like(key)
  grad_value = torch.empty_like(value)

  _wrapped_flash_attn_backward(
    grad_out,
    query,
    key,
    value,
    out,
    lse,
    grad_query,
    grad_key,
    grad_value,
    ctx.dropout_p,
    ctx.scale,
    ctx.is_causal,
    ctx.window_size[0],
    ctx.window_size[1],
    ctx.softcap,
    ctx.alibi_slopes,
    ctx.deterministic,
    rng_state,
  )

  grad_query = grad_query[..., :grad_out.shape[-1]]
  grad_key = grad_key[..., :grad_out.shape[-1]]
  grad_value = grad_value[..., :grad_out.shape[-1]]

  return grad_query, grad_key, grad_value


if _flash_attn_available:

  @_AttnBackendRegistry.register(
    _AttnBackend.FLASH,
    constraints=[],
    supports_context_parallel=True,
  )
  def _flash_attention(
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
    if attn_mask is not None:
      raise ValueError("`attn_mask` is not supported for flash-attn 2.")
    if enable_gqa:
      raise ValueError("`enable_gqa` is not yet supported for flash-attn 2.")

    if _cp_config is None:
      out = flash_attn_func(
        q=query,
        k=key,
        v=value,
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=is_causal,
        return_attn_probs=return_lse,
      )
      if return_lse:
        out, lse, *_ = out
    else:
      out = _context_parallel_attention(
        query,
        key,
        value,
        None,
        dropout_p,
        is_causal,
        scale,
        False,
        return_lse,
        forward_op=_flash_attention_forward_op,
        backward_op=_flash_attention_backward_op,
        _cp_config=_cp_config,
      )
      if return_lse:
        out, lse = out

    return (out, lse) if return_lse else out

else:
  _flash_attention = None  # type: ignore[assignment]


def _flash_attention_3_forward_op(
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
  if attn_mask is not None:
    raise ValueError("`attn_mask` is not yet supported for flash-attn 3.")
  if enable_gqa:
    raise ValueError("`enable_gqa` is not yet supported for flash-attn 3.")
  if dropout_p > 0.0:
    raise ValueError("`dropout_p` > 0 is not yet supported for flash-attn 3.")

  if scale is None:
    scale = query.shape[-1] ** (-0.5)

  if _save_ctx:
    pass

  window_size = (-1, -1)
  softcap = 0.0
  deterministic = False

  out = flash_attn_3_func(
    q=query,
    k=key,
    v=value,
    softmax_scale=scale,
    causal=is_causal,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=window_size,
    attention_chunk=0,
    softcap=softcap,
    num_splits=1,
    pack_gqa=None,
    deterministic=deterministic,
    sm_margin=0,
    return_attn_probs=return_lse,
  )
  if return_lse:
    out, lse = out
    lse = lse.permute(0, 2, 1)
    return out, lse
  else:
    return out


if _flash_attn_3_available:

  @_AttnBackendRegistry.register(
    _AttnBackend._FLASH_3,
    constraints=[],
    supports_context_parallel=True,
  )
  def _flash_attention_3(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = False,
    return_lse: bool = False,
    _cp_config: Optional["_ContextParallelConfig"] = None,
  ) -> torch.Tensor:
    lse = None
    if _cp_config is None:
      window_size = (-1, -1)
      softcap = 0.0
      deterministic = False
      out = flash_attn_3_func(
        q=query,
        k=key,
        v=value,
        softmax_scale=scale,
        causal=is_causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=window_size,
        attention_chunk=0,
        softcap=softcap,
        num_splits=1,
        pack_gqa=None,
        deterministic=deterministic,
        sm_margin=0,
        return_attn_probs=return_lse,
      )
      if return_lse:
        out, lse = out
        lse = lse.permute(0, 2, 1)
    else:
      out = _context_parallel_attention(
        query,
        key,
        value,
        None,
        0.0,
        is_causal,
        scale,
        False,
        return_lse,
        forward_op=_flash_attention_3_forward_op,
        backward_op=None,
        _cp_config=_cp_config,
      )
      if return_lse:
        out, lse = out

    return (out, lse) if return_lse else out

else:
  _flash_attention_3 = None  # type: ignore[assignment]
