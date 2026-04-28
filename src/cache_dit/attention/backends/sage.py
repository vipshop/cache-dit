from typing import Optional

import torch

from .register import (
  _AttnBackend,
  _AttnBackendRegistry,
  _context_parallel_attention,
  _ContextParallelConfig,
)

try:
  from sageattention import sageattn
except Exception:
  sageattn = None


def _sage_attention_forward_op(
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
    raise ValueError("`attn_mask` is not yet supported for Sage attention.")
  if dropout_p > 0.0:
    raise ValueError("`dropout_p` is not yet supported for Sage attention.")
  if enable_gqa:
    raise ValueError("`enable_gqa` is not yet supported for Sage attention.")
  if sageattn is None:
    raise RuntimeError(
      "Sage attention backend is not available. Please install `sageattention` to use it.")

  out = sageattn(
    q=query,
    k=key,
    v=value,
    tensor_layout="NHD",
    is_causal=is_causal,
    sm_scale=scale,
    return_lse=return_lse,
  )
  lse = None
  if return_lse:
    out, lse, *_ = out
    lse = lse.permute(0, 2, 1)

  return (out, lse) if return_lse else out


def _sage_attention_backward_op(
  ctx: torch.autograd.function.FunctionCtx,
  grad_out: torch.Tensor,
  *args,
):
  raise NotImplementedError("Backward pass is not implemented for Sage attention.")


@_AttnBackendRegistry.register(
  _AttnBackend.SAGE,
  constraints=[],
  supports_context_parallel=True,
)
def _sage_attention(
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
  if attn_mask is not None:
    raise ValueError("`attn_mask` is not yet supported for Sage attention.")
  if dropout_p > 0.0:
    raise ValueError("`dropout_p` is not yet supported for Sage attention.")
  if enable_gqa:
    raise ValueError("`enable_gqa` is not yet supported for Sage attention.")

  lse = None
  if _cp_config is None:
    if sageattn is None:
      raise RuntimeError(
        "Sage attention backend is not available. Please install `sageattention` to use it.")
    out = sageattn(
      q=query,
      k=key,
      v=value,
      tensor_layout="NHD",
      is_causal=is_causal,
      sm_scale=scale,
      return_lse=return_lse,
    )
    if return_lse:
      out, lse, *_ = out
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
      forward_op=_sage_attention_forward_op,
      backward_op=_sage_attention_backward_op,
      _cp_config=_cp_config,
    )
    if return_lse:
      out, lse = out

  return (out, lse) if return_lse else out
