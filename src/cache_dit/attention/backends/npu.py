import math
from typing import Optional

import torch

from .register import (
  _AttnBackend,
  _AttnBackendRegistry,
  _context_parallel_attention,
  _ContextParallelConfig,
)

try:
  from torch_npu import npu_fusion_attention
except Exception:
  npu_fusion_attention = None

try:
  from torch_npu import npu_fused_infer_attention_score
except Exception:
  npu_fused_infer_attention_score = None


def _maybe_modify_attn_mask_npu(query: torch.Tensor,
                                key: torch.Tensor,
                                attn_mask: Optional[torch.Tensor] = None):
  if attn_mask is not None and torch.all(attn_mask != 0):
    attn_mask = None

  if (attn_mask is not None and attn_mask.ndim == 2 and attn_mask.shape[0] == query.shape[0]
      and attn_mask.shape[1] == key.shape[1]):
    B, Sq, Skv = attn_mask.shape[0], query.shape[1], key.shape[1]
    attn_mask = ~attn_mask.to(torch.bool)
    attn_mask = attn_mask.unsqueeze(1).expand(B, Sq, Skv).unsqueeze(1).contiguous()

  return attn_mask


@_AttnBackendRegistry.register(
  _AttnBackend._NATIVE_NPU,
  constraints=[],
  supports_context_parallel=True,
)
def _native_npu_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  scale: Optional[float] = None,
  return_lse: bool = False,
  is_causal: bool = False,
  enable_gqa: bool = False,
  _cp_config: Optional["_ContextParallelConfig"] = None,
) -> torch.Tensor:
  if return_lse:
    raise ValueError("NPU attention backend does not support setting `return_lse=True`.")
  if _cp_config is None:
    attn_mask = _maybe_modify_attn_mask_npu(query, key, attn_mask)
    out = npu_fusion_attention(
      query,
      key,
      value,
      atten_mask=attn_mask,
      input_layout="BSND",
      scale=1.0 / math.sqrt(query.shape[-1]) if scale is None else scale,
      pre_tockens=2147483647,
      next_tockens=2147483647,
      head_num=query.size(2),
    )[0]
  else:
    out = _context_parallel_attention(
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      None,
      1.0 / math.sqrt(query.shape[-1]) if scale is None else scale,
      None,
      return_lse,
      forward_op=None,
      backward_op=None,
      _cp_config=_cp_config,
    )
  return out


@_AttnBackendRegistry.register(
  _AttnBackend._NPU_FIA,
  constraints=[],
  supports_context_parallel=True,
)
def _npu_fused_infer_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  scale: Optional[float] = None,
  return_lse: bool = False,
  is_causal: bool = False,
  enable_gqa: bool = False,
  _cp_config: Optional["_ContextParallelConfig"] = None,
) -> torch.Tensor:
  if _cp_config is None:
    attn_mask = _maybe_modify_attn_mask_npu(query, key, attn_mask)
    out = npu_fused_infer_attention_score(
      query,
      key,
      value,
      atten_mask=attn_mask,
      input_layout="BSND",
      scale=1.0 / math.sqrt(query.shape[-1]) if scale is None else scale,
      pre_tokens=2147483647,
      next_tokens=2147483647,
      num_heads=query.size(2),
    )
  else:
    out = _context_parallel_attention(
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      None,
      1.0 / math.sqrt(query.shape[-1]) if scale is None else scale,
      None,
      return_lse,
      forward_op=None,
      backward_op=None,
      _cp_config=_cp_config,
    )
  return out


try:
  from mindiesd.layers import attention_forward

  _mindiesd_available = True
except Exception:
  _mindiesd_available = False
  attention_forward = None


if _mindiesd_available:

  @_AttnBackendRegistry.register(
    _AttnBackend._MINDIESD_LASER,
    constraints=[],
    supports_context_parallel=True,
  )
  def _mindiesd_laser_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    return_lse: bool = False,
    is_causal: bool = False,
    enable_gqa: bool = False,
    _cp_config: Optional["_ContextParallelConfig"] = None,
  ) -> torch.Tensor:
    if return_lse:
      raise ValueError(
        "MindIE-SD laser attention backend does not support setting `return_lse=True`.")
    scale_val = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
    return attention_forward(
      query,
      key,
      value,
      attn_mask=attn_mask,
      scale=scale_val,
      fused=True,
      head_first=False,
      opt_mode="manual",
      op_type="ascend_laser_attention",
    )
