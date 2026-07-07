from typing import Optional

import torch
import torch.nn.functional as F

from .register import (
  _AttnBackend,
  _AttnBackendRegistry,
  _ContextParallelConfig,
  _context_parallel_attention,
)

try:
  from flash_attn import flash_attn_varlen_func
  from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
  _flash_varlen_available = True
except Exception:
  flash_attn_varlen_func = None
  index_first_axis = None
  pad_input = None
  unpad_input = None
  _flash_varlen_available = False


def _mask_to_2d(attn_mask: torch.Tensor) -> torch.Tensor:
  if attn_mask.dim() == 2:
    return attn_mask.to(torch.bool)
  if attn_mask.dim() != 3:
    raise ValueError(
      f"flash_varlen only supports 2D or 3D attention masks, got {attn_mask.dim()}D.")

  batch_size, sequence_length, _ = attn_mask.shape
  lengths = torch.diagonal(attn_mask, dim1=-2, dim2=-1).sum(dim=-1, dtype=torch.int32)
  mask_2d = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=attn_mask.device)
  for i in range(batch_size):
    length = int(lengths[i].item())
    if length > 0:
      mask_2d[i, :length] = True
  return mask_2d


def _get_unpad_data(mask_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
  seqlens_in_batch = mask_2d.sum(dim=-1, dtype=torch.int32)
  indices = torch.nonzero(mask_2d.flatten(), as_tuple=False).flatten()
  max_seqlen_in_batch = seqlens_in_batch.max().item()
  cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
  return indices, cu_seqlens, max_seqlen_in_batch


def _flash_varlen_attention_impl(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
) -> torch.Tensor:
  if flash_attn_varlen_func is None:
    raise RuntimeError("flash_varlen attention backend is not available.")

  batch_size, query_length, num_q_heads, head_dim = query.shape
  key_length = key.shape[1]
  num_kv_heads = key.shape[2]

  if attn_mask is None:
    query_states = query.reshape(batch_size * query_length, num_q_heads, head_dim)
    key_states = key.reshape(batch_size * key_length, num_kv_heads, head_dim)
    value_states = value.reshape(batch_size * key_length, num_kv_heads, head_dim)
    cu_seqlens_q = torch.arange(
      0,
      batch_size * query_length + 1,
      query_length,
      device=query.device,
      dtype=torch.int32,
    )
    cu_seqlens_k = torch.arange(
      0,
      batch_size * key_length + 1,
      key_length,
      device=query.device,
      dtype=torch.int32,
    )
    max_seqlen_q = query_length
    max_seqlen_k = key_length
    indices_q = None
  else:
    if index_first_axis is None or pad_input is None or unpad_input is None:
      raise RuntimeError("flash_attn bert_padding helpers are required for masked flash_varlen.")
    mask_2d = _mask_to_2d(attn_mask)
    indices_k, cu_seqlens_k, max_seqlen_k = _get_unpad_data(mask_2d)
    key_states = index_first_axis(key.reshape(batch_size * key_length, num_kv_heads, head_dim),
                                  indices_k)
    value_states = index_first_axis(value.reshape(batch_size * key_length, num_kv_heads, head_dim),
                                    indices_k)

    if query_length == key_length:
      query_states = index_first_axis(query.reshape(batch_size * key_length, num_q_heads, head_dim),
                                      indices_k)
      cu_seqlens_q = cu_seqlens_k
      max_seqlen_q = max_seqlen_k
      indices_q = indices_k
    elif query_length == 1:
      query_states = query.squeeze(1)
      cu_seqlens_q = torch.arange(batch_size + 1, device=query.device, dtype=torch.int32)
      max_seqlen_q = 1
      indices_q = cu_seqlens_q[:-1]
    else:
      query_mask = mask_2d[:, -query_length:]
      query_states, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(query, query_mask)

  out = flash_attn_varlen_func(
    query_states,
    key_states,
    value_states,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    dropout_p=dropout_p,
    softmax_scale=scale,
    causal=is_causal,
  )
  if indices_q is None:
    return out.view(batch_size, query_length, num_q_heads, head_dim)
  return pad_input(out, indices_q, batch_size, query_length)


def _flash_varlen_attention_forward_op(
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
    raise ValueError("flash_varlen attention does not support return_lse=True.")
  return _flash_varlen_attention_impl(
    query,
    key,
    value,
    attn_mask=attn_mask,
    dropout_p=dropout_p,
    is_causal=is_causal,
    scale=scale,
  )


def _flash_varlen_attention_backward_op(
  ctx: torch.autograd.function.FunctionCtx,
  grad_out: torch.Tensor,
  *args,
  **kwargs,
):
  raise NotImplementedError("Backward for flash_varlen attention is not implemented yet.")


if _flash_varlen_available:

  @_AttnBackendRegistry.register(
    _AttnBackend.FLASH_VARLEN,
    constraints=[],
    supports_context_parallel=True,
  )
  def _flash_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    cp_gqa_strategy: Optional[str] = None,
    _cp_config: Optional["_ContextParallelConfig"] = None,
  ) -> torch.Tensor:
    if return_lse:
      raise ValueError("flash_varlen attention does not support return_lse=True.")
    if _cp_config is None:
      return _flash_varlen_attention_impl(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
      )
    return _context_parallel_attention(
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      enable_gqa,
      return_lse,
      cp_gqa_strategy,
      forward_op=_flash_varlen_attention_forward_op,
      backward_op=_flash_varlen_attention_backward_op,
      _cp_config=_cp_config,
    )

else:
  _flash_varlen_attention = None  # type: ignore[assignment]
