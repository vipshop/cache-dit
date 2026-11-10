"""Attention processor that runs the q/k norm and rotary embedding on the fused kernel.

The attention call itself is *not* fused: it still goes to whichever backend
:func:`cache_dit.attention._dispatch_attention_fn` selects. Only the projection epilogue
(`norm_q` / `norm_k` + RoPE) is replaced.


`_qwenimage21_prepare_qkv` is a module-level function, so a subclass cannot intercept it by
overriding a method: the whole `__call__` has to be rewritten. The rewrite is deliberately
literal — every attention argument (`attention_mask`, `rotary_emb`, `layer_cache`,
`kv_cache_mode`, `cache_write_slice`, `segments`, `key_valid`) and every trailing step
(`to_out[0]`, `to_out[1]`, the sequence trim and the `flatten(2, 3).type_as(query)` cast) is
copied unchanged, so the attention call sees exactly the tensors the stock pipeline gives it.

The attention call itself goes through cache-dit's
:func:`cache_dit.attention._dispatch_attention_fn`, not diffusers', so `--attn` /
`attention_backend` keeps choosing the kernel: the fused path must not silently fall back to
diffusers' native SDPA. When a caller pinned a *diffusers* backend on the processor instead,
the whole call is handed back to the stock processor rather than ignoring that choice.

This is the one place where a fused kernel can beat `torch.compile` on this model:
inductor explicitly refuses to codegen complex64, so the eager chain materializes a complex
intermediate per head per layer.
"""

from typing import Any, Optional

import torch
from diffusers.models.transformers.transformer_qwenimage21 import (
  QwenImage21AttnProcessor,
  QwenImage21KVLayerCache,
)

from ....attention import _dispatch_attention_fn
from .fused_ops import apply_qk_norm_rope


def _fused_prepare_qkv(
  attn: torch.nn.Module,
  hidden_states: torch.Tensor,
  rotary_emb: Optional[torch.Tensor],
  layer_cache: Optional[QwenImage21KVLayerCache],
  kv_cache_mode: Optional[str],
  cache_write_slice: Optional[slice],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
  """QKV projection, fused per-head norm + RoPE, and KV-cache bookkeeping.

  :param attn: The `QwenImage21Attention` module owning the projections and norms.
  :param hidden_states: `(B, S, D)` block input.
  :param rotary_emb: `(S, D // 2)` complex64 table.
  :param layer_cache: Per-layer prefix KV cache, or `None`.
  :param kv_cache_mode: `"extract"`, `"cached"` or `None`.
  :param cache_write_slice: Prefix slice to store on the extract step.
  :returns: `(query, key, value, seq_len_q)`.
  """

  query = attn.to_q(hidden_states)
  key = attn.to_k(hidden_states)
  value = attn.to_v(hidden_states)

  # Split by `head_dim` rather than by `attn.heads`: under tensor parallelism each rank holds
  # only its share of the heads while `heads`/`inner_dim` keep their full values.
  head_dim = attn.inner_dim // attn.heads
  query = query.unflatten(-1, (-1, head_dim))
  key = key.unflatten(-1, (-1, head_dim))
  value = value.unflatten(-1, (-1, head_dim))

  query = apply_qk_norm_rope(query, attn.norm_q.weight, rotary_emb, attn.norm_q.eps)
  key = apply_qk_norm_rope(key, attn.norm_k.weight, rotary_emb, attn.norm_k.eps)
  query = query.to(value.dtype)
  key = key.to(value.dtype)

  if layer_cache is not None:
    if kv_cache_mode == "extract" and cache_write_slice is not None:
      # `clone()`, not `contiguous()`: at batch size 1 the prefix slice already counts as
      # contiguous, so `contiguous()` would return the same view and pin the whole prefill
      # K/V for every step of the denoising loop.
      layer_cache.store(
        key[:, cache_write_slice].clone(),
        value[:, cache_write_slice].clone(),
      )
    elif kv_cache_mode == "cached":
      cached_k, cached_v = layer_cache.get()
      key = torch.cat([cached_k, key], dim=1)
      value = torch.cat([cached_v, value], dim=1)

  return query, key, value, query.shape[1]


class FusedQwenImage21AttnProcessor(QwenImage21AttnProcessor):
  """Stock processor with the fused q/k norm + RoPE in front of the same attention call."""

  def __call__(
    self,
    attn: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Any | None = None,
    rotary_emb: Optional[torch.Tensor] = None,
    layer_cache: Optional[QwenImage21KVLayerCache] = None,
    kv_cache_mode: Optional[str] = None,
    cache_write_slice: Optional[slice] = None,
    segments: Optional[list[tuple[int, int, bool]]] = None,
    key_valid: Optional[torch.Tensor] = None,
    cp_meta: Optional[tuple] = None,
  ) -> torch.Tensor:
    if cp_meta is not None or self._attention_backend is not None:
      # Context parallelism replaces `QwenImage21AttnProcessor.__call__` wholesale (it
      # redistributes Q/K/V and shards the prefix by head), and a diffusers backend pinned on
      # the processor only makes sense for diffusers' dispatcher, so both cases are handed
      # back instead of being bypassed.
      return super().__call__(
        attn,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        rotary_emb=rotary_emb,
        layer_cache=layer_cache,
        kv_cache_mode=kv_cache_mode,
        cache_write_slice=cache_write_slice,
        segments=segments,
        key_valid=key_valid,
        cp_meta=cp_meta,
      )

    query, key, value, seq_len_q = _fused_prepare_qkv(attn, hidden_states, rotary_emb, layer_cache,
                                                      kv_cache_mode, cache_write_slice)

    if segments is None:
      # decode: full attention over [cached prefix, target]
      hidden_states = _dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
      )
    else:
      # prefill: every segment attends to the keys `[0, end)`; text segments additionally get
      # a causal triangle over their own keys; padded text keys are dropped.
      prefix_len = segments[-1][1] if segments else 0
      outputs = []
      for start, end, is_text in segments:
        seg_mask = None
        if is_text:
          seg_len = end - start
          seg_mask = torch.cat(
            [
              torch.ones(seg_len, start, dtype=torch.bool, device=query.device),
              torch.tril(torch.ones(seg_len, seg_len, dtype=torch.bool, device=query.device)),
            ],
            dim=1,
          )[None, None]
        if key_valid is not None:
          seg_key_valid = key_valid[:, None, None, :end]
          seg_mask = seg_key_valid if seg_mask is None else (seg_mask & seg_key_valid)
        outputs.append(
          _dispatch_attention_fn(
            query[:, start:end],
            key[:, :end],
            value[:, :end],
            attn_mask=seg_mask,
            dropout_p=0.0,
          ))
      outputs.append(
        _dispatch_attention_fn(
          query[:, prefix_len:],
          key,
          value,
          attn_mask=None if key_valid is None else key_valid[:, None, None, :],
          dropout_p=0.0,
        ))
      hidden_states = torch.cat(outputs, dim=1)

    hidden_states = hidden_states[:, :seq_len_q]
    hidden_states = hidden_states.flatten(2, 3).type_as(query)

    hidden_states = attn.to_out[0](hidden_states)
    return attn.to_out[1](hidden_states)


__all__ = ["FusedQwenImage21AttnProcessor"]
