"""Fused q/k norm + RoPE, installed at the one seam every processor goes through.

The attention call itself is *not* fused: it still goes to whichever backend
:func:`cache_dit.attention._dispatch_attention_fn` selects, and the processor classes are left
alone.

diffusers routes both `QwenImage21AttnProcessor` and `QwenImage21FlexAttnProcessor` through the
module-level `_qwenimage21_prepare_qkv`, and cache-dit's context-parallel `cp_call` imports and
calls that same function. Patching the module global is therefore the only seam that composes
with every mode:

* plain eager / compiled blocks -> fused,
* flex attention -> fused (its `__call__` is untouched, so its BlockMask still drives attention),
* context parallelism -> fused (its `__call__` is untouched, so the U/All-to-All sharding and the
  block-causal segments still work),
* tensor parallelism -> fused (the fused op still splits by `inner_dim // heads`).

Subclassing the processor was the alternative and it broke CP: cache-dit replaces
`QwenImage21AttnProcessor.__call__` wholesale, so a subclass `__call__` either bypassed the
sharding or had to delegate to it - and delegating means no fusion at all under CP.

Fallback is per call: `apply_qk_norm_rope` re-checks dtype/layout/preconditions and evaluates
the literal upstream expression when any of them fails, so a patched call can never produce a
different value than the unpatched one.
"""

from typing import Any, Optional

import torch
from diffusers.models.transformers.transformer_qwenimage21 import (
  QwenImage21KVLayerCache, )

from .fused_ops import apply_qk_norm_rope

_PATCHED = False
_ORIGINALS: dict[str, Any] = {}
_TARGETS = ("diffusers.models.transformers.transformer_qwenimage21",
            "cache_dit.distributed.transformers.qwen_image21")


def _fused_prepare_qkv(
  attn: torch.nn.Module,
  hidden_states: torch.Tensor,
  rotary_emb: Optional[torch.Tensor],
  layer_cache: Optional[QwenImage21KVLayerCache],
  kv_cache_mode: Optional[str],
  cache_write_slice: Optional[slice],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
  """QKV projection, fused per-head norm + RoPE, and KV-cache bookkeeping.

  Same contract as the diffusers function it replaces.

  :param attn: The `QwenImage21Attention` module owning the projections and norms.
  :param hidden_states: `(B, S, D)` block input.
  :param rotary_emb: `(S, D // 2)` complex64 table, or `None` to skip the rotation.
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

  # Upstream rounds to `value.dtype` between the norm and the rotation; the fused kernel does
  # norm and rotation in one pass over bf16, which is the same thing only when the dtypes
  # already agree. Keep the upstream order otherwise.
  if rotary_emb is not None and query.dtype == value.dtype:
    query = apply_qk_norm_rope(query, attn.norm_q.weight, rotary_emb, attn.norm_q.eps)
    key = apply_qk_norm_rope(key, attn.norm_k.weight, rotary_emb, attn.norm_k.eps)
  else:
    query = attn.norm_q(query).to(value.dtype)
    key = attn.norm_k(key).to(value.dtype)
    if rotary_emb is not None:
      from diffusers.models.transformers.transformer_qwenimage21 import apply_rotary_emb_qwen

      query = apply_rotary_emb_qwen(query, rotary_emb, use_real=False)
      key = apply_rotary_emb_qwen(key, rotary_emb, use_real=False)

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


def install_qkv_patch() -> None:
  """Install :func:`_fused_prepare_qkv` on every module that calls the original.

  Idempotent.
  """

  global _PATCHED
  if _PATCHED:
    return

  import importlib

  for name in _TARGETS:
    try:
      module = importlib.import_module(name)
    except ImportError:
      # Cache-dit's parallel package is optional; the diffusers module alone already covers
      # the non-context-parallel paths.
      continue
    original = getattr(module, "_qwenimage21_prepare_qkv", None)
    if original is None:
      continue
    _ORIGINALS[name] = original
    module._qwenimage21_prepare_qkv = _fused_prepare_qkv
  _PATCHED = True


def uninstall_qkv_patch() -> None:
  """Restore the original functions.

  Used by tests and by the operator switch.
  """

  global _PATCHED
  import importlib

  for name, original in _ORIGINALS.items():
    module = importlib.import_module(name)
    module._qwenimage21_prepare_qkv = original
  _ORIGINALS.clear()
  _PATCHED = False


def qkv_patch_is_installed() -> bool:
  """Return whether the fused `prepare_qkv` is currently installed.

  :returns: `True` when at least one module still points at the fused function.
  """

  import importlib

  return any(
    getattr(importlib.import_module(name), "_qwenimage21_prepare_qkv", None) is _fused_prepare_qkv
    for name in _TARGETS)


__all__ = [
  "install_qkv_patch",
  "qkv_patch_is_installed",
  "uninstall_qkv_patch",
]
