"""Fused q/k norm + RoPE installed on the attention processor instances.

The attention call itself is *not* fused: it still goes to whichever backend
`dispatch_attention_fn` selects, with the processor's own `_attention_backend` and
`_parallel_config`. Only the two lines that normalize and rotate q/k change.

Why an instance-level processor swap instead of a class patch:

* a class-level `Flux3AttnProcessor.__call__` assignment would leak into every FLUX.3 user in
  the same process, and the operator switch is per model instance;
* context parallelism does *not* replace `__call__` for FLUX.3 - the Ulysses/ring all-to-all
  happens inside the attention backend (`attention/diffusers_bridge.py`), and cache-dit only
  annotates the processor **instance** with `_cp_config` / `_parallel_config`
  (`distributed/parallel/runtime.py`), which a subclass instance receives like any other.

Fallback is per call: :func:`apply_qk_norm_rope` re-checks dtype/layout/preconditions and
evaluates the literal upstream expression when any of them fails, so a patched call can never
produce a different value than the unpatched one.
"""

from typing import Optional

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux3 import (
  Flux3Attention,
  Flux3AttnProcessor,
  _get_qkv_projections,
)

from ....logger import init_logger
from .fused_ops import apply_qk_norm_rope

logger = init_logger(__name__)


class _Flux3FusedAttnProcessor(Flux3AttnProcessor):
  """`Flux3AttnProcessor` with the q/k RMSNorm and RoPE merged into one kernel.

  Everything else (the QKV projections, the attention backend dispatch, the output projection) is
  the stock code path.
  """

  def __call__(
    self,
    attn: Flux3Attention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
  ) -> torch.Tensor:
    """Run attention with fused per-head q/k norm and RoPE.

    :param attn: The `Flux3Attention` module owning the projections and norms.
    :param hidden_states: `(B, S, D)` block input.
    :param attention_mask: Optional attention mask, forwarded unchanged.
    :param image_rotary_emb: `(cos, sin)` fp32 tables, or `None` to skip the rotation.
    :returns: `(B, S, D)` attention output.
    """

    if torch.compiler.is_compiling():
      # Same reasoning as the block forwards: compiled graphs get the stock processor so the
      # fused pass cannot drift from stock while buying no speed.
      return super().__call__(attn, hidden_states, attention_mask, image_rotary_emb)

    query, key, value = _get_qkv_projections(attn, hidden_states)

    query = query.unflatten(-1, (attn.heads, -1))
    key = key.unflatten(-1, (attn.heads, -1))
    value = value.unflatten(-1, (attn.heads, -1))

    # Layerwise offload leaves `norm_q.weight` on the host between forwards; the module call is
    # what makes accelerate's pre-forward hook move it, so that case keeps the upstream path
    # instead of handing a host tensor to the kernel.
    fuse_qk = attn.norm_q.weight.is_cuda and attn.norm_k.weight.is_cuda

    if fuse_qk and image_rotary_emb is not None:
      cos, sin = image_rotary_emb
      query = apply_qk_norm_rope(query, attn.norm_q.weight, cos, sin, attn.norm_q.eps)
      key = apply_qk_norm_rope(key, attn.norm_k.weight, cos, sin, attn.norm_k.eps)
    else:
      query = attn.norm_q(query.float())
      key = attn.norm_k(key.float())
      if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
        key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
      query, key = query.to(value.dtype), key.to(value.dtype)

    hidden_states = dispatch_attention_fn(
      query,
      key,
      value,
      attn_mask=attention_mask,
      backend=self._attention_backend,
      parallel_config=self._parallel_config,
    )
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states.to(query.dtype)

    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)

    return hidden_states


def _swap_processors(transformer: torch.nn.Module, fused: bool) -> int:
  """Swap every stock FLUX.3 attention processor for the fused one, or back.

  Processors that are neither the stock class nor the fused class (a user-supplied flex or
  custom processor) are left alone and reported once.

  :param transformer: FLUX.3 transformer to walk.
  :param fused: `True` installs the fused processor, `False` restores the stock one.
  :returns: The number of processors that changed.
  """

  changed = 0
  skipped = 0
  for module in transformer.modules():
    if not isinstance(module, Flux3Attention):
      continue
    processor = module.processor
    if fused:
      if isinstance(processor, _Flux3FusedAttnProcessor):
        continue
      if not isinstance(processor, Flux3AttnProcessor):
        skipped += 1
        continue
      replacement = _Flux3FusedAttnProcessor()
    else:
      if not isinstance(processor, _Flux3FusedAttnProcessor):
        continue
      replacement = Flux3AttnProcessor()
    # Context parallelism annotates the processor instance; keep those annotations so the
    # swap does not silently drop the communication config.
    for attribute in ("_cp_config", "_parallel_config"):
      value = getattr(processor, attribute, None)
      if value is not None:
        setattr(replacement, attribute, value)
    module.set_processor(replacement)
    changed += 1

  if fused and skipped:
    logger.warning(f"{skipped} FLUX.3 attention processors are not the stock class and were "
                   "left untouched; the fused q/k norm + RoPE only covers stock processors.")
  return changed


def install_fused_processors(transformer: torch.nn.Module) -> int:
  """Install the fused attention processor on every stock FLUX.3 attention module.

  :param transformer: FLUX.3 transformer whose processors should be replaced.
  :returns: The number of processors replaced.
  """

  return _swap_processors(transformer, fused=True)


def restore_processors(transformer: torch.nn.Module) -> int:
  """Put the stock FLUX.3 attention processor back.

  :param transformer: FLUX.3 transformer whose processors should be restored.
  :returns: The number of processors restored.
  """

  return _swap_processors(transformer, fused=False)


def fused_processors_installed(transformer: torch.nn.Module) -> bool:
  """Return whether every FLUX.3 attention module runs the fused processor.

  :param transformer: FLUX.3 transformer to inspect.
  :returns: `True` when at least one attention module exists and all of them are fused.
  """

  modules = [m for m in transformer.modules() if isinstance(m, Flux3Attention)]
  if not modules:
    return False
  return all(isinstance(m.processor, _Flux3FusedAttnProcessor) for m in modules)


__all__ = [
  "install_fused_processors",
  "fused_processors_installed",
  "restore_processors",
]
