"""Patch BriaFibo2Transformer2DModel — stash each injection block's text bundle on the block.

Every injection layer of Fibo 2.0 reads a *different* slice of `encoder_hidden_states` inside the
block loop (`encoder_hidden_states[:, bundle_index[layer_id]]`). Once cache-dit folds the loop into a
single cached-blocks call, the per-layer slice is lost: the wrapper receives whatever the collapsed
call passes and reuses it for every block.

The patch keeps the upstream forward untouched apart from a thin prologue that distributes the
bundles, so the cached loop still feeds every block its own conditioning. Blocks only switch to the
stash while the loop is collapsed: an intact loop already hands each block its own bundle, and
overriding there would make a gradient-checkpointing recompute read the stash of a later forward.
This follows the `BriaFiboPatchFunctor` pattern.
"""

import torch
from typing import Optional

from .functor_base import PatchFunctor
from ...logger import init_logger

logger = init_logger(__name__)


class BriaFibo2PatchFunctor(PatchFunctor):
  """Distribute Fibo 2.0's per-injection-layer text bundles before the cached block loop."""

  def _apply(
    self,
    transformer: torch.nn.Module,
    **kwargs,
  ) -> torch.nn.Module:
    from diffusers.models.transformers.transformer_bria_fibo2 import BriaFibo2Transformer2DModel

    assert isinstance(transformer, BriaFibo2Transformer2DModel)

    # Snapshot the real layers: under cache `transformer.layers` is replaced by a single wrapper
    # module for the duration of each forward, so the stash pass must not walk `self.layers`.
    orig_layers = list(transformer.layers)
    bundle_index = {
      layer_id: index + 1
      for index, layer_id in enumerate(transformer.config.injection_layer_ids)
    }

    for layer_id, block in enumerate(orig_layers):
      block._orig_forward = block.forward
      block._text_bundle = None
      block._text_bundle_index = bundle_index.get(layer_id, None)
      block._text_mask = None
      block._use_text_bundle = False
      block.forward = _patched_block_forward.__get__(block)

    transformer._orig_layers = orig_layers
    transformer._text_bundle_index = bundle_index
    transformer._orig_forward = transformer.forward
    transformer.forward = _patched_transformer_forward.__get__(transformer)
    if hasattr(transformer, "_old_forward"):
      # accelerate's offload hook dispatches through `_old_forward`, which was captured before this
      # patch; leaving it stale would run the upstream forward and lose the stash pass.
      transformer._old_forward = transformer.forward
    transformer._is_patched = True
    return transformer


def _patched_transformer_forward(
  self: torch.nn.Module,
  hidden_states: torch.Tensor,
  timestep: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  encoder_attention_mask: Optional[torch.Tensor] = None,
  context_latents: Optional[list[torch.Tensor]] = None,
  return_dict: bool = True,
):
  # Bundle 0 feeds the Perceiver, bundle `k + 1` feeds `injection_layer_ids[k]`; both are consumed
  # by the forward body, so only the injection bundles need to be stashed here. The stash only has
  # to override the caller's text when the loop is collapsed: otherwise the upstream loop already
  # hands every block its own bundle, and rewriting would make a gradient-checkpointing recompute
  # read whatever the *last* forward stashed instead of the values captured for this graph.
  collapsed = len(self.layers) != len(self._orig_layers)
  for layer_id, index in self._text_bundle_index.items():
    block = self._orig_layers[layer_id]
    block._text_bundle = encoder_hidden_states[:, index]
    block._text_mask = encoder_attention_mask
    block._use_text_bundle = collapsed
  return self._orig_forward(
    hidden_states,
    timestep,
    encoder_hidden_states,
    encoder_attention_mask=encoder_attention_mask,
    context_latents=context_latents,
    return_dict=return_dict,
  )


# Tail of `BriaFibo2TransformerBlock.forward` in declaration order (the three leading parameters
# `x` / `attn_mask` / `freqs_cis` are always passed positionally).
_FORWARD_TAIL_ARGS = (
  "adaln_input",
  "noise_mask",
  "adaln_noisy",
  "adaln_clean",
  "encoder_hidden_states",
  "encoder_attention_mask",
)


def _patched_block_forward(
  self: torch.nn.Module,
  x: torch.Tensor,
  attn_mask: Optional[torch.Tensor],
  freqs_cis: torch.Tensor,
  *args,
  **kwargs,
) -> torch.Tensor:
  # The loop passes the first three parameters positionally and the rest as keywords, while the
  # gradient-checkpointing branch passes all nine positionally. Normalize the tail into keywords
  # first: under cache the collapsed loop only ever indexes layer 0, so a positional
  # `encoder_hidden_states` belongs to the wrong block (usually `None`).
  if args:
    if len(args) > len(_FORWARD_TAIL_ARGS):
      raise TypeError(f"BriaFibo2TransformerBlock got {len(args)} positional arguments after "
                      f"`freqs_cis`, expected at most {len(_FORWARD_TAIL_ARGS)}.")
    kwargs = {**dict(zip(_FORWARD_TAIL_ARGS, args)), **kwargs}
  if self._use_text_bundle:
    kwargs["encoder_hidden_states"] = self._text_bundle
    kwargs["encoder_attention_mask"] = self._text_mask
  return self._orig_forward(x, attn_mask, freqs_cis, **kwargs)
