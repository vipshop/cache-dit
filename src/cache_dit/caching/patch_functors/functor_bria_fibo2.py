"""Patch BriaFibo2Transformer2DModel — stash each injection block's text bundle on the block.

Every injection layer of Fibo 2.0 reads a *different* slice of `encoder_hidden_states` inside the
block loop (`encoder_hidden_states[:, bundle_index[layer_id]]`). Once cache-dit folds the loop into a
single cached-blocks call, the per-layer slice is lost: the wrapper receives whatever the collapsed
call passes and reuses it for every block.

The patch keeps the upstream forward untouched apart from a thin prologue that distributes the
bundles, so the cached loop still feeds every block its own conditioning. This follows the
`BriaFiboPatchFunctor` pattern.
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
  # by the forward body, so only the injection bundles need to be stashed here.
  for layer_id, index in self._text_bundle_index.items():
    block = self._orig_layers[layer_id]
    block._text_bundle = encoder_hidden_states[:, index]
    block._text_mask = encoder_attention_mask
  return self._orig_forward(
    hidden_states,
    timestep,
    encoder_hidden_states,
    encoder_attention_mask=encoder_attention_mask,
    context_latents=context_latents,
    return_dict=return_dict,
  )


def _patched_block_forward(
  self: torch.nn.Module,
  x: torch.Tensor,
  attn_mask: torch.Tensor,
  freqs_cis: torch.Tensor,
  *args,
  **kwargs,
) -> torch.Tensor:
  # The block has 9 parameters. The loop passes the first three positionally and the rest as
  # keywords, whereas the gradient-checkpointing path passes all nine positionally
  # (`_gradient_checkpointing_func(block, x, attention_mask, freqs_cis, None, noise_mask, ...)`);
  # only rewrite the keyword form, which is the only one an inference run with cache takes.
  if self._text_bundle_index is not None and not args:
    kwargs["encoder_hidden_states"] = self._text_bundle
    kwargs["encoder_attention_mask"] = self._text_mask
  return self._orig_forward(x, attn_mask, freqs_cis, *args, **kwargs)
