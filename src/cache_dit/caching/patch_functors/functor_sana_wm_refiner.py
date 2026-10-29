"""PatchFunctor for the SANA-WM stage-2 refiner (`SanaWMLTX2RefinerTransformer3DModel`).

The refiner is a chunk-causal autoregressive LTX-2 DiT: `SanaWMLTX2Refiner.__call__` walks the
latents in `block_size`-frame chunks and runs, per chunk, one `kv_cache_mode="inject"` denoising
forward per sigma step (three with the distilled 3-step schedule; generally `len(sigmas) - 1`)
that only *reads* the sliding-window K/V, plus one `inject_and_capture_post_rope` pass that
*writes* the refined K/V into the cache. A separate `capture_pre_rope` pass seeds the sink once at
the very start.

Block-level caching is legal only for the `inject` forwards: skipping a block during a capture
pass would leave that layer's K/V entry missing, and the read-back in
`SanaWMRefinerKVCache.get_captured_*` raises. Three things therefore have to be fixed up before
the generic cached wrapper can drive this model:

1. **KV layer indexing.** Upstream calls `block(..., kv_cache=kv_cache.get(i))` with the layer
   index from the block loop. After `enable_cache` the loop runs once (the block list holds a
   single wrapper), so `i` is always 0 and every layer would read/write layer 0's cache. The
   patched forward hands the *whole* cache to the wrapper, and each block resolves its own slice
   through an instance-level bind.
2. **KV-writing passes must not be cached.** They replay the original blocks directly, which also
   keeps the forward bit-exact against upstream.
3. **Chunk boundaries.** Every chunk re-noises different frames at a different sink offset, so the
   residual of the previous chunk is meaningless for the next one. The cache context is refreshed
   with `force_refresh_step_hint` right before the first `inject` forward of each chunk, which
   clears the stored residuals and restarts the step counters.

`patch_functors/functor_ltx2.py` solves (2) the same way for the LTX-2 pipelines, and its
`off_cache` replay is the template used here.
"""

import types

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_sana_wm_refiner import SanaWMRefinerKVCache
from diffusers.utils import apply_lora_scale

from .functor_base import PatchFunctor
from ...logger import init_logger

logger = init_logger(__name__)


def _is_shared_kv_cache(kv_cache) -> bool:
  """Whether `kv_cache` is the whole per-layer container (rather than one layer's slice)."""
  return isinstance(kv_cache, SanaWMRefinerKVCache)


def _bind_layer_kv_cache(block, layer_idx: int):
  """Bind `block.forward` so a whole-cache `kv_cache` argument resolves to this layer's slice."""
  original_forward = block.forward

  def forward(self, hidden_states, *args, kv_cache=None, **kwargs):
    if _is_shared_kv_cache(kv_cache):
      kv_cache = kv_cache.get(layer_idx)
    return original_forward(hidden_states, *args, kv_cache=kv_cache, **kwargs)

  block.forward = types.MethodType(forward, block)
  block._sana_wm_layer_idx = layer_idx


def _live_cache_context(wrapper):
  """Resolve the cached context the wrapper will activate on its next call.

  `UnifiedBlocks.cache_context` is the context *name*, not the context object; the live
  `CachedContext` lives in the shared `ContextManager`. Mutating the config on the object returned
  here is what `set_context` (the wrapper's first statement) will read in `maybe_refresh`.
  """
  context_manager = getattr(wrapper, "context_manager", None)
  context_name = getattr(wrapper, "cache_context", None)
  if context_manager is None or not isinstance(context_name, str):
    return None
  try:
    return context_manager.get_context(context_name)
  except Exception:  # noqa: BLE001 - the context may not exist yet on the very first call
    return None


def _refresh_cache_context_at_chunk_start(transformer, wrapper, kv_cache_mode) -> None:
  """Clear the cached residuals when a new autoregressive chunk starts.

  The refiner reports a chunk boundary as the transition from a KV-writing pass (or the very
  beginning of the run) back to `inject`. `force_refresh_step_hint` is the supported way to make
  the wrapper rebuild its context: `set_context` refreshes when the current transformer step
  equals `hint - 1`, and that check runs before this forward advances the counter.
  """
  previous_mode = getattr(transformer, "_sana_wm_last_kv_cache_mode", None)
  transformer._sana_wm_last_kv_cache_mode = kv_cache_mode
  if kv_cache_mode != "inject" or previous_mode == "inject":
    return

  cache_context = _live_cache_context(wrapper)
  cache_config = getattr(cache_context, "cache_config", None)
  if cache_config is None:
    return
  if cache_config.num_inference_steps is not None:
    # `maybe_refresh` also refreshes once the executed-step counter passes
    # `num_inference_steps`, which would fire in the middle of a chunk (the counter never sees
    # the KV-writing passes). The chunk-boundary refresh below is the only one this model wants.
    logger.warning_once(
      "SANA-WM refiner: `num_inference_steps` is set in the cache config, which refreshes the "
      "context by forward count and can split an autoregressive chunk. Leave it unset and let "
      "the chunk boundary drive the refresh.")
  current_step = cache_context.get_current_transformer_step()
  cache_config.force_refresh_step_hint = current_step + 1
  logger.info_once("SANA-WM refiner: new autoregressive chunk, refreshing the cache context "
                   f"(step {current_step} -> full compute for this chunk's first step).")


@apply_lora_scale("attention_kwargs")
def _patched_forward(
  self,
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  timestep: torch.Tensor,
  video_rotary_emb,
  encoder_attention_mask: torch.Tensor | None = None,
  kv_cache=None,
  kv_cache_mode: str | None = None,
  attention_kwargs: dict | None = None,
  return_dict: bool = True,
):
  """Upstream `SanaWMLTX2RefinerTransformer3DModel.forward` with a cache-aware block loop.

  The `@apply_lora_scale("attention_kwargs")` wrapper is re-applied here because replacing
  `forward` drops the one upstream installed, which would silently ignore adapter scales.
  """
  raw_blocks = getattr(self, "_sana_wm_raw_blocks", None)
  blocks = self.transformer_blocks
  # After `enable_cache`, `transformer_blocks` holds one wrapper that iterates the originals.
  wrapped = len(blocks) == 1 and hasattr(blocks[0], "transformer_blocks")
  if raw_blocks is None:
    raw_blocks = blocks[0].transformer_blocks if wrapped else blocks

  if kv_cache_mode is not None:
    if kv_cache_mode not in ("inject", "capture_pre_rope", "inject_and_capture_post_rope"):
      raise ValueError("`kv_cache_mode` must be one of 'inject', 'capture_pre_rope', "
                       f"'inject_and_capture_post_rope' or `None`, got {kv_cache_mode!r}.")
    if kv_cache is None:
      raise ValueError(f"`kv_cache_mode={kv_cache_mode!r}` requires a `SanaWMRefinerKVCache`.")
    # Upstream compares against `len(self.transformer_blocks)`; the collapsed loop would make that
    # 1 and reject every real cache.
    if len(kv_cache) != len(raw_blocks):
      raise ValueError(
        f"`kv_cache` holds {len(kv_cache)} layer caches but the model has {len(raw_blocks)} "
        "transformer blocks.")

  batch_size = hidden_states.size(0)

  if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
    encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
    encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

  hidden_states = self.proj_in(hidden_states)

  temb, embedded_timestep = self.time_embed(
    timestep.flatten(),
    batch_size=batch_size,
    hidden_dtype=hidden_states.dtype,
  )
  temb = temb.view(batch_size, -1, temb.size(-1))
  embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

  encoder_hidden_states = self.caption_projection(encoder_hidden_states)
  encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

  sink_rotary_emb = kv_cache.sink_pe if kv_cache is not None else None

  # A KV-writing pass has to run every block: it records the sliding-window K/V that later chunks
  # attend to, and the read-back raises if a layer was skipped. `inject` only *reads* the cache,
  # so those are the forwards block-level caching may skip.
  off_cache = kv_cache_mode is not None and kv_cache_mode != "inject"
  if kv_cache_mode == "inject" and wrapped:
    _refresh_cache_context_at_chunk_start(self, blocks[0], kv_cache_mode)

  block_kwargs = dict(
    encoder_hidden_states=encoder_hidden_states,
    temb=temb,
    video_rotary_emb=video_rotary_emb,
    encoder_attention_mask=encoder_attention_mask,
    sink_rotary_emb=sink_rotary_emb,
    kv_cache_mode=kv_cache_mode,
  )
  if off_cache or not wrapped:
    for layer_idx, block in enumerate(raw_blocks):
      hidden_states = block(
        hidden_states=hidden_states,
        kv_cache=kv_cache.get(layer_idx) if kv_cache is not None else None,
        **block_kwargs,
      )
  else:
    # The wrapper hands the whole cache to every block; each block resolves its own layer slice.
    hidden_states = blocks[0](
      hidden_states=hidden_states,
      kv_cache=kv_cache,
      **block_kwargs,
    )

  scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
  shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
  hidden_states = self.norm_out(hidden_states)
  hidden_states = hidden_states * (1 + scale) + shift
  output = self.proj_out(hidden_states)

  if not return_dict:
    return (output, )
  return Transformer2DModelOutput(sample=output)


class SanaWMRefinerPatchFunctor(PatchFunctor):
  """Make the SANA-WM stage-2 refiner compatible with block-level caching.

  Patches are instance-level (each block's `forward`, the transformer's `forward`) so the class
  trees shared with the LTX-2 pipelines (`SanaWMLTX2Attention`, the perturbed / audio-video
  processors) are untouched.
  """

  def _apply(
    self,
    transformer: torch.nn.Module,
    **kwargs,
  ) -> torch.nn.Module:
    if getattr(transformer, "_is_patched", False):
      return transformer

    blocks = transformer.transformer_blocks
    # A plain list, not the `ModuleList`: assigning a module container would register the same
    # blocks twice (`state_dict` collisions, doubled `named_modules`/offload traversal).
    object.__setattr__(transformer, "_sana_wm_raw_blocks", list(blocks))
    for layer_idx, block in enumerate(blocks):
      _bind_layer_kv_cache(block, layer_idx)

    transformer.forward = types.MethodType(_patched_forward, transformer)
    transformer._is_patched = True
    return transformer
