"""Cache wiring guards for a modular pipeline that inspects `transformer.forward`.

Pika H3 Flex's denoise block builds the transformer kwargs by filtering its packed-layout fields
against `inspect.signature(transformer.forward).parameters` (`vendored denoise.py`):

```python
layout_kwargs = {
  name: value for name, value in block_state.denoiser_input_fields.items()
  if name in inspect.signature(transformer.forward).parameters
}
```

`enable_cache` replaces `transformer.forward` with a `functools.partial`. Without the
`__signature__` propagation in `cache_adapter.py`, that signature collapses to `(*args, **kwargs)`,
the filter selects nothing, and the layout tensors are dropped **without an exception** — the same
failure family as a missing input, one layer deeper. These tests run on a 1/128-width
`MiniMaxH3Transformer3DModel` (same module tree, no checkpoint) and pin both halves of the contract:

1. the patched forward still advertises every packed-layout parameter;
2. the adapter still points at the same block instances after the patch, so the cache wraps the real
   stack instead of a copy.
"""
import inspect

import cache_dit
from cache_dit import BlockAdapter, DBCacheConfig, ForwardPattern
from cache_dit.caching.block_adapters.block_adapters import FakeDiffusionPipeline
from diffusers import MiniMaxH3Transformer3DModel

# The fields the vendored denoise block hands the transformer through `denoiser_input_fields`.
_LAYOUT_PARAMS = (
  "position_ids",
  "token_tags",
  "video_indices",
  "audio_indices",
  "text_indices",
  "timestep_indices",
)


def _tiny_transformer() -> MiniMaxH3Transformer3DModel:
  return MiniMaxH3Transformer3DModel(
    num_attention_heads=2,
    attention_head_dim=8,
    hidden_size=16,
    num_layers=2,
    num_refiner_layers=1,
    ffn_dim=32,
    in_channels=4,
    audio_in_channels=4,
    text_dim=32,
    freq_dim=16,
    time_embed_hidden_dim=32,
    time_embed_dim=16,
    rope_freq_dim=4,
  )


def _enable(transformer: MiniMaxH3Transformer3DModel, cache_config: DBCacheConfig):
  cache_dit.enable_cache(
    BlockAdapter(
      pipe=FakeDiffusionPipeline(transformer),
      transformer=transformer,
      blocks=transformer.transformer_blocks,
      blocks_name="transformer_blocks",
      forward_pattern=ForwardPattern.Pattern_3,
    ),
    cache_config=cache_config,
  )


def _six_step_config() -> DBCacheConfig:
  return DBCacheConfig(
    Fn_compute_blocks=1,
    Bn_compute_blocks=0,
    max_warmup_steps=1,
    num_inference_steps=6,
    residual_diff_threshold=0.08,
  )


def test_cache_patch_keeps_the_packed_layout_signature():
  transformer = _tiny_transformer()
  before = list(inspect.signature(transformer.forward).parameters)

  _enable(transformer, _six_step_config())

  after = list(inspect.signature(transformer.forward).parameters)
  assert after == before
  # The filter in the denoise block would silently select nothing if this regressed.
  missing = [name for name in _LAYOUT_PARAMS if name not in after]
  assert not missing, f"the packed-layout inputs {missing} would be dropped from the call"
  assert getattr(transformer, "_is_cached", False) is True


def test_cache_patch_wraps_the_real_block_instances():
  transformer = _tiny_transformer()
  blocks_before = [id(module) for module in transformer.transformer_blocks]

  _enable(transformer, _six_step_config())

  assert [id(module) for module in transformer.transformer_blocks] == blocks_before
  assert getattr(transformer, "_original_forward", None) is not None
  # The recipe fingerprint the run logs must carry the six-step ladder and the one-step warmup: they
  # are what decide whether any step can be skipped at all.
  fingerprint = _six_step_config().strify()
  assert "F1B0" in fingerprint and "W1I1" in fingerprint and "N6" in fingerprint
