"""Regression tests for the SANA-WM stage-2 refiner cache integration.

`SanaWMLTX2Refiner` is chunk-causal and autoregressive: per chunk it runs three
`kv_cache_mode="inject"` denoising forwards (read-only on the sliding-window K/V) plus one
`inject_and_capture_post_rope` pass that *writes* the refined K/V into a per-layer cache. The
`SanaWMRefinerPatchFunctor` exists because the generic cached wrapper breaks two things:

* upstream calls `block(..., kv_cache=kv_cache.get(i))`; after the block list collapses to a
  single wrapper, `i` is always 0, so every layer would read and write layer 0's slot;
* a KV-writing pass must run every block, or the next chunk reads an empty cache and raises.

These CPU tests pin down the dispatch, the per-layer resolution, and bit-exactness.
"""
import pytest
import torch

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.caching.block_adapters import BlockAdapterRegister
from cache_dit.caching.forward_pattern import ForwardPattern

diffusers = pytest.importorskip("diffusers")

try:
  from diffusers.models.transformers.transformer_sana_wm_refiner import (
    SanaWMRefinerKVCache,
    SanaWMLTX2RefinerTransformer3DModel,
  )
except ImportError:  # pragma: no cover - older diffusers
  pytest.skip("SANA-WM requires a newer diffusers", allow_module_level=True)

_IN_CHANNELS = 4
_HEADS = 2
_HEAD_DIM = 8
# `cross_attention_dim` has to equal the inner dim the caption projection emits; the raw caption
# width is a separate argument (`caption_channels`).
_CROSS_DIM = _HEADS * _HEAD_DIM
_CAPTION_DIM = 8
_FRAMES = 2
_LATENT_H = 2
_LATENT_W = 2
_TEXT_LEN = 3


def _tiny_refiner(num_layers: int = 2):
  torch.manual_seed(0)
  model = SanaWMLTX2RefinerTransformer3DModel(
    in_channels=_IN_CHANNELS,
    out_channels=_IN_CHANNELS,
    patch_size=1,
    patch_size_t=1,
    num_attention_heads=_HEADS,
    attention_head_dim=_HEAD_DIM,
    cross_attention_dim=_CROSS_DIM,
    caption_channels=_CAPTION_DIM,
    num_layers=num_layers,
    audio_in_channels=_IN_CHANNELS,
    audio_out_channels=_IN_CHANNELS,
    audio_num_attention_heads=_HEADS,
    audio_attention_head_dim=_HEAD_DIM,
    audio_cross_attention_dim=_CROSS_DIM,
  )
  return model.eval()


class _RefinerLikePipe:
  """Minimal stand-in for `SanaWMLTX2Refiner`: a transformer plus `__call__`."""

  def __init__(self, transformer):
    self.transformer = transformer

    def __call__(self):
      return None

    self.__call__ = __call__


def _inputs(model, batch_size: int = 1, seed: int = 1):
  """Patchified latents `(B, F*H*W, C)`, as the refiner's forward expects them."""
  generator = torch.Generator().manual_seed(seed)
  num_tokens = _FRAMES * _LATENT_H * _LATENT_W
  return dict(
    hidden_states=torch.randn(batch_size, num_tokens, _IN_CHANNELS, generator=generator),
    encoder_hidden_states=torch.randn(batch_size, _TEXT_LEN, _CAPTION_DIM, generator=generator),
    timestep=torch.full((batch_size, num_tokens), 0.5),
    video_rotary_emb=model.build_rotary_emb_for_absolute_positions(
      batch_size=batch_size,
      frame_positions=list(range(_FRAMES)),
      height=_LATENT_H,
      width=_LATENT_W,
      device=torch.device("cpu"),
      fps=16.0,
    ),
    return_dict=False,
  )


def test_refiner_transformer_dispatches_to_the_sana_wm_adapter():
  from cache_dit.caching.block_adapters import sana_wm_adapter

  prefixes = BlockAdapterRegister.supported_pipelines()[1]
  matches = [name for name in prefixes if "SanaWMLTX2RefinerTransformer3DModel".startswith(name)]
  assert matches and matches[0] == "SanaWM"

  transformer = _tiny_refiner()
  adapter = sana_wm_adapter(_RefinerLikePipe(transformer))
  assert adapter.blocks is transformer.transformer_blocks
  assert adapter.blocks_name == "transformer_blocks"
  assert adapter.forward_pattern is ForwardPattern.Pattern_3
  # The functor runs before the adapter reads `blocks`, and keeps a handle on the originals.
  assert getattr(transformer, "_is_patched", False) is True
  # A plain list on purpose: a second module container would register every block twice.
  assert isinstance(transformer._sana_wm_raw_blocks, list)
  assert len(transformer._sana_wm_raw_blocks) == len(transformer.transformer_blocks)
  for layer_idx, block in enumerate(transformer.transformer_blocks):
    assert block._sana_wm_layer_idx == layer_idx
    assert transformer._sana_wm_raw_blocks[layer_idx] is block


def test_inject_reads_each_layers_own_cache():
  # Seed every layer with a *different* sink so a mixed-up index changes the result.
  def _seeded_cache(num_layers, rope):
    cache = SanaWMRefinerKVCache(num_layers)
    cache.sink_pe = rope
    for idx in range(num_layers):
      generator = torch.Generator().manual_seed(100 + idx)
      cache.get(idx).store_sink(
        torch.randn(1, 1, _HEADS * _HEAD_DIM, generator=generator),
        torch.randn(1, 1, _HEADS * _HEAD_DIM, generator=generator),
      )
    return cache

  reference = _tiny_refiner(num_layers=2)
  inputs = _inputs(reference)
  rope = inputs["video_rotary_emb"]
  with torch.no_grad():
    expected = reference(**inputs, kv_cache=_seeded_cache(2, rope), kv_cache_mode="inject")[0]

  transformer = _tiny_refiner(num_layers=2)
  # `get_adapter` builds the `FakeDiffusionPipeline` that owns the cache contexts.
  adapter = BlockAdapterRegister.get_adapter(transformer)
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=1,
      max_warmup_steps=0,
      max_cached_steps=0,
      num_inference_steps=3,
      residual_diff_threshold=0.0,
    ),
  )
  # The wrapper hands the whole container to every block; each block resolves its own slot.
  with torch.no_grad():
    patched = transformer(**inputs, kv_cache=_seeded_cache(2, rope), kv_cache_mode="inject")[0]
  assert torch.allclose(patched, expected, atol=1e-6)


def test_no_cache_run_is_bit_exact():
  from cache_dit.caching.block_adapters import sana_wm_adapter

  reference = _tiny_refiner()
  inputs = _inputs(reference)
  with torch.no_grad():
    expected = reference(**inputs)[0]

  transformer = _tiny_refiner()
  # The functor alone (no cache) must reproduce upstream bit for bit.
  sana_wm_adapter(_RefinerLikePipe(transformer))
  with torch.no_grad():
    patched = transformer(**inputs)[0]

  assert torch.equal(patched, expected), "the patched forward must reproduce upstream"


def test_cached_run_is_bit_exact_when_no_step_is_cached():
  reference = _tiny_refiner()
  inputs = _inputs(reference)
  with torch.no_grad():
    expected = reference(**inputs)[0]

  transformer = _tiny_refiner()
  adapter = BlockAdapterRegister.get_adapter(transformer)
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=1,
      max_warmup_steps=0,
      max_cached_steps=0,
      num_inference_steps=3,
      residual_diff_threshold=0.0,
    ),
  )

  with torch.no_grad():
    cached = transformer(**inputs)[0]
  assert torch.allclose(cached, expected, atol=1e-6)


def test_inject_forwards_go_through_the_cached_wrapper():
  # `BlockAdapterRegister.get_adapter` builds the `FakeDiffusionPipeline` that owns the cache
  # contexts; enabling the cache on a hand-rolled pipe object silently skips every step instead.
  # With three layers, `Fn_compute_blocks=1` + `Bn_compute_blocks=1` leave the middle layer as the
  # only skippable one, so counting its executions distinguishes "served from cache" from
  # "replayed through the originals".
  transformer = _tiny_refiner(num_layers=3)
  adapter = BlockAdapterRegister.get_adapter(transformer)
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=1,
      max_warmup_steps=0,
      max_cached_steps=-1,
      # Identical inputs every call -> the middle block can be served from cache.
      residual_diff_threshold=1e9,
    ),
  )

  calls = [0]
  transformer._sana_wm_raw_blocks[1].register_forward_pre_hook(
    lambda module, args, kwargs=None: calls.__setitem__(0, calls[0] + 1))

  kv_cache = SanaWMRefinerKVCache(len(transformer._sana_wm_raw_blocks))
  inputs = _inputs(transformer)
  with torch.no_grad():
    for _ in range(3):
      transformer(**inputs, kv_cache=kv_cache, kv_cache_mode="inject")

  assert calls[0] == 1, (
    "the middle block ran "
    f"{calls[0]} times over three identical `inject` forwards; `inject` must go through the "
    "cached wrapper so the repeated steps can be served from cache")


def test_capture_pass_is_replayed_and_len_check_uses_the_real_block_count():
  transformer = _tiny_refiner()
  adapter = BlockAdapterRegister.get_adapter(transformer)
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=1,
      max_warmup_steps=0,
      max_cached_steps=-1,
      num_inference_steps=3,
      # A huge threshold would otherwise skip blocks on every call, including the KV capture.
      residual_diff_threshold=1e9,
    ),
  )

  kv_cache = SanaWMRefinerKVCache(len(transformer._sana_wm_raw_blocks))
  inputs = _inputs(transformer)
  with torch.no_grad():
    transformer(**inputs, kv_cache=kv_cache, kv_cache_mode="capture_pre_rope")
  for idx, cache in enumerate(kv_cache.layer_caches):
    assert cache.captured_k_pre is not None, f"layer {idx} was skipped during a KV capture pass"
  # The upstream length check compares against the real block count, not the collapsed list.
  with torch.no_grad(), pytest.raises(ValueError):
    transformer(**inputs, kv_cache=SanaWMRefinerKVCache(1), kv_cache_mode="inject")
