"""Regression tests for the Wan-Animate-2 cache integration.

Wan-Animate-2 keeps one reference K/V cache per transformer block (`kv_cache.get(idx)` in the block
loop) and skips layer 9 on the unconditional CFG branch. Cache-dit replaces `transformer.blocks`
with a single-element wrapper driven by one shared kwargs set, so both behaviours must be replayed
inside the block by `WanAnimate2PatchFunctor`: otherwise every layer reads layer 0's K/V (the output
collapses to colour noise without raising) and the unconditional branch runs one extra layer.

The reference pass (`kv_cache_mode="extract"`) also has to bypass the wrapper, because it must run
every layer to fill the cache and must not advance the cache step counter.
"""
import copy
import types

import pytest
import torch

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.caching.forward_pattern import ForwardPattern
from cache_dit.caching.patch_functors import WanAnimate2PatchFunctor

diffusers = pytest.importorskip("diffusers")

if not hasattr(diffusers, "WanAnimate2Transformer3DModel"):
  pytest.skip("Wan-Animate-2 requires a newer diffusers", allow_module_level=True)

from diffusers.models.transformers.transformer_wan_animate_2 import (  # noqa: E402
  WanAnimate2KVCache, WanAnimate2Transformer3DModel,
)

NUM_LAYERS = 12  # > 9, so the unconditional skip layer exists
LATENT_CHANNELS = 16
CONDITION_CHANNELS = 20  # `in_dim = 36 = 16 + 20`
GRID = 4  # per-axis patch grid
SEQ_LEN = GRID * GRID


def _tiny_transformer() -> WanAnimate2Transformer3DModel:
  return WanAnimate2Transformer3DModel(
    patch_size=(1, 2, 2),
    in_dim=LATENT_CHANNELS + CONDITION_CHANNELS,
    out_dim=LATENT_CHANNELS,
    dim=64,
    ffn_dim=128,
    freq_dim=32,
    text_dim=16,
    text_len=8,
    num_heads=4,
    num_layers=NUM_LAYERS,
    use_img_emb=False,
  ).eval()


def _latents(seed: int = 0) -> dict:
  generator = torch.Generator(device="cpu").manual_seed(seed)
  return dict(
    hidden_states=[
      torch.randn(LATENT_CHANNELS, 1, 2 * GRID, 2 * GRID, generator=generator).to(torch.float32)
    ],
    # The reference stream is the driving clip; the generation stream reuses the same geometry here.
    condition_latents=[
      torch.randn(CONDITION_CHANNELS, 1, 2 * GRID, 2 * GRID, generator=generator).to(torch.float32)
    ],
    encoder_hidden_states=[torch.randn(4, 16, generator=generator)],
  )


def _grid_sizes() -> torch.Tensor:
  return torch.tensor([[1, GRID, GRID]], dtype=torch.long)


@torch.no_grad()
def _extract(model, kv_cache, latents, grid: torch.Tensor = None):
  if grid is None:
    grid = _grid_sizes()
  return model(
    **latents,
    timestep=torch.tensor([1.0], device=grid.device),
    kv_cache=kv_cache,
    kv_cache_mode="extract",
    seq_len=SEQ_LEN,
    offset_grid_sizes=grid,
    return_dict=False,
  )[0]


@torch.no_grad()
def _cached(model, kv_cache, latents, is_uncondtion: bool = False):
  return model(
    **latents,
    timestep=torch.tensor([0.5]),
    kv_cache=kv_cache,
    kv_cache_mode="cached",
    seq_len=SEQ_LEN,
    reference_grid_sizes=_grid_sizes(),
    origin_len=1,
    origin_area=[8 * 2 * GRID, 8 * 2 * GRID],
    is_uncondtion=is_uncondtion,
    return_dict=False,
  )[0]


def test_patch_keeps_numerics_bit_exact():
  """The patched blocks must reproduce the upstream loop exactly, both passes and both branches."""
  reference = _tiny_transformer()
  patched = copy.deepcopy(reference)
  WanAnimate2PatchFunctor().apply(patched)

  reference_cache = WanAnimate2KVCache(NUM_LAYERS)
  patched_cache = WanAnimate2KVCache(NUM_LAYERS)
  latents = _latents()

  torch.testing.assert_close(
    _extract(patched, patched_cache, latents)[0],
    _extract(reference, reference_cache, latents)[0],
    atol=0,
    rtol=0,
  )
  for layer in range(NUM_LAYERS):
    # Per-layer resolution: every layer stored its own K/V instead of aliasing layer 0.
    assert patched_cache.get(layer).key is not None
    torch.testing.assert_close(patched_cache.get(layer).key,
                               reference_cache.get(layer).key,
                               atol=0,
                               rtol=0)
  distinct = {id(patched_cache.get(layer).key) for layer in range(NUM_LAYERS)}
  assert len(distinct) == NUM_LAYERS

  for is_uncondtion in (False, True):
    torch.testing.assert_close(
      _cached(patched, patched_cache, latents, is_uncondtion)[0],
      _cached(reference, reference_cache, latents, is_uncondtion)[0],
      atol=0,
      rtol=0,
      msg=f"cached pass diverged (is_uncondtion={is_uncondtion})",
    )


def test_uncond_branch_skips_layer_9():
  """The unconditional branch must skip block 9, matching the upstream `continue`."""
  patched = _tiny_transformer()
  WanAnimate2PatchFunctor().apply(patched)
  cache = WanAnimate2KVCache(NUM_LAYERS)
  latents = _latents()
  _extract(patched, cache, latents)

  calls = []
  original = patched.blocks[9]._wan_animate2_block_forward

  def tracked(*args, **kwargs):
    calls.append(1)
    return original(*args, **kwargs)

  patched.blocks[9]._wan_animate2_block_forward = tracked

  _cached(patched, cache, latents, is_uncondtion=True)
  assert len(calls) == 0, "layer 9 ran on the unconditional branch"

  _cached(patched, cache, latents, is_uncondtion=False)
  assert len(calls) == 1, "layer 9 must run on the conditional branch"


def test_extract_bypasses_cached_wrapper():
  """With the cache installed, the reference pass still has to fill every layer."""
  transformer = _tiny_transformer()
  cache_dit.enable_cache(
    transformer,
    cache_config=DBCacheConfig(
      num_inference_steps=2,
      Fn_compute_blocks=1,
      residual_diff_threshold=0.0,
      enable_separate_cfg=False,
    ),
  )
  assert getattr(transformer, "_is_patched", False)
  assert hasattr(transformer, "_context_manager")

  cache = WanAnimate2KVCache(NUM_LAYERS)
  _extract(transformer, cache, _latents())

  missing = [layer for layer in range(NUM_LAYERS) if cache.get(layer).key is None]
  assert not missing, f"layers {missing} did not run during the reference pass"
  # The reference pass must not advance the denoising step counter.
  context = transformer._context_manager.get_context(transformer._context_names[0])
  assert context.transformer_executed_steps == 0


def test_adapter_uses_pattern_3_and_reads_guider():
  """The adapter must use the single-stream pattern and derive separate-CFG from the guider."""
  from cache_dit.caching.block_adapters.adapters import wan_animate_2_adapter

  transformer = _tiny_transformer()
  base_stub = types.SimpleNamespace(
    transformer=transformer,
    guider=types.SimpleNamespace(num_conditions=2),
  )
  adapter = wan_animate_2_adapter(base_stub)
  assert adapter.forward_pattern is ForwardPattern.Pattern_3
  assert adapter.transformer is transformer
  assert adapter.blocks is transformer.blocks
  assert adapter.blocks_name == "blocks"
  assert adapter.has_separate_cfg is True

  distilled_stub = types.SimpleNamespace(
    transformer=_tiny_transformer(),
    guider=types.SimpleNamespace(num_conditions=1),
  )
  assert wan_animate_2_adapter(distilled_stub).has_separate_cfg is False


def test_registry_prefers_wan_animate_2_over_wan():
  """The adapter registry matches the first prefix, so "WanAnimate2" must precede "Wan".

  The distilled checkpoint also has to keep the adapter's `False`: the prefix fallback list must not
  contain "Wan", otherwise `has_separate_cfg` reports two passes for a single-pass checkpoint and
  the
  cache bookkeeping halves the effective step count.
  """
  from cache_dit.caching.block_adapters import BlockAdapterRegister

  keys = list(BlockAdapterRegister._adapters.keys())
  assert keys.index("WanAnimate2") < keys.index("Wan")
  assert "Wan" not in BlockAdapterRegister._predefined_adapters_has_separate_cfg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cached pass runs flex attention")
def test_cached_pass_matches_reference_through_wrapper():
  """A cached pass with nothing skippable must be bit-exact, i.e. `kv_cache` pierces the wrapper."""
  reference = _tiny_transformer().cuda()
  patched = copy.deepcopy(reference)

  cache_dit.enable_cache(
    patched,
    cache_config=DBCacheConfig(
      num_inference_steps=2,
      Fn_compute_blocks=1,
      residual_diff_threshold=0.0,
      enable_separate_cfg=False,
    ),
  )
  assert getattr(patched, "_is_patched", False)

  reference_cache = WanAnimate2KVCache(NUM_LAYERS)
  patched_cache = WanAnimate2KVCache(NUM_LAYERS)
  latents = {k: [t.cuda() for t in v] for k, v in _latents().items()}
  grid = _grid_sizes().cuda()

  _extract(reference, reference_cache, latents, grid)
  _extract(patched, patched_cache, latents, grid)
  for is_uncondtion in (False, True):
    got = patched(
      **latents,
      timestep=torch.tensor([0.5], device="cuda"),
      kv_cache=patched_cache,
      kv_cache_mode="cached",
      seq_len=SEQ_LEN,
      reference_grid_sizes=grid,
      origin_len=1,
      origin_area=[8 * 2 * GRID, 8 * 2 * GRID],
      is_uncondtion=is_uncondtion,
      return_dict=False,
    )[0]
    expected = reference(
      **latents,
      timestep=torch.tensor([0.5], device="cuda"),
      kv_cache=reference_cache,
      kv_cache_mode="cached",
      seq_len=SEQ_LEN,
      reference_grid_sizes=grid,
      origin_len=1,
      origin_area=[8 * 2 * GRID, 8 * 2 * GRID],
      is_uncondtion=is_uncondtion,
      return_dict=False,
    )[0]
    torch.testing.assert_close(got[0], expected[0], atol=0, rtol=0)
