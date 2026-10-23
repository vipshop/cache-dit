"""Regression tests for the Bria Fibo 2.0 cache integration.

Fibo 2.0 is a regular diffusers pipeline whose transformer (`BriaFibo2Transformer2DModel`) keeps a
single-stream `layers` stack. Two properties need locking down:

1. The adapter must be registered under `"BriaFibo2"` *before* the `"BriaFibo"` (Fibo 1.0) prefix,
   otherwise the Fibo 1.0 adapter matches first and its strict type assertion fires.
2. Every injection layer reads its own slice of `encoder_hidden_states`, which the collapsed cached
   block loop cannot express — `BriaFibo2PatchFunctor` stashes the bundles on the blocks. These CPU
   tests check that the cached run is bit-exact against the unpatched transformer, which fails
   loudly if a bundle is dropped.
"""
import inspect
import types

import pytest
import torch

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.caching.block_adapters import BlockAdapterRegister
from cache_dit.caching.forward_pattern import ForwardPattern

diffusers = pytest.importorskip("diffusers")

if not hasattr(diffusers, "BriaFibo2Transformer2DModel"):
  pytest.skip("Bria Fibo 2.0 requires a newer diffusers", allow_module_level=True)

_TEXT_LEN = 6
_GRID = 8  # 16x16 latents with patch 2x2 -> 8x8 = 64 tokens
_GIST = 8
_IN_CHANNELS = 8
_DIM = 32
_CAP_FEAT_DIM = 48
_INJECTION_LAYER_IDS = (1, )


def _tiny_transformer():
  from diffusers import BriaFibo2Transformer2DModel

  torch.manual_seed(0)
  transformer = BriaFibo2Transformer2DModel(
    in_channels=_IN_CHANNELS,
    patch_size=2,
    dim=_DIM,
    n_layers=3,
    n_refiner_layers=1,
    n_heads=2,
    cap_feat_dim=_CAP_FEAT_DIM,
    injection_layer_ids=_INJECTION_LAYER_IDS,
    perceiver_num_layers=1,
    min_num_gist_tokens=_GIST,
    max_num_gist_tokens=_GIST,
    gist_step=_GIST,
    gist_min_text_len=2,
    gist_max_text_len=4,
    # `sum(axes_dims)` must equal `dim // n_heads` (16 here).
    axes_dims=(4, 6, 6),
    axes_lens=(64, 32, 32),
  )
  return transformer.eval()


class _BriaFibo2LikePipe:
  """Minimal stand-in for `BriaFibo2Pipeline`: a transformer plus a `__call__` default."""

  def __init__(self, transformer, guidance_scale: float = 5.0):
    self.transformer = transformer
    self._guidance_scale = guidance_scale

    def __call__(self, guidance_scale: float = guidance_scale):  # noqa: ARG001
      return None

    self.__call__ = __call__


def _inputs(batch_size: int = 1):
  generator = torch.Generator().manual_seed(1)
  return dict(
    hidden_states=torch.randn(batch_size, _IN_CHANNELS, 2 * _GRID, 2 * _GRID, generator=generator),
    timestep=torch.tensor([0.5] * batch_size),
    # Bundle 0 feeds the Perceiver, bundle k + 1 feeds `injection_layer_ids[k]`.
    encoder_hidden_states=torch.randn(batch_size,
                                      1 + len(_INJECTION_LAYER_IDS),
                                      _TEXT_LEN,
                                      _CAP_FEAT_DIM,
                                      generator=generator),
  )


def test_bria_fibo2_prefix_wins_over_the_fibo_1_prefixes():
  prefixes = BlockAdapterRegister.supported_pipelines()[1]
  assert "BriaFibo2" in prefixes
  # The registry takes the first registered prefix, so `BriaFibo2` must precede `BriaFibo` / `Bria`.
  matches = [name for name in prefixes if "BriaFibo2Pipeline".startswith(name)]
  assert matches[0] == "BriaFibo2", f"Bria Fibo 2.0 dispatched to the wrong adapter: {matches}"


@pytest.mark.parametrize(("guidance_scale", "expected"), [(5.0, True), (1.0, False)])
def test_has_separate_cfg_follows_pipeline_default(guidance_scale, expected):
  from cache_dit.caching.block_adapters import bria_fibo2_adapter

  pipe = _BriaFibo2LikePipe(_tiny_transformer(), guidance_scale=guidance_scale)
  adapter = bria_fibo2_adapter(pipe)
  assert adapter.has_separate_cfg is expected
  assert adapter.blocks is pipe.transformer.layers
  assert adapter.forward_pattern is ForwardPattern.Pattern_3


def test_has_separate_cfg_survives_a_pipe_without_call_defaults():
  from cache_dit.caching.block_adapters import bria_fibo2_adapter

  # The adapter is also built from objects whose `__call__` carries no `guidance_scale` (e.g. the
  # `FakeDiffusionPipeline` wrapper of the transformer-only path); an undiscoverable default keeps
  # the guidance-enabled behavior instead of raising.
  pipe = types.SimpleNamespace(transformer=_tiny_transformer())
  adapter = bria_fibo2_adapter(pipe)
  assert adapter.has_separate_cfg is True


def test_patch_functor_routes_one_bundle_per_injection_layer():
  from cache_dit.caching.patch_functors import BriaFibo2PatchFunctor

  transformer = _tiny_transformer()
  BriaFibo2PatchFunctor().apply(transformer)
  with torch.no_grad():
    transformer(**_inputs(), return_dict=False)

  bundles = [block._text_bundle for block in transformer.layers if block._text_bundle_index]
  assert len(bundles) == len(_INJECTION_LAYER_IDS)
  assert all(bundle is not None for bundle in bundles)
  assert transformer.layers[_INJECTION_LAYER_IDS[0]]._text_bundle.shape == (1, _TEXT_LEN,
                                                                            _CAP_FEAT_DIM)


def test_cp_planner_applies_the_patch_functor():
  from cache_dit.distributed.config import ParallelismConfig
  from cache_dit.distributed.transformers.bria_fibo2 import (
    BriaFibo2ContextParallelismPlanner, )

  # Parallelism can be enabled before the cache. `PatchFunctor.apply` skips an already parallelized
  # transformer, so the planner itself has to arm the functor; otherwise the cached loop loses the
  # per-injection-layer text bundles without raising.
  transformer = _tiny_transformer()
  plan = BriaFibo2ContextParallelismPlanner()._apply(
    transformer=transformer,
    parallelism_config=ParallelismConfig(ulysses_size=2),
  )
  assert transformer._is_patched is True
  assert "layers.*" in plan


def test_cache_run_is_bit_exact_when_no_step_is_cached():
  from cache_dit.caching.block_adapters import bria_fibo2_adapter

  reference = _tiny_transformer()
  inputs = _inputs()
  with torch.no_grad():
    expected = reference(**inputs, return_dict=False)[0]

  transformer = _tiny_transformer()
  adapter = bria_fibo2_adapter(_BriaFibo2LikePipe(transformer))
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=0,
      max_cached_steps=0,
      num_inference_steps=4,
      enable_separate_cfg=True,
      residual_diff_threshold=0.0,
    ),
  )
  signature = str(inspect.signature(transformer.forward))
  with torch.no_grad():
    output = transformer(**inputs, return_dict=False)[0]

  assert str(inspect.signature(transformer.forward)) == signature
  assert torch.equal(output, expected), (
    "the cached (never-skipping) run must match the unpatched transformer, including every "
    "injection layer's own text bundle")
