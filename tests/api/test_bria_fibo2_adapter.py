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
import unittest.mock

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
# Several bundles, including layer 0 (the block `Fn_compute_blocks=1` re-runs on *every* step) and a
# skippable layer, so the bundle routing is exercised on both sides of a cache hit.
_INJECTION_LAYER_IDS = (0, 2)


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

    def __call__(self, guidance_scale: float = guidance_scale):
      return None

    self.__call__ = __call__


class _LoopBriaFibo2Pipe:
  """Tiny denoising loop: one `__call__` runs every step inside a single cache context.

  A class-level `__call__` is what `enable_cache` wraps, and one call has to hold all the steps —
  otherwise every step opens a fresh context, sees no history and can never be served from cache.
  `step` publishes the current index so the cross-attention recorder can label its calls.
  """

  def __init__(self, transformer, step: dict = None):
    self.transformer = transformer
    self.step = step if step is not None else {"value": 0}

  def __call__(self, steps, guidance_scale: float = 1.0):
    outputs = []
    for step_id, (hidden_states, text) in enumerate(steps):
      self.step["value"] = step_id
      outputs.append(
        self.transformer(
          hidden_states=hidden_states,
          timestep=torch.tensor([0.5]),
          encoder_hidden_states=text,
          return_dict=False,
        ))
    return outputs


def _inputs(batch_size: int = 1, seed: int = 1):
  generator = torch.Generator().manual_seed(seed)
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


def _bundle(inputs, layer_id: int):
  """Text bundle `layer_id` must read: `encoder_hidden_states[:, k + 1]`."""
  return inputs["encoder_hidden_states"][:, 1 + _INJECTION_LAYER_IDS.index(layer_id)]


class _CrossAttentionRecorder(torch.nn.Module):
  """Pass-through wrapper that logs `(step, text)` for every cross-attention call it sees."""

  def __init__(self, attention: torch.nn.Module, sink: dict, layer_id: int, step: dict):
    super().__init__()
    self.attention = attention
    self.sink = sink
    self.layer_id = layer_id
    self.step = step

  def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, freqs_cis=None):
    self.sink.setdefault(self.layer_id, []).append((self.step["value"], encoder_hidden_states))
    return self.attention(
      hidden_states,
      encoder_hidden_states=encoder_hidden_states,
      attention_mask=attention_mask,
      freqs_cis=freqs_cis,
    )


def _record_cross_attention(transformer, step: dict = None):
  """Log, per injection layer, the `(step, text)` of every executed cross-attention call."""
  step = step if step is not None else {"value": 0}
  seen = {}
  for layer_id, block in enumerate(transformer.layers):
    if block.cross_attention is not None:
      block.cross_attention = _CrossAttentionRecorder(block.cross_attention, seen, layer_id, step)
  return seen


def _capture_block_call(transformer, layer_id: int, inputs):
  """Record the arguments the upstream loop hands to `layer_id`."""
  captured = {}

  def pre_hook(module, args, kwargs):
    captured["args"] = args
    captured["kwargs"] = kwargs

  handle = transformer.layers[layer_id].register_forward_pre_hook(pre_hook, with_kwargs=True)
  try:
    with torch.no_grad():
      transformer(**inputs, return_dict=False)
  finally:
    handle.remove()
  return captured["args"], captured["kwargs"]


def _collapsed_forward(transformer, inputs):
  """Run one forward with `layers` collapsed to a single wrapper, the way cache-dit calls it."""
  with unittest.mock.patch.object(transformer, "layers",
                                  torch.nn.ModuleList([transformer._orig_layers[0]])):
    with torch.no_grad():
      transformer(**inputs, return_dict=False)


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
  inputs = _inputs()
  BriaFibo2PatchFunctor().apply(transformer)
  seen = _record_cross_attention(transformer)
  with torch.no_grad():
    transformer(**inputs, return_dict=False)

  for layer_id in _INJECTION_LAYER_IDS:
    block = transformer.layers[layer_id]
    assert torch.equal(block._text_bundle, _bundle(inputs, layer_id))
    # What the block actually cross-attends to, not just what was stashed on it.
    assert torch.equal(seen[layer_id][-1][1], _bundle(inputs, layer_id))
    # The loop is not collapsed here, so upstream already supplies the right per-layer text and the
    # stash stays inert.
    assert block._use_text_bundle is False
  plain = [
    layer_id for layer_id in range(len(transformer.layers)) if layer_id not in _INJECTION_LAYER_IDS
  ]
  assert plain, "the fixture needs at least one block without cross-attention"
  assert all(transformer.layers[layer_id]._text_bundle is None for layer_id in plain)


def _positional_call_args(block, kwargs):
  """Rebuild the upstream gradient-checkpointing call: all nine parameters, positionally."""
  params = [name for name in inspect.signature(type(block).forward).parameters if name != "self"]
  assert params[:3] == ["x", "attn_mask", "freqs_cis"]
  tail = [kwargs.get(name, None) for name in params[3:]]
  text_index = params[3:].index("encoder_hidden_states")
  tail[text_index] = torch.zeros_like(tail[text_index])
  return params, tail, text_index


def test_positional_block_call_uses_the_stash_when_the_loop_is_collapsed():
  from cache_dit.caching.patch_functors.functor_bria_fibo2 import _FORWARD_TAIL_ARGS
  from cache_dit.caching.patch_functors import BriaFibo2PatchFunctor

  layer_id = _INJECTION_LAYER_IDS[-1]
  transformer = _tiny_transformer()
  inputs = _inputs()
  args, kwargs = _capture_block_call(transformer, layer_id, inputs)

  BriaFibo2PatchFunctor().apply(transformer)
  _collapsed_forward(transformer, inputs)  # what a cached step looks like to the prologue
  block = transformer.layers[layer_id]
  assert block._use_text_bundle is True
  seen = _record_cross_attention(transformer)

  params, tail, _ = _positional_call_args(block, kwargs)
  assert tuple(params[3:]) == _FORWARD_TAIL_ARGS, "the tail order drifted from upstream"
  with torch.no_grad():
    block(*args, *tail)

  # The collapsed loop only ever indexes layer 0, so the positional text is layer 0's bundle or
  # `None`; the stash (zeroed sentinel above) has to win.
  assert torch.equal(seen[layer_id][-1][1], _bundle(inputs, layer_id))


def test_positional_block_call_passes_through_when_the_loop_is_not_collapsed():
  from cache_dit.caching.patch_functors import BriaFibo2PatchFunctor

  # Without the collapse the upstream loop hands each block its own text, and a gradient-checkpointing
  # recompute must re-read *those* values, not the live stash of some later forward.
  layer_id = _INJECTION_LAYER_IDS[-1]
  transformer = _tiny_transformer()
  inputs = _inputs()
  args, kwargs = _capture_block_call(transformer, layer_id, inputs)

  BriaFibo2PatchFunctor().apply(transformer)
  with torch.no_grad():
    transformer(**inputs, return_dict=False)
  block = transformer.layers[layer_id]
  assert block._use_text_bundle is False
  seen = _record_cross_attention(transformer)

  _, tail, text_index = _positional_call_args(block, kwargs)
  with torch.no_grad():
    block(*args, *tail)

  assert torch.equal(seen[layer_id][-1][1], tail[text_index])


def test_cached_steps_refresh_every_injection_bundle():
  from cache_dit.caching.block_adapters import bria_fibo2_adapter

  num_steps = 4
  # A different prompt and different latents on every step, so a reused bundle is unmistakable.
  steps = [_inputs(seed=100 + step) for step in range(num_steps)]
  bundles = [{
    layer_id: _bundle(step_inputs, layer_id)
    for layer_id in _INJECTION_LAYER_IDS
  } for step_inputs in steps]

  transformer = _tiny_transformer()
  step = {"value": 0}
  pipe = _LoopBriaFibo2Pipe(transformer, step=step)
  adapter = bria_fibo2_adapter(pipe)
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=0,
      max_cached_steps=-1,
      # One cached step at most in a row, so a block has to resume computing after a cache hit.
      max_continuous_cached_steps=1,
      num_inference_steps=num_steps,
      enable_separate_cfg=False,
      residual_diff_threshold=10.0,
    ),
  )
  seen = _record_cross_attention(transformer, step)
  with torch.no_grad():
    pipe([(step_inputs["hidden_states"], step_inputs["encoder_hidden_states"])
          for step_inputs in steps])

  # Every executed cross-attention must read the bundle of the step it ran in, skipped steps
  # included in the sense that a block resuming afterwards must not reuse the last computed text.
  for layer_id in _INJECTION_LAYER_IDS:
    for step_id, logged in seen[layer_id]:
      assert torch.equal(logged, bundles[step_id][layer_id]), (
        f"step {step_id}: layer {layer_id} cross-attended to a stale bundle")

  # Layer 0 sits inside `Fn_compute_blocks=1`, so it runs on every step, in order.
  assert [step_id for step_id, _ in seen[0]] == list(range(num_steps))

  skippable = [layer_id for layer_id in _INJECTION_LAYER_IDS if layer_id != 0]
  assert skippable, "the fixture needs an injection layer behind `Fn_compute_blocks`"
  for layer_id in skippable:
    executed = [step_id for step_id, _ in seen[layer_id]]
    assert len(executed) < num_steps, "no step was served from cache"
    assert len(executed) > 1, f"layer {layer_id} never resumed after a cache hit"

  # The prologue refreshes the stash on every call, cached ones included.
  for layer_id in _INJECTION_LAYER_IDS:
    assert torch.equal(transformer.layers[layer_id]._text_bundle, bundles[-1][layer_id])


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
