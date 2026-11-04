"""Regression tests for the SANA-Video cache integration.

`SanaVideoTransformer3DModel` (diffusers, SANA-Video 2B 480p) runs its denoising stack in
`transformer.transformer_blocks`. Each block takes nine positional arguments
`(hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, timestep, frames,
height, width, rotary_emb)` and returns a single tensor, so `Pattern_3` caching is only correct if
the wrapper replays the eight extras untouched. These CPU tests pin down:

1. The `"SanaVideo"` prefix wins over `"Sana"` without stealing `"SanaVideo2"` (the registry takes
   the first matching class-name prefix), and the untouched `"Sana"` adapter still rejects the video
   transformer instead of silently accepting it.
2. The adapter reports the right block contract.
3. A cached run is bit-exact against the unpatched transformer when nothing is cached — i.e. the
   positional extras survive the collapsed loop.
4. A fully cached step reproduces the computed one.
"""

import inspect

import pytest
import torch

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.caching.block_adapters import BlockAdapterRegister
from cache_dit.caching.forward_pattern import ForwardPattern

diffusers = pytest.importorskip("diffusers")

if not hasattr(diffusers, "SanaVideoTransformer3DModel"):
  pytest.skip("SANA-Video requires a newer diffusers", allow_module_level=True)

_IN_CHANNELS = 4
_HEADS = 2
_HEAD_DIM = 8
_FRAMES = 2
_LATENT_H = 8
_LATENT_W = 8
_TEXT_LEN = 5
_CAPTION_CHANNELS = 8
_CROSS_ATTENTION_DIM = _HEADS * _HEAD_DIM


def _tiny_transformer(num_layers: int = 2):
  from diffusers import SanaVideoTransformer3DModel

  torch.manual_seed(0)
  transformer = SanaVideoTransformer3DModel(
    in_channels=_IN_CHANNELS,
    out_channels=_IN_CHANNELS,
    num_attention_heads=_HEADS,
    attention_head_dim=_HEAD_DIM,
    num_layers=num_layers,
    num_cross_attention_heads=_HEADS,
    cross_attention_head_dim=_HEAD_DIM,
    cross_attention_dim=_CROSS_ATTENTION_DIM,
    caption_channels=_CAPTION_CHANNELS,
    mlp_ratio=2.0,
    sample_size=_LATENT_H,
    patch_size=(1, 2, 2),
  )
  return transformer.eval()


class _SanaVideoLikePipe:
  """Minimal stand-in for `SanaVideoPipeline`: a transformer plus `__call__` defaults."""

  def __init__(self, transformer, guidance_scale: float = 6.0):
    self.transformer = transformer
    self.guidance_scale = guidance_scale

    def __call__(self, **kwargs):
      return None

    self.__call__ = __call__


def _inputs(batch_size: int = 1, seed: int = 1):
  generator = torch.Generator().manual_seed(seed)
  return dict(
    hidden_states=torch.randn(batch_size,
                              _IN_CHANNELS,
                              _FRAMES,
                              _LATENT_H,
                              _LATENT_W,
                              generator=generator),
    encoder_hidden_states=torch.randn(batch_size, _TEXT_LEN, _CAPTION_CHANNELS,
                                      generator=generator),
    timestep=torch.zeros(batch_size),
    encoder_attention_mask=torch.ones(batch_size, _TEXT_LEN, dtype=torch.long),
    return_dict=False,
  )


def test_sana_video_prefix_wins_without_stealing_sana_video2():
  prefixes = BlockAdapterRegister.supported_pipelines()[1]
  assert "SanaVideo" in prefixes

  matches = [name for name in prefixes if "SanaVideoTransformer3DModel".startswith(name)]
  assert matches and matches[0] == "SanaVideo", f"wrong adapter for SANA-Video: {matches}"

  # The I2V *pipeline* class is `SanaImageToVideoPipeline`, which shares no prefix with `SanaVideo`
  # and would otherwise fall through to the 2D `Sana` adapter.
  matches_i2v = [name for name in prefixes if "SanaImageToVideoPipeline".startswith(name)]
  assert matches_i2v and matches_i2v[
    0] == "SanaImageToVideo", f"wrong adapter for I2V: {matches_i2v}"

  # SANA-Video 2.0 shares the `SanaVideo` prefix and must keep its own (vendored) adapter.
  matches_v2 = [name for name in prefixes if "SanaVideo2Transformer3DModel".startswith(name)]
  assert matches_v2 and matches_v2[
    0] == "SanaVideo2", f"wrong adapter for SANA-Video 2.0: {matches_v2}"

  # The 2D Sana adapter must stay reachable for `SanaTransformer2DModel`.
  matches_sana = [name for name in prefixes if "SanaTransformer2DModel".startswith(name)]
  assert matches_sana and matches_sana[0] == "Sana", f"wrong adapter for Sana: {matches_sana}"


def test_sana_adapter_still_rejects_the_video_transformer():
  from cache_dit.caching.block_adapters import sana_adapter

  # Resolving through the registry is what a user gets; the strict `Sana` assertion must not be
  # relaxed just to admit the video model.
  with pytest.raises(AssertionError):
    sana_adapter(_SanaVideoLikePipe(_tiny_transformer()))


def test_adapter_reports_the_block_contract():
  from cache_dit.caching.block_adapters import sana_video_adapter

  transformer = _tiny_transformer()
  adapter = sana_video_adapter(_SanaVideoLikePipe(transformer))
  assert adapter.blocks is transformer.transformer_blocks
  assert adapter.blocks_name == "transformer_blocks"
  assert adapter.forward_pattern is ForwardPattern.Pattern_3
  assert adapter.check_forward_pattern is True
  # CFG is folded into the batch axis: one `transformer.forward()` per denoising step.
  assert adapter.has_separate_cfg is False
  # The block call is positional and the loop body holds no extra operator.
  assert adapter.patch_functor is None


def test_adapter_declares_the_repeated_block_stack():
  """Diffusers ships `_repeated_blocks = []`, which would widen compile and quantization."""

  from cache_dit.caching.block_adapters import sana_video_adapter

  transformer = _tiny_transformer()
  assert not getattr(transformer, "_repeated_blocks", None)
  sana_video_adapter(_SanaVideoLikePipe(transformer))
  assert transformer._repeated_blocks == ["SanaVideoTransformerBlock"]


def test_cache_run_is_bit_exact_when_no_step_is_cached():
  reference = _tiny_transformer()
  inputs = _inputs()
  with torch.no_grad():
    expected = reference(**inputs)[0]

  transformer = _tiny_transformer()
  adapter = BlockAdapterRegister.get_adapter(transformer)
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=1,
      max_warmup_steps=0,
      max_cached_steps=0,
      num_inference_steps=4,
      residual_diff_threshold=0.0,
    ),
  )
  signature = str(inspect.signature(transformer.forward))
  with torch.no_grad():
    cached = transformer(**inputs)[0]

  assert str(inspect.signature(transformer.forward)) == signature
  assert torch.equal(cached, expected), "the cached run must reproduce the velocity"


def test_fully_cached_step_reproduces_the_computed_one():
  # Three layers so `Fn_compute_blocks=1` + `Bn_compute_blocks=1` leave the middle layer as the
  # only skippable one: block 0 always runs and cannot prove anything about the cache.
  transformer = _tiny_transformer(num_layers=3)
  adapter = BlockAdapterRegister.get_adapter(transformer)
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=1,
      max_warmup_steps=0,
      max_cached_steps=-1,
      num_inference_steps=4,
      # Identical inputs on the second call -> residual diff is 0, so the middle block is skipped.
      residual_diff_threshold=1e9,
    ),
  )

  calls = [0]
  transformer.transformer_blocks[1].register_forward_pre_hook(
    lambda module, args, kwargs=None: calls.__setitem__(0, calls[0] + 1))

  inputs = _inputs(seed=7)
  with torch.no_grad():
    first = transformer(**inputs)[0]
    second = transformer(**inputs)[0]

  assert calls[0] == 1, (f"the middle block ran {calls[0]} times over two identical forwards; "
                         "the second forward was not served from cache")
  assert torch.allclose(first, second, atol=1e-6), (
    "a fully cached step must reproduce the computed one; the positional extras (modulation, "
    "frames/height/width, RoPE) did not survive the collapsed block loop")
