"""Regression tests for the Echo (JoyAI-Echo) cache integration.

Echo is a diffusers modular pipeline (`EchoModularPipeline`) built on the LTX-2 audiovisual
backbone, so it reuses the LTX-2 block contract and `LTX2PatchFunctor`. What has to be locked down
here:

1. The adapter is registered under `"Echo"`. The registry matches *pipeline* class-name prefixes, so
   `"LTX"` — registered for the regular LTX pipelines — never matches `EchoModularPipeline`.
2. `"Echo"` is one of the verified modular pipelines, so `enable_cache` does not warn about an
   unverified dispatch.
3. Echo is guidance-free (one forward per stochastic DMD step), so `has_separate_cfg` stays False.
4. `enable_cache` must keep the LTX-2 forward signature, because the modular denoise step passes the
   transformer exactly the keywords it finds there.
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

if not hasattr(diffusers, "EchoModularPipeline"):
  pytest.skip("Echo requires a newer diffusers", allow_module_level=True)

_INNER_DIM = 32
_VIDEO_TOKENS = 8
_AUDIO_TOKENS = 8
_TEXT_TOKENS = 4

# Keywords the Echo denoise step passes to the transformer (`EchoLoopDenoiser.__call__`).
_ECHO_DENOISER_INPUTS = (
  "hidden_states",
  "audio_hidden_states",
  "encoder_hidden_states",
  "audio_encoder_hidden_states",
  "timestep",
  "audio_timestep",
  "sigma",
  "audio_sigma",
  "encoder_attention_mask",
  "audio_encoder_attention_mask",
  "num_frames",
  "height",
  "width",
  "fps",
  "audio_num_frames",
  "video_coords",
  "audio_coords",
  "isolate_modalities",
  "use_cross_timestep",
  "attention_kwargs",
)


def _tiny_transformer():
  from diffusers import LTX2VideoTransformer3DModel

  torch.manual_seed(0)
  transformer = LTX2VideoTransformer3DModel(
    in_channels=8,
    out_channels=8,
    patch_size=1,
    patch_size_t=1,
    num_attention_heads=4,
    attention_head_dim=8,
    cross_attention_dim=_INNER_DIM,
    vae_scale_factors=(1, 1, 1),
    pos_embed_max_pos=8,
    base_height=8,
    base_width=8,
    gated_attn=True,
    cross_attn_mod=True,
    audio_in_channels=8,
    audio_out_channels=8,
    audio_patch_size=1,
    audio_patch_size_t=1,
    audio_num_attention_heads=4,
    audio_attention_head_dim=8,
    audio_cross_attention_dim=_INNER_DIM,
    audio_scale_factor=1,
    audio_pos_embed_max_pos=8,
    audio_gated_attn=True,
    audio_cross_attn_mod=True,
    num_layers=2,
    caption_channels=_INNER_DIM,
    rope_type="split",
    use_prompt_embeddings=False,
    perturbed_attn=True,
    ff_bias=False,
    audio_ff_bias=False,
    use_prompt_adaln_single=True,
    use_keyframes_abs_pos_embedding=False,
  )
  return transformer.eval()


class _EchoLikePipe:
  """Minimal stand-in for `EchoModularPipeline`: a transformer, no guider by default."""

  def __init__(self, transformer, num_conditions=None):
    self.transformer = transformer
    if num_conditions is not None:
      self.guider = types.SimpleNamespace(num_conditions=num_conditions)

  def __call__(self, **kwargs):  # noqa: ARG002 - the cache adapter wraps this
    return None


def test_echo_prefix_does_not_collide_with_ltx():
  prefixes = BlockAdapterRegister.supported_pipelines()[1]
  assert "Echo" in prefixes
  matches = [name for name in prefixes if "EchoModularPipeline".startswith(name)]
  assert matches == ["Echo"], f"ambiguous adapter prefixes for Echo: {matches}"


def test_echo_is_registered_as_a_verified_modular_pipeline():
  from cache_dit.caching.block_adapters import echo_adapter

  # `"Echo"` maps to the Echo factory and is whitelisted, so `enable_cache` neither dispatches the
  # modular pipeline to another family nor warns about an unverified one.
  assert BlockAdapterRegister._adapters["Echo"] is echo_adapter
  assert "Echo" in BlockAdapterRegister._verified_modular_pipelines


def test_adapter_wiring():
  from cache_dit.caching.block_adapters import echo_adapter
  from cache_dit.caching.patch_functors import LTX2PatchFunctor

  pipe = _EchoLikePipe(_tiny_transformer())
  adapter = echo_adapter(pipe)
  assert adapter.blocks is pipe.transformer.transformer_blocks
  assert adapter.forward_pattern is ForwardPattern.Pattern_0
  assert isinstance(adapter.patch_functor, LTX2PatchFunctor)
  # Guidance-free: one transformer forward per DMD step.
  assert adapter.has_separate_cfg is False


def test_has_separate_cfg_follows_a_guider_when_present():
  from cache_dit.caching.block_adapters import echo_adapter

  assert echo_adapter(_EchoLikePipe(_tiny_transformer(), num_conditions=2)).has_separate_cfg is True


def test_enable_cache_keeps_the_echo_denoiser_inputs():
  from cache_dit.caching.block_adapters import echo_adapter

  transformer = _tiny_transformer()
  adapter = echo_adapter(_EchoLikePipe(transformer))
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=1,
      num_inference_steps=8,
      enable_separate_cfg=False,
      residual_diff_threshold=0.0,
    ),
  )
  parameters = set(inspect.signature(transformer.forward).parameters)
  missing = [name for name in _ECHO_DENOISER_INPUTS if name not in parameters]
  assert not missing, f"enable_cache dropped the Echo denoiser inputs {missing}"


def _echo_inputs(transformer):
  generator = torch.Generator().manual_seed(3)
  device = torch.device("cpu")
  return dict(
    hidden_states=torch.randn(1, _VIDEO_TOKENS, 8, generator=generator),
    audio_hidden_states=torch.randn(1, _AUDIO_TOKENS, 8, generator=generator),
    encoder_hidden_states=torch.randn(1, _TEXT_TOKENS, _INNER_DIM, generator=generator),
    audio_encoder_hidden_states=torch.randn(1, _TEXT_TOKENS, _INNER_DIM, generator=generator),
    timestep=torch.full((1, _VIDEO_TOKENS), 750.0),
    # Echo ships a per-token audio timestep; the scalar LTX-2 form must stay covered by its own test.
    audio_timestep=torch.full((1, _AUDIO_TOKENS), 500.0),
    sigma=torch.full((1, ), 0.75),
    audio_sigma=torch.full((1, ), 0.25),
    video_coords=transformer.rope.prepare_video_coords(1, 2, 2, 2, device),
    audio_coords=transformer.audio_rope.prepare_audio_coords(1, _AUDIO_TOKENS, device),
    fps=24.0,
    use_cross_timestep=True,
  )


def test_cache_run_is_bit_exact_when_no_step_is_cached():
  from cache_dit.caching.block_adapters import echo_adapter

  reference = _tiny_transformer()
  inputs = _echo_inputs(reference)
  with torch.no_grad():
    expected = [t for t in reference(**inputs, return_dict=False)]

  transformer = _tiny_transformer()
  transformer.load_state_dict(reference.state_dict())
  adapter = echo_adapter(_EchoLikePipe(transformer))
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=0,
      max_cached_steps=0,
      num_inference_steps=8,
      enable_separate_cfg=False,
      residual_diff_threshold=0.0,
    ),
  )
  with torch.no_grad():
    output = [t for t in transformer(**inputs, return_dict=False)]

  torch.testing.assert_close(output[0], expected[0], rtol=1e-5, atol=1e-6)
  torch.testing.assert_close(output[1], expected[1], rtol=1e-5, atol=1e-6)
