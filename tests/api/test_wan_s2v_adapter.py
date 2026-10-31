"""Regression tests for the Wan2.2-S2V cache integration.

Wan2.2-S2V injects audio residuals after 12 of the 40 blocks, and the upstream loop does it in the
loop body. Cache-dit replaces `transformer.blocks` with a single-element wrapper driven by one
shared kwargs set, so a loop-body injection survives exactly one call per step and every cached
step silently loses the audio conditioning. The injection therefore lives inside the block, keyed by
`block.block_idx`, and the transformer's own loop must stay free of per-layer logic.

Usage:
  python -m pytest tests/api/test_wan_s2v_adapter.py -q
"""

import inspect
import types

import torch

from cache_dit.caching.forward_pattern import ForwardPattern
from cache_dit.models.pipelines.wan_s2v.transformer import (
  AudioInjector,
  WanS2VTransformer3DModel,
)

NUM_LAYERS = 8
AUDIO_INJECT_LAYERS = (0, 2, 5, 7)
LATENT_CHANNELS = 16
GRID = 8
VIDEO_FRAMES = 5
# One audio frame per latent frame. The causal encoder downsamples the 73 motion frames plus
# `AUDIO_FRAMES` bucket frames by 4 and then drops the motion frames, so 20 bucket frames give
# exactly 5 latent frames.
AUDIO_FRAMES = 20
MOTION_FRAMES = 73
LATENT_MOTION_FRAMES = 19


def _tiny_transformer() -> WanS2VTransformer3DModel:
  torch.manual_seed(0)
  return WanS2VTransformer3DModel(
    patch_size=(1, 2, 2),
    num_attention_heads=4,
    attention_head_dim=16,
    in_channels=LATENT_CHANNELS,
    out_channels=LATENT_CHANNELS,
    text_dim=32,
    freq_dim=32,
    audio_dim=1024,
    audio_inject_layers=AUDIO_INJECT_LAYERS,
    enable_adain=True,
    adain_mode="attn_norm",
    pose_dim=16,
    ffn_dim=128,
    num_layers=NUM_LAYERS,
    num_weighted_avg_layers=25,
    cross_attn_norm=True,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
    added_kv_proj_dim=None,
    rope_max_seq_len=1024,
    enable_framepack=True,
    framepack_drop_mode="padd",
    add_last_motion=True,
    zero_timestep=True,
  ).eval()


def _inputs(seed: int = 0) -> dict:
  generator = torch.Generator().manual_seed(seed)
  return dict(
    hidden_states=torch.randn(1, LATENT_CHANNELS, VIDEO_FRAMES, GRID, GRID, generator=generator),
    timestep=torch.tensor([500.0]),
    encoder_hidden_states=torch.randn(1, 8, 32, generator=generator),
    motion_latents=torch.randn(1,
                               LATENT_CHANNELS,
                               LATENT_MOTION_FRAMES,
                               GRID,
                               GRID,
                               generator=generator),
    audio_embeds=torch.randn(1, 25, 1024, AUDIO_FRAMES, generator=generator),
    image_latents=torch.randn(1, LATENT_CHANNELS, 1, GRID, GRID, generator=generator),
    pose_latents=torch.zeros(1, 16, VIDEO_FRAMES, GRID, GRID),
    motion_frames=[MOTION_FRAMES, LATENT_MOTION_FRAMES],
    drop_motion_frames=False,
    return_dict=False,
  )


def test_blocks_carry_their_index():
  """The cached wrapper hides the loop position, so the block itself has to know it."""
  transformer = _tiny_transformer()
  assert [block.block_idx for block in transformer.blocks] == list(range(NUM_LAYERS))
  assert transformer.audio_injector.injected_block_id == {
    layer: index
    for index, layer in enumerate(AUDIO_INJECT_LAYERS)
  }


def test_audio_injection_runs_once_per_inject_layer():
  """Injection must fire for exactly the configured layers, in ascending order, from the blocks."""
  transformer = _tiny_transformer()
  calls = []
  original_forward = AudioInjector.forward

  def _record(self, block_idx, hidden_states, audio_ctx):
    calls.append(block_idx)
    return original_forward(self, block_idx, hidden_states, audio_ctx)

  AudioInjector.forward = _record
  try:
    with torch.no_grad():
      out = transformer(**_inputs())[0]
  finally:
    AudioInjector.forward = original_forward

  assert calls == list(AUDIO_INJECT_LAYERS)
  assert out.shape == (1, LATENT_CHANNELS, VIDEO_FRAMES, GRID, GRID)


def test_transformer_loop_has_no_per_layer_logic():
  """A `for block in self.blocks` body that touches the injector would be skipped on cached
  steps."""
  source = inspect.getsource(WanS2VTransformer3DModel.forward)
  assert "audio_injector(" not in source
  assert "enumerate(self.blocks)" not in source


def test_adapter_uses_pattern_2_with_separate_cfg():
  from cache_dit.caching.block_adapters.adapters import wan_s2v_adapter

  transformer = _tiny_transformer()
  adapter = wan_s2v_adapter(types.SimpleNamespace(transformer=transformer))

  assert adapter.forward_pattern is ForwardPattern.Pattern_2
  assert adapter.transformer is transformer
  assert adapter.blocks is transformer.blocks
  assert adapter.blocks_name == "blocks"
  assert adapter.check_forward_pattern is True
  # `guidance_scale > 1` runs the conditional and the unconditional pass as two transformer calls.
  assert adapter.has_separate_cfg is True


def test_adapter_rejects_a_foreign_transformer():
  from cache_dit.caching.block_adapters.adapters import wan_s2v_adapter

  with torch.no_grad():
    foreign = torch.nn.Module()
    foreign.blocks = torch.nn.ModuleList([torch.nn.Identity()])
    import pytest

    with pytest.raises(AssertionError, match="WanS2VTransformer3DModel"):
      wan_s2v_adapter(types.SimpleNamespace(transformer=foreign))


def test_registry_prefers_wan_s2v_over_the_wan_family():
  """`_relaxed_assert` only warns for cache-dit-owned modules, so the prefix order is the guard."""
  from cache_dit.caching.block_adapters import BlockAdapterRegister

  keys = list(BlockAdapterRegister._adapters.keys())
  assert keys.index("WanS2V") < keys.index("WanSpeechToVideo") < keys.index("WanAnimate2")
  assert keys.index("WanAnimate2") < keys.index("Wan")


def test_pipeline_and_transformer_names_resolve_to_the_s2v_adapter():
  """The registry matches the *class name* prefix, so both names must hit this factory first."""
  from cache_dit.caching.block_adapters import BlockAdapterRegister
  from cache_dit.caching.block_adapters.adapters import wan_s2v_adapter
  from cache_dit.models.pipelines.wan_s2v import (
    WanS2VTransformer3DModel,
    WanSpeechToVideoPipeline,
  )

  for cls in (WanSpeechToVideoPipeline, WanS2VTransformer3DModel):
    first_match = next(p for p in BlockAdapterRegister._adapters if cls.__name__.startswith(p))
    assert BlockAdapterRegister._adapters[first_match] is wan_s2v_adapter, cls.__name__
