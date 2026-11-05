"""CPU tests for the MOVA fused stack and its block adapter.

They build a tiny stack (no weights, no GPU) and pin the three contracts that fail silently
otherwise: the per-block ``index`` (the cached wrapper collapses the list to one element), the
"tail block returns the audio stream unchanged" rule, and the explicit type guard that keeps
``get_adapter`` from falling through to the ``Wan`` prefix.
"""

import inspect

import pytest
import torch

from cache_dit.caching.block_adapters.adapters import mova_adapter
from cache_dit.caching.forward_pattern import ForwardPattern
from cache_dit.models.pipelines.mova.bridge import DualTowerConditionalBridge
from cache_dit.models.pipelines.mova.stack import MOVAFusedBlock, MOVAStack
from cache_dit.models.pipelines.mova.wan_video_dit import (
  WanAudioModel,
  WanModel,
  audio_freqs,
  video_freqs,
)

_VIDEO_CFG = dict(
  dim=32,
  in_dim=36,
  ffn_dim=64,
  out_dim=16,
  text_dim=16,
  freq_dim=8,
  eps=1e-6,
  patch_size=(1, 2, 2),
  num_heads=4,
  num_layers=3,
  has_image_input=False,
)
_AUDIO_CFG = dict(
  # `head_dim = dim // num_heads = 12` keeps the 1-D rotary table wide enough for `chunk(3)`; a
  # narrower table returns two chunks and the third buffer cannot be registered.
  dim=24,
  in_dim=8,
  ffn_dim=32,
  out_dim=8,
  text_dim=16,
  freq_dim=8,
  eps=1e-6,
  patch_size=(1, ),
  num_heads=2,
  num_layers=2,
  has_image_input=False,
  vae_type="dac",
)
_BRIDGE_CFG = dict(
  visual_layers=3,
  audio_layers=2,
  visual_hidden_dim=32,
  audio_hidden_dim=24,
  head_dim=8,
  interaction_strategy="full",
  apply_cross_rope=True,
)


def _tiny_stack() -> MOVAStack:
  """Build a deterministic tiny stack with a video-only tail block.

  :returns: A `MOVAStack` with 3 video layers and 2 audio layers.
  """

  torch.manual_seed(0)
  stack = MOVAStack(
    video_dit=WanModel(**_VIDEO_CFG),
    video_dit_2=WanModel(**_VIDEO_CFG),
    audio_dit=WanAudioModel(**_AUDIO_CFG),
    dual_tower_bridge=DualTowerConditionalBridge(**_BRIDGE_CFG),
  )
  return stack.eval()


def test_blocks_are_indexed_and_ordered():
  """Every fused block carries its own zero-based index (the wrapper hides the position)."""

  stack = _tiny_stack()
  assert [block.index for block in stack.blocks] == list(range(3))
  assert stack.config.num_blocks == 3
  assert stack.visible_layers() == 2
  assert all(isinstance(block, MOVAFusedBlock) for block in stack.blocks)


def test_tail_block_has_no_audio_and_passes_it_through():
  """The video-only tail returns the *same* audio tensor (``None`` would freeze its cache
  buffer)."""

  stack = _tiny_stack()
  tail = stack.blocks[2]
  assert tail.audio_block is None
  assert tail.a2v_conditioner is None and tail.v2a_conditioner is None

  visual = torch.randn(1, 4, 32)
  audio = torch.randn(1, 3, 16)
  with torch.no_grad():
    out_visual, out_audio = tail(
      visual,
      audio,
      torch.randn(1, 2, 32),
      torch.randn(1, 2, 16),
      torch.randn(1, 6, 32),
      torch.randn(1, 6, 24),
      # The rotary tables are complex (`head_dim / 2` wide), so build them with the released helpers
      # instead of guessing the width.
      video_freqs(stack.video_dit.freqs, 2, 2, 1, visual.device),
      audio_freqs(stack.audio_dit.freqs, 3, audio.device),
      None,
      None,
      False,
      None,
    )
  assert out_visual.shape == visual.shape
  assert out_audio is audio


def test_first_block_conditions_both_directions():
  """Layer 0 carries both bridge directions and both towers' blocks."""

  stack = _tiny_stack()
  block = stack.blocks[0]
  assert block.audio_block is not None
  assert block.a2v_conditioner is not None and block.v2a_conditioner is not None
  assert stack.blocks[0].video_block_2 is not None


def test_stack_forward_returns_both_streams():
  """One step returns the video and audio velocities in latent layout."""

  stack = _tiny_stack()
  visual_latents = torch.randn(1, 36, 3, 4, 4)
  audio_latents = torch.randn(1, 8, 5)
  with torch.no_grad():
    visual_out, audio_out = stack(visual_latents, audio_latents, torch.randn(1, 4, 16),
                                  torch.tensor([500.0]))
  # The video input carries the conditioning channels (36 = 16 latents + 20), the velocity keeps the
  # latent width (`out_dim`), which is what the scheduler steps.
  assert visual_out.shape == (1, 16, 3, 4, 4)
  assert audio_out.shape == audio_latents.shape
  assert torch.isfinite(visual_out).all() and torch.isfinite(audio_out).all()


def test_enumerate_is_not_used_for_the_block_index():
  """The index must come from the block, never from enumerating the (wrapped) list."""

  for func in (MOVAStack.forward, MOVAStack._run_blocks):
    source = inspect.getsource(func)
    assert "enumerate" not in source, f"{func.__name__} must not derive the block index from enumerate"


def test_stack_declares_the_repeated_blocks():
  """Regional quantization and block-level compilation discover the stack from this attribute."""

  assert MOVAStack._repeated_blocks == ["MOVAFusedBlock"]


def test_trainable_condition_scale_is_rejected():
  """A learned per-model conditioning scale has no place in the fused block's constant path."""

  bridge = DualTowerConditionalBridge(**{**_BRIDGE_CFG, "trainable_condition_scale": True})
  with pytest.raises(NotImplementedError):
    MOVAStack(
      video_dit=WanModel(**_VIDEO_CFG),
      video_dit_2=None,
      audio_dit=WanAudioModel(**_AUDIO_CFG),
      dual_tower_bridge=bridge,
    )


class _Wrapper:
  """Minimal pipeline stand-in exposing only a `transformer`."""

  def __init__(self, transformer):
    self.transformer = transformer


def test_adapter_contract():
  """The adapter binds `blocks` with `Pattern_0`, separate CFG and the vendored stack."""

  stack = _tiny_stack()
  adapter = mova_adapter(_Wrapper(stack))
  assert adapter.forward_pattern == ForwardPattern.Pattern_0
  assert adapter.has_separate_cfg is True
  assert adapter.check_forward_pattern is True
  assert adapter.blocks_name == "blocks"


def test_adapter_rejects_a_foreign_transformer():
  """A non-MOVAStack must fail loudly instead of falling through to the `Wan` adapter."""

  with pytest.raises(TypeError):
    mova_adapter(_Wrapper(WanModel(**_VIDEO_CFG)))


def _cache_inputs(seed: int = 11):
  """Fixed stack inputs shared by the cached and the plain run.

  :param seed: Input seed.
  :returns: Positional inputs of `MOVAStack.forward`.
  """

  generator = torch.Generator().manual_seed(seed)
  return (
    torch.randn(1, 36, 3, 4, 4, generator=generator),
    torch.randn(1, 8, 5, generator=generator),
    torch.randn(1, 4, 16, generator=generator),
    torch.tensor([500.0]),
  )


def _fake_adapter(stack: MOVAStack):
  """Adapt a bare stack the way the transformer-only entry point does.

  `FakeDiffusionPipeline` is what makes the cache contexts persistent; with a hand-rolled pipe object
  the wrapper logs "Cache context not exist" and silently recomputes every step.

  :param stack: The stack to adapt.
  :returns: The block adapter.
  """

  from cache_dit.caching.block_adapters.block_adapters import FakeDiffusionPipeline

  return mova_adapter(FakeDiffusionPipeline(stack))


def _cache_config(**kwargs):
  """Cache config that keeps both streams in the picture.

  :param kwargs: Overrides of the shared defaults.
  :returns: The `DBCacheConfig`.
  """

  from cache_dit import DBCacheConfig

  base = dict(Fn_compute_blocks=1,
              Bn_compute_blocks=1,
              max_warmup_steps=0,
              num_inference_steps=2,
              enable_separate_cfg=True)
  base.update(kwargs)
  return DBCacheConfig(**base)


def test_cached_run_is_bit_exact_for_both_streams_when_no_step_is_cached():
  """With caching disabled the wrapper must be numerically transparent for video **and** audio.

  The audio stream travels through the same block as `encoder_hidden_states`, so a wrapper that only
  forwards the video tensor would keep every shape legal while quietly dropping the audio residual.
  """

  import copy

  import cache_dit

  reference = _tiny_stack()
  stack = copy.deepcopy(reference)
  inputs = _cache_inputs()
  with torch.no_grad():
    expected_video, expected_audio = reference(*inputs)

  cache_dit.enable_cache(_fake_adapter(stack),
                         cache_config=_cache_config(max_cached_steps=0,
                                                    residual_diff_threshold=0.0))
  with torch.no_grad():
    actual_video, actual_audio = stack(*inputs)

  assert torch.equal(actual_video,
                     expected_video), "the cached run must reproduce the video velocity"
  assert torch.equal(actual_audio,
                     expected_audio), "the cached run must reproduce the audio velocity"


def test_forced_cache_hit_replays_both_streams():
  """A forced hit must add the audio residual too, not only the video one.

  Every forward sees identical inputs, so a correctly cached step replays the previous full output
  on both streams. If the audio residual were dropped, the audio output would be the partial (Fn+Bn)
  result of a step whose later blocks never ran.
  """

  import copy

  import cache_dit

  reference = _tiny_stack()
  stack = copy.deepcopy(reference)
  inputs = _cache_inputs()
  cache_dit.enable_cache(_fake_adapter(stack),
                         cache_config=_cache_config(residual_diff_threshold=1e9))

  # With three fused blocks, `Fn=1` + `Bn=1` leave block 1 as the only skippable one, so counting it
  # separates "served from the cache" from "replayed through the originals".
  middle_calls = [0]
  stack.blocks[1].register_forward_pre_hook(
    lambda module, args, kwargs=None: middle_calls.__setitem__(0, middle_calls[0] + 1))

  outputs = []
  for _ in range(4):  # two steps, one conditional and one unconditional forward each
    with torch.no_grad():
      outputs.append(stack(*inputs))

  cached = None
  for _, module in stack.named_modules():
    if hasattr(module, "_accumulated_cached_steps"):
      cached = int(module._accumulated_cached_steps)
      break
  assert cached is not None and cached > 0, "the forced threshold must produce a cached step"
  assert middle_calls[0] == 2, (
    f"the middle block ran {middle_calls[0]} times over four identical forwards; a served step must "
    "skip it, so only the two warmup halves may reach it")

  for step, half in ((2, 0), (3, 1)):
    hit_video, hit_audio = outputs[step]
    exp_video, exp_audio = outputs[half]
    assert hit_audio is not None, "the audio stream must survive the cache"
    torch.testing.assert_close(hit_video, exp_video, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(hit_audio, exp_audio, atol=1e-5, rtol=1e-4)
