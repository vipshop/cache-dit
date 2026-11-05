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
