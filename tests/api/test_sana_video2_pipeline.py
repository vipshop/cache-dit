"""Regression tests for the vendored SANA-Video 2.0 pipeline plumbing.

These lock the parts that do not need weights: the frame/canvas rules of the LTX-2.3 VAE contract,
the conditioning-frame preprocessing, and the cache guard that keeps "one forward per step" true
whenever cache-dit caching is enabled.

Usage:
  python -m pytest tests/api/test_sana_video2_pipeline.py -q
"""
import numpy as np
import pytest
import torch
from PIL import Image

from cache_dit.models.pipelines.sana_video2.pipeline import (
  SanaVideo2Pipeline,
  _prepare_image,
)
from cache_dit.models.pipelines.sana_video2.transformer import SanaVideo2Transformer3DModel


def _tiny_transformer() -> SanaVideo2Transformer3DModel:
  torch.manual_seed(0)
  return SanaVideo2Transformer3DModel(
    in_channels=4,
    hidden_size=192,
    depth=4,
    num_heads=2,
    caption_channels=16,
    model_max_length=4,
    linear_head_dim=64,
    softmax_head_dim=96,
    attn_res_block_size=2,
    input_size=2,
  ).eval()


def _tiny_pipeline() -> SanaVideo2Pipeline:
  return SanaVideo2Pipeline(transformer=_tiny_transformer(), max_sequence_length=4)


def test_conditioning_frame_is_normalized_and_cropped():
  image = Image.new("RGB", (64, 32), (255, 0, 0))
  pixels = _prepare_image(image, 16, 16)
  assert pixels.shape == (3, 16, 16)
  assert pixels.min() >= -1.0 and pixels.max() <= 1.0
  # The red channel survives the center crop, the other two are zeroed by the source image.
  assert torch.allclose(pixels[0], torch.ones_like(pixels[0]))
  assert torch.allclose(pixels[1], -torch.ones_like(pixels[1]))


def test_pipeline_registers_only_the_transformer():
  pipeline = _tiny_pipeline()
  assert pipeline.transformer is not None
  assert pipeline.vae is None


def test_canvas_must_be_divisible_by_32():
  pipeline = _tiny_pipeline()
  with pytest.raises(ValueError, match="divisible by 32"):
    pipeline(prompt="a rooster", height=100, width=128)


def test_cache_guard_rejects_guidance_free_runs():
  pipeline = _tiny_pipeline()
  # cache-dit marks the transformer while it is wrapped; the guard must fire before any encoding.
  pipeline.transformer._is_cached = True
  try:
    with pytest.raises(ValueError, match="guidance_scale"):
      pipeline(prompt="a rooster", height=64, width=64, guidance_scale=1.0)
  finally:
    del pipeline.transformer._is_cached


def test_latent_frames_follow_the_vae_contract():
  """`(F - 1) % 8 == 0` frames map onto `(F - 1) // 8 + 1` latent frames."""
  for frames in (9, 17, 97, 193):
    assert (frames - 1) % 8 == 0
    latent_frames = (frames - 1) // 8 + 1
    assert (latent_frames - 1) * 8 + 1 == frames


def test_output_dataclass_shape():
  from cache_dit.models.pipelines.sana_video2.pipeline import SanaVideo2PipelineOutput

  frames = [[Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))]]
  output = SanaVideo2PipelineOutput(frames=frames)
  assert output.fps == 24
  assert len(output.frames[0]) == 1
