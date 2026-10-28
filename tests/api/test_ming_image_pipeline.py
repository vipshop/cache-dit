from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image
from torch import nn

from cache_dit.models.pipelines.ming_image.pipeline import MingImagePipeline
from cache_dit.models.pipelines.ming_image.transformer import MingImageTransformer2DModel


class _FakeVAE(nn.Module):

  def __init__(self):
    super().__init__()
    self.config = SimpleNamespace(
      input_channels=4,
      block_out_channels=[64, 128, 256, 512],
      temperal_downsample=[False, True, True],
      scaling_factor=1.0,
      shift_factor=0.0,
    )
    self.dtype = torch.float32
    self.last_reference = None

  def encode(self, image):
    self.last_reference = image
    latent = image.repeat_interleave(4, dim=1)
    latent = torch.nn.functional.avg_pool3d(latent, kernel_size=(1, 8, 8))
    return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

  def decode(self, latents, return_dict=False):
    images = latents[:, :4].squeeze(2)
    images = torch.nn.functional.interpolate(images, scale_factor=8, mode="nearest")
    return (images, )


def _transformer(layered=False):
  return MingImageTransformer2DModel(
    in_channels=16,
    dim=64,
    n_layers=1,
    n_refiner_layers=1,
    n_heads=2,
    n_kv_heads=2,
    cap_feat_dim=8,
    axes_dims=(8, 12, 12),
    axes_lens=(256, 64, 64),
    alignment_padding_mode="learned" if layered else "zero_masked",
    multi_frame_output=layered,
  )


def _pipeline(layered=False):
  vae = _FakeVAE()
  pipe = MingImagePipeline(
    transformer=_transformer(layered),
    scheduler=FlowMatchEulerDiscreteScheduler(shift=6.0, use_dynamic_shifting=False),
    vae=vae,
  )
  return pipe, vae


def _conditions():
  return [torch.randn(4, 8)], torch.randn(1, 2, 64)


def test_design_generates_one_rgba_image_without_mllm():
  pipe, _ = _pipeline()
  prompt_embeds, direct = _conditions()

  output = pipe(
    prompt_embeds=prompt_embeds,
    direct_condition=direct,
    height=32,
    width=32,
    num_inference_steps=2,
    generator=torch.Generator().manual_seed(0),
  )

  assert len(output.images) == 1
  assert output.images[0].mode == "RGBA"
  assert output.images[0].size == (32, 32)
  assert set(pipe.stage_durations) == {"conditioning", "denoising", "vae"}


def test_layered_pipeline_returns_composite_and_every_layer():
  pipe, vae = _pipeline(layered=True)
  prompt_embeds, direct = _conditions()
  reference = Image.new("RGBA", (32, 32), (255, 0, 0, 127))

  output = pipe(
    image=reference,
    prompt_embeds=prompt_embeds,
    direct_condition=direct,
    height=32,
    width=32,
    num_layers=2,
    num_inference_steps=2,
    guidance_scale=2.0,
    generator=torch.Generator().manual_seed(0),
  )

  assert len(output.images) == 3
  assert all(image.mode == "RGBA" for image in output.images)
  assert vae.last_reference.shape == (1, 4, 1, 32, 32)
  assert vae.last_reference[0, 3].min().item() < 0


def test_reference_aspect_ratio_drives_output_geometry():
  pipe, vae = _pipeline(layered=True)
  prompt_embeds, direct = _conditions()
  reference = Image.new("RGBA", (64, 32), (255, 0, 0, 255))

  output = pipe(
    image=reference,
    prompt_embeds=prompt_embeds,
    direct_condition=direct,
    height=64,
    width=64,
    num_layers=1,
    num_inference_steps=1,
    guidance_scale=1.0,
  )

  assert output.images[0].size == (96, 48)
  assert vae.last_reference.shape == (1, 4, 1, 48, 96)


def test_latent_output_uses_vae_resolution_and_frame_axis():
  pipe, _ = _pipeline()
  prompt_embeds, direct = _conditions()

  latents = pipe(
    prompt_embeds=prompt_embeds,
    direct_condition=direct,
    height=32,
    width=48,
    num_inference_steps=1,
    output_type="latent",
  ).images

  assert latents.shape == (1, 16, 1, 4, 6)
  assert latents.dtype == torch.float32


def test_input_constraints_fail_fast():
  pipe, _ = _pipeline()
  prompt_embeds, direct = _conditions()
  with pytest.raises(ValueError, match="negative_prompt"):
    pipe(prompt_embeds=prompt_embeds, direct_condition=direct, negative_prompt="bad")
  with pytest.raises(ValueError, match="exactly one output frame"):
    pipe(prompt_embeds=prompt_embeds, direct_condition=direct, num_layers=2)
  with pytest.raises(ValueError, match="multiples of 16"):
    pipe(prompt_embeds=prompt_embeds, direct_condition=direct, height=30)
  with pytest.raises(ValueError, match="direct_condition"):
    pipe(prompt_embeds=prompt_embeds)
  with pytest.raises(ValueError, match="supplied together"):
    pipe(query_hidden_states=torch.randn(1, 256, 2048))
  with pytest.raises(ValueError, match="not both"):
    pipe(
      prompt_embeds=prompt_embeds,
      direct_condition=direct,
      query_hidden_states=torch.randn(1, 256, 2048),
      direct_hidden_states=torch.randn(1, 1, 6144),
    )

  layered, _ = _pipeline(layered=True)
  with pytest.raises(ValueError, match="requires a reference"):
    layered(prompt_embeds=prompt_embeds, direct_condition=direct)
  with pytest.raises(ValueError, match="exactly one reference"):
    layered(
      image=[Image.new("RGBA", (32, 32)), Image.new("RGBA", (32, 32))],
      prompt_embeds=prompt_embeds,
      direct_condition=direct,
    )
