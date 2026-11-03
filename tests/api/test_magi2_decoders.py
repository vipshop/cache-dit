"""CPU contracts of the MAGI-2 decoder glue (no weights, no GPU)."""

import torch

from cache_dit.models.pipelines.magi2.pipeline import preview_output_to_numpy, preview_output_to_pil
from cache_dit.models.pipelines.magi2.wan_vae import (
  WAN22_LATENT_MEAN,
  WAN22_LATENT_STD,
  denormalize_wan22_latent,
  wan22_vae_key_mapping,
)


def test_stage_keys_follow_the_released_nesting():
  """Encoder/decoder stage keys must land on the diffusers block lists, not on a second stage
  name."""

  renamed = wan22_vae_key_mapping({
    "encoder.downsamples.1.downsamples.0.residual.0.gamma": 1,
    "encoder.downsamples.1.downsamples.2.time_conv.weight": 2,
    "decoder.upsamples.0.upsamples.3.time_conv.weight": 3,
    "conv1.weight": 4,
    "encoder.middle.1.to_qkv.weight": 5,
  })

  assert renamed["encoder.down_blocks.1.resnets.0.norm1.gamma"] == 1
  assert renamed["encoder.down_blocks.1.downsampler.time_conv.weight"] == 2
  assert renamed["decoder.up_blocks.0.upsampler.time_conv.weight"] == 3
  assert renamed["quant_conv.weight"] == 4
  assert renamed["encoder.mid_block.attentions.0.to_qkv.weight"] == 5


def test_latent_statistics_have_one_entry_per_channel():
  """The de-normalization must be built from 48 per-channel statistics."""

  assert len(WAN22_LATENT_MEAN) == 48
  assert len(WAN22_LATENT_STD) == 48

  latent = torch.zeros(1, 48, 1, 2, 2)
  restored = denormalize_wan22_latent(latent)
  assert restored.shape == latent.shape
  assert torch.equal(restored[0, :, 0, 0, 0], torch.tensor(WAN22_LATENT_MEAN))


def test_decoder_output_converts_to_frames():
  """The ``[-1, 1]`` decoder output must become uint8 frames, not a double-scaled image."""

  video = torch.zeros(1, 3, 2, 4, 6)
  frames = preview_output_to_pil(video)
  assert len(frames) == 2
  assert frames[0].size == (6, 4)
  assert frames[0].mode == "RGB"

  array = preview_output_to_numpy(video)
  assert array.shape == (2, 4, 6, 3)
  assert array.dtype.name == "uint8"
  assert int(array.max()) == 127
