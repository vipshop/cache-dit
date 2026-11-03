# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the cache-dit project
"""Wan2.2 VAE loading for MAGI-2 Preview.

The released checkpoint ships the Wan2.2 video VAE (``vae/Wan2.2_VAE.pth``) in the official Wan
naming. Diffusers already implements the same architecture as `AutoencoderKLWan`, so the weights are
converted here instead of vendoring another decoder; the conversion follows
``diffusers/scripts/convert_wan_to_diffusers.py::convert_vae_22`` (the released config and the latent
statistics are copied from that script, not re-derived).

MAGI-2 renders video through a distilled decoder (``turbo_vae``), so decoding the released latents
with the full Wan2.2 VAE is faithful to the same latent space but not bit-identical to that distilled
decoder. Every comparison row therefore has to use this single decode path.
"""

from pathlib import Path

import torch

__all__ = [
  "WAN22_LATENT_MEAN",
  "WAN22_LATENT_STD",
  "WAN22_VAE_CONFIG",
  "load_wan22_vae",
  "wan22_vae_key_mapping",
]

# Copied from diffusers `scripts/convert_wan_to_diffusers.py::vae22_diffusers_config`.
WAN22_VAE_CONFIG = {
  "base_dim": 160,
  "decoder_base_dim": 256,
  "z_dim": 48,
  "is_residual": True,
  "in_channels": 12,
  "out_channels": 12,
  "scale_factor_temporal": 4,
  "scale_factor_spatial": 16,
  "patch_size": 2,
  "dim_mult": [1, 2, 4, 4],
  "num_res_blocks": 2,
  "attn_scales": [],
  # The conversion script ships `[True, True, False]`, which leaves the two temporal convolutions on
  # stages this checkpoint does not contain (mismatched shapes). This checkpoint places them on the
  # two deeper stages, and `[False, True, True]` is the only ordering under which all 196 released
  # tensors match the module by name *and* shape - verified before the weights are assigned.
  "temperal_downsample": [False, True, True],
}

WAN22_LATENT_MEAN = [
  -0.2289,
  -0.0052,
  -0.1323,
  -0.2339,
  -0.2799,
  0.0174,
  0.1838,
  0.1557,
  -0.1382,
  0.0542,
  0.2813,
  0.0891,
  0.1570,
  -0.0098,
  0.0375,
  -0.1825,
  -0.2246,
  -0.1207,
  -0.0698,
  0.5109,
  0.2665,
  -0.2108,
  -0.2158,
  0.2502,
  -0.2055,
  -0.0322,
  0.1109,
  0.1567,
  -0.0729,
  0.0899,
  -0.2799,
  -0.1230,
  -0.0313,
  -0.1649,
  0.0117,
  0.0723,
  -0.2839,
  -0.2083,
  -0.0520,
  0.3748,
  0.0152,
  0.1957,
  0.1433,
  -0.2944,
  0.3573,
  -0.0548,
  -0.1681,
  -0.0667,
]

WAN22_LATENT_STD = [
  0.4765,
  1.0364,
  0.4514,
  1.1677,
  0.5313,
  0.4990,
  0.4818,
  0.5013,
  0.8158,
  1.0344,
  0.5894,
  1.0901,
  0.6885,
  0.6165,
  0.8454,
  0.4978,
  0.5759,
  0.3523,
  0.7135,
  0.6804,
  0.5833,
  1.4146,
  0.8986,
  0.5659,
  0.7069,
  0.5338,
  0.4889,
  0.4917,
  0.4069,
  0.4999,
  0.6866,
  0.4093,
  0.5709,
  0.6065,
  0.6415,
  0.4944,
  0.5726,
  1.2042,
  0.5458,
  1.6887,
  0.3971,
  1.0600,
  0.3943,
  0.5537,
  0.5444,
  0.4089,
  0.7468,
  0.7744,
]

_MIDDLE_MAPPING = {
  "encoder.middle.0.residual.0.gamma": "encoder.mid_block.resnets.0.norm1.gamma",
  "encoder.middle.0.residual.2.bias": "encoder.mid_block.resnets.0.conv1.bias",
  "encoder.middle.0.residual.2.weight": "encoder.mid_block.resnets.0.conv1.weight",
  "encoder.middle.0.residual.3.gamma": "encoder.mid_block.resnets.0.norm2.gamma",
  "encoder.middle.0.residual.6.bias": "encoder.mid_block.resnets.0.conv2.bias",
  "encoder.middle.0.residual.6.weight": "encoder.mid_block.resnets.0.conv2.weight",
  "encoder.middle.2.residual.0.gamma": "encoder.mid_block.resnets.1.norm1.gamma",
  "encoder.middle.2.residual.2.bias": "encoder.mid_block.resnets.1.conv1.bias",
  "encoder.middle.2.residual.2.weight": "encoder.mid_block.resnets.1.conv1.weight",
  "encoder.middle.2.residual.3.gamma": "encoder.mid_block.resnets.1.norm2.gamma",
  "encoder.middle.2.residual.6.bias": "encoder.mid_block.resnets.1.conv2.bias",
  "encoder.middle.2.residual.6.weight": "encoder.mid_block.resnets.1.conv2.weight",
  "decoder.middle.0.residual.0.gamma": "decoder.mid_block.resnets.0.norm1.gamma",
  "decoder.middle.0.residual.2.bias": "decoder.mid_block.resnets.0.conv1.bias",
  "decoder.middle.0.residual.2.weight": "decoder.mid_block.resnets.0.conv1.weight",
  "decoder.middle.0.residual.3.gamma": "decoder.mid_block.resnets.0.norm2.gamma",
  "decoder.middle.0.residual.6.bias": "decoder.mid_block.resnets.0.conv2.bias",
  "decoder.middle.0.residual.6.weight": "decoder.mid_block.resnets.0.conv2.weight",
  "decoder.middle.2.residual.0.gamma": "decoder.mid_block.resnets.1.norm1.gamma",
  "decoder.middle.2.residual.2.bias": "decoder.mid_block.resnets.1.conv1.bias",
  "decoder.middle.2.residual.2.weight": "decoder.mid_block.resnets.1.conv1.weight",
  "decoder.middle.2.residual.3.gamma": "decoder.mid_block.resnets.1.norm2.gamma",
  "decoder.middle.2.residual.6.bias": "decoder.mid_block.resnets.1.conv2.bias",
  "decoder.middle.2.residual.6.weight": "decoder.mid_block.resnets.1.conv2.weight",
  "encoder.middle.1.norm.gamma": "encoder.mid_block.attentions.0.norm.gamma",
  "encoder.middle.1.to_qkv.weight": "encoder.mid_block.attentions.0.to_qkv.weight",
  "encoder.middle.1.to_qkv.bias": "encoder.mid_block.attentions.0.to_qkv.bias",
  "encoder.middle.1.proj.weight": "encoder.mid_block.attentions.0.proj.weight",
  "encoder.middle.1.proj.bias": "encoder.mid_block.attentions.0.proj.bias",
  "decoder.middle.1.norm.gamma": "decoder.mid_block.attentions.0.norm.gamma",
  "decoder.middle.1.to_qkv.weight": "decoder.mid_block.attentions.0.to_qkv.weight",
  "decoder.middle.1.to_qkv.bias": "decoder.mid_block.attentions.0.to_qkv.bias",
  "decoder.middle.1.proj.weight": "decoder.mid_block.attentions.0.proj.weight",
  "decoder.middle.1.proj.bias": "decoder.mid_block.attentions.0.proj.bias",
  "encoder.head.0.gamma": "encoder.norm_out.gamma",
  "encoder.head.2.bias": "encoder.conv_out.bias",
  "encoder.head.2.weight": "encoder.conv_out.weight",
  "decoder.head.0.gamma": "decoder.norm_out.gamma",
  "decoder.head.2.bias": "decoder.conv_out.bias",
  "decoder.head.2.weight": "decoder.conv_out.weight",
  "conv1.weight": "quant_conv.weight",
  "conv1.bias": "quant_conv.bias",
  "conv2.weight": "post_quant_conv.weight",
  "conv2.bias": "post_quant_conv.bias",
  "encoder.conv1.weight": "encoder.conv_in.weight",
  "encoder.conv1.bias": "encoder.conv_in.bias",
  "decoder.conv1.weight": "decoder.conv_in.weight",
  "decoder.conv1.bias": "decoder.conv_in.bias",
}

_RESIDUAL_KEYS = {
  ".residual.0.gamma": ".norm1.gamma",
  ".residual.2.weight": ".conv1.weight",
  ".residual.2.bias": ".conv1.bias",
  ".residual.3.gamma": ".norm2.gamma",
  ".residual.6.weight": ".conv2.weight",
  ".residual.6.bias": ".conv2.bias",
  ".shortcut.weight": ".conv_shortcut.weight",
  ".shortcut.bias": ".conv_shortcut.bias",
}


def _rename_stage(key: str, stage: str) -> str:
  """Rename one ``downsamples`` / ``upsamples`` key into the diffusers layout.

  The anchored prefix (``encoder.downsamples.``) becomes the block list and the *inner* unanchored
  occurrence becomes ``resnets`` (residual keys) or ``downsampler``/``upsampler`` (resample keys),
  which is what the released nesting maps onto.

  :param key: Original checkpoint key.
  :param stage: ``"down"`` or ``"up"``.
  :returns: Diffusers key.
  """

  blocks = "down_blocks" if stage == "down" else "up_blocks"
  resampler = "downsampler" if stage == "down" else "upsampler"
  owner = "encoder" if stage == "down" else "decoder"
  new_key = key.replace(f"{owner}.{stage}samples.", f"{owner}.{blocks}.")
  if "residual" in new_key or "shortcut" in new_key:
    new_key = new_key.replace(f".{stage}samples.", ".resnets.")
    for suffix, replacement in _RESIDUAL_KEYS.items():
      new_key = new_key.replace(suffix, replacement)
    return new_key
  parts = new_key.split(".")
  if len(parts) >= 5 and parts[3] == f"{stage}samples":
    return ".".join(parts[:3] + [resampler] + parts[5:])
  return new_key


def wan22_vae_key_mapping(state_dict: dict) -> dict:
  """Map the released Wan2.2 VAE checkpoint onto `AutoencoderKLWan` parameter names.

  :param state_dict: Released ``vae/Wan2.2_VAE.pth`` state dict.
  :returns: Renamed state dict.
  """

  renamed = {}
  for key, value in state_dict.items():
    if key in _MIDDLE_MAPPING:
      renamed[_MIDDLE_MAPPING[key]] = value
    elif key.startswith("encoder.downsamples."):
      renamed[_rename_stage(key, "down")] = value
    elif key.startswith("decoder.upsamples."):
      renamed[_rename_stage(key, "up")] = value
    else:
      renamed[key] = value
  return renamed


def denormalize_wan22_latent(latent: torch.Tensor) -> torch.Tensor:
  """Map a normalized MAGI-2 video latent back to the VAE latent scale.

  The sampler runs in the Wan2.2 normalized space (`(z - mean) / std`); `AutoencoderKLWan.decode`
  expects the raw VAE latent, so the statistics are applied here.

  :param latent: ``[B, 48, T, H, W]`` normalized latent.
  :returns: Raw VAE latent, same shape.
  """

  shape = (1, -1) + (1, ) * (latent.ndim - 2)
  mean = torch.tensor(WAN22_LATENT_MEAN, dtype=latent.dtype, device=latent.device).view(shape)
  std = torch.tensor(WAN22_LATENT_STD, dtype=latent.dtype, device=latent.device).view(shape)
  return latent * std + mean


def load_wan22_vae(path, *, dtype=torch.float32, device="cpu"):
  """Build `AutoencoderKLWan` and strictly load the released Wan2.2 VAE weights.

  :param path: ``vae/Wan2.2_VAE.pth`` of the MAGI-2 preview checkpoint.
  :param dtype: Parameter dtype.
  :param device: Device to materialize the decoder on.
  :returns: Loaded `AutoencoderKLWan`.
  :raises FileNotFoundError: When ``path`` does not exist.
  """

  from diffusers import AutoencoderKLWan

  path = Path(path)
  if not path.is_file():
    raise FileNotFoundError(f"MAGI-2 image VAE does not exist: {path}")
  state_dict = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
  vae = AutoencoderKLWan(
    **WAN22_VAE_CONFIG,
    latents_mean=list(WAN22_LATENT_MEAN),
    latents_std=list(WAN22_LATENT_STD),
  )
  vae.load_state_dict(wan22_vae_key_mapping(state_dict), strict=True, assign=True)
  return vae.to(device=device, dtype=dtype).eval()
