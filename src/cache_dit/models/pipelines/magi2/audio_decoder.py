# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the cache-dit project
"""Stable Audio Oobleck decoder for MAGI-2 Preview.

The released checkpoint embeds Stable Audio's Oobleck VAE under the ``pretransform.model`` prefix
with ``stable-audio-tools`` module names. Diffusers ships the same architecture as
`AutoencoderOobleck`, so only the decoder tensors are renamed and loaded here instead of vendoring a
second audio VAE. The key mapping follows
``vllm-omni/vllm_omni/diffusion/models/magi2/audio_decoder.py`` (Apache-2.0, SandAI), which is the
reference implementation this model ships with.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn

__all__ = [
  "Magi2AudioDecoder",
  "convert_stable_audio_decoder_key",
  "convert_stable_audio_decoder_state_dict",
]

_CHECKPOINT_PREFIX = "pretransform.model.decoder."
_BLOCK_COMPONENTS = {
  0: "snake1",
  1: "conv_t1",
}
_RESIDUAL_COMPONENTS = {
  0: "snake1",
  1: "conv1",
  2: "snake2",
  3: "conv2",
}


def convert_stable_audio_decoder_key(key: str):
  """Map one Stable Audio decoder key onto Diffusers' Oobleck decoder.

  :param key: Checkpoint key.
  :returns: Diffusers key, or ``None`` for a tensor outside the decoder namespace.
  :raises ValueError: When a decoder key does not match the released layout.
  """

  if not key.startswith(_CHECKPOINT_PREFIX):
    return None
  suffix = key[len(_CHECKPOINT_PREFIX):]
  parts = suffix.split(".")
  if len(parts) < 3 or parts[0] != "layers":
    raise ValueError(f"Unsupported Stable Audio decoder key: {key}")

  try:
    layer = int(parts[1])
  except ValueError as exc:
    raise ValueError(f"Unsupported Stable Audio decoder key: {key}") from exc
  tail = parts[2:]

  if layer == 0 and len(tail) == 1:
    return f"conv1.{tail[0]}"
  if 1 <= layer <= 5:
    block = layer - 1
    if len(tail) == 3 and tail[0] == "layers":
      try:
        component = _BLOCK_COMPONENTS[int(tail[1])]
      except (KeyError, ValueError) as exc:
        raise ValueError(f"Unsupported Stable Audio decoder key: {key}") from exc
      return f"block.{block}.{component}.{tail[2]}"
    if len(tail) == 5 and tail[0] == "layers" and tail[2] == "layers":
      try:
        residual = int(tail[1]) - 1
        component = _RESIDUAL_COMPONENTS[int(tail[3])]
      except (KeyError, ValueError) as exc:
        raise ValueError(f"Unsupported Stable Audio decoder key: {key}") from exc
      if residual not in (1, 2, 3):
        raise ValueError(f"Unsupported Stable Audio decoder key: {key}")
      return f"block.{block}.res_unit{residual}.{component}.{tail[4]}"
  if layer == 6 and len(tail) == 1:
    return f"snake1.{tail[0]}"
  if layer == 7 and len(tail) == 1:
    return f"conv2.{tail[0]}"
  raise ValueError(f"Unsupported Stable Audio decoder key: {key}")


def convert_stable_audio_decoder_state_dict(state_dict: dict) -> dict:
  """Convert and filter a Stable Audio checkpoint state dict.

  :param state_dict: Released decoder tensors.
  :returns: Tensors renamed for `AutoencoderOobleck`.
  :raises ValueError: On a duplicate converted key.
  """

  converted = {}
  for key, value in state_dict.items():
    target_key = convert_stable_audio_decoder_key(key)
    if target_key is None:
      continue
    if target_key in converted:
      raise ValueError(f"Duplicate converted Stable Audio key: {target_key}")
    if target_key.endswith((".alpha", ".beta")):
      value = value.reshape(1, -1, 1)
    converted[target_key] = value
  return converted


def _load_oobleck_config(model_path: Path):
  """Read the released Oobleck architecture and validate that Diffusers can express it.

  :param model_path: Directory holding ``model_config.json``.
  :returns: ``(AutoencoderOobleck kwargs, downsampling ratio)``.
  :raises FileNotFoundError: When the config is absent.
  :raises ValueError: When the released architecture is not Oobleck-compatible.
  """

  config_path = model_path / "model_config.json"
  if not config_path.is_file():
    raise FileNotFoundError(f"Stable Audio config does not exist: {config_path}")
  with config_path.open(encoding="utf-8") as file:
    full_config = json.load(file)

  try:
    pretransform = full_config["model"]["pretransform"]
    config = pretransform["config"]
    decoder = config["decoder"]
    decoder_config = decoder["config"]
    sample_rate = int(full_config["sample_rate"])
  except (KeyError, TypeError, ValueError) as exc:
    raise ValueError(f"Invalid Stable Audio model config: {config_path}") from exc

  if pretransform.get("type") != "autoencoder" or decoder.get("type") != "oobleck":
    raise ValueError("MAGI-2 requires an Oobleck Stable Audio pretransform")
  if decoder_config.get("use_nearest_upsample", False):
    raise ValueError("Diffusers Oobleck does not support nearest-neighbor decoder upsampling")
  if decoder_config.get("final_tanh", True):
    raise ValueError("MAGI-2 requires the Stable Audio decoder with final_tanh=false")

  strides = list(decoder_config["strides"])
  downsampling_ratio = int(config["downsampling_ratio"])
  if int(torch.tensor(strides).prod().item()) != downsampling_ratio:
    raise ValueError("Stable Audio strides do not match downsampling_ratio")

  kwargs = {
    "encoder_hidden_size": int(config["encoder"]["config"]["channels"]),
    "downsampling_ratios": strides,
    "channel_multiples": list(decoder_config["c_mults"]),
    "decoder_channels": int(decoder_config["channels"]),
    "decoder_input_channels": int(decoder_config["latent_dim"]),
    "audio_channels": int(decoder_config["out_channels"]),
    "sampling_rate": sample_rate,
  }
  return kwargs, downsampling_ratio


class Magi2AudioDecoder(nn.Module):
  """Decode MAGI-2 audio latents with the bundled Stable Audio VAE decoder.

  :param model_path: ``stable-audio-open-1.0`` directory of the checkpoint.
  :param device: Device for the decoder.
  :param dtype: Decoder dtype.
  :raises FileNotFoundError: When ``model.safetensors`` is missing.
  """

  def __init__(self, model_path, *, device="cpu", dtype=torch.float32) -> None:
    super().__init__()
    from diffusers import AutoencoderOobleck
    from safetensors import safe_open

    model_path = Path(model_path)
    if not model_path.is_dir():
      raise ValueError(f"Stable Audio model path must be a local directory: {model_path}")

    oobleck_kwargs, self.downsampling_ratio = _load_oobleck_config(model_path)
    self.sample_rate = int(oobleck_kwargs["sampling_rate"])
    # Build the released architecture, then keep only the decoder: MAGI-2 has no audio input encoder.
    autoencoder = AutoencoderOobleck(**oobleck_kwargs)
    self.decoder = autoencoder.decoder.to(device=device, dtype=dtype)
    del autoencoder

    weights_path = model_path / "model.safetensors"
    if not weights_path.is_file():
      raise FileNotFoundError(f"Stable Audio weights do not exist: {weights_path}")
    selected = {}
    with safe_open(weights_path, framework="pt", device=str(device)) as checkpoint:
      for key in checkpoint.keys():
        if key.startswith(_CHECKPOINT_PREFIX):
          selected[key] = checkpoint.get_tensor(key)
    converted = convert_stable_audio_decoder_state_dict(selected)
    converted = {key: value.to(dtype=dtype) for key, value in converted.items()}
    self.decoder.load_state_dict(converted, strict=True)
    self.eval().requires_grad_(False)

  @property
  def latent_fps(self) -> float:
    """Latent frames per second of the released audio timeline."""

    return self.sample_rate / self.downsampling_ratio

  @torch.inference_mode()
  def decode(self, latents: torch.Tensor) -> torch.Tensor:
    """Decode ``[B, C, T]`` audio latents into a ``[B, channels, samples]`` waveform.

    :param latents: Audio latents with the channel axis second.
    :returns: Waveform.
    """

    parameter = next(self.decoder.parameters())
    return self.decoder(latents.to(device=parameter.device, dtype=parameter.dtype))

  forward = decode
