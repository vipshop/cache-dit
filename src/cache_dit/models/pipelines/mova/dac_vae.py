"""Audio decoder of MOVA (a DAC continuous VAE), vendored from OpenMOSS/MOVA ``dac_vae.py``.

The released file is HunyuanVideo-Foley's DAC implementation (Apache-2.0) and inherits from
``audiotools.ml.BaseModel`` / ``CodecMixin``; ``audiotools`` is not a cache-dit dependency, so those
mixins are dropped here.

Scope: **decode only**. The released checkpoint is ``continuous=True`` and the MOVA pipeline never
encodes audio, so the encoder and the ``quant_conv`` posterior projection are not vendored; their
weights exist in ``audio_vae/diffusion_pytorch_model.safetensors`` and the loader drops exactly the
``encoder.*`` / ``quant_conv.*`` keys and asserts that set. Everything on the decode path keeps its
upstream module names, so ``decoder.*`` / ``post_quant_conv.*`` keys load verbatim.

Two style-level differences, both numerically neutral: the convolutions are plain ``nn.Conv1d`` (the
upstream builds them with ``weight_norm`` and then fuses it during ``__init__`` because the shipped
config has ``use_weight_norm=False``, which leaves exactly ``weight`` / ``bias``), and ``snake`` is
plain Python instead of ``torch.jit.script``.
"""

import math
from typing import List

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

__all__ = ["MovaAudioDAC", "DAC_KEY_PREFIXES_TO_DROP"]

# Keys whose weights are present in the checkpoint but belong to the (unused) encode path.
DAC_KEY_PREFIXES_TO_DROP = ("encoder.", "quant_conv.")


def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
  """Snake activation (upstream formula, ``x + sin(alpha * x)^2 / alpha``).

  :param x: Activations.
  :param alpha: Per-channel learnable frequency.
  :returns: Activated tensor.
  """

  shape = x.shape
  x = x.reshape(shape[0], shape[1], -1)
  x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
  return x.reshape(shape)


class Snake1d(nn.Module):
  """Per-channel Snake activation.

  :param channels: Channel count.
  """

  def __init__(self, channels: int):
    super().__init__()
    self.alpha = nn.Parameter(torch.ones(1, channels, 1))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply the activation.

    :param x: ``[B, C, T]``.
    :returns: Same shape.
    """

    return snake(x, self.alpha)


class ResidualUnit(nn.Module):
  """Dilated residual unit of the DAC decoder.

  :param dim: Channel count.
  :param dilation: Dilation of the first convolution.
  """

  def __init__(self, dim: int = 16, dilation: int = 1):
    super().__init__()
    pad = ((7 - 1) * dilation) // 2
    self.block = nn.Sequential(
      Snake1d(dim),
      nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
      Snake1d(dim),
      nn.Conv1d(dim, dim, kernel_size=1),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Run the residual unit.

    :param x: ``[B, C, T]``.
    :returns: Same shape.
    """

    y = self.block(x)
    pad = (x.shape[-1] - y.shape[-1]) // 2
    if pad > 0:
      x = x[..., pad:-pad]
    return x + y


class DecoderBlock(nn.Module):
  """Upsampling block: Snake, transposed convolution, three residual units.

  :param input_dim: Input channels.
  :param output_dim: Output channels.
  :param stride: Upsampling factor.
  """

  def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
    super().__init__()
    self.block = nn.Sequential(
      Snake1d(input_dim),
      nn.ConvTranspose1d(
        input_dim,
        output_dim,
        kernel_size=2 * stride,
        stride=stride,
        padding=math.ceil(stride / 2),
        output_padding=stride % 2,
      ),
      ResidualUnit(output_dim, dilation=1),
      ResidualUnit(output_dim, dilation=3),
      ResidualUnit(output_dim, dilation=9),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Run the block.

    :param x: ``[B, C_in, T]``.
    :returns: ``[B, C_out, T * stride]``.
    """

    return self.block(x)


class Decoder(nn.Module):
  """Waveform decoder of the DAC VAE.

  :param input_channel: Latent channel count.
  :param channels: Base channel count (halved at every upsampling block).
  :param rates: Upsampling factors.
  :param d_out: Output channel count.
  """

  def __init__(self, input_channel: int, channels: int, rates: List[int], d_out: int = 1):
    super().__init__()
    layers = [nn.Conv1d(input_channel, channels, kernel_size=7, padding=3)]
    for i, stride in enumerate(rates):
      input_dim = channels // 2 ** i
      output_dim = channels // 2 ** (i + 1)
      layers += [DecoderBlock(input_dim, output_dim, stride)]
    layers += [
      Snake1d(output_dim),
      nn.Conv1d(output_dim, d_out, kernel_size=7, padding=3),
      nn.Tanh(),
    ]
    self.model = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Decode latents to a waveform.

    :param x: ``[B, latent_dim, T]``.
    :returns: ``[B, 1, T * prod(rates)]``.
    """

    return self.model(x)


class MovaAudioDAC(ModelMixin, ConfigMixin):
  """MOVA's audio VAE, decode path only.

  The class is named ``MovaAudioDAC`` instead of upstream's ``DAC`` so it cannot be confused with
  ``descript-audio-codec``; the checkpoint's ``audio_vae/config.json`` is read as plain scalars by the
  pipeline loader, not through ``_class_name``.

  :param encoder_dim: Latent channel count of the (unused) encoder.
  :param encoder_rates: Strides of the (unused) encoder; their product is the hop length.
  :param latent_dim: Latent channel count.
  :param decoder_dim: Base channel count of the decoder.
  :param decoder_rates: Upsampling factors of the decoder.
  :param n_codebooks: Kept for config compatibility (unused when continuous).
  :param codebook_size: Kept for config compatibility.
  :param codebook_dim: Kept for config compatibility.
  :param quantizer_dropout: Kept for config compatibility.
  :param sample_rate: Waveform sample rate.
  :param continuous: Whether the VAE is continuous (always true for MOVA).
  :param use_weight_norm: Kept for config compatibility; the shipped weights are already fused.
  """

  @register_to_config
  def __init__(
    self,
    encoder_dim: int = 128,
    encoder_rates: List[int] = (2, 3, 4, 5, 8),
    latent_dim: int = 128,
    decoder_dim: int = 2048,
    decoder_rates: List[int] = (8, 5, 4, 3, 2),
    n_codebooks: int = 9,
    codebook_size: int = 1024,
    codebook_dim: int = 8,
    quantizer_dropout: bool = False,
    sample_rate: int = 48000,
    continuous: bool = True,
    use_weight_norm: bool = False,
  ):
    super().__init__()
    self.encoder_dim = encoder_dim
    self.encoder_rates = list(encoder_rates)
    self.decoder_dim = decoder_dim
    self.decoder_rates = list(decoder_rates)
    self.sample_rate = sample_rate
    self.continuous = continuous
    self.use_weight_norm = use_weight_norm
    self.latent_dim = latent_dim
    self.hop_length = 1
    for rate in self.encoder_rates:
      self.hop_length *= rate

    self.post_quant_conv = nn.Conv1d(latent_dim, latent_dim, 1)
    self.decoder = Decoder(latent_dim, decoder_dim, self.decoder_rates)

  @property
  def dtype(self) -> torch.dtype:
    """Dtype of the first parameter."""

    for param in self.parameters():
      return param.dtype
    return torch.float32

  @property
  def device(self) -> torch.device:
    """Device of the first parameter."""

    for param in self.parameters():
      return param.device
    return torch.device("cpu")

  def decode(self, z: torch.Tensor) -> torch.Tensor:
    """Decode audio latents to a waveform.

    :param z: ``[B, latent_dim, T]`` latents produced by the audio tower.
    :returns: ``[B, 1, samples]`` waveform in ``[-1, 1]``.
    """

    return self.decoder(self.post_quant_conv(z))
