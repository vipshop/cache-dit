"""SANA-Video 2.0 building blocks that are not plain ``nn.Linear``.

Copied from NVlabs/Sana (Apache-2.0, Copyright 2026 NVIDIA CORPORATION & AFFILIATES),
``diffusion/model/nets/sana_blocks.py`` and ``diffusion/model/norms.py`` at commit of the
SANA-Video 2.0 release (PR #439), then reformatted to this repository's style. The parameter
names and the numerics are unchanged, because the released checkpoint is a merged ``state_dict``
that is loaded with ``strict=True``.

Intentional differences from the upstream copy:

* The 3D RoPE frequency table is a plain attribute in upstream and is re-uploaded on every
  forward; here it is cached per device (a 720p clip re-uploads a 2 MB complex128 table 30 times
  per sampling step otherwise).
* :meth:`T2IFinalLayer.forward` keeps both the batch-timestep and the per-frame-timestep paths,
  since the TI2V branch feeds a ``(B, 1, F, D)`` conditioning.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


def t2i_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
  """Apply the AdaLN shift/scale pair to a normalized tensor.

  :param x: Normalized activations.
  :param shift: Additive term, broadcastable to ``x``.
  :param scale: Multiplicative term, broadcastable to ``x``.
  :returns: ``x * (1 + scale) + shift``.
  """
  return x * (1 + scale) + shift


def get_1d_rotary_pos_embed(dim: int,
                            max_seq_len: int,
                            theta: float = 10000.0,
                            freqs_dtype: torch.dtype = torch.float64) -> torch.Tensor:
  """Precompute complex rotary frequencies for one axis.

  :param dim: Axis dimension (even).
  :param max_seq_len: Number of cached positions.
  :param theta: RoPE base.
  :param freqs_dtype: Dtype of the frequency computation; upstream uses float64.
  :returns: Complex frequency table ``[max_seq_len, dim // 2]``.
  """
  if dim % 2:
    raise ValueError(f"RoPE axis dimension must be even, got {dim}.")
  pos = torch.arange(max_seq_len)
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype)[:(dim // 2)] / dim))
  freqs = torch.outer(pos, freqs)
  return torch.polar(torch.ones_like(freqs), freqs)


class _RMSNorm(nn.Module):
  """Upstream ``RMSNorm``: fp32 normalization, ``weight`` multiplied in fp32.

  :param dim: Normalized size.
  :param scale_factor: Initial value of ``weight``.
  :param eps: Epsilon of the mean-square.
  :param norm_dim: Dimension the mean is taken over.
  """

  def __init__(self, dim: int, scale_factor: float = 1.0, eps: float = 1e-6, norm_dim: int = -1):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim) * scale_factor)
    self.norm_dim = norm_dim

  def _norm(self, x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(self.norm_dim, keepdim=True) + self.eps)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    weight_shape = [1] * x.dim()
    weight_shape[self.norm_dim] = -1
    weight = self.weight.view(*weight_shape)
    return (weight * self._norm(x.float())).type_as(x)


class DepthRMSNorm(nn.Module):
  """Parameter-free RMS normalization along the hidden dimension.

  :param eps: Epsilon of the mean-square. Upstream does not promote to fp32 here.
  """

  def __init__(self, eps: float = 1e-6):
    super().__init__()
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


def _get_same_padding(kernel_size: Sequence[int]) -> list[int]:
  return [int(size) // 2 for size in kernel_size]


class PatchEmbedMS3D(nn.Module):
  """3D patch embedding, parameter names ``proj.weight`` / ``proj.bias``.

  :param patch_size: ``(t, h, w)`` patch stride; the temporal entry must be 1.
  :param in_chans: Latent channels.
  :param embed_dim: Hidden size.
  :param kernel_size: Convolution kernel, defaults to ``patch_size``.
  :param bias: Whether the projection keeps a bias.
  """

  def __init__(self,
               patch_size: Tuple[int, int, int] = (1, 2, 2),
               in_chans: int = 3,
               embed_dim: int = 768,
               kernel_size: Optional[Tuple[int, int, int]] = None,
               bias: bool = True):
    super().__init__()
    kernel_size = tuple(kernel_size or patch_size)
    patch_size = tuple(patch_size)
    if patch_size[0] != 1:
      raise ValueError(f"Patch size for 3D embedding must be (1, *, *), got {patch_size}.")
    self.kernel_size = kernel_size
    self.patch_size = patch_size
    padding = 0
    if kernel_size[-1] % 2 > 0:
      padding = _get_same_padding(kernel_size)
    self.proj = nn.Conv3d(in_chans,
                          embed_dim,
                          kernel_size=kernel_size,
                          stride=patch_size,
                          padding=padding,
                          bias=bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Project latents to tokens.

    :param x: Latents ``[B, C, T, H, W]``.
    :returns: Tokens ``[B, T*H*W, C]``.
    """
    x = self.proj(x)
    return x.flatten(2).transpose(1, 2)


class TimestepEmbedder(nn.Module):
  """Sinusoidal timestep embedding followed by a two-layer MLP (``mlp.0/2``).

  :param hidden_size: Output size.
  :param frequency_embedding_size: Number of sinusoidal features.
  """

  def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True),
    )
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000) -> torch.Tensor:
    """Build sinusoidal embeddings for fractional timesteps.

    :param t: 1-D tensor of timesteps.
    :param dim: Embedding dimension.
    :param max_period: Smallest frequency.
    :returns: ``[N, dim]`` sinusoidal features.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  @property
  def dtype(self) -> torch.dtype:
    return next(self.parameters()).dtype

  def forward(self, t: torch.Tensor) -> torch.Tensor:
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
    return self.mlp(t_freq)


class _UpstreamMlp(nn.Module):
  """Timm-style MLP used by ``y_embedder.y_proj`` (``fc1``/``fc2``, both with bias).

  :param in_features: Input size.
  :param hidden_features: Hidden size.
  :param out_features: Output size.
  """

  def __init__(self, in_features: int, hidden_features: int, out_features: int):
    super().__init__()
    self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
    self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class CaptionEmbedder(nn.Module):
  """Projects text features to the hidden size and holds the null-caption table.

  Upstream keeps the null table as a buffer named ``y_embedding`` and only uses it for training
  dropout, so inference is a single projection.

  :param in_channels: Text feature size (Gemma-2-2B hidden size).
  :param hidden_size: Output size.
  :param token_num: Number of cached null-caption embeddings.
  """

  def __init__(self, in_channels: int, hidden_size: int, token_num: int = 300):
    super().__init__()
    self.y_proj = _UpstreamMlp(in_channels, hidden_size, hidden_size)
    self.register_buffer("y_embedding",
                         nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))

  def forward(self, caption: torch.Tensor) -> torch.Tensor:
    return self.y_proj(caption)


class T2IFinalLayer(nn.Module):
  """Final AdaLN + projection, with ``scale_shift_table`` owned by this module.

  :param hidden_size: Hidden size.
  :param patch_size: ``(t, h, w)`` patch size, used for the output width.
  :param out_channels: Latent channels of the prediction.
  """

  def __init__(self, hidden_size: int, patch_size: Sequence[int], out_channels: int):
    super().__init__()
    self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.linear = nn.Linear(hidden_size, math.prod(patch_size) * out_channels, bias=True)
    self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)
    self.out_channels = out_channels

  def forward_frame_aware(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """:param x: Tokens ``[B, F*tokens, C]``. :param t: Modulation ``[B, 1, F, C]``."""
    batch, num_tokens, channels = x.shape
    num_frames = t.shape[2]
    shift, scale = (self.scale_shift_table[None, None, :, :] + t.transpose(1, 2)).chunk(2, dim=-2)
    x = t2i_modulate(self.norm_final(x).reshape(batch, num_frames, -1, channels), shift,
                     scale).reshape(batch, num_tokens, channels)
    return self.linear(x)

  def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if t.dim() > 2:
      return self.forward_frame_aware(x, t)
    shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
    return self.linear(t2i_modulate(self.norm_final(x), shift, scale))


class WanRotaryPosEmbed(nn.Module):
  """3D rotary position embedding split over the temporal / height / width axes.

  :param attention_head_dim: Head dimension the table is built for.
  :param patch_size: ``(t, h, w)`` patch size; the table is built for the latent grid.
  :param max_seq_len: Number of cached positions per axis.
  :param theta: RoPE base.
  """

  def __init__(self,
               attention_head_dim: int,
               patch_size: Tuple[int, int, int],
               max_seq_len: int,
               theta: float = 10000.0):
    super().__init__()
    self.attention_head_dim = attention_head_dim
    self.patch_size = tuple(patch_size)
    self.max_seq_len = max_seq_len
    self.theta = theta
    h_dim = w_dim = 2 * (attention_head_dim // 6)
    t_dim = attention_head_dim - h_dim - w_dim
    self._freqs_dims = (t_dim, h_dim, w_dim)
    # Upstream keeps this as a plain attribute, never as a buffer: it is not part of the
    # checkpoint and it stays in fp64 on the host.
    self.freqs = self._build_freqs()
    self._freqs_device = None

  def _build_freqs(self) -> torch.Tensor:
    freqs = [
      get_1d_rotary_pos_embed(dim, self.max_seq_len, self.theta, freqs_dtype=torch.float64)
      for dim in self._freqs_dims
    ]
    return torch.cat(freqs, dim=1)

  def rebuild_freqs(self) -> None:
    """Materialize the table again after the module was built on the meta device."""
    self.freqs = self._build_freqs()
    self._freqs_device = None

  def forward(self, fhw: Union[torch.Tensor, Tuple[int, int, int]],
              device: torch.device) -> torch.Tensor:
    """Build the per-token frequency table for one latent grid.

    :param fhw: ``(frames, height, width)`` of the latent grid.
    :param device: Device the table is moved to.
    :returns: Complex frequencies ``[1, 1, frames*height*width, attention_head_dim // 2]``.
    """
    ppf, pph, ppw = (int(value) for value in fhw)
    if self._freqs_device != device:
      self.freqs = self.freqs.to(device)
      self._freqs_device = device
    dim = self.attention_head_dim
    t_size = dim // 2 - 2 * (dim // 6)
    h_size = dim // 6
    w_size = dim // 6
    freqs_t, freqs_h, freqs_w = self.freqs.split_with_sizes([t_size, h_size, w_size], dim=1)
    freqs_t = freqs_t[:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
    freqs_h = freqs_h[:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
    freqs_w = freqs_w[:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
    return torch.cat([freqs_t, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)


def apply_rope_channel_first(hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
  """Apply complex RoPE to a ``[B, heads, head_dim, tokens]`` tensor.

  :param hidden_states: Query or key states in channel-first layout.
  :param freqs: Complex frequencies broadcastable to the token axis.
  :returns: Rotated states in the input layout and dtype.
  """
  rotated = torch.view_as_complex(
    hidden_states.permute(0, 1, 3, 2).to(torch.float64).unflatten(3, (-1, 2)))
  return torch.view_as_real(rotated * freqs).flatten(3, 4).permute(0, 1, 3,
                                                                   2).type_as(hidden_states)


def apply_rope_token_first(hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
  """Apply complex RoPE to a ``[B, tokens, heads, head_dim]`` tensor.

  :param hidden_states: Query or key states in token-first layout.
  :param freqs: Complex frequencies broadcastable to the token axis.
  :returns: Rotated states in the input layout and dtype.
  """
  rotated = torch.view_as_complex(
    hidden_states.transpose(1, 2).to(torch.float64).unflatten(3, (-1, 2)))
  return torch.view_as_real(rotated * freqs).flatten(3, 4).transpose(1, 2).type_as(hidden_states)


__all__ = [
  "CaptionEmbedder",
  "DepthRMSNorm",
  "PatchEmbedMS3D",
  "T2IFinalLayer",
  "TimestepEmbedder",
  "WanRotaryPosEmbed",
  "apply_rope_channel_first",
  "apply_rope_token_first",
  "get_1d_rotary_pos_embed",
  "t2i_modulate",
]
