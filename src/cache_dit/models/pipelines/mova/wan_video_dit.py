"""Video tower of MOVA (``WanModel``), vendored from OpenMOSS/MOVA.

Copied from ``mova/diffusion/models/wan_video_dit.py`` (Apache-2.0). The released class shares its
name with diffusers' Wan transformer, but nothing else: it is a DiffSynth-style tower whose denoising
loop lives in the *pipeline*, not in ``forward``. Only the pieces the pipeline actually calls are kept
here (``patchify`` / ``unpatchify`` / ``head`` / the three embeddings / the rotary tables); the
``forward`` that loops over ``blocks`` is deliberately not vendored, because the fused stack owns the
loop in cache-dit.

Intentional differences from upstream:

* the rotary tables are non-persistent buffers instead of plain tuple attributes, so ``.to(device)``
  and the offload helpers move them with the module and they stay out of the state dict;
* ``time_embed`` is factored out (xDiT does the same) so the stack can build the timestep embedding
  before the cached region.
"""

from typing import Tuple

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange

from .attention import precompute_freqs_cis_3d, precompute_freqs_cis_1d, legacy_precompute_freqs_cis_1d
from .blocks import DiTBlock, Head, MLP, sinusoidal_embedding_1d

__all__ = ["WanModel", "video_freqs", "audio_freqs"]


def video_freqs(freqs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], f: int, h: int, w: int,
                device: torch.device) -> torch.Tensor:
  """Expand the 3-D rotary tables onto the ``(frame, height, width)`` token grid.

  :param freqs: The three complex tables built at construction time.
  :param f: Latent frames.
  :param h: Latent patch rows.
  :param w: Latent patch columns.
  :param device: Device to build the table on.
  :returns: ``[f * h * w, 1, head_dim]`` in complex64.
  """

  return torch.cat([
    freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
    freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
    freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
  ],
                   dim=-1).reshape(f * h * w, 1, -1).to(device)


def audio_freqs(freqs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], f: int,
                device: torch.device) -> torch.Tensor:
  """Expand the 1-D rotary table onto the audio token axis.

  :param freqs: The three complex tables built at construction time.
  :param f: Number of audio latent frames.
  :param device: Device to build the table on.
  :returns: ``[f, 1, head_dim]`` in complex64.
  """

  return torch.cat([
    freqs[0][:f].view(f, -1).expand(f, -1),
    freqs[1][:f].view(f, -1).expand(f, -1),
    freqs[2][:f].view(f, -1).expand(f, -1),
  ],
                   dim=-1).reshape(f, 1, -1).to(device)


class WanModel(ModelMixin, ConfigMixin):
  """Video tower: patch embedding, prompt/timestep embeddings, block stack, output head.

  The class name and every state-dict key match the released checkpoint.

  :param dim: Hidden width.
  :param in_dim: Latent channels fed to the patch embedding (16 latents + 20 conditioning channels).
  :param ffn_dim: FFN hidden width.
  :param out_dim: Output latent channels.
  :param text_dim: Prompt embedding width.
  :param freq_dim: Sinusoidal timestep width.
  :param eps: Normalization epsilon.
  :param patch_size: ``(t, h, w)`` patch shape.
  :param num_heads: Attention head count.
  :param num_layers: Block count.
  :param has_image_input: Unused by the 360p checkpoint, kept for config compatibility.
  """

  _repeated_blocks = ("DiTBlock", )

  @register_to_config
  def __init__(
    self,
    dim: int = 5120,
    in_dim: int = 36,
    ffn_dim: int = 13824,
    out_dim: int = 16,
    text_dim: int = 4096,
    freq_dim: int = 256,
    eps: float = 1e-6,
    patch_size: Tuple[int, int, int] = (1, 2, 2),
    num_heads: int = 40,
    num_layers: int = 40,
    has_image_input: bool = False,
    has_image_pos_emb: bool = False,
    has_ref_conv: bool = False,
    add_control_adapter: bool = False,
    in_dim_control_adapter: int = 24,
    seperated_timestep: bool = False,
    require_vae_embedding: bool = True,
    require_clip_embedding: bool = True,
    fuse_vae_embedding_in_latents: bool = False,
  ):
    super().__init__()
    self.dim = dim
    self.freq_dim = freq_dim
    self.has_image_input = has_image_input
    self.patch_size = tuple(patch_size)
    self.seperated_timestep = seperated_timestep
    self.require_vae_embedding = require_vae_embedding
    self.require_clip_embedding = require_clip_embedding
    self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
    self.has_image_pos_emb = has_image_pos_emb
    self.has_ref_conv = has_ref_conv

    self.patch_embedding = nn.Conv3d(in_dim,
                                     dim,
                                     kernel_size=self.patch_size,
                                     stride=self.patch_size)
    self.text_embedding = nn.Sequential(
      nn.Linear(text_dim, dim),
      nn.GELU(approximate="tanh"),
      nn.Linear(dim, dim),
    )
    self.time_embedding = nn.Sequential(
      nn.Linear(freq_dim, dim),
      nn.SiLU(),
      nn.Linear(dim, dim),
    )
    self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
    self.blocks = nn.ModuleList([DiTBlock(dim, num_heads, ffn_dim, eps) for _ in range(num_layers)])
    self.head = Head(dim, out_dim, self.patch_size, eps)

    head_dim = dim // num_heads
    freqs = precompute_freqs_cis_3d(head_dim)
    self.register_buffer("freqs_0", freqs[0], persistent=False)
    self.register_buffer("freqs_1", freqs[1], persistent=False)
    self.register_buffer("freqs_2", freqs[2], persistent=False)

    if has_image_input:
      self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)
    if has_ref_conv:
      self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))

  @property
  def freqs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three rotary tables (kept as buffers so they follow ``.to(device)``)."""

    return (self.freqs_0, self.freqs_1, self.freqs_2)

  def time_embed(self, timestep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the timestep embedding and its 6-way modulation.

    :param timestep: ``[B]`` timestep in the scheduler's units.
    :returns: ``(t, t_mod)`` with shapes ``[B, dim]`` and ``[B, 6, dim]``.
    """

    t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
    t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
    return t, t_mod

  def patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Patchify a ``[B, C, T, H, W]`` latent grid.

    :param x: Latent grid.
    :returns: ``([B, L, dim], (t, h, w))``.
    """

    x = x.contiguous(memory_format=torch.channels_last_3d)
    x = self.patch_embedding(x)
    grid_size = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
    return x, grid_size

  def unpatchify(self, x: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
    """Invert :meth:`patchify`.

    :param x: ``[B, L, out_dim * prod(patch_size)]``.
    :param grid_size: ``(t, h, w)`` returned by :meth:`patchify`.
    :returns: ``[B, out_dim, T, H, W]``.
    """

    return rearrange(
      x,
      "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
      f=grid_size[0],
      h=grid_size[1],
      w=grid_size[2],
      x=self.patch_size[0],
      y=self.patch_size[1],
      z=self.patch_size[2],
    )


class WanAudioModel(ModelMixin, ConfigMixin):
  """Audio tower: same block stack as the video tower, 1-D patch embedding and 1-D rotary table.

  A standalone class rather than a subclass of :class:`WanModel`, so ``config`` / ``_class_name``
  stay exactly what the released ``audio_dit/config.json`` declares.

  :param vae_type: ``"dac"`` selects the plain 1-D table; ``"oobleck"`` the legacy scaled one.
  """

  _repeated_blocks = ("DiTBlock", )

  @register_to_config
  def __init__(
    self,
    dim: int = 1536,
    in_dim: int = 128,
    ffn_dim: int = 8960,
    out_dim: int = 128,
    text_dim: int = 4096,
    freq_dim: int = 256,
    eps: float = 1e-6,
    patch_size: Tuple[int] = (1, ),
    num_heads: int = 12,
    num_layers: int = 30,
    has_image_input: bool = False,
    has_image_pos_emb: bool = False,
    has_ref_conv: bool = False,
    add_control_adapter: bool = False,
    in_dim_control_adapter: int = 24,
    seperated_timestep: bool = False,
    require_vae_embedding: bool = True,
    require_clip_embedding: bool = True,
    fuse_vae_embedding_in_latents: bool = False,
    vae_type: str = "dac",
  ):
    super().__init__()
    self.dim = dim
    self.freq_dim = freq_dim
    self.has_image_input = has_image_input
    self.patch_size = tuple(patch_size)
    self.seperated_timestep = seperated_timestep
    self.require_vae_embedding = require_vae_embedding
    self.require_clip_embedding = require_clip_embedding
    self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
    self.has_image_pos_emb = has_image_pos_emb
    self.has_ref_conv = has_ref_conv
    self.vae_type = vae_type

    self.patch_embedding = nn.Conv1d(in_dim,
                                     dim,
                                     kernel_size=self.patch_size,
                                     stride=self.patch_size)
    self.text_embedding = nn.Sequential(
      nn.Linear(text_dim, dim),
      nn.GELU(approximate="tanh"),
      nn.Linear(dim, dim),
    )
    self.time_embedding = nn.Sequential(
      nn.Linear(freq_dim, dim),
      nn.SiLU(),
      nn.Linear(dim, dim),
    )
    self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
    self.blocks = nn.ModuleList([DiTBlock(dim, num_heads, ffn_dim, eps) for _ in range(num_layers)])
    self.head = Head(dim, out_dim, self.patch_size, eps)

    head_dim = dim // num_heads
    if vae_type == "oobleck":
      freqs = legacy_precompute_freqs_cis_1d(head_dim, base_tps=4.0, target_tps=44100 / 2048)
    elif vae_type == "dac":
      freqs = precompute_freqs_cis_1d(head_dim)
    else:
      raise ValueError(f"Invalid audio VAE type: {vae_type}")
    self.register_buffer("freqs_0", freqs[0], persistent=False)
    self.register_buffer("freqs_1", freqs[1], persistent=False)
    self.register_buffer("freqs_2", freqs[2], persistent=False)

    if has_image_input:
      self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)
    if has_ref_conv:
      self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))

  @property
  def freqs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three rotary tables (kept as buffers so they follow ``.to(device)``)."""

    return (self.freqs_0, self.freqs_1, self.freqs_2)

  def time_embed(self, timestep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the timestep embedding and its 6-way modulation.

    :param timestep: ``[B]`` timestep in the scheduler's units.
    :returns: ``(t, t_mod)`` with shapes ``[B, dim]`` and ``[B, 6, dim]``.
    """

    t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
    t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
    return t, t_mod

  def patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int]]:
    """Patchify a ``[B, C, T]`` audio latent.

    :param x: Audio latent.
    :returns: ``([B, T, dim], (T, ))``.
    """

    x = self.patch_embedding(x)
    grid_size = x.shape[2:]
    x = rearrange(x, "b c f -> b f c").contiguous()
    return x, grid_size

  def unpatchify(self, x: torch.Tensor, grid_size: Tuple[int]) -> torch.Tensor:
    """Invert :meth:`patchify`.

    :param x: ``[B, T, out_dim * prod(patch_size)]``.
    :param grid_size: ``(T, )`` returned by :meth:`patchify`.
    :returns: ``[B, out_dim, T]``.
    """

    return rearrange(x, "b f (p c) -> b c (f p)", f=grid_size[0], p=self.patch_size[0])
