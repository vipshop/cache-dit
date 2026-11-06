"""Transformer block, output head and small helpers shared by both MOVA towers.

Copied from OpenMOSS/MOVA ``mova/diffusion/models/wan_video_dit.py`` / ``wan_audio_dit.py``
(Apache-2.0). The math is unchanged; the only edits are

* ``modulate`` is no longer wrapped in ``torch.compile(fullgraph=True)`` — an inner compiled region
  fights the outer ``--compile`` and the per-block graphs cache-dit builds;
* ``cp_config`` is threaded through the block so the self attention can join the context-parallel
  collective while the text cross attention stays local.

The audio tower reuses this very class (upstream imports ``DiTBlock`` from the video module), which is
why the block carries no tower-specific state beyond the shapes it is constructed with.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import CrossAttention, SelfAttention

__all__ = [
  "DiTBlock",
  "Head",
  "MLP",
  "modulate",
  "sinusoidal_embedding_1d",
]


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
  """Encode scalar timesteps into a 1-D sinusoidal embedding.

  :param dim: Embedding width (must be even).
  :param position: Timestep tensor.
  :returns: ``[B, dim]`` float embedding, in ``position``'s dtype.
  """

  sinusoid = torch.outer(
    position.type(torch.float64),
    torch.pow(10000,
              -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)),
  )
  x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
  return x.to(position.dtype)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
  """Apply the adaLN scale/shift pair (upstream formula, unchanged).

  :param x: Normalized activations.
  :param shift: Additive term.
  :param scale: Multiplicative term.
  :returns: ``x * (1 + scale) + shift``.
  """

  return x * (1 + scale) + shift


class GateModule(nn.Module):
  """Residual gate used by the block's two sub-layers."""

  def forward(self, x: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Add a gated residual.

    :param x: Residual stream.
    :param gate: Per-channel gate.
    :param residual: Sub-layer output.
    :returns: The updated residual stream.
    """

    return x + gate * residual


class MLP(nn.Module):
  """Small projection head used for CLIP features (unused by the 360p checkpoint).

  :param in_dim: Input width.
  :param out_dim: Output width.
  :param has_pos_emb: Whether to add the learned positional table.
  """

  def __init__(self, in_dim: int, out_dim: int, has_pos_emb: bool = False):
    super().__init__()
    self.proj = nn.Sequential(
      nn.LayerNorm(in_dim),
      nn.Linear(in_dim, in_dim),
      nn.GELU(),
      nn.Linear(in_dim, out_dim),
      nn.LayerNorm(out_dim),
    )
    self.has_pos_emb = has_pos_emb
    if has_pos_emb:
      self.emb_pos = nn.Parameter(torch.zeros((1, 514, 1280)))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Project the features.

    :param x: Input features.
    :returns: Projected features.
    """

    if self.has_pos_emb:
      x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
    return self.proj(x)


class Head(nn.Module):
  """Final adaLN-modulated projection that maps hidden states back to latent patches.

  :param dim: Hidden width.
  :param out_dim: Latent channels.
  :param patch_size: Patch shape; the output width is ``out_dim * prod(patch_size)``.
  :param eps: LayerNorm epsilon.
  """

  def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, ...], eps: float):
    super().__init__()
    self.dim = dim
    self.patch_size = patch_size
    self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
    self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
    self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

  def forward(self, x: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
    """Apply the head.

    :param x: Hidden states, ``[B, S, dim]``.
    :param t_mod: Timestep embedding, ``[B, dim]``.
    :returns: ``[B, S, out_dim * prod(patch_size)]``.
    """

    shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) +
                    t_mod.unsqueeze(1)).chunk(2, dim=1)
    return self.head(self.norm(x) * (1 + scale) + shift)


class DiTBlock(nn.Module):
  """One denoising block: gated self attention, text cross attention, gated FFN.

  :param dim: Hidden width.
  :param num_heads: Head count (divided by tensor parallelism).
  :param ffn_dim: FFN hidden width.
  :param eps: Normalization epsilon.
  """

  def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
    super().__init__()
    self.dim = dim
    self.num_heads = num_heads
    self.ffn_dim = ffn_dim

    self.self_attn = SelfAttention(dim, num_heads, eps)
    self.cross_attn = CrossAttention(dim, num_heads, eps)
    self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
    self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
    self.norm3 = nn.LayerNorm(dim, eps=eps)
    self.ffn = nn.Sequential(
      nn.Linear(dim, ffn_dim),
      nn.GELU(approximate="tanh"),
      nn.Linear(ffn_dim, dim),
    )
    self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)
    self.gate = GateModule()

  def forward(self,
              x: torch.Tensor,
              context: torch.Tensor,
              t_mod: torch.Tensor,
              freqs: torch.Tensor,
              cp_config: Optional[object] = None) -> torch.Tensor:
    """Run one block.

    :param x: Hidden states, ``[B, S, dim]``.
    :param context: Prompt embeddings, ``[B, L, dim]``.
    :param t_mod: Timestep modulation, ``[B, 6, dim]``.
    :param freqs: Rotary table of this tower.
    :param cp_config: Armed context-parallel config, or None.
    :returns: Updated hidden states, ``[B, S, dim]``.
    """

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
      self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
    input_x = modulate(self.norm1(x), shift_msa, scale_msa)
    x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, cp_config=cp_config))
    x = x + self.cross_attn(self.norm3(x), context)
    input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
    x = self.gate(x, gate_mlp, self.ffn(input_x))
    return x
