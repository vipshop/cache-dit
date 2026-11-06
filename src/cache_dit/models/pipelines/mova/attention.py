"""Self / cross attention of the vendored MOVA towers.

Copied from OpenMOSS/MOVA ``mova/diffusion/models/wan_video_dit.py`` (Apache-2.0) and changed in two
ways:

* the four-way backend fallback (`flash_attn_3` / `flash_attn_2` / `sageattention` / SDPA) is replaced
  by cache-dit's dispatcher, so ``--attn`` selects the kernel and context parallelism rides the same
  call. The released code falls back to SDPA when none of the three optional packages is installed,
  which is also what the dispatcher's native backend runs;
* every projection is a plain ``nn.Linear`` and the QK norms are ``torch.nn.RMSNorm``, exactly as
  upstream, so the state-dict keys are unchanged. ``head_dim`` is *not* divided by tensor
  parallelism; only ``num_heads`` is (see ``distributed/transformers/mova.py``).

Layout: the dispatcher takes ``[B, S, H, D]`` while the projections emit ``[B, S, H*D]``, so every
call rearranges in and out.
"""

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch.distributed.tensor import DTensor
from torch.nn import RMSNorm

from ....attention import _dispatch_attention_fn

__all__ = [
  "AttentionModule",
  "SelfAttention",
  "CrossAttention",
  "rope_apply_head_dim",
  "precompute_freqs_cis",
  "precompute_freqs_cis_3d",
  "precompute_freqs_cis_1d",
  "legacy_precompute_freqs_cis_1d",
]


def precompute_freqs_cis(dim: int, end: int = 16384, theta: float = 10000.0, s: float = 1.0):
  """Build one axis of the rotary table.

  :param dim: Axis width, half of it is used.
  :param end: Number of positions.
  :param theta: RoPE base.
  :param s: Position scale (used by the Oobleck-style 1-D table).
  :returns: A complex64 tensor of shape ``[end, dim // 2]``.
  """

  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].double() / dim))
  pos = torch.arange(end, dtype=torch.float64, device=freqs.device) * s
  freqs = torch.outer(pos, freqs)
  return torch.polar(torch.ones_like(freqs), freqs)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
  """Build the ``(frame, height, width)`` rotary tables of the video tower.

  :param dim: Head dimension of the video tower.
  :param end: Number of positions per axis.
  :param theta: RoPE base.
  :returns: Three complex64 tensors.
  """

  f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
  h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
  w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
  return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis_1d(dim: int, end: int = 16384, theta: float = 10000.0):
  """Build the audio tower's rotary table and split it into the three concatenated slots.

  :param dim: Head dimension of the audio tower.
  :param end: Number of positions.
  :param theta: RoPE base.
  :returns: Three complex64 tensors.
  """

  return precompute_freqs_cis(dim, end, theta).chunk(3, dim=-1)


def legacy_precompute_freqs_cis_1d(dim: int,
                                   end: int = 16384,
                                   theta: float = 10000.0,
                                   base_tps: float = 4.0,
                                   target_tps: float = 44100 / 2048):
  """Build the Oobleck-flavoured 1-D table (kept for config compatibility only).

  :param dim: Head dimension of the audio tower.
  :param end: Number of positions.
  :param theta: RoPE base.
  :param base_tps: Source frames per second.
  :param target_tps: Target frames per second.
  :returns: Three complex64 tensors; the last two are all-ones.
  """

  s = float(base_tps) / float(target_tps)
  f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta, s)
  no_freqs_cis = precompute_freqs_cis(dim // 3, end, theta, s)
  no_freqs_cis = torch.ones_like(no_freqs_cis)
  return f_freqs_cis, no_freqs_cis, no_freqs_cis


def rope_apply_head_dim(x: torch.Tensor, freqs: torch.Tensor, head_dim: int):
  """Rotate ``[B, S, H*D]`` with a complex table of shape ``[S, 1, D // 2]``.

  :param x: Query or key projection output.
  :param freqs: Rotary table already expanded to the token layout.
  :param head_dim: Head width; frozen under tensor parallelism.
  :returns: The rotated tensor, same shape and dtype as ``x``.
  """

  x = rearrange(x, "b s (n d) -> b s n d", d=head_dim)
  x_out = torch.view_as_complex(
    x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
  x_out = torch.view_as_real(x_out * freqs).flatten(2)
  return x_out.to(x.dtype)


class AttentionModule(nn.Module):
  """Thin wrapper that hands the projections to cache-dit's attention dispatcher.

  :param num_heads: Head count of the *local* shard (tensor parallelism divides it).
  """

  def __init__(self, num_heads: int):
    super().__init__()
    self.num_heads = num_heads

  def forward(self,
              q: torch.Tensor,
              k: torch.Tensor,
              v: torch.Tensor,
              cp_config=None) -> torch.Tensor:
    """Run scaled dot-product attention.

    :param q: Query, ``[B, S, H*D]``.
    :param k: Key, ``[B, S_k, H*D]``.
    :param v: Value, ``[B, S_k, H*D]``.
    :param cp_config: Armed context-parallel config, or None for a local call.
    :returns: Attention output, ``[B, S, H*D]``.
    """

    q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
    k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
    v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)
    x = _dispatch_attention_fn(
      q,
      k,
      v,
      dropout_p=0.0,
      is_causal=False,
      cp_config=cp_config,
    )
    return rearrange(x, "b s n d -> b s (n d)", n=self.num_heads)


class SelfAttention(nn.Module):
  """Self attention with per-head RMSNorm and rotary embedding.

  :param dim: Hidden width of the tower.
  :param num_heads: Head count (divided by tensor parallelism).
  :param eps: RMSNorm epsilon.
  """

  def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
    super().__init__()
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads

    self.q = nn.Linear(dim, dim)
    self.k = nn.Linear(dim, dim)
    self.v = nn.Linear(dim, dim)
    self.o = nn.Linear(dim, dim)
    self.norm_q = RMSNorm(dim, eps=eps)
    self.norm_k = RMSNorm(dim, eps=eps)

    self.attn = AttentionModule(self.num_heads)

  def forward(self, x: torch.Tensor, freqs: torch.Tensor, cp_config=None) -> torch.Tensor:
    """Attend over the tower's own sequence.

    :param x: Hidden states, ``[B, S, dim]``.
    :param freqs: Rotary table for this tower's token layout.
    :param cp_config: Armed context-parallel config, or None.
    :returns: Attention output, ``[B, S, dim]``.
    """

    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(x))
    v = self.v(x)
    if isinstance(freqs, DTensor):
      freqs = freqs.to_local()
    q = rope_apply_head_dim(q, freqs, self.head_dim)
    k = rope_apply_head_dim(k, freqs, self.head_dim)
    return self.o(self.attn(q, k, v, cp_config=cp_config))


class CrossAttention(nn.Module):
  """Text cross attention (query from the tower, key/value from the prompt embeddings).

  :param dim: Hidden width of the tower.
  :param num_heads: Head count (divided by tensor parallelism).
  :param eps: RMSNorm epsilon.
  """

  def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
    super().__init__()
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads

    self.q = nn.Linear(dim, dim)
    self.k = nn.Linear(dim, dim)
    self.v = nn.Linear(dim, dim)
    self.o = nn.Linear(dim, dim)
    self.norm_q = RMSNorm(dim, eps=eps)
    self.norm_k = RMSNorm(dim, eps=eps)

    self.attn = AttentionModule(self.num_heads)

  def forward(self,
              x: torch.Tensor,
              context: torch.Tensor,
              cp_config: Optional[object] = None) -> torch.Tensor:
    """Attend from the tower tokens to the prompt embeddings.

    The prompt stream is replicated, so this call never participates in the context-parallel
    collective even when ``cp_config`` is armed for the tower's own sequence.

    :param x: Hidden states, ``[B, S, dim]``.
    :param context: Prompt embeddings, ``[B, L, dim]``.
    :param cp_config: Ignored; kept so the call site is uniform.
    :returns: Attention output, ``[B, S, dim]``.
    """

    del cp_config
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(context))
    v = self.v(context)
    return self.o(self.attn(q, k, v))
