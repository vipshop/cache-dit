"""Rotary positional embeddings of the LongCat-Video-Avatar transformer.

Copied from `meituan-longcat/LongCat-Video` (main, 2026-11-03),
`longcat_video/modules/avatar/rope_3d.py`.

Intentional differences from the upstream copy:

* the official 2-D (H/W) context-parallel split (`cp_split_hw` + `context_parallel_util`) is gone;
  cache-dit splits the frame axis, so `forward` slices the **frame** range it is given instead and
  keeps the table global-grid based;
* the `frame_index` / `num_ref_latents` reference-frame path is gone together with the long-video
  (autoregressive) mode it served;
* the frequency table cache stores CPU tensors and moves them on use, so a device change does not
  leave a stale table behind.
"""

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat


def broadcat(tensors, dim: int = -1) -> torch.Tensor:
  """Concatenate broadcastable tensors along `dim`.

  :param tensors: tensors with the same rank, differing only along broadcastable axes.
  :param dim: axis to concatenate on.
  :returns: the concatenated tensor.
  """

  num_tensors = len(tensors)
  shape_lens = set(len(t.shape) for t in tensors)
  assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
  shape_len = list(shape_lens)[0]
  dim = (dim + shape_len) if dim < 0 else dim
  dims = list(zip(*[list(t.shape) for t in tensors]))
  expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
  assert all(len(set(t[1])) <= 2
             for t in expandable_dims), "invalid dimensions for broadcastable concatenation"
  max_dims = [(t[0], max(t[1])) for t in expandable_dims]
  expanded_dims = [(t[0], (t[1], ) * num_tensors) for t in max_dims]
  expanded_dims.insert(dim, (dim, dims[dim]))
  expandable_shapes = list(zip(*[t[1] for t in expanded_dims]))
  tensors = [t[0].expand(*t[1]) for t in zip(tensors, expandable_shapes)]
  return torch.cat(tensors, dim=dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
  """Rotate the last axis in half-turns, the standard RoPE helper.

  :param x: `[..., d]` tensor with an even `d`.
  :returns: `[..., d]` rotated tensor.
  """

  x = rearrange(x, "... (d r) -> ... d r", r=2)
  x1, x2 = x.unbind(dim=-1)
  x = torch.stack((-x2, x1), dim=-1)
  return rearrange(x, "... d r -> ... (d r)")


class RotaryPositionalEmbedding(nn.Module):
  """3-D (frame, height, width) rotary embedding for the self-attention heads.

  The frequency table is built on the **global** token grid and cached per grid shape; under
  context parallelism `forward` slices the frame range owned by the current rank, so every rank sees
  globally continuous positions.

  :param head_dim: attention head width, must be a multiple of 8.
  """

  def __init__(self, head_dim):
    super().__init__()
    self.head_dim = head_dim
    assert self.head_dim % 8 == 0, "head_dim must be a multiple of 8 for the 3-D RoPE"
    self.base = 10000
    self.freqs_dict = {}

  def precompute_freqs_cis_3d(self, grid_size) -> torch.Tensor:
    """Build the frequency table of one token grid.

    :param grid_size:`(num_frames, height, width)` of the global grid.
    :returns:`[num_frames * height * width, head_dim]` angular frequencies on CPU.
    """

    num_frames, height, width = grid_size
    dim_t = self.head_dim - 4 * (self.head_dim // 6)
    dim_h = 2 * (self.head_dim // 6)
    dim_w = 2 * (self.head_dim // 6)
    freqs_t = 1.0 / (self.base ** (torch.arange(0, dim_t, 2)[:(dim_t // 2)].float() / dim_t))
    freqs_h = 1.0 / (self.base ** (torch.arange(0, dim_h, 2)[:(dim_h // 2)].float() / dim_h))
    freqs_w = 1.0 / (self.base ** (torch.arange(0, dim_w, 2)[:(dim_w // 2)].float() / dim_w))

    grid_t = torch.from_numpy(
      np.linspace(0, num_frames, num_frames, endpoint=False, dtype=np.float32))
    grid_h = torch.from_numpy(np.linspace(0, height, height, endpoint=False, dtype=np.float32))
    grid_w = torch.from_numpy(np.linspace(0, width, width, endpoint=False, dtype=np.float32))

    freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
    freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
    freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)
    freqs_t = repeat(freqs_t, "... n -> ... (n r)", r=2)
    freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
    freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)

    freqs = broadcat(
      (freqs_t[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1)
    return rearrange(freqs, "T H W D -> (T H W) D")

  def forward(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    grid_size,
    local_frame_range=None,
  ):
    """Apply 3-D RoPE to queries and keys.

    :param q:`[B, heads, S, head_dim]` queries.
    :param k:`[B, heads, S, head_dim]` keys.
    :param grid_size:`(num_frames, height, width)` of the **global** token grid.
    :param local_frame_range: optional `(start, end)` frame range of the current rank; the table is
        sliced to those frames before it is applied.
    :returns: the rotated `(q, k)` pair.
    """

    key_name = "x".join(str(i) for i in grid_size)
    if key_name not in self.freqs_dict:
      self.freqs_dict[key_name] = self.precompute_freqs_cis_3d(grid_size)

    freqs_cis = self.freqs_dict[key_name]
    if local_frame_range is not None:
      tokens_per_frame = grid_size[1] * grid_size[2]
      start, end = local_frame_range
      freqs_cis = freqs_cis[start * tokens_per_frame:end * tokens_per_frame]

    freqs_cis = freqs_cis.to(device=q.device, dtype=torch.float32)
    cos = rearrange(freqs_cis.cos(), "n d -> 1 1 n d")
    sin = rearrange(freqs_cis.sin(), "n d -> 1 1 n d")

    q_ = q.float()
    k_ = k.float()
    q_ = (q_ * cos) + (rotate_half(q_) * sin)
    k_ = (k_ * cos) + (rotate_half(k_) * sin)
    return q_.type_as(q), k_.type_as(k)


class RotaryPositionalEmbedding1D(nn.Module):
  """1-D rotary embedding (kept for parity with the upstream module layout).

  :param head_dim: attention head width.
  """

  def __init__(self, head_dim):
    super().__init__()
    self.head_dim = head_dim
    self.base = 10000

  def precompute_freqs_cis_1d(self, pos_indices: torch.Tensor) -> torch.Tensor:
    """Build the frequency table of a 1-D position vector.

    :param pos_indices:`[n]` positions.
    :returns:`[n, head_dim]` angular frequencies.
    """

    freqs = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2)[:(self.head_dim // 2)].float() /
                                 self.head_dim))
    freqs = freqs.to(pos_indices.device)
    freqs = torch.einsum("..., f -> ... f", pos_indices.float(), freqs)
    return repeat(freqs, "... n -> ... (n r)", r=2)

  def forward(self, x: torch.Tensor, pos_indices: torch.Tensor) -> torch.Tensor:
    """Apply 1-D RoPE.

    :param x:`[B, heads, S, head_dim]` activations.
    :param pos_indices:`[S]` positions.
    :returns: the rotated activations.
    """

    freqs_cis = self.precompute_freqs_cis_1d(pos_indices).to(device=x.device, dtype=torch.float32)
    cos = rearrange(freqs_cis.cos(), "n d -> 1 1 n d")
    sin = rearrange(freqs_cis.sin(), "n d -> 1 1 n d")
    x_ = x.float()
    x_ = (x_ * cos) + (rotate_half(x_) * sin)
    return x_.type_as(x)
