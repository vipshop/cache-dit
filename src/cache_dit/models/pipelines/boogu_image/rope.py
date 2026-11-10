"""Boogu-Image rotary position embeddings.

Adapted from Boogu-Image (`boogu/models/transformers/rope.py`), Apache-2.0; the module itself
derives from the OmniGen2 and diffusers RoPE helpers.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from einops import repeat

# The released checkpoints carry `axes_lens = (2048, 1664, 1664)`. Instruction
# position ids are the token index on all three axes, so a long prompt (a long
# instruction plus the VLM's reference-image tokens) can exceed the 1664 budget
# of axes 1/2 and `torch.gather` then runs out of bounds, which surfaces as a
# CUDA device-side assert rather than an index error. The tables are tiny, so
# every axis is widened to a floor instead of truncating the prompt.
_MIN_AXIS_LEN = 4096


def _resolve_axes_lens(axes_lens: Tuple[int, int, int]) -> Tuple[int, int, int]:
  """Widen the per-axis RoPE table budget to at least `_MIN_AXIS_LEN`.

  :param axes_lens: Per-axis table lengths from the checkpoint config.
  :returns: The per-axis lengths with each entry floored at `_MIN_AXIS_LEN`.
  """

  return tuple(max(int(length), _MIN_AXIS_LEN) for length in axes_lens)


class BooguImageRotaryPosEmbed(nn.Module):

  def __init__(
      self,
      theta: int,
      axes_dim: Tuple[int, int, int],
      axes_lens: Tuple[int, int, int] = (300, 512, 512),
      patch_size: int = 2,
  ):
    super().__init__()
    self.theta = theta
    self.axes_dim = axes_dim
    self.axes_lens = _resolve_axes_lens(axes_lens)
    self.patch_size = patch_size

  @staticmethod
  def get_freqs_cis(axes_dim: Tuple[int, int, int], axes_lens: Tuple[int, int, int],
                    theta: int) -> List[torch.Tensor]:
    freqs_cis = []
    freqs_dtype = (torch.float32 if torch.backends.mps.is_available() else torch.float64)
    for i, (d, e) in enumerate(zip(axes_dim, _resolve_axes_lens(axes_lens))):
      emb = get_1d_rotary_pos_embed(d, e, theta=theta, freqs_dtype=freqs_dtype)
      freqs_cis.append(emb)
    return freqs_cis

  def _get_freqs_cis(self, freqs_cis, ids: torch.Tensor) -> torch.Tensor:
    device = ids.device
    if ids.device.type == "mps":
      ids = ids.to("cpu")

    result = []
    for i in range(len(self.axes_dim)):
      freqs = freqs_cis[i].to(ids.device)
      index = ids[:, :, i:i + 1].repeat(1, 1, freqs.shape[-1]).to(torch.int64)
      result.append(
        torch.gather(freqs.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index))
    return torch.cat(result, dim=-1).to(device)

  def forward(
    self,
    freqs_cis,
    attention_mask,
    l_effective_ref_img_len,
    l_effective_img_len,
    ref_img_sizes,
    img_sizes,
    device,
  ):
    batch_size = len(attention_mask)
    p = self.patch_size

    encoder_seq_len = attention_mask.shape[1]
    l_effective_cap_len = attention_mask.sum(dim=1).tolist()

    seq_lengths = [
      cap_len + sum(ref_img_len) + img_len for cap_len, ref_img_len, img_len in zip(
        l_effective_cap_len, l_effective_ref_img_len, l_effective_img_len)
    ]

    max_seq_len = max(seq_lengths)
    max_ref_img_len = max([sum(ref_img_len) for ref_img_len in l_effective_ref_img_len])
    max_img_len = max(l_effective_img_len)

    # Create position IDs
    position_ids = torch.zeros(batch_size, max_seq_len, 3, dtype=torch.int32, device=device)

    for i, (cap_seq_len, seq_len) in enumerate(zip(l_effective_cap_len, seq_lengths)):
      # add text position ids
      position_ids[i, :cap_seq_len] = repeat(
        torch.arange(cap_seq_len, dtype=torch.int32, device=device), "l -> l 3")

      pe_shift = cap_seq_len
      pe_shift_len = cap_seq_len

      if ref_img_sizes[i] is not None:
        for ref_img_size, ref_img_len in zip(ref_img_sizes[i], l_effective_ref_img_len[i]):
          H, W = ref_img_size
          ref_H_tokens, ref_W_tokens = H // p, W // p
          assert ref_H_tokens * ref_W_tokens == ref_img_len
          # add image position ids

          row_ids = repeat(
            torch.arange(ref_H_tokens, dtype=torch.int32, device=device),
            "h -> h w",
            w=ref_W_tokens,
          ).flatten()
          col_ids = repeat(
            torch.arange(ref_W_tokens, dtype=torch.int32, device=device),
            "w -> h w",
            h=ref_H_tokens,
          ).flatten()
          position_ids[i, pe_shift_len:pe_shift_len + ref_img_len, 0] = (pe_shift)
          position_ids[i, pe_shift_len:pe_shift_len + ref_img_len, 1] = (row_ids)
          position_ids[i, pe_shift_len:pe_shift_len + ref_img_len, 2] = (col_ids)

          pe_shift += max(ref_H_tokens, ref_W_tokens)
          pe_shift_len += ref_img_len

      H, W = img_sizes[i]
      H_tokens, W_tokens = H // p, W // p
      assert H_tokens * W_tokens == l_effective_img_len[i]

      row_ids = repeat(
        torch.arange(H_tokens, dtype=torch.int32, device=device),
        "h -> h w",
        w=W_tokens,
      ).flatten()
      col_ids = repeat(
        torch.arange(W_tokens, dtype=torch.int32, device=device),
        "w -> h w",
        h=H_tokens,
      ).flatten()

      assert pe_shift_len + l_effective_img_len[i] == seq_len
      position_ids[i, pe_shift_len:seq_len, 0] = pe_shift
      position_ids[i, pe_shift_len:seq_len, 1] = row_ids
      position_ids[i, pe_shift_len:seq_len, 2] = col_ids

    # Get combined rotary embeddings
    freqs_cis = self._get_freqs_cis(freqs_cis, position_ids)

    # create separate rotary embeddings for captions and images
    cap_freqs_cis = torch.zeros(
      batch_size,
      encoder_seq_len,
      freqs_cis.shape[-1],
      device=device,
      dtype=freqs_cis.dtype,
    )
    ref_img_freqs_cis = torch.zeros(
      batch_size,
      max_ref_img_len,
      freqs_cis.shape[-1],
      device=device,
      dtype=freqs_cis.dtype,
    )
    img_freqs_cis = torch.zeros(
      batch_size,
      max_img_len,
      freqs_cis.shape[-1],
      device=device,
      dtype=freqs_cis.dtype,
    )

    for i, (cap_seq_len, ref_img_len, img_len, seq_len) in enumerate(
        zip(
          l_effective_cap_len,
          l_effective_ref_img_len,
          l_effective_img_len,
          seq_lengths,
        )):
      cap_freqs_cis[i, :cap_seq_len] = freqs_cis[i, :cap_seq_len]
      ref_img_freqs_cis[i, :sum(ref_img_len)] = freqs_cis[i, cap_seq_len:cap_seq_len +
                                                          sum(ref_img_len)]
      img_freqs_cis[i, :img_len] = freqs_cis[
        i,
        cap_seq_len + sum(ref_img_len):cap_seq_len + sum(ref_img_len) + img_len,
      ]

    return (
      cap_freqs_cis,
      ref_img_freqs_cis,
      img_freqs_cis,
      freqs_cis,
      l_effective_cap_len,
      seq_lengths,
    )


class BooguImageDoubleStreamRotaryPosEmbed(nn.Module):

  def __init__(
      self,
      theta: int,
      axes_dim: Tuple[int, int, int],
      axes_lens: Tuple[int, int, int] = (300, 512, 512),
      patch_size: int = 2,
  ):
    super().__init__()
    self.theta = theta
    self.axes_dim = axes_dim
    self.axes_lens = _resolve_axes_lens(axes_lens)
    self.patch_size = patch_size
    self._rope_cache = {}

  @staticmethod
  def _rope_cache_key(freqs_cis, attention_mask, l_effective_ref_img_len, l_effective_img_len,
                      ref_img_sizes, img_sizes, device):
    # The output depends on the mask only through its per-row sums
    # (l_effective_cap_len); hash those instead of data_ptr, which the
    # caching allocator can reuse for a different mask of equal shape.
    ref_sizes = tuple(tuple(s) if s is not None else None for s in ref_img_sizes)
    ref_lens = tuple(tuple(ln) for ln in l_effective_ref_img_len)
    return (
      tuple(f.data_ptr() for f in freqs_cis),
      tuple(attention_mask.shape),
      tuple(attention_mask.sum(dim=1).tolist()),
      ref_lens,
      tuple(l_effective_img_len),
      ref_sizes,
      tuple(img_sizes),
      str(device),
    )

  def forward(
    self,
    freqs_cis,
    attention_mask,
    l_effective_ref_img_len,
    l_effective_img_len,
    ref_img_sizes,
    img_sizes,
    device,
  ):
    """Return rotary embeddings for captions / ref images / images.

    Diffusion sampling repeats this pure function with identical inputs
    every step; cache the output to skip position-id/gather/split work.
    Returned tensors are shared across steps and must be treated as
    read-only by callers.

    :param freqs_cis: per-axis rotary tables from :meth:`get_freqs_cis`
    :param attention_mask: [B, S] instruction padding mask
    :param l_effective_ref_img_len: per-sample list of ref-image lengths
    :param l_effective_img_len: per-sample image token lengths
    :param ref_img_sizes: per-sample list of (H, W) or None
    :param img_sizes: per-sample (H, W)
    :param device: target device
    :returns: (cap, ref_img, img, joint rotary tensors, cap lens,
      seq lengths, combined img rotary, combined img lens)
    """
    key = self._rope_cache_key(freqs_cis, attention_mask, l_effective_ref_img_len,
                               l_effective_img_len, ref_img_sizes, img_sizes, device)
    cached = self._rope_cache.get(key)
    if cached is not None:
      return cached
    out = self._forward_impl(freqs_cis, attention_mask, l_effective_ref_img_len,
                             l_effective_img_len, ref_img_sizes, img_sizes, device)
    if len(self._rope_cache) >= 8:
      self._rope_cache.clear()
    self._rope_cache[key] = out
    return out

  @staticmethod
  def get_freqs_cis(axes_dim: Tuple[int, int, int], axes_lens: Tuple[int, int, int],
                    theta: int) -> List[torch.Tensor]:
    freqs_cis = []
    freqs_dtype = (torch.float32 if torch.backends.mps.is_available() else torch.float64)
    for i, (d, e) in enumerate(zip(axes_dim, axes_lens)):
      emb = get_1d_rotary_pos_embed(d, e, theta=theta, freqs_dtype=freqs_dtype)
      freqs_cis.append(emb)
    return freqs_cis

  def _get_freqs_cis(self, freqs_cis, ids: torch.Tensor) -> torch.Tensor:
    device = ids.device
    if ids.device.type == "mps":
      ids = ids.to("cpu")

    result = []
    for i in range(len(self.axes_dim)):
      freqs = freqs_cis[i].to(ids.device)
      index = ids[:, :, i:i + 1].repeat(1, 1, freqs.shape[-1]).to(torch.int64)
      result.append(
        torch.gather(freqs.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index))
    return torch.cat(result, dim=-1).to(device)

  def _forward_impl(
    self,
    freqs_cis,
    attention_mask,
    l_effective_ref_img_len,
    l_effective_img_len,
    ref_img_sizes,
    img_sizes,
    device,
  ):
    batch_size = len(attention_mask)
    p = self.patch_size

    encoder_seq_len = attention_mask.shape[1]
    l_effective_cap_len = attention_mask.sum(dim=1).tolist()

    seq_lengths = [
      cap_len + sum(ref_img_len) + img_len for cap_len, ref_img_len, img_len in zip(
        l_effective_cap_len, l_effective_ref_img_len, l_effective_img_len)
    ]

    max_seq_len = max(seq_lengths)
    max_ref_img_len = max([sum(ref_img_len) for ref_img_len in l_effective_ref_img_len])
    max_img_len = max(l_effective_img_len)

    # Create position IDs
    position_ids = torch.zeros(batch_size, max_seq_len, 3, dtype=torch.int32, device=device)

    for i, (cap_seq_len, seq_len) in enumerate(zip(l_effective_cap_len, seq_lengths)):
      # add text position ids
      position_ids[i, :cap_seq_len] = repeat(
        torch.arange(cap_seq_len, dtype=torch.int32, device=device), "l -> l 3")

      pe_shift = cap_seq_len
      pe_shift_len = cap_seq_len

      if ref_img_sizes[i] is not None:
        for ref_img_size, ref_img_len in zip(ref_img_sizes[i], l_effective_ref_img_len[i]):
          H, W = ref_img_size
          ref_H_tokens, ref_W_tokens = H // p, W // p
          assert ref_H_tokens * ref_W_tokens == ref_img_len
          # add image position ids

          row_ids = repeat(
            torch.arange(ref_H_tokens, dtype=torch.int32, device=device),
            "h -> h w",
            w=ref_W_tokens,
          ).flatten()
          col_ids = repeat(
            torch.arange(ref_W_tokens, dtype=torch.int32, device=device),
            "w -> h w",
            h=ref_H_tokens,
          ).flatten()
          position_ids[i, pe_shift_len:pe_shift_len + ref_img_len, 0] = (pe_shift)
          position_ids[i, pe_shift_len:pe_shift_len + ref_img_len, 1] = (row_ids)
          position_ids[i, pe_shift_len:pe_shift_len + ref_img_len, 2] = (col_ids)

          pe_shift += max(ref_H_tokens, ref_W_tokens)
          pe_shift_len += ref_img_len

      H, W = img_sizes[i]
      H_tokens, W_tokens = H // p, W // p
      assert H_tokens * W_tokens == l_effective_img_len[i]

      row_ids = repeat(
        torch.arange(H_tokens, dtype=torch.int32, device=device),
        "h -> h w",
        w=W_tokens,
      ).flatten()
      col_ids = repeat(
        torch.arange(W_tokens, dtype=torch.int32, device=device),
        "w -> h w",
        h=H_tokens,
      ).flatten()

      assert pe_shift_len + l_effective_img_len[i] == seq_len
      position_ids[i, pe_shift_len:seq_len, 0] = pe_shift
      position_ids[i, pe_shift_len:seq_len, 1] = row_ids
      position_ids[i, pe_shift_len:seq_len, 2] = col_ids

    # Get combined rotary embeddings
    freqs_cis = self._get_freqs_cis(freqs_cis, position_ids)

    # create separate rotary embeddings for captions and images
    cap_freqs_cis = torch.zeros(
      batch_size,
      encoder_seq_len,
      freqs_cis.shape[-1],
      device=device,
      dtype=freqs_cis.dtype,
    )
    ref_img_freqs_cis = torch.zeros(
      batch_size,
      max_ref_img_len,
      freqs_cis.shape[-1],
      device=device,
      dtype=freqs_cis.dtype,
    )
    img_freqs_cis = torch.zeros(
      batch_size,
      max_img_len,
      freqs_cis.shape[-1],
      device=device,
      dtype=freqs_cis.dtype,
    )

    # Calculate combined image sequence lengths (ref_img + img) for each sample
    combined_img_seq_lengths = [
      sum(ref_img_len) + img_len
      for ref_img_len, img_len in zip(l_effective_ref_img_len, l_effective_img_len)
    ]
    max_combined_img_len = max(combined_img_seq_lengths)

    # Create combined image rotary embeddings
    combined_img_freqs_cis = torch.zeros(
      batch_size,
      max_combined_img_len,
      freqs_cis.shape[-1],
      device=device,
      dtype=freqs_cis.dtype,
    )

    for i, (cap_seq_len, ref_img_len, img_len, seq_len) in enumerate(
        zip(
          l_effective_cap_len,
          l_effective_ref_img_len,
          l_effective_img_len,
          seq_lengths,
        )):
      cap_freqs_cis[i, :cap_seq_len] = freqs_cis[i, :cap_seq_len]
      ref_img_freqs_cis[i, :sum(ref_img_len)] = freqs_cis[i, cap_seq_len:cap_seq_len +
                                                          sum(ref_img_len)]
      img_freqs_cis[i, :img_len] = freqs_cis[
        i,
        cap_seq_len + sum(ref_img_len):cap_seq_len + sum(ref_img_len) + img_len,
      ]

      # Combined image rotary embeddings: ref_img + img (same order as img_patch_embed_and_refine)
      combined_img_freqs_cis[i, :sum(ref_img_len)] = freqs_cis[i, cap_seq_len:cap_seq_len +
                                                               sum(ref_img_len)]
      combined_img_freqs_cis[i, sum(ref_img_len):sum(ref_img_len) + img_len] = (freqs_cis[
        i,
        cap_seq_len + sum(ref_img_len):cap_seq_len + sum(ref_img_len) + img_len,
      ])

    return (
      cap_freqs_cis,
      ref_img_freqs_cis,
      img_freqs_cis,
      freqs_cis,
      l_effective_cap_len,
      seq_lengths,
      combined_img_freqs_cis,
      combined_img_seq_lengths,
    )


class BooguImagePromptTuningRotaryPosEmbed(nn.Module):
  """Rotary Position Embedding for Prompt Tuning tokens.

  This class generates rotary position embeddings specifically for prompt tuning tokens.
  Since prompt tokens are treated as text tokens, we use text-style position encoding
  with a fixed sequence length equal to num_trainable_prompt_tokens.

  Args:
      theta: Base frequency for rotary embeddings
      axes_dim: Dimensions for each axis (tuple like (32, 32, 32))
      num_trainable_prompt_tokens: Number of trainable prompt tokens
  """

  def __init__(self, theta: int, dim: int, num_trainable_prompt_tokens: int):
    super().__init__()
    self.theta = theta
    self.num_trainable_prompt_tokens = num_trainable_prompt_tokens
    # For text tokens, only use the first dimension (text/temporal dimension)
    self.dim = dim  # Extract text dimension from tuple

  def forward(self,
              batch_size: int,
              device: torch.device,
              use_causal_mask: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate rotary position embeddings and attention mask for prompt tuning.

    Args:
        batch_size: Batch size
        device: Target device for tensors
        use_causal_mask: Whether to use causal attention mask

    Returns:
        Tuple of (rotary_embeddings, attention_mask)
        - rotary_embeddings: [B, num_tokens, instruction_dim//2] - RoPE embeddings for prompt tokens (complex form)
        - attention_mask: [B, num_tokens] or [B, num_tokens, num_tokens] - Attention mask
    """
    # Generate 1D rotary embeddings for text-style tokens
    freqs_dtype = (torch.float32 if torch.backends.mps.is_available() else torch.float64)

    # get_1d_rotary_pos_embed(dim, seq_len) returns [seq_len, dim//2]
    # Because RoPE uses complex representation, each dimension is split into sin/cos pairs
    text_freqs_cis = get_1d_rotary_pos_embed(
      self.dim,  # This should be 32 (text dimension)
      self.num_trainable_prompt_tokens,  # Sequence length
      theta=self.theta,
      freqs_dtype=freqs_dtype,
    )

    # For prompt tuning, we create simple sequential position embeddings
    # Each prompt token gets a unique position ID: 0, 1, 2, ..., num_tokens-1
    position_indices = torch.arange(
      self.num_trainable_prompt_tokens,
      dtype=torch.int64,
      device=text_freqs_cis.device,
    )

    # Select the appropriate rotary embeddings for each position
    # text_freqs_cis is [num_tokens, instruction_dim//2], we want [num_tokens, instruction_dim//2]
    rotary_emb = text_freqs_cis[position_indices]  # [num_tokens, instruction_dim//2]

    # Expand to batch size and move to target device
    rotary_emb = (rotary_emb.unsqueeze(0).expand(batch_size, -1, -1).to(device)
                  )  # [B, num_tokens, instruction_dim//2]

    # Create attention mask based on use_causal_mask parameter
    if use_causal_mask:
      # Create causal mask: only future tokens can attend to past tokens
      # Lower triangular matrix where mask[i, j] = True if i >= j
      causal_mask = torch.tril(
        torch.ones(
          self.num_trainable_prompt_tokens,
          self.num_trainable_prompt_tokens,
          dtype=torch.bool,
          device=device,
        ))  # [num_tokens, num_tokens]

      # Expand to batch size [B, num_tokens, num_tokens]
      attention_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
    else:
      # Non-causal mask: all tokens can attend to each other (all True)
      attention_mask = torch.ones(
        batch_size,
        self.num_trainable_prompt_tokens,
        dtype=torch.bool,
        device=device,
      )  # [B, num_tokens]

    return rotary_emb, attention_mask
