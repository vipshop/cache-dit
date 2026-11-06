"""Segment KV cache and the attention plan that reads it.

Copied from ``sglang.multimodal_gen.runtime.models.dits.nomas_kv_cache`` (Apache-2.0), itself
adapted from Tencent-Hunyuan/Hy-Nomas.  Values, masks and the commit/evict protocol are unchanged;
the ``msgspec`` structs became dataclasses.

The cache stores committed segments (video tokens then text tokens) with their visibility.  The
plan turns history visibility, text padding and action windows into one of three equivalent
executions:

* ``fa3``: dense FlashAttention-3 over the visible prefix ``[visible history | current video | head
  text]``, plus one small pass per action over that action's text, merged by log-sum-exp.
* ``flex``: a FlexAttention block mask (CUDA).
* ``sdpa``: a bounded dense mask, the CPU and test reference.

``NomasAttentionPlan.__call__`` is the **single** place in this package allowed to bypass
cache-dit's attention dispatcher: ``_dispatch_attention_fn`` only accepts a dense ``attn_mask``
tensor, while this model needs a varlen log-sum-exp merge (``fa3``) or a FlexAttention
``BlockMask`` (``flex``).  :mod:`tests.api.test_hy_nomas_transformer` guards that scope.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F


@dataclass
class NomasCache:
  """Committed K/V of earlier segments plus their visibility.

  :param layers: One ``{"k", "v"}`` tensor pair per transformer layer.
  :param visible: ``[1, capacity]`` visibility mask of the committed rows.
  :param valid_length: Number of committed rows.
  :param frame_offset: Absolute latent-frame offset of the next segment (drives RoPE).
  :param action_tails: ``(start, end)`` token ranges that may be hidden while denoising.
  :param segment_lengths: Token length of every committed segment, oldest first.
  :param failed: Set when a commit or eviction failed; the session must be reset.
  """

  layers: List[Dict[str, torch.Tensor]]
  visible: torch.Tensor
  valid_length: int = 0
  frame_offset: int = 0
  action_tails: List[Tuple[int, int]] = field(default_factory=list)
  segment_lengths: List[int] = field(default_factory=list)
  failed: bool = False

  @classmethod
  def allocate(
    cls,
    config,
    capacity: int,
    *,
    device,
    dtype,
    num_heads: Optional[int] = None,
  ) -> "NomasCache":
    """Allocate a zeroed cache.

    :param config: Transformer config exposing ``num_layers``, ``num_attention_heads`` and
      ``hidden_size``.
    :param capacity: Row capacity in tokens.
    :param device: Device of the K/V tensors.
    :param dtype: Dtype of the K/V tensors.
    :param num_heads: Per-rank head count under tensor or sequence parallelism.
    :returns: An empty cache.
    """
    shape = (
      1,
      capacity,
      num_heads or config.num_attention_heads,
      config.hidden_size // config.num_attention_heads,
    )
    layers = [{
      key: torch.zeros(shape, device=device, dtype=dtype)
      for key in ("k", "v")
    } for _ in range(config.num_layers)]
    return cls(layers, torch.zeros(1, capacity, device=device, dtype=torch.bool))

  @property
  def capacity(self) -> int:
    return self.visible.shape[1]

  def _check_usable(self) -> None:
    if self.failed:
      raise RuntimeError("cache is invalid after a failed commit; reset the session")

  def reserve(self, tokens: int) -> None:
    self._check_usable()
    if tokens <= 0 or self.valid_length + tokens > self.capacity:
      raise ValueError("Nomas cache capacity exceeded; evict older segments or raise the budget")

  def history_mask(self, hide_action_tails: bool) -> torch.Tensor:
    self._check_usable()
    mask = self.visible[:, :self.valid_length].clone()
    if hide_action_tails:
      for start, end in self.action_tails:
        mask[:, start:end] = False
    return mask

  def commit(
    self,
    entries: Sequence[Dict[str, torch.Tensor]],
    text_mask: torch.Tensor,
    video_tokens: int,
    frames: int,
    action_start: int = -1,
  ) -> None:
    """Publish a whole segment only after every model layer succeeded.

    :param entries: One ``{"k", "v"}`` pair per transformer layer, in layer order.
    :param text_mask: ``[1, S]`` boolean mask of the committed segment's text tokens.
    :param video_tokens: Video token count of the segment.
    :param frames: Latent frame count of the segment.
    :param action_start: Offset of the action text inside the text block, ``-1`` for none.
    :raises ValueError: When the entry count, shape, dtype or device does not match the cache.
    """
    tokens = video_tokens + text_mask.shape[1]
    self.reserve(tokens)
    if len(entries) != len(self.layers):
      raise ValueError("cache layer count mismatch")
    start, end = self.valid_length, self.valid_length + tokens
    for target, entry in zip(self.layers, entries):
      for key in ("k", "v"):
        expected = target[key][:, start:end]
        value = entry[key]
        if (value.shape != expected.shape or value.device != expected.device
            or value.dtype != expected.dtype):
          raise ValueError("cache entry shape, dtype or device mismatch")
    try:
      for target, entry in zip(self.layers, entries):
        for key in ("k", "v"):
          target[key][:, start:end].copy_(entry[key])
      self.visible[:, start:start + video_tokens] = True
      self.visible[:, start + video_tokens:end] = text_mask
      if action_start >= 0:
        self.action_tails.append((start + video_tokens + action_start, end))
      self.segment_lengths.append(tokens)
      self.valid_length = end
      self.frame_offset += frames
    except Exception:
      self.failed = True
      raise

  def evict_oldest_segment(self) -> int:
    """Drop the oldest committed segment, as upstream ``causal_ar`` does.

    Surviving K/V and visibility rows shift left; action tails shift with them and tails that fell
    out are dropped.  The RoPE frame offset keeps counting, so later segments keep their absolute
    positions.

    :returns: Number of evicted tokens.
    """
    self._check_usable()
    if not self.segment_lengths:
      return 0
    evicted = self.segment_lengths.pop(0)
    remaining = self.valid_length - evicted
    try:
      if remaining > 0:
        for layer in self.layers:
          for key in ("k", "v"):
            layer[key][:, :remaining].copy_(layer[key][:, evicted:self.valid_length].clone())
        self.visible[:, :remaining] = self.visible[:, evicted:self.valid_length].clone()
      self.visible[:, remaining:] = False
      self.action_tails = [(
        max(start - evicted, 0),
        end - evicted,
      ) for start, end in self.action_tails if end - evicted > 0]
      self.valid_length = remaining
    except Exception:
      self.failed = True
      raise
    return evicted

  def grow(self, capacity: int) -> None:
    """Reallocate to a larger capacity, keeping committed rows.

    :param capacity: New row capacity; a no-op when it is not larger.
    """
    if capacity <= self.capacity:
      return
    self._check_usable()
    for layer in self.layers:
      for key in ("k", "v"):
        old = layer[key]
        new = torch.zeros((old.shape[0], capacity, *old.shape[2:]),
                          device=old.device,
                          dtype=old.dtype)
        new[:, :self.valid_length].copy_(old[:, :self.valid_length])
        layer[key] = new
    visible = torch.zeros(1, capacity, device=self.visible.device, dtype=torch.bool)
    visible[:, :self.valid_length] = self.visible[:, :self.valid_length]
    self.visible = visible

  def evict_for(self, tokens: int) -> int:
    """Evict oldest segments until ``tokens`` fit.

    :param tokens: Token count of the segment about to be committed.
    :returns: Number of evicted tokens.
    """
    evicted = 0
    while self.segment_lengths and self.valid_length + tokens > self.capacity:
      evicted += self.evict_oldest_segment()
    return evicted

  def reset(self) -> None:
    # Uncommitted storage is never visible; clearing K/V is unnecessary.
    self.visible.zero_()
    self.valid_length = self.frame_offset = 0
    self.action_tails.clear()
    self.segment_lengths.clear()
    self.failed = False


@lru_cache(maxsize=1)
def _flex_kernel() -> Callable:
  from torch.nn.attention.flex_attention import flex_attention

  return torch.compile(flex_attention, dynamic=True, fullgraph=True)


def _varlen_lse(q, k, v, cu_q, cu_k, max_q, max_k, heads):
  """Varlen flash attention that also returns the log-sum-exp as ``[heads, total_q]``.

  Both wheels already return that layout: FlashAttention-2 allocates ``torch::empty({num_heads,
  total_q})`` (``flash-attention/csrc/flash_attn/flash_api.cpp``) and FlashAttention-3 does the same
  in ``hopper/flash_api.cpp``.  The layout is therefore taken from the contract and asserted, never
  guessed from the shape: with ``total_q == heads`` a guess would silently transpose the merge
  weights.  Only the keyword that *exposes* the log-sum-exp differs (``return_attn_probs`` vs
  ``return_softmax_lse``), which is what the two branches below handle.

  :param q: ``[total_q, heads, head_dim]`` packed queries.
  :param k: ``[total_k, heads, head_dim]`` packed keys.
  :param v: ``[total_k, heads, head_dim]`` packed values.
  :param cu_q: Cumulative query lengths (int32).
  :param cu_k: Cumulative key lengths (int32).
  :param max_q: Number of query rows.
  :param max_k: Number of key rows.
  :param heads: Head count of ``q``.
  :returns: ``(out, lse)``.
  """
  try:
    from flash_attn_interface import flash_attn_varlen_func as varlen
  except ImportError:
    from flash_attn import flash_attn_varlen_func as varlen

    out, lse, *_ = varlen(
      q,
      k,
      v,
      cu_q,
      cu_k,
      max_q,
      max_k,
      return_attn_probs=True,
    )
  else:
    out, lse, *_ = varlen(
      q,
      k,
      v,
      cu_q,
      cu_k,
      max_q,
      max_k,
      return_softmax_lse=True,
    )
  if tuple(lse.shape) != (heads, max_q):
    raise RuntimeError(f"varlen attention returned log-sum-exp of shape {tuple(lse.shape)}, "
                       f"expected {(heads, max_q)}; the installed flash-attention wheel uses an "
                       f"unsupported layout")
  return out, lse


@lru_cache(maxsize=1)
def _block_mask_builder() -> Callable:
  from torch.nn.attention.flex_attention import create_block_mask

  # Eager create_block_mask materializes a Q_LEN x KV_LEN boolean tensor.  Compile the reduction so
  # that only block-level metadata is materialized.  History length grows on every segment; keep it
  # symbolic, and fullgraph prevents a graph break from silently executing the dense eager builder.
  return torch.compile(create_block_mask, dynamic=True, fullgraph=True)


class NomasAttentionPlan:
  """Masked attention over ``[history | current video | text]`` for one segment.

  :param text_mask: ``[1, S]`` boolean mask with a valid prefix and right padding.
  :param history_mask: ``[1, H]`` visibility of the committed rows.
  :param frames: Latent frames of the current segment.
  :param spatial_tokens: Tokens per latent frame (``H_lat * W_lat``).
  :param text_offsets: Optional partition of head/action text, ending at the valid text length.
  :param video_windows: Optional ``[[start_frame, end_frame, action_index], ...]`` tiling.
  :param backend: ``fa3``, ``flex`` or ``sdpa``.
  """

  def __init__(
    self,
    text_mask: torch.Tensor,
    history_mask: torch.Tensor,
    frames: int,
    spatial_tokens: int,
    text_offsets: Optional[Sequence[int]] = None,
    video_windows: Optional[Sequence[Sequence[int]]] = None,
    backend: str = "flex",
  ):
    if backend not in ("fa3", "flex", "sdpa"):
      raise ValueError("Nomas attention backend must be fa3, flex or sdpa")
    if (text_mask.ndim != 2 or text_mask.shape[0] != 1 or text_mask.dtype != torch.bool):
      raise ValueError("Nomas requires a boolean text mask with batch size 1")
    valid = int(text_mask.sum().item())
    if valid == 0 or not torch.equal(
        text_mask[0],
        torch.arange(text_mask.shape[1], device=text_mask.device) < valid,
    ):
      raise ValueError("text mask must contain a nonempty valid prefix and right padding")
    device = text_mask.device
    self.video_tokens = frames * spatial_tokens
    n = self.video_tokens + text_mask.shape[1]
    self.query_valid = torch.cat((
      torch.ones(1, self.video_tokens, device=device, dtype=torch.bool),
      text_mask,
    ), 1)
    key_valid = torch.cat((history_mask, self.query_valid), 1)[0]
    query_action = torch.full((n, ), -1, device=device, dtype=torch.int32)
    current_key_action = torch.full_like(query_action, -1)
    if (text_offsets is None) != (video_windows is None):
      raise ValueError("text offsets and video windows must be supplied together")
    if text_offsets is not None:
      offsets = list(text_offsets)
      if (len(offsets) < 2 or any(type(x) is not int for x in offsets) or offsets[0] < 0
          or offsets[-1] != valid or any(a > b for a, b in zip(offsets, offsets[1:]))):
        raise ValueError("text offsets must partition head/action text and end at the valid "
                         "text length")
      cursor = 0
      for window in video_windows:
        if len(window) != 3 or any(type(x) is not int for x in window):
          raise ValueError("each video window must be [start_frame, end_frame, action_index]")
        start, end, action = window
        if (start != cursor or not start < end <= frames or not -1 <= action < len(offsets) - 1):
          raise ValueError("video windows must tile the segment without gaps/overlap")
        query_action[start * spatial_tokens:end * spatial_tokens] = action
        cursor = end
      if cursor != frames:
        raise ValueError("video windows must cover all latent frames")
      for action, (start, end) in enumerate(zip(offsets, offsets[1:])):
        sl = slice(self.video_tokens + start, self.video_tokens + end)
        query_action[sl] = action
        current_key_action[sl] = action
    key_action = torch.cat((
      torch.full((history_mask.shape[1], ), -1, device=device, dtype=torch.int32),
      current_key_action,
    ))

    def mask_mod(b, h, q, k):
      # Block-mask construction also visits padded block positions.
      qi, ki = q.clamp(max=n - 1), k.clamp(max=key_valid.numel() - 1)
      return ((q < n) & (k < key_valid.numel()) & key_valid[ki]
              & ((key_action[ki] < 0) | (query_action[qi] == key_action[ki])))

    self.backend = backend
    self.history_length = history_mask.shape[1]
    if backend == "fa3":
      self._prepare_fa3(history_mask, valid, text_offsets, video_windows, spatial_tokens)
      return
    if backend == "flex":
      if device.type != "cuda":
        raise ValueError("FlexAttention requires CUDA; use sdpa for the CPU reference")
      self.mask = _block_mask_builder()(mask_mod, 1, None, n, key_valid.numel(), device=str(device))
    else:
      # Bounded debug/reference backend, never silently allocate an enormous mask.
      if n * key_valid.numel() > 16_777_216:
        raise ValueError("SDPA reference mask is too large; select flex")
      self.mask = mask_mod(0, 0,
                           torch.arange(n, device=device)[:, None],
                           torch.arange(key_valid.numel(), device=device)[None, :])

  def _prepare_fa3(self, history_mask, valid, offsets, windows, spatial_tokens) -> None:
    """Index metadata for the compact layout ``[visible history | video | valid text]``."""
    device = history_mask.device
    self.history_index = history_mask[0].nonzero().squeeze(-1)
    hvis = self.history_index.numel()
    video = self.video_tokens
    self.compact_tokens = video + valid
    head_end = valid if offsets is None else offsets[0]
    self.prefix_tokens = hvis + video + head_end
    n = self.query_valid.shape[1]
    int32 = dict(device=device, dtype=torch.int32)
    self.cu_q = torch.tensor([0, n], **int32)
    self.cu_prefix = torch.tensor([0, self.prefix_tokens], **int32)
    self.max_q = n
    # Each action: its video window rows plus its own text rows see that text.  Both are contiguous
    # query ranges, so they are passed as views.
    self.actions = []
    for action, (start, end) in enumerate(zip(offsets or (), (offsets or ())[1:])):
      if end <= start:
        continue
      rows = [(a * spatial_tokens, b * spatial_tokens) for a, b, index in windows
              if index == action]
      rows.append((video + start, video + end))
      keys = (hvis + video + start, hvis + video + end)
      self.actions.extend((q_range, keys, torch.tensor([0, q_range[1] - q_range[0]], **int32),
                           torch.tensor([0, keys[1] - keys[0]], **int32)) for q_range in rows
                          if q_range[1] > q_range[0])

  def __call__(self, q, k, v, layer_cache=None, history_length=0):
    """Attend over the current segment plus the cached history.

    ``k`` / ``v`` hold the current segment; history is read from ``layer_cache``.

    :param q: ``[1, N, H, D]`` queries (video then text), per-rank heads under parallelism.
    :param k: ``[1, N, H, D]`` keys of the current segment.
    :param v: ``[1, N, H, D]`` values of the current segment.
    :param layer_cache: ``{"k", "v"}`` history of this layer, or ``None``.
    :param history_length: Committed rows of ``layer_cache`` to attend to.
    :returns: ``[1, N, H, D]``; padded text queries are zero.
    """
    if history_length != self.history_length:
      raise ValueError("attention plan was built for a different history length")
    if self.backend == "fa3":
      return self._fa3(q, k, v, layer_cache)
    if history_length:
      k = torch.cat((layer_cache["k"][:, :history_length], k), 1)
      v = torch.cat((layer_cache["v"][:, :history_length], v), 1)
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    if self.backend == "flex":
      result = _flex_kernel()(q, k, v, block_mask=self.mask)
    else:
      result = F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask)
    return result.transpose(1, 2) * self.query_valid[:, :, None, None]

  def _fa3(self, q, k, v, layer_cache):

    def compact(current, key):
      # [visible history | current video | valid text]: padding and hidden action tails are dropped
      # instead of masked.  History is gathered straight into the buffer.
      current = current[0, :self.compact_tokens]
      history = self.history_index.numel()
      if history == 0:
        return current.contiguous()
      buffer = current.new_empty((history + current.shape[0], *current.shape[1:]))
      torch.index_select(layer_cache[key][0], 0, self.history_index, out=buffer[:history])
      buffer[history:].copy_(current)
      return buffer

    k, v = compact(k, "k"), compact(v, "v")
    q = q[0].contiguous()
    heads = q.shape[1]
    prefix = self.prefix_tokens
    out, lse = _varlen_lse(q, k[:prefix], v[:prefix], self.cu_q, self.cu_prefix, self.max_q, prefix,
                           heads)
    # Action rows also attend to their own action text: merge the two softmax partitions exactly
    # through their log-sum-exp, in place.
    for (q_start, q_end), (k_start, k_end), cu_rows, cu_keys in self.actions:
      extra, extra_lse = _varlen_lse(q[q_start:q_end], k[k_start:k_end], v[k_start:k_end], cu_rows,
                                     cu_keys, q_end - q_start, k_end - k_start, heads)
      base_lse, extra_lse = lse[:, q_start:q_end].float(), extra_lse.float()
      top = torch.maximum(base_lse, extra_lse)
      w_base, w_extra = torch.exp(base_lse - top), torch.exp(extra_lse - top)
      total = (w_base + w_extra).T[..., None]
      rows = out[q_start:q_end]
      blended = (rows.float() * w_base.T[..., None] + extra.float() * w_extra.T[..., None]) / total
      rows.copy_(blended)
    # Padded text queries see nothing; x * 0 here matches the masked backends.
    out[self.compact_tokens:].mul_(0)
    return out[None]
