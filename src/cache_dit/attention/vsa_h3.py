"""Video sparse attention (VSA) for the FastH3 MiniMax-H3 students.

The FastH3 preview students are distilled with a 90 % sparse video attention: a query row of the
generated video keeps the full dense prefix (text + audio) plus the top-k 3-D tiles of video keys,
scored by the mean-pooled tile representatives. A learned per-head gate then adds a *compression*
branch, dense attention over the pooled tile means::

    out = sparse_attention(q, k, v, selected tiles)
    out += gate * softmax(pooled_q @ pooled_k.T * scale) @ pooled_v

Both branches exist in the reference implementations (FastVideo ``VIDEO_SPARSE_ATTN_H3``, xDiT
``FLEX_VSA_H3``); this module reproduces their semantics with stock PyTorch so it runs on any CUDA
device, including sm_120, which neither project's Blackwell block-sparse kernel covers.

Two interchangeable kernels, same math:

* ``impl="flex"`` (default where supported) hands a `BlockMask` built from k/v block index tensors
  to `torch.nn.attention.flex_attention`, which keeps one fused block-sparse kernel per layer.
  Requires torch >= 2.12: the block-sparse Triton template of torch 2.11 does not compile
  ("Loop-carried variable kv_offset has initial type int32 but is re-assigned to int64").
* ``impl="gather"`` is the portable baseline: query tiles are gathered per chunk with only the
  selected key rows and handed to `scaled_dot_product_attention` with a padding mask. It is exact
  but launch-bound (one small attention per query-tile chunk), which is why the fused path exists.

``impl="auto"`` picks flex when the interpreter's torch has a working block-sparse template and
gather otherwise. Both paths share the tile selection, the fp32 pooling and the gate fold, so an
A/B between them isolates the kernel.

Implementation notes:

* Selection runs on **device tensors only** — no `.item()`, no `.tolist()` — so a step never
  synchronizes with the host.
* The block lists have static widths taken from the geometry (its full and partial tile counts), so
  neither path recompiles when the selection changes.
* Padding inside a partial tile is masked by a `mask_mod` closure over ``VsaH3Meta.tiled_valid``;
  full tiles are handed to the kernel as unmasked blocks.
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import Optional

import torch
from torch.nn import functional as F

from .vsa_h3_geometry import VsaH3Meta

__all__ = [
  "build_vsa_h3_keep_matrix",
  "compute_vsa_h3_topk",
  "pool_vsa_h3_tiles",
  "vsa_h3_impl",
  "vsa_h3_attention",
  "vsa_h3_attention_reference",
]

VSA_H3_IMPL_ENV = "CACHE_DIT_VSA_H3_IMPL"
_DEFAULT_QUERY_TILE_CHUNK = 4


def _torch_has_block_sparse_flex() -> bool:
  """Whether this torch's FlexAttention block-sparse template compiles.

  torch 2.11 promotes the block-index operands inconsistently and the Triton template fails to
  build; 2.12 and newer are known good, and every dev environment here pins one of the two.
  """
  raw = torch.__version__.split("+")[0].split(".")
  version = tuple(int(part) for part in raw[:2])
  return version >= (2, 12)


def vsa_h3_impl(impl: Optional[str] = None) -> str:
  """Resolve which kernel implementation to use.

  :param impl: ``"flex"``, ``"gather"`` or ``"auto"``/``None``; ``"auto"`` follows the torch
    version, and `CACHE_DIT_VSA_H3_IMPL` overrides the default.
  :returns: ``"flex"`` or ``"gather"``.
  """
  requested = (impl or os.getenv(VSA_H3_IMPL_ENV) or "auto").lower()
  if requested not in ("auto", "flex", "gather"):
    raise ValueError(f"{VSA_H3_IMPL_ENV} must be 'flex', 'gather' or 'auto', got {requested!r}.")
  if requested == "flex" and not _torch_has_block_sparse_flex():
    raise ValueError(f"torch {torch.__version__} cannot run FlexAttention block sparsity: its "
                     "Triton template compiles but returns wrong values here (checked against the "
                     "gather baseline on torch 2.11). Use torch >= 2.12 or "
                     f"{VSA_H3_IMPL_ENV}=gather.")
  if requested == "auto":
    return "flex" if _torch_has_block_sparse_flex() else "gather"
  return requested


def compute_vsa_h3_topk(sparsity: float, num_video_tiles: int) -> int:
  """Number of video tiles a sparse query keeps.

  :param sparsity: fraction of video tiles to drop, e.g. ``0.9`` for the 4-step students.
  :param num_video_tiles: number of tiles the video grid holds.
  :returns: ``max(1, ceil((1 - sparsity) * num_video_tiles))``, clamped to ``num_video_tiles``.
  """
  if not 0.0 <= float(sparsity) <= 1.0:
    raise ValueError(f"VSA-H3 sparsity must be in [0, 1], got {sparsity}.")
  if num_video_tiles < 1:
    raise ValueError(f"VSA-H3 needs at least one video tile, got {num_video_tiles}.")
  return max(1, min(math.ceil((1.0 - float(sparsity)) * num_video_tiles), num_video_tiles))


def pool_vsa_h3_tiles(x: torch.Tensor, meta: VsaH3Meta) -> torch.Tensor:
  """Mean-pool a padded tile buffer to one representative per tile, in float32.

  :param x: padded tile buffer, shape ``(B, H, padded, D)``; padding slots must be zero.
  :param meta: tile geometry of this request.
  :returns: pooled tiles, shape ``(B, H, num_tiles, D)``, float32.
  """
  batch, heads, padded, dim = x.shape
  if padded != meta.padded_seq_length:
    raise ValueError(f"VSA-H3 pooling expects a padded sequence of {meta.padded_seq_length} rows, "
                     f"got {padded}.")
  pooled = x.view(batch, heads, meta.num_tiles, meta.tile_elements, dim).sum(dim=3,
                                                                             dtype=torch.float32)
  return pooled / meta.variable_block_sizes.view(1, 1, -1, 1)


def build_vsa_h3_keep_matrix(pooled_query: torch.Tensor, pooled_key: torch.Tensor, meta: VsaH3Meta,
                             sparsity: float, scale: float) -> torch.Tensor:
  """Boolean tile-level keep matrix of one attention layer at one step.

  Prefix queries keep every tile; video queries keep every prefix tile plus their own top-k video
  tiles, ranked by the pooled tile scores.

  :param pooled_query: pooled queries, shape ``(B, H, num_tiles, D)``.
  :param pooled_key: pooled keys, shape ``(B, H, num_tiles, D)``.
  :param meta: tile geometry of this request.
  :param sparsity: fraction of video tiles to drop.
  :param scale: attention scale applied to the tile scores.
  :returns: keep matrix, shape ``(B, H, num_tiles, num_tiles)``.
  """
  scores = torch.matmul(pooled_query, pooled_key.transpose(-1, -2)) * scale
  num_prefix = meta.num_prefix_tiles
  num_video = meta.num_video_tiles
  keep = torch.zeros_like(scores, dtype=torch.bool)
  keep[..., :num_prefix] = True
  topk = compute_vsa_h3_topk(sparsity, num_video)
  if topk >= num_video:
    keep[..., num_prefix:] = True
  else:
    indices = scores[..., num_prefix:].topk(topk, dim=-1).indices + num_prefix
    keep.scatter_(-1, indices, True)
  keep[:, :, :num_prefix, :] = True
  return keep


def _to_tiled(x: torch.Tensor, meta: VsaH3Meta) -> torch.Tensor:
  """Gather a packed ``(B, S, H, D)`` tensor into the padded tile buffer, zeroing padding."""
  tiled = x.index_select(1, meta.tiled_to_packed)
  valid = meta.tiled_valid.view(1, -1, 1, 1)
  return torch.where(valid, tiled, torch.zeros((), dtype=tiled.dtype, device=tiled.device))


def _from_tiled(x: torch.Tensor, meta: VsaH3Meta) -> torch.Tensor:
  """Scatter a padded tile buffer back to packed order."""
  out = torch.zeros(x.shape[0],
                    meta.total_seq_length,
                    x.shape[2],
                    x.shape[3],
                    dtype=x.dtype,
                    device=x.device)
  src = x.index_select(1, meta.tiled_valid_slots)
  index = meta.packed_row_order.view(1, -1, 1, 1).expand_as(src)
  return out.scatter(1, index, src)


def _block_index_lists(keep: torch.Tensor, meta: VsaH3Meta):
  """Split the keep matrix into FlexAttention's full-block and masked-block index lists.

  A tile whose ``variable_block_sizes`` entry equals the tile volume carries no padding, so it can
  skip the mask; only partial tiles need ``mask_mod``. The list widths are the geometry's static
  full and partial tile counts, which keeps both the shapes and the compiled graph stable.

  :param keep: boolean keep matrix, shape ``(B, H, num_tiles, num_tiles)``.
  :param meta: tile geometry of this request.
  :returns: ``(full_num, full_indices, part_num, part_indices)``, all int32.
  """
  is_full = meta.variable_block_sizes == meta.tile_elements
  full_keep = keep & is_full.view(1, 1, 1, -1)
  part_keep = keep & ~is_full.view(1, 1, 1, -1)
  full_num = full_keep.sum(dim=-1, dtype=torch.int32)
  part_num = part_keep.sum(dim=-1, dtype=torch.int32)
  # int32 throughout: `topk` hands back int64 indices, and a mixed int32 count / int64 index pair
  # makes the block-sparse Triton template fail to compile ("Loop-carried variable kv_offset has
  # initial type int32 but is re-assigned to int64").
  full_indices = full_keep.to(torch.int32).topk(max(meta.num_full_tiles, 1),
                                                dim=-1).indices.to(torch.int32)
  part_indices = part_keep.to(torch.int32).topk(max(meta.num_partial_tiles, 1),
                                                dim=-1).indices.to(torch.int32)
  return full_num, full_indices, part_num, part_indices


def _compiled_flex():
  from torch.nn.attention.flex_attention import flex_attention

  return torch.compile(flex_attention, dynamic=False)


_FLEX_COMPILED = None


def _flex_call(*args, **kwargs):
  """Run FlexAttention, reusing one compiled wrapper and never nesting one in another graph."""
  global _FLEX_COMPILED
  from torch.nn.attention.flex_attention import flex_attention

  if torch.compiler.is_compiling():
    return flex_attention(*args, **kwargs)
  if _FLEX_COMPILED is None:
    _FLEX_COMPILED = _compiled_flex()
  return _FLEX_COMPILED(*args, **kwargs)


def _flex_sparse_output(query_t, key_t, value_t, meta, keep, scale):
  """Block-sparse attention through a FlexAttention `BlockMask`."""
  from torch.nn.attention.flex_attention import BlockMask

  full_num, full_indices, part_num, part_indices = _block_index_lists(keep, meta)
  valid = meta.tiled_valid

  def padding_mask_mod(batch, head, q_index, kv_index):
    return valid[kv_index]

  block_mask = BlockMask.from_kv_blocks(
    part_num,
    part_indices,
    full_num,
    full_indices,
    BLOCK_SIZE=meta.tile_elements,
    mask_mod=padding_mask_mod,
    seq_lengths=(meta.padded_seq_length, meta.padded_seq_length),
    compute_q_blocks=False,
  )
  return _flex_call(query_t, key_t, value_t, block_mask=block_mask, scale=scale)


def _gather_rows(kv_tiled: torch.Tensor, tile_ids: torch.Tensor,
                 tile_elements: int) -> torch.Tensor:
  """Gather every row of the given tiles.

  :param kv_tiled: padded tile buffer, shape ``(B, H, padded, D)``.
  :param tile_ids: tile ids, shape ``(B, H, num_q, num_k)``.
  :param tile_elements: slots per tile.
  :returns: gathered rows, shape ``(B, H, num_q, num_k * tile_elements, D)``.
  """
  offsets = torch.arange(tile_elements, device=tile_ids.device)
  rows = (tile_ids.unsqueeze(-1) * tile_elements + offsets).flatten(-2)
  batch, heads, padded, dim = kv_tiled.shape
  flat = kv_tiled.reshape(batch * heads, padded, dim)
  index = rows.reshape(batch * heads, -1).unsqueeze(-1).expand(-1, -1, dim)
  return torch.gather(flat, 1, index).reshape(batch, heads, *rows.shape[2:], dim)


def _dense_prefix_output(query, key, value, meta, scale):
  """Dense attention for the prefix rows, computed in packed order so no padding is involved."""
  prefix = meta.num_prefix_tokens
  out = F.scaled_dot_product_attention(query[:, :prefix].transpose(1, 2),
                                       key.transpose(1, 2),
                                       value.transpose(1, 2),
                                       scale=scale)
  return out.transpose(1, 2)


def _gather_sparse_output(query_t, key_t, value_t, prefix_out, meta, keep, scale,
                          query_tile_chunk: int) -> torch.Tensor:
  """Chunked gather + `scaled_dot_product_attention` baseline, exact but launch-bound."""
  out = torch.zeros_like(query_t)
  prefix_slots = meta.tiled_valid_slots[:meta.num_prefix_tokens]
  out[:, :, prefix_slots] = prefix_out.transpose(1, 2)

  num_prefix = meta.num_prefix_tiles
  num_video = meta.num_video_tiles
  width = num_prefix + compute_vsa_h3_topk(meta.sparsity, num_video)
  selected = keep[:, :, num_prefix:, :].to(torch.int32).topk(width, dim=-1).indices
  valid = meta.tiled_valid
  chunk = max(1, int(query_tile_chunk))

  for start in range(0, num_video, chunk):
    stop = min(start + chunk, num_video)
    ids = selected[:, :, start:stop]
    keys = _gather_rows(key_t, ids, meta.tile_elements)
    values = _gather_rows(value_t, ids, meta.tile_elements)
    rows = (ids.unsqueeze(-1) * meta.tile_elements +
            torch.arange(meta.tile_elements, device=ids.device)).flatten(-2)
    keep_mask = valid[rows]
    batch, heads, tiles, token_rows, dim = keys.shape
    slot_lo = (num_prefix + start) * meta.tile_elements
    slot_hi = (num_prefix + stop) * meta.tile_elements
    queries = query_t[:, :, slot_lo:slot_hi]
    attended = F.scaled_dot_product_attention(
      queries.reshape(batch * heads * tiles, meta.tile_elements, dim),
      keys.reshape(batch * heads * tiles, token_rows, dim),
      values.reshape(batch * heads * tiles, token_rows, dim),
      attn_mask=keep_mask.reshape(batch * heads * tiles, 1, token_rows),
      scale=scale)
    out[:, :, slot_lo:slot_hi] = attended.reshape(batch, heads, tiles * meta.tile_elements, dim)
  return out


def vsa_h3_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  meta: Optional[VsaH3Meta],
  gate: Optional[torch.Tensor] = None,
  sparsity: Optional[float] = None,
  scale: Optional[float] = None,
  impl: Optional[str] = None,
  query_tile_chunk: int = _DEFAULT_QUERY_TILE_CHUNK,
) -> torch.Tensor:
  """Block-sparse video attention over the tile geometry of ``meta``.

  :param query: packed queries, shape ``(B, S, H, D)``.
  :param key: packed keys, shape ``(B, S, H, D)``.
  :param value: packed values, shape ``(B, S, H, D)``.
  :param meta: tile geometry of this request; ``None`` falls back to dense attention.
  :param gate: per-head compression gate in packed order, shape ``(B, S, H, D)``, or ``None``.
  :param sparsity: fraction of video tiles to drop; defaults to ``meta.sparsity``.
  :param scale: attention scale; defaults to ``D ** -0.5``.
  :param impl: ``"flex"``, ``"gather"`` or ``None`` for :func:`vsa_h3_impl`.
  :param query_tile_chunk: query tiles per gathered attention call, gather path only.
  :returns: attention output in packed order, shape ``(B, S, H, D)``.
  """
  head_dim = query.shape[-1]
  scale = head_dim ** -0.5 if scale is None else float(scale)
  if meta is None:
    return F.scaled_dot_product_attention(query.transpose(1, 2),
                                          key.transpose(1, 2),
                                          value.transpose(1, 2),
                                          scale=scale).transpose(1, 2)
  sparsity = meta.sparsity if sparsity is None else float(sparsity)
  if sparsity <= 0.0 and gate is None:
    return F.scaled_dot_product_attention(query.transpose(1, 2),
                                          key.transpose(1, 2),
                                          value.transpose(1, 2),
                                          scale=scale).transpose(1, 2)
  if meta.num_prefix_tokens < 1:
    raise ValueError("VSA-H3 expects at least one dense prefix row (text / audio).")

  query_t = _to_tiled(query, meta).transpose(1, 2)
  key_t = _to_tiled(key, meta).transpose(1, 2)
  value_t = _to_tiled(value, meta).transpose(1, 2)
  pooled_query = pool_vsa_h3_tiles(query_t, meta)
  pooled_key = pool_vsa_h3_tiles(key_t, meta)
  scoped_meta = meta if sparsity == meta.sparsity else _with_sparsity(meta, sparsity)
  keep = build_vsa_h3_keep_matrix(pooled_query, pooled_key, scoped_meta, sparsity, scale)

  if vsa_h3_impl(impl) == "flex":
    out = _flex_sparse_output(query_t, key_t, value_t, scoped_meta, keep, scale)
  else:
    prefix_out = _dense_prefix_output(query, key, value, scoped_meta, scale)
    out = _gather_sparse_output(query_t, key_t, value_t, prefix_out, scoped_meta, keep, scale,
                                query_tile_chunk)

  out = out.transpose(1, 2)
  if gate is not None:
    pooled_value = pool_vsa_h3_tiles(value_t, meta)
    compressed = F.scaled_dot_product_attention(pooled_query.to(out.dtype),
                                                pooled_key.to(out.dtype),
                                                pooled_value.to(out.dtype))
    tile_of_slot = meta.tile_of_slot()
    compressed = compressed.transpose(1, 2).index_select(1, tile_of_slot)
    gate_tiled = _to_tiled(gate, meta)
    out = out + compressed * gate_tiled
  return _from_tiled(out, meta)


def _with_sparsity(meta: VsaH3Meta, sparsity: float) -> VsaH3Meta:
  """Copy of the geometry with a different sparsity, for one-off sweeps."""
  return dataclasses.replace(meta, sparsity=float(sparsity))


def vsa_h3_attention_reference(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  meta: Optional[VsaH3Meta],
  gate: Optional[torch.Tensor] = None,
  sparsity: Optional[float] = None,
  scale: Optional[float] = None,
) -> torch.Tensor:
  """Exact, loop-based reference of :func:`vsa_h3_attention` for tests (``B == 1`` only).

  Each query tile is materialized on its own with the kept key tiles gathered explicitly, which is
  far slower than either kernel and shares no code with them.

  :param query: packed queries, shape ``(1, S, H, D)``.
  :param key: packed keys, shape ``(1, S, H, D)``.
  :param value: packed values, shape ``(1, S, H, D)``.
  :param meta: tile geometry of this request.
  :param gate: per-head compression gate in packed order, shape ``(1, S, H, D)``, or ``None``.
  :param sparsity: fraction of video tiles to drop; defaults to ``meta.sparsity``.
  :param scale: attention scale; defaults to ``D ** -0.5``.
  :returns: attention output in packed order, shape ``(1, S, H, D)``.
  """
  if query.shape[0] != 1:
    raise ValueError("The VSA-H3 reference path only supports a single packed request.")
  head_dim = query.shape[-1]
  scale = head_dim ** -0.5 if scale is None else float(scale)
  if meta is None:
    return F.scaled_dot_product_attention(query.transpose(1, 2),
                                          key.transpose(1, 2),
                                          value.transpose(1, 2),
                                          scale=scale).transpose(1, 2)

  sparsity = meta.sparsity if sparsity is None else float(sparsity)
  prefix = meta.num_prefix_tokens
  prefix_slots = meta.tiled_valid_slots[:prefix]
  query_t = _to_tiled(query, meta).transpose(1, 2)
  key_t = _to_tiled(key, meta).transpose(1, 2)
  value_t = _to_tiled(value, meta).transpose(1, 2)
  pooled_query = pool_vsa_h3_tiles(query_t, meta)
  pooled_key = pool_vsa_h3_tiles(key_t, meta)
  keep = build_vsa_h3_keep_matrix(pooled_query, pooled_key, meta, sparsity, scale)

  sizes = meta.variable_block_sizes.tolist()
  valid = meta.tiled_valid.tolist()
  out_tiled = torch.zeros_like(query_t)
  out_tiled[:, :, prefix_slots] = _dense_prefix_output(query, key, value, meta,
                                                       scale).transpose(1, 2)
  # The tile selection is per head, so the oracle walks heads as well; it is only used on the tiny
  # unit-test geometries.
  for tile in range(meta.num_prefix_tiles, meta.num_tiles):
    start = tile * meta.tile_elements
    q_rows = list(range(start, start + sizes[tile]))
    for head in range(query.shape[2]):
      kept = keep[0, head, tile].nonzero().flatten().tolist()
      rows = [
        slot for k in kept for slot in range(k * meta.tile_elements, (k + 1) * meta.tile_elements)
        if valid[slot]
      ]
      out_tiled[:, head:head + 1,
                q_rows] = F.scaled_dot_product_attention(query_t[:, head:head + 1, q_rows],
                                                         key_t[:, head:head + 1, rows],
                                                         value_t[:, head:head + 1, rows],
                                                         scale=scale)

  if gate is not None:
    pooled_value = pool_vsa_h3_tiles(value_t, meta)
    compressed = F.scaled_dot_product_attention(pooled_query.to(out_tiled.dtype),
                                                pooled_key.to(out_tiled.dtype),
                                                pooled_value.to(out_tiled.dtype))
    compressed = compressed.index_select(2, meta.tile_of_slot())
    gate_tiled = _to_tiled(gate, meta).transpose(1, 2)
    out_tiled = out_tiled + compressed * gate_tiled

  return _from_tiled(out_tiled.transpose(1, 2), meta)
