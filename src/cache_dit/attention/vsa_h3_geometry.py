"""Packed-row to 64-token tile geometry for FastH3's video sparse attention.

FastH3 students (the `FastVideo-FastH3-*` DMD2 releases of MiniMax-H3) run a **video sparse
attention** whose selection unit is a 3-D tile of ``(4, 4, 4) = 64`` latent tokens. The packed
sequence that MiniMax-H3 denoises mixes modalities::

    [ text rows | conditioning rows | audio rows | generated video rows ]

Only the generated video rows carry a ``(t, h, w)`` grid, so only they can be tiled in 3-D; every
other row is a *dense prefix* that a sparse query always attends to in full. This module owns the
row <-> tile mapping and the invariants the block-sparse attention relies on:

* **Dense segments never share a tile.** Text and audio are chunked separately, so a tile never
  straddles a modality boundary (the reference tiler keeps them independent too).
* **Tiles are 64 slots wide and partial tiles are packed at the end of their tile.** Every tile
  contributes exactly ``variable_block_sizes[i]`` valid rows; the remaining slots are padding that
  the attention must mask out.
* **The mapping is a permutation.** ``packed_to_tiled`` and ``tiled_to_packed`` are inverse on the
  valid slots, which is what lets the attention scatter into the padded tile buffer and gather the
  result back without a host-side index list.

The geometry only depends on the request layout (row counts and the video grid), so it is built
once per request and shared by all blocks and steps.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional, Sequence

import torch

__all__ = [
  "VsaH3Meta",
  "VSA_H3_TILE_ELEMENTS",
  "VSA_H3_TILE_SHAPE",
  "build_vsa_h3_meta",
  "vsa_h3_meta_from_layout",
]

VSA_H3_TILE_SHAPE: tuple[int, int, int] = (4, 4, 4)
VSA_H3_TILE_ELEMENTS: int = math.prod(VSA_H3_TILE_SHAPE)


@dataclasses.dataclass(frozen=True, eq=False)
class VsaH3Meta:
  """Row layout of one FastH3 request in both packed order and tile order.

  ``eq=False`` keeps identity semantics: the tensor fields would otherwise make ``__eq__``/
  ``__hash__`` raise, and the cache below hands out one shared instance per layout anyway.

  :param total_seq_length: number of rows of the packed sequence.
  :param prefix_segments: dense (non-video) row counts in packed order, after zero-length removal.
  :param video_shape: generated video grid ``(t, h, w)`` in latent-patch units.
  :param tile_shape: tile shape, always ``(4, 4, 4)`` for the shipped students.
  :param num_prefix_tiles: leading tiles that hold the dense segments.
  :param num_video_tiles: tiles that hold the 3-D video grid.
  :param num_full_tiles: tiles whose ``variable_block_sizes`` entry equals the tile volume.
  :param num_partial_tiles: tiles with padding, i.e. ``num_tiles - num_full_tiles``.
  :param variable_block_sizes: valid rows per tile, shape ``(num_tiles,)``.
  :param packed_to_tiled: packed row -> slot of the padded tile buffer, shape ``(total,)``.
  :param tiled_to_packed: slot of the padded tile buffer -> packed row, shape ``(padded,)``; the
    padding slots repeat row 0 and must be masked by ``tiled_valid``.
  :param tiled_valid: ``True`` for slots that hold a real row, shape ``(padded,)``.
  """

  total_seq_length: int
  prefix_segments: tuple[int, ...]
  video_shape: tuple[int, int, int]
  tile_shape: tuple[int, int, int]
  num_prefix_tiles: int
  num_video_tiles: int
  num_full_tiles: int
  num_partial_tiles: int
  variable_block_sizes: torch.Tensor
  packed_to_tiled: torch.Tensor
  tiled_to_packed: torch.Tensor
  tiled_valid: torch.Tensor
  tiled_valid_slots: torch.Tensor
  packed_row_order: torch.Tensor
  sparsity: float
  tile_elements: int = VSA_H3_TILE_ELEMENTS

  @property
  def num_tiles(self) -> int:
    """Number of tiles, dense prefix tiles included."""
    return self.num_prefix_tiles + self.num_video_tiles

  @property
  def padded_seq_length(self) -> int:
    """Length of the padded tile buffer, ``num_tiles * tile_elements``."""
    return self.num_tiles * self.tile_elements

  @property
  def num_prefix_tokens(self) -> int:
    """Number of dense prefix rows, i.e. rows whose queries stay dense."""
    return sum(self.prefix_segments)

  def tile_of_slot(self) -> torch.Tensor:
    """Tile id of every slot of the padded tile buffer, shape ``(padded,)``."""
    return torch.arange(self.padded_seq_length,
                        device=self.tiled_valid.device) // self.tile_elements


def _axis_tile_sizes(length: int, tile: int, device: torch.device) -> list[int]:
  count = math.ceil(length / tile)
  sizes = [tile] * count
  sizes[-1] = length - (count - 1) * tile
  return sizes


def _video_tile_rows(
  video_shape: tuple[int, int, int],
  tile_shape: tuple[int, int, int],
  device: torch.device,
) -> torch.Tensor:
  """Row order of the video grid, one contiguous run per ``(t, h, w)`` tile."""
  t, h, w = video_shape
  tile_t, tile_h, tile_w = tile_shape
  rows = torch.arange(t * h * w, device=device).reshape(t, h, w)
  tiles = [
    rows[t0:t0 + tile_t, h0:h0 + tile_h, w0:w0 + tile_w].flatten() for t0 in range(0, t, tile_t)
    for h0 in range(0, h, tile_h) for w0 in range(0, w, tile_w)
  ]
  return torch.cat(tiles)


def _dense_tile_rows(num_rows: int, tile_elements: int, device: torch.device) -> torch.Tensor:
  """Row order of one dense segment, chunked into tiles without straddling the segment end."""
  if num_rows == 0:
    return torch.empty(0, dtype=torch.long, device=device)
  return torch.arange(num_rows, device=device)


def build_vsa_h3_meta(
  prefix_segments: Sequence[int],
  video_shape: tuple[int, int, int],
  device: torch.device,
  sparsity: float,
  tile_shape: tuple[int, int, int] = VSA_H3_TILE_SHAPE,
  video_start: Optional[int] = None,
) -> VsaH3Meta:
  """Build the packed-row / tile mapping of one FastH3 request.

  Rows are permuted so that every dense chunk lands in a leading prefix tile and the video grid
  contributes one contiguous run of 3-D tiles after them.

  :param prefix_segments: dense row counts in packed order (zero-length entries are dropped).
  :param video_shape: generated video grid ``(t, h, w)``.
  :param device: device of the produced index tensors.
  :param sparsity: fraction of video tiles the attention drops, kept on the metadata for logging.
  :param tile_shape: tile shape, must be ``(4, 4, 4)``.
  :param video_start: packed row of the first video row; defaults to the prefix length.
  :returns: The geometry of this request.
  """
  tile_elements = math.prod(tile_shape)
  if tile_elements != VSA_H3_TILE_ELEMENTS:
    raise ValueError(f"VSA-H3 requires {VSA_H3_TILE_ELEMENTS}-token tiles, i.e. tile shape "
                     f"{VSA_H3_TILE_SHAPE}, got {tuple(tile_shape)}.")
  segments = tuple(int(s) for s in prefix_segments if int(s) > 0)
  if any(segment < 0 for segment in prefix_segments):
    raise ValueError(f"VSA-H3 prefix segments must be non-negative, got {tuple(prefix_segments)}.")
  if len(video_shape) != 3 or any(int(dim) < 1 for dim in video_shape):
    raise ValueError(f"VSA-H3 video grid must be three positive axis lengths, got {video_shape}.")

  video_shape = tuple(int(dim) for dim in video_shape)
  num_video_rows = math.prod(video_shape)
  prefix_rows = sum(segments)
  if video_start is None:
    video_start = prefix_rows
  if int(video_start) != prefix_rows:
    raise ValueError(f"VSA-H3 expects the packed video region to start at row {prefix_rows} "
                     f"(the prefix length), got {video_start}. Interleaved layouts are not "
                     "supported.")

  prefix_rows_index = [
    _dense_tile_rows(segment, tile_elements, device) + offset
    for segment, offset in zip(segments, _prefix_offsets(segments))
  ]
  video_tile_rows = _video_tile_rows(video_shape, tile_shape, device) + prefix_rows
  tile_rows = torch.cat([*prefix_rows_index, video_tile_rows], dim=0)

  prefix_sizes = [
    size for segment in segments for size in _axis_tile_sizes(segment, tile_elements, device)
  ]
  t, h, w = video_shape
  tile_t, tile_h, tile_w = tile_shape
  t_sizes = _axis_tile_sizes(t, tile_t, device)
  h_sizes = _axis_tile_sizes(h, tile_h, device)
  w_sizes = _axis_tile_sizes(w, tile_w, device)
  video_sizes = [ts * hs * ws for ts in t_sizes for hs in h_sizes for ws in w_sizes]
  block_sizes = torch.tensor([*prefix_sizes, *video_sizes], dtype=torch.long, device=device)

  num_prefix_tiles = len(prefix_sizes)
  num_video_tiles = len(video_sizes)
  num_tiles = num_prefix_tiles + num_video_tiles
  padded = num_tiles * tile_elements
  total = prefix_rows + num_video_rows

  tiled_valid = torch.zeros(padded, dtype=torch.bool, device=device)
  tiled_to_packed = torch.zeros(padded, dtype=torch.long, device=device)
  packed_to_tiled = torch.zeros(total, dtype=torch.long, device=device)

  # Every tile owns the fixed slot window [i * tile_elements, (i + 1) * tile_elements); only its
  # first `size_i` slots hold a row, the rest is padding at the end of the tile.
  sizes = block_sizes.tolist()
  valid_slots = torch.cat([
    torch.arange(tile * tile_elements, tile * tile_elements + size, device=device)
    for tile, size in enumerate(sizes)
  ])
  tiled_valid[valid_slots] = True
  tiled_to_packed[valid_slots] = tile_rows
  packed_to_tiled[tile_rows] = valid_slots

  num_full_tiles = sum(1 for size in [*prefix_sizes, *video_sizes] if size == tile_elements)
  meta = VsaH3Meta(
    total_seq_length=total,
    prefix_segments=segments,
    video_shape=video_shape,
    tile_shape=tuple(int(dim) for dim in tile_shape),
    num_prefix_tiles=num_prefix_tiles,
    num_video_tiles=num_video_tiles,
    num_full_tiles=num_full_tiles,
    num_partial_tiles=num_tiles - num_full_tiles,
    variable_block_sizes=block_sizes,
    packed_to_tiled=packed_to_tiled,
    tiled_to_packed=tiled_to_packed,
    tiled_valid=tiled_valid,
    tiled_valid_slots=valid_slots,
    packed_row_order=tile_rows,
    sparsity=float(sparsity),
    tile_elements=tile_elements,
  )
  _validate_vsa_h3_meta(meta)
  return meta


def _prefix_offsets(segments: Sequence[int]) -> list[int]:
  offsets = []
  running = 0
  for segment in segments:
    offsets.append(running)
    running += int(segment)
  return offsets


def _validate_vsa_h3_meta(meta: VsaH3Meta) -> None:
  """Fail synchronously on a geometry the block-sparse attention cannot represent."""
  sizes = meta.variable_block_sizes
  total = int(meta.total_seq_length)
  sizes_min = int(sizes.min())
  sizes_max = int(sizes.max())
  sizes_sum = int(sizes.sum())
  if sizes_min < 1 or sizes_max > meta.tile_elements or sizes_sum != total:
    raise ValueError(f"VSA-H3 tile sizes are out of bounds: min={sizes_min}, max={sizes_max}, "
                     f"sum={sizes_sum}, expected sum={total}, tile={meta.tile_elements}.")
  if int(meta.tiled_valid.sum()) != total:
    raise ValueError(f"VSA-H3 tile buffer holds {int(meta.tiled_valid.sum())} valid slots for a "
                     f"packed sequence of {total} rows.")
  packed_to_tiled = meta.packed_to_tiled
  if packed_to_tiled.numel() != total:
    raise ValueError(
      f"VSA-H3 packed_to_tiled has {packed_to_tiled.numel()} entries for {total} rows.")
  if int(torch.unique(packed_to_tiled).numel()) != total:
    raise ValueError("VSA-H3 packed_to_tiled is not injective; the tile ordering dropped a row.")


def vsa_h3_meta_from_layout(
  position_ids: torch.Tensor,
  video_indices: torch.Tensor,
  audio_indices: torch.Tensor,
  text_indices: torch.Tensor,
  sparsity: float,
  tile_shape: tuple[int, int, int] = VSA_H3_TILE_SHAPE,
) -> VsaH3Meta:
  """Derive the geometry from the packed layout that MiniMax-H3 already publishes per step.

  The layout is the ``t2va`` one: ``[text | audio | generated video]``. Conditioning-video
  partitions (``fl2va`` / ``ref2va``) put extra rows between the text and audio segments and are
  rejected here, because the FastH3 preview students are distilled on ``t2va`` only.

  :param position_ids: packed rotary positions, shape ``(seq, 3)``.
  :param video_indices: packed rows of the video region, ascending.
  :param audio_indices: packed rows of the audio region, ascending.
  :param text_indices: packed rows of the text region, ascending.
  :param sparsity: fraction of video tiles to drop.
  :param tile_shape: tile shape, must be ``(4, 4, 4)``.
  :returns: The geometry of this request.
  """
  num_video = int(video_indices.numel())
  num_text = int(text_indices.numel())
  num_audio = int(audio_indices.numel())
  if num_video < 1:
    raise ValueError("VSA-H3 needs at least one generated video row.")
  video_start = int(video_indices[0])
  if video_start != num_text + num_audio:
    raise ValueError(f"VSA-H3 only supports the t2va packed order [text | audio | video], which "
                     f"puts the video region at row {num_text + num_audio}; this request has "
                     f"conditioning rows (video starts at {video_start}).")
  expected = torch.arange(video_start,
                          video_start + num_video,
                          device=video_indices.device,
                          dtype=video_indices.dtype)
  if not torch.equal(video_indices, expected):
    raise ValueError("VSA-H3 expects the video region to be one contiguous ascending run.")

  grid_positions = position_ids[video_indices].to(torch.float64)
  video_shape = tuple(int(axis.unique().numel()) for axis in grid_positions.unbind(dim=1))
  if math.prod(video_shape) != num_video:
    raise ValueError(f"VSA-H3 cannot recover the (t, h, w) video grid from `position_ids`: "
                     f"unique counts {video_shape} multiply to {math.prod(video_shape)}, but the "
                     f"video region has {num_video} rows.")

  return build_vsa_h3_meta(
    prefix_segments=(num_text, num_audio),
    video_shape=video_shape,
    device=video_indices.device,
    sparsity=sparsity,
    tile_shape=tile_shape,
  )
