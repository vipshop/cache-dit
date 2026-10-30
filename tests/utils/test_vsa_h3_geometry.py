"""Geometry tests for FastH3's video sparse attention.

The geometry is the only part of VSA that a wrong implementation can get away with: an off-by-one
in the tile order still produces a video, just one that attends to the wrong rows. These tests pin
the invariants the block-sparse kernels trust, plus the `t2va`-only restriction.
"""
import math

import pytest
import torch

from cache_dit.attention.vsa_h3_geometry import (
  build_vsa_h3_meta,
  vsa_h3_meta_from_layout,
)

LAYOUTS = [
  ((5, 3), (2, 4, 4)),
  ((1024, 207), (37, 24, 42)),
  ((64, 0), (8, 4, 8)),
  ((100, 65), (5, 9, 7)),
]


@pytest.mark.parametrize("prefix,video", LAYOUTS)
def test_tile_geometry_is_a_permutation(prefix, video):
  meta = build_vsa_h3_meta(prefix, video, torch.device("cpu"), sparsity=0.9)
  total = sum(prefix) + math.prod(video)

  assert meta.total_seq_length == total
  assert int(meta.variable_block_sizes.sum()) == total
  assert int(meta.tiled_valid.sum()) == total
  assert meta.packed_to_tiled.unique().numel() == total
  assert torch.equal(meta.tiled_to_packed[meta.packed_to_tiled], torch.arange(total))
  assert torch.equal(meta.tiled_to_packed[meta.tiled_valid_slots], meta.packed_row_order)
  assert meta.num_tiles == meta.num_prefix_tiles + meta.num_video_tiles
  assert meta.num_full_tiles + meta.num_partial_tiles == meta.num_tiles


@pytest.mark.parametrize("prefix,video", LAYOUTS)
def test_dense_segments_never_share_a_tile(prefix, video):
  meta = build_vsa_h3_meta(prefix, video, torch.device("cpu"), sparsity=0.9)
  sizes = meta.variable_block_sizes.tolist()
  expected = [
    size for segment in prefix if segment > 0
    for size in [64] * (segment // 64) + ([segment % 64] if segment % 64 else [])
  ]
  assert sizes[:meta.num_prefix_tiles] == expected


def test_partial_tiles_keep_their_real_row_count():
  meta = build_vsa_h3_meta((100, 65), (5, 9, 7), torch.device("cpu"), sparsity=0.9)
  sizes = meta.variable_block_sizes.tolist()
  assert min(sizes) >= 1
  assert max(sizes) <= meta.tile_elements
  partial = [size for size in sizes if size < meta.tile_elements]
  assert partial, "this layout must exercise padding"


def test_meta_rejects_a_non_64_token_tile():
  with pytest.raises(ValueError, match="64-token tiles"):
    build_vsa_h3_meta((5, 3), (2, 4, 4), torch.device("cpu"), sparsity=0.9, tile_shape=(4, 8, 8))


def _layout(video_shape, text=4, audio=2, condition=0):
  num_video = math.prod(video_shape)
  text_indices = torch.arange(text)
  audio_indices = torch.arange(text + condition, text + condition + audio)
  video_indices = torch.arange(text + condition + audio, text + condition + audio + num_video)
  position_ids = torch.zeros(text + condition + audio + num_video, 3, dtype=torch.float64)
  dims = [torch.arange(size, dtype=torch.float64) for size in video_shape]
  grid = torch.cartesian_prod(dims[0], dims[1], dims[2])
  position_ids[video_indices, 0] = grid[:, 0]
  position_ids[video_indices, 1] = grid[:, 1]
  position_ids[video_indices, 2] = grid[:, 2]
  return position_ids, video_indices, audio_indices, text_indices


def test_meta_from_layout_recovers_the_video_grid():
  position_ids, video_indices, audio_indices, text_indices = _layout((3, 4, 5))
  meta = vsa_h3_meta_from_layout(position_ids,
                                 video_indices,
                                 audio_indices,
                                 text_indices,
                                 sparsity=0.9)
  assert meta.video_shape == (3, 4, 5)
  assert meta.prefix_segments == (4, 2)
  # (t=3, h=4, w=5) with (4, 4, 4) tiles: one partial tile on t, one on h, two on w.
  assert meta.num_video_tiles == 1 * 1 * 2
  assert int(meta.variable_block_sizes[meta.num_prefix_tiles:].sum()) == 3 * 4 * 5


def test_meta_from_layout_rejects_conditioning_rows():
  position_ids, video_indices, audio_indices, text_indices = _layout((3, 4, 5), condition=8)
  with pytest.raises(ValueError, match="t2va packed order"):
    vsa_h3_meta_from_layout(position_ids, video_indices, audio_indices, text_indices, sparsity=0.9)


def test_meta_from_layout_rejects_an_unrecoverable_grid():
  position_ids, video_indices, audio_indices, text_indices = _layout((3, 4, 5))
  position_ids[video_indices, 2] = 0.0
  with pytest.raises(ValueError, match="cannot recover"):
    vsa_h3_meta_from_layout(position_ids, video_indices, audio_indices, text_indices, sparsity=0.9)
