"""Auto-encoder data-parallel planner for `AutoencoderKLMiniMaxH3` (VAE-P).

The MiniMax-H3 video VAE decodes a clip of latent frames with a causal 3D decoder. `_decode_clip`
lays 256-pixel tiles with a 64-pixel overlap over the pixel grid of **one clip** and stitches them
back with `_stitch_tiles`. A tile cannot be split further without changing the result, so the tile is
the only independent unit of work — the same structure as the MiniMax-H3-Image VAE, one dimension up.

This planner distributes the tiles of every clip round-robin and moves the decoded tiles back to rank
0, which is the rank that stitches and then hands the clip down the chain. The temporal chunk loop in
`_decode` stays replicated, so the output is **bit-identical** to the serial path whatever the tile
count or the world size.

Three details matter:

* the tile grid is recomputed with `auto_encoder._split_tiles(...)` on every rank instead of being
  re-derived here, so the last tile of an axis may be shorter and still matches;
* the exchange goes through `TileSafeP2PComm` with `set_dims(5)`, because a decoded tile is
  `[Batch, Channels, Frames, Height, Width]` and tiles are not guaranteed to be equally shaped;
* `_decode` calls `_decode_clip` once per temporal chunk, so a 124-frame clip posts one tile round per
  chunk (about seven rounds at the released `clip_length`); the synchronisation is part of the cost
  and the per-rank speedup stays below the planar ideal.
"""

import functools

import torch
import torch.distributed as dist

from ...logger import init_logger
from ..config import ParallelismConfig
from .register import (
  AutoEncoderDataParallelismPlanner,
  AutoEncoderDataParallelismPlannerRegister,
)
from .utils import TileSafeP2PComm

logger = init_logger(__name__)


@AutoEncoderDataParallelismPlannerRegister.register("AutoencoderKLMiniMaxH3")
class AutoencoderKLMiniMaxH3DataParallelismPlanner(AutoEncoderDataParallelismPlanner):
  """Tile-parallel decoder for the MiniMax-H3 video VAE.

  Registered under the exact class name: the auto-encoder registry matches names exactly, so the
  generic `AutoencoderKL` planner (which patches `tiled_decode` / `blend_v`, neither of which this
  model has) never applies.
  """

  def _apply(
    self,
    auto_encoder: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> torch.nn.Module:
    from diffusers import AutoencoderKLMiniMaxH3

    assert isinstance(auto_encoder, AutoencoderKLMiniMaxH3), (
      "AutoencoderKLMiniMaxH3DataParallelismPlanner can only be applied to AutoencoderKLMiniMaxH3")
    dp_mesh = self.mesh(parallelism_config=parallelism_config)
    return self.parallelize_tiling(auto_encoder=auto_encoder, dp_mesh=dp_mesh)

  def parallelize_tiling(
    self,
    auto_encoder: torch.nn.Module,
    dp_mesh: dist.DeviceMesh,
  ) -> torch.nn.Module:
    """Distribute the model's own tiles across the DP ranks.

    :param auto_encoder: MiniMax-H3 video VAE.
    :param dp_mesh: Data-parallel mesh of the auto-encoder group.
    :returns: The patched auto-encoder.
    """

    group = dp_mesh.get_group()
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    comm = TileSafeP2PComm()
    comm.set_dims(5)  # [B, C, T, H, W]: the decoder returns the frames of one clip

    def _assert_same_input(shape, owner: str) -> None:
      """Fail fast when the ranks disagree on the clip they are tiling.

      The tile grid (and therefore the number of messages every rank posts) is derived from the
      clip shape, so a rank that decodes a differently shaped clip would make the tile exchange
      hang instead of raising: rank 0 would wait for a message nobody sends.
      """

      if world_size == 1:
        return
      shapes = [None] * world_size
      dist.all_gather_object(shapes, tuple(shape), group=group)
      if any(candidate != shapes[0] for candidate in shapes):
        raise ValueError(f"{owner} needs the same latents on every rank, got {shapes}.")

    def _gather_tiles(rows, device):
      """Collect every tile of the clip on rank 0; other ranks only send."""

      if rank == 0:
        count = 0
        for i in range(len(rows)):
          for j in range(len(rows[i])):
            if count % world_size != rank:
              rows[i][j] = comm.recv_tensor(count % world_size, group, device=device)
            count += 1
      else:
        for row in rows:
          for tile in row:
            if tile is not None:
              comm.send_tensor(tile, 0, group)
      comm.sync()
      return rows

    def _send_result(result, device):
      """Hand rank 0's stitched clip down the chain; every rank returns it."""

      if rank != 0:
        result = comm.recv_tensor(rank - 1, group, device=device)
      if rank < world_size - 1:
        comm.send_tensor(result, rank + 1, group)
      comm.sync()
      return result

    original_decode_clip = auto_encoder.__class__._decode_clip

    @functools.wraps(original_decode_clip)
    def new_decode_clip(self, z, *args, **kwargs):
      if not self.use_tiling or world_size == 1:
        if not self.use_tiling and world_size > 1:
          # Serial decode on every rank still produces the right clip, so this is a warning rather than
          # an error — but it is the one way this planner can be enabled and quietly do nothing.
          logger.warning("MiniMax-H3 VAE-P is enabled but `use_tiling` is off: decoding serially.")
        return original_decode_clip(self, z, *args, **kwargs)

      ratio = self.spatial_compression_ratio
      _assert_same_input(z.shape, "decode")
      # Tiles are laid out in pixel space and then mapped back onto the latent grid.
      height = z.shape[-2] * ratio
      width = z.shape[-1] * ratio
      # Ask the model for the grid instead of rebuilding it: the last tile of an axis is trimmed to
      # the frame and would otherwise be sized wrongly here.
      y_indices, y_lengths, y_overlaps = self._split_tiles(height, self.tile_sample_min_height,
                                                           self.tile_sample_min_overlap_height)
      x_indices, x_lengths, x_overlaps = self._split_tiles(width, self.tile_sample_min_width,
                                                           self.tile_sample_min_overlap_width)

      rows = []
      count = 0
      for i_pos, i_len in zip(y_indices, y_lengths):
        row = []
        for j_pos, j_len in zip(x_indices, x_lengths):
          if count % world_size == rank:
            tile = z[
              ...,
              i_pos // ratio:i_pos // ratio + i_len // ratio,
              j_pos // ratio:j_pos // ratio + j_len // ratio,
            ]
            row.append(self.decoder(self.post_quant_conv(tile)))
          else:
            row.append(None)
          count += 1
        rows.append(row)

      rows = _gather_tiles(rows, z.device)
      decoded = self._stitch_tiles(rows, y_overlaps, x_overlaps) if rank == 0 else None
      return _send_result(decoded, z.device)

    # Bind on the instance, not the class: `dispatch.py` only refuses to apply twice through
    # `_is_parallelized`, so a class-level patch would leak into every other
    # `AutoencoderKLMiniMaxH3` living in the same process with this rank's group and comm.
    auto_encoder._decode_clip = new_decode_clip.__get__(auto_encoder)
    return auto_encoder
