"""Auto-encoder data-parallel planner for `AutoencoderKLMiniMaxH3Image` (VAE-P).

The MiniMax-H3-Image VAE is a causal 3D CNN encoder followed by a **2D ViT decoder**. Its decoder
attends over the whole latent grid of a tile, so a tile cannot be split further without changing the
result: the only independent unit of work is a tile.

The model already ships that tiling (`use_tiling = True` by default): `_encode` / `_decode` lay
256-pixel tiles with a 64-pixel overlap over the image and stitch them back with `_stitch_tiles`.
This planner distributes those tiles round-robin and moves the decoded tensors back to rank 0,
which is the rank that stitches and then hands the result down the chain — so the output is
**bit-identical** to the serial path, whatever the tile count or the world size.

Two details make it correct rather than merely similar to the serial run:

* the tile grid is recomputed from `auto_encoder._split_tiles(...)` on every rank instead of being
  re-derived here, so the last tile of an axis may be shorter and still matches;
* the exchange goes through `TileSafeP2PComm`, the comm helper that transmits each tile's shape
  (and dtype) ahead of the payload, because tiles are not guaranteed to be equally shaped.
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


@AutoEncoderDataParallelismPlannerRegister.register("AutoencoderKLMiniMaxH3Image")
class AutoencoderKLMiniMaxH3ImageDataParallelismPlanner(AutoEncoderDataParallelismPlanner):
  """Tile-parallel encoder / decoder for the MiniMax-H3-Image VAE.

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
    from diffusers import AutoencoderKLMiniMaxH3Image

    assert isinstance(auto_encoder, AutoencoderKLMiniMaxH3Image), (
      "AutoencoderKLMiniMaxH3ImageDataParallelismPlanner can only be applied to "
      "AutoencoderKLMiniMaxH3Image")
    dp_mesh = self.mesh(parallelism_config=parallelism_config)
    return self.parallelize_tiling(auto_encoder=auto_encoder, dp_mesh=dp_mesh)

  def parallelize_tiling(
    self,
    auto_encoder: torch.nn.Module,
    dp_mesh: dist.DeviceMesh,
  ) -> torch.nn.Module:
    """Distribute the model's own tiles across the DP ranks.

    :param auto_encoder: MiniMax-H3-Image VAE.
    :param dp_mesh: Data-parallel mesh of the auto-encoder group.
    :returns: The patched auto-encoder.
    """
    group = dp_mesh.get_group()
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    comm = TileSafeP2PComm()
    comm.set_dims(4)  # [1, C, H, W]: the model decodes one image at a time

    def _assert_same_input(shape, owner: str) -> None:
      """Fail fast when the ranks disagree on the tensor they are tiling.

      The tile grid (and therefore the number of messages every rank posts) is derived from the
      input shape, so a rank that decodes a differently shaped latent would make the tile exchange
      hang instead of raising: rank 0 would wait for a message nobody sends.
      """
      if world_size == 1:
        return
      shapes = [None] * world_size
      dist.all_gather_object(shapes, tuple(shape), group=group)
      if any(candidate != shapes[0] for candidate in shapes):
        raise ValueError(f"{owner} needs the same latents on every rank, got {shapes}.")

    def _gather_tiles(rows, device):
      """Collect every tile on rank 0; other ranks only send."""

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
      """Hand rank 0's stitched tensor down the chain; every rank returns it."""

      if rank != 0:
        result = comm.recv_tensor(rank - 1, group, device=device)
      if rank < world_size - 1:
        comm.send_tensor(result, rank + 1, group)
      comm.sync()
      return result

    original_decode = auto_encoder.__class__._decode
    original_encode = auto_encoder.__class__._encode

    @functools.wraps(original_decode)
    def new_decode(self, z, *args, **kwargs):
      if not self.use_tiling or world_size == 1:
        return original_decode(self, z, *args, **kwargs)

      ratio = self.spatial_compression_ratio
      _assert_same_input(z.shape, "decode")
      height = z.shape[-2] * ratio
      width = z.shape[-1] * ratio
      # Ask the model for the grid instead of rebuilding it: the last tile of an axis is trimmed to
      # the image and would otherwise be sized wrongly here.
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

    @functools.wraps(original_encode)
    def new_encode(self, x, *args, **kwargs):
      if not self.use_tiling or world_size == 1:
        return original_encode(self, x, *args, **kwargs)

      height = x.shape[-2]
      width = x.shape[-1]
      _assert_same_input(x.shape, "encode")
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
            tile = x[..., i_pos:i_pos + i_len, j_pos:j_pos + j_len]
            row.append(self.quant_conv(self.encoder(tile[:, :, None]))[:, :, -1])
          else:
            row.append(None)
          count += 1
        rows.append(row)

      rows = _gather_tiles(rows, x.device)
      ratio = self.spatial_compression_ratio
      encoded = None
      if rank == 0:
        encoded = self._stitch_tiles(
          rows,
          [overlap // ratio for overlap in y_overlaps],
          [overlap // ratio for overlap in x_overlaps],
        )
      return _send_result(encoded, x.device)

    auto_encoder._decode = new_decode.__get__(auto_encoder)
    auto_encoder._encode = new_encode.__get__(auto_encoder)
    return auto_encoder
