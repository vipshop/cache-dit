import functools

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLQwenImage
from diffusers.models.autoencoders.vae import DecoderOutput

from ..config import ParallelismConfig
from ...logger import init_logger
from .register import (
  AutoEncoderDataParallelismPlanner,
  AutoEncoderDataParallelismPlannerRegister,
)
from .utils import TileBatchedP2PComm

logger = init_logger(__name__)


def _set_tile_geometry(
  auto_encoder: torch.nn.Module,
  height: int,
  width: int,
  tile_size: int,
  tile_stride: int,
  default_geometry: tuple[int, int, int, int],
  optimized_geometry: tuple[int, int, int, int] | None = None,
) -> None:
  """Set VAE tile geometry and restore the original geometry for smaller images.

  :param auto_encoder: VAE instance whose tiling attributes are updated.
  :param height: Input image height in pixels.
  :param width: Input image width in pixels.
  :param tile_size: Default square tile size for large images.
  :param tile_stride: Default square tile stride for large images.
  :param default_geometry: Original `(tile_h, tile_w, stride_h, stride_w)` before parallelization.
  :param optimized_geometry: Optional `(target_h, target_w, tile, stride)` override.
  """

  if min(height, width) < 1024:
    tile_height, tile_width, stride_height, stride_width = default_geometry
  else:
    tile_height = tile_width = tile_size
    stride_height = stride_width = tile_stride
    if optimized_geometry is not None:
      target_height, target_width, target_tile, target_stride = optimized_geometry
      if height == target_height and width == target_width:
        tile_height = tile_width = target_tile
        stride_height = stride_width = target_stride
  auto_encoder.tile_sample_min_height = tile_height
  auto_encoder.tile_sample_min_width = tile_width
  auto_encoder.tile_sample_stride_height = stride_height
  auto_encoder.tile_sample_stride_width = stride_width


@AutoEncoderDataParallelismPlannerRegister.register("AutoencoderKLQwenImage")
class AutoencoderKLQwenImageDataParallelismPlanner(AutoEncoderDataParallelismPlanner):

  # The Qwen-Image 2.1 decoder takes a `first_chunk` flag for the first temporal chunk of every
  # tile; the 1.x decoder has no such argument and must not receive it.
  decoder_first_chunk: bool = False
  # 2.1 also clamps the decoded range like its own `tiled_decode`; 1.x does not.
  clamp_decoded_output: bool = False
  large_image_encode_tile_size: int = 512
  large_image_encode_tile_stride: int = 384
  large_image_decode_tile_size: int = 512
  large_image_decode_tile_stride: int = 384
  optimized_decode_geometry: tuple[int, int, int, int] | None = None

  def _apply(
    self,
    auto_encoder: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> torch.nn.Module:
    assert isinstance(
      auto_encoder, AutoencoderKLQwenImage
    ), "AutoencoderKLQwenImageDataParallelismPlanner can only be applied to AutoencoderKLQwenImage"
    dp_mesh = self.mesh(parallelism_config=parallelism_config)
    auto_encoder = self.parallelize_tiling(
      auto_encoder=auto_encoder,
      dp_mesh=dp_mesh,
    )

    return auto_encoder

  def parallelize_tiling(
    self,
    auto_encoder: AutoencoderKLQwenImage,
    dp_mesh: dist.DeviceMesh,
  ):
    group = dp_mesh.get_group()
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    # captured here: `self` inside the wrappers below is the VAE, not the planner
    decoder_first_chunk = self.decoder_first_chunk
    encode_tile_size = self.large_image_encode_tile_size
    encode_tile_stride = self.large_image_encode_tile_stride
    decode_tile_size = self.large_image_decode_tile_size
    decode_tile_stride = self.large_image_decode_tile_stride
    optimized_decode_geometry = self.optimized_decode_geometry

    auto_encoder.enable_tiling()
    default_tile_geometry = (
      auto_encoder.tile_sample_min_height,
      auto_encoder.tile_sample_min_width,
      auto_encoder.tile_sample_stride_height,
      auto_encoder.tile_sample_stride_width,
    )

    clamp_decoded_output = self.clamp_decoded_output
    comm = TileBatchedP2PComm()
    comm.set_dims(5)  # [1, 3, T, H, W]

    @functools.wraps(auto_encoder.__class__.tiled_encode)
    def new_tiled_encode(
      self: AutoencoderKLQwenImage,
      x: torch.Tensor,
      *args,
      **kwargs,
    ):
      _, _, num_frames, height, width = x.shape

      _set_tile_geometry(self, height, width, encode_tile_size, encode_tile_stride,
                         default_tile_geometry)

      latent_height = height // self.spatial_compression_ratio
      latent_width = width // self.spatial_compression_ratio

      tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
      tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
      tile_latent_stride_height = (self.tile_sample_stride_height // self.spatial_compression_ratio)
      tile_latent_stride_width = (self.tile_sample_stride_width // self.spatial_compression_ratio)

      blend_height = tile_latent_min_height - tile_latent_stride_height
      blend_width = tile_latent_min_width - tile_latent_stride_width

      # Split x into overlapping tiles and encode them separately.
      count = 0
      rows = []
      for i in range(0, height, self.tile_sample_stride_height):
        row = []
        for j in range(0, width, self.tile_sample_stride_width):
          if count % world_size == rank:
            # num_frames = 1 for image model
            self.clear_cache()
            time = []
            frame_range = 1 + (num_frames - 1) // 4
            for k in range(frame_range):
              self._enc_conv_idx = [0]
              if k == 0:
                tile = x[
                  :,
                  :,
                  :1,
                  i:i + self.tile_sample_min_height,
                  j:j + self.tile_sample_min_width,
                ]
              else:
                tile = x[
                  :,
                  :,
                  1 + 4 * (k - 1):1 + 4 * k,
                  i:i + self.tile_sample_min_height,
                  j:j + self.tile_sample_min_width,
                ]
              tile = self.encoder(tile, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
              tile = self.quant_conv(tile)
              time.append(tile)
            tile = torch.cat(time, dim=2)
          else:
            tile = None
          row.append(tile)
          count += 1
        rows.append(row)
      self.clear_cache()

      # Gather all tiles to rank 0
      if rank == 0:
        count = 0
        for i in range(len(rows)):
          for j in range(len(rows[i])):
            if count % world_size != rank:
              rows[i][j] = comm.recv_tensor(count % world_size,
                                            group,
                                            device=x.device,
                                            dtype=x.dtype)
            count += 1
      else:
        for i in range(len(rows)):
          for j in range(len(rows[i])):
            tile = rows[i][j]
            if tile is not None:
              comm.send_tensor(tile, 0, group)

      comm.sync()
      # Blend tiles on rank 0
      if rank == 0:
        result_rows = []
        for i, row in enumerate(rows):
          result_row = []
          for j, tile in enumerate(row):
            # blend the above tile and the left tile
            # to the current tile and add the current tile to the result row
            if i > 0:
              tile = self.blend_v(rows[i - 1][j], tile, blend_height)
            if j > 0:
              tile = self.blend_h(row[j - 1], tile, blend_width)
            result_row.append(tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width])
          result_rows.append(torch.cat(result_row, dim=-1))

        enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
      else:
        enc = comm.recv_tensor(rank - 1, group, device=x.device, dtype=x.dtype)

      # Propagate result through all ranks
      if rank < world_size - 1:
        comm.send_tensor(enc, rank + 1, group)

      comm.sync()
      return enc

    auto_encoder.tiled_encode = new_tiled_encode.__get__(auto_encoder)

    @functools.wraps(auto_encoder.__class__.tiled_decode)
    def new_tiled_decode(
      self: AutoencoderKLQwenImage,
      z: torch.Tensor,
      *args,
      return_dict: bool = True,
      **kwargs,
    ):
      _, _, num_frames, height, width = z.shape

      sample_height = height * self.spatial_compression_ratio
      sample_width = width * self.spatial_compression_ratio

      _set_tile_geometry(
        self,
        sample_height,
        sample_width,
        decode_tile_size,
        decode_tile_stride,
        default_tile_geometry,
        optimized_decode_geometry,
      )

      tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
      tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
      tile_latent_stride_height = (self.tile_sample_stride_height // self.spatial_compression_ratio)
      tile_latent_stride_width = (self.tile_sample_stride_width // self.spatial_compression_ratio)

      blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
      blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

      # Split z into overlapping tiles and decode them separately.
      count = 0
      rows = []
      for i in range(0, height, tile_latent_stride_height):
        row = []
        for j in range(0, width, tile_latent_stride_width):
          if count % world_size == rank:
            # num_frames = 1 for image model
            self.clear_cache()
            time = []
            for k in range(num_frames):
              self._conv_idx = [0]
              tile = z[
                :,
                :,
                k:k + 1,
                i:i + tile_latent_min_height,
                j:j + tile_latent_min_width,
              ]
              tile = self.post_quant_conv(tile)
              decoder_kwargs = {"feat_cache": self._feat_map, "feat_idx": self._conv_idx}
              if decoder_first_chunk:
                decoder_kwargs["first_chunk"] = (k == 0)
              decoded = self.decoder(tile, **decoder_kwargs)
              time.append(decoded)
            decoded = torch.cat(time, dim=2)
          else:
            decoded = None
          row.append(decoded)
          count += 1
        rows.append(row)
      self.clear_cache()

      # Gather all tiles to rank 0
      if rank == 0:
        count = 0
        for i in range(len(rows)):
          for j in range(len(rows[i])):
            if count % world_size != rank:
              rows[i][j] = comm.recv_tensor(count % world_size,
                                            group,
                                            device=z.device,
                                            dtype=z.dtype)
            count += 1
      else:
        for i in range(len(rows)):
          for j in range(len(rows[i])):
            decoded = rows[i][j]
            if decoded is not None:
              comm.send_tensor(decoded, 0, group)

      comm.sync()
      # Blend tiles on rank 0
      if rank == 0:
        result_rows = []
        for i, row in enumerate(rows):
          result_row = []
          for j, tile in enumerate(row):
            # blend the above tile and the left tile
            # to the current tile and add the current tile to the result row
            if i > 0:
              tile = self.blend_v(rows[i - 1][j], tile, blend_height)
            if j > 0:
              tile = self.blend_h(row[j - 1], tile, blend_width)
            result_row.append(tile[
              :,
              :,
              :,
              :self.tile_sample_stride_height,
              :self.tile_sample_stride_width,
            ])
          result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]
      else:
        dec = comm.recv_tensor(rank - 1, group, device=z.device, dtype=z.dtype)

      # Propagate result through all ranks
      if rank < world_size - 1:
        comm.send_tensor(dec, rank + 1, group)

      comm.sync()
      if clamp_decoded_output:
        dec = torch.clamp(dec, -1.0, 1.0)
      if not return_dict:
        return (dec, )

      return DecoderOutput(sample=dec)

    auto_encoder.tiled_decode = new_tiled_decode.__get__(auto_encoder)

    return auto_encoder
