import functools

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLQwenImage
from diffusers.models.autoencoders.vae import DecoderOutput

from cache_dit.logger import init_logger
from cache_dit.parallelism.config import ParallelismConfig
from .dp_plan_registers import (
    AutoEncoderDataParallelismPlanner,
    AutoEncoderDataParallelismPlannerRegister,
)
from .utils import TileBatchedP2PComm

logger = init_logger(__name__)


@AutoEncoderDataParallelismPlannerRegister.register("AutoencoderKLQwenImage")
class AutoencoderKLQwenImageDataParallelismPlanner(AutoEncoderDataParallelismPlanner):
    def apply(
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

        auto_encoder.enable_tiling()

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

            # Overwrite tile size and stride for better performance while
            # still reducing memory usage.
            if min(height, width) >= 1024:
                self.tile_sample_min_height = 512
                self.tile_sample_min_width = 512
                self.tile_sample_stride_height = 384
                self.tile_sample_stride_width = 384

            latent_height = height // self.spatial_compression_ratio
            latent_width = width // self.spatial_compression_ratio

            tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
            tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
            tile_latent_stride_height = (
                self.tile_sample_stride_height // self.spatial_compression_ratio
            )
            tile_latent_stride_width = (
                self.tile_sample_stride_width // self.spatial_compression_ratio
            )

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
                                    i : i + self.tile_sample_min_height,
                                    j : j + self.tile_sample_min_width,
                                ]
                            else:
                                tile = x[
                                    :,
                                    :,
                                    1 + 4 * (k - 1) : 1 + 4 * k,
                                    i : i + self.tile_sample_min_height,
                                    j : j + self.tile_sample_min_width,
                                ]
                            tile = self.encoder(
                                tile, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx
                            )
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
                            rows[i][j] = comm.recv_tensor(
                                count % world_size, group, device=x.device, dtype=x.dtype
                            )
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
                        result_row.append(
                            tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width]
                        )
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

            # Overwrite tile size and stride for better performance while
            # still reducing memory usage.
            if min(sample_height, sample_width) >= 1024:
                self.tile_sample_min_height = 512
                self.tile_sample_min_width = 512
                self.tile_sample_stride_height = 384
                self.tile_sample_stride_width = 384

            tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
            tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
            tile_latent_stride_height = (
                self.tile_sample_stride_height // self.spatial_compression_ratio
            )
            tile_latent_stride_width = (
                self.tile_sample_stride_width // self.spatial_compression_ratio
            )

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
                                k : k + 1,
                                i : i + tile_latent_min_height,
                                j : j + tile_latent_min_width,
                            ]
                            tile = self.post_quant_conv(tile)
                            decoded = self.decoder(
                                tile, feat_cache=self._feat_map, feat_idx=self._conv_idx
                            )
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
                            rows[i][j] = comm.recv_tensor(
                                count % world_size, group, device=z.device, dtype=z.dtype
                            )
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
                        result_row.append(
                            tile[
                                :,
                                :,
                                :,
                                : self.tile_sample_stride_height,
                                : self.tile_sample_stride_width,
                            ]
                        )
                    result_rows.append(torch.cat(result_row, dim=-1))

                dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]
            else:
                dec = comm.recv_tensor(rank - 1, group, device=z.device, dtype=z.dtype)

            # Propagate result through all ranks
            if rank < world_size - 1:
                comm.send_tensor(dec, rank + 1, group)

            comm.sync()
            if not return_dict:
                return (dec,)

            return DecoderOutput(sample=dec)

        auto_encoder.tiled_decode = new_tiled_decode.__get__(auto_encoder)

        return auto_encoder
