import functools

import torch
import torch.distributed as dist
from typing import Optional
from diffusers import AutoencoderKLLTX2Video
from diffusers.models.autoencoders.vae import DecoderOutput

from cache_dit.parallelism.config import ParallelismConfig
from .dp_plan_registers import (
    AutoEncoderDataParallelismPlanner,
    AutoEncoderDataParallelismPlannerRegister,
)
from .utils import TileBatchedP2PComm


@AutoEncoderDataParallelismPlannerRegister.register("AutoencoderKLLTX2Video")
class AutoencoderKLLTX2VideoDataParallelismPlanner(AutoEncoderDataParallelismPlanner):
    def apply(
        self,
        auto_encoder: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        assert isinstance(
            auto_encoder, AutoencoderKLLTX2Video
        ), "AutoencoderKLLTX2VideoDataParallelismPlanner can only be applied to AutoencoderKLLTX2Video"
        dp_mesh = self.mesh(parallelism_config=parallelism_config)
        auto_encoder = self.parallelize_tiling(
            auto_encoder=auto_encoder,
            dp_mesh=dp_mesh,
        )

        return auto_encoder

    def parallelize_tiling(
        self,
        auto_encoder: AutoencoderKLLTX2Video,
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
            self: AutoencoderKLLTX2Video,
            x: torch.Tensor,
            causal=None,
            *args,
            **kwargs,
        ):
            _, _, num_frames, height, width = x.shape
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

            count = 0
            rows = []
            for i in range(0, height, self.tile_sample_stride_height):
                row = []
                for j in range(0, width, self.tile_sample_stride_width):
                    if count % world_size == rank:
                        tile = x[
                            :,
                            :,
                            :,
                            i : i + self.tile_sample_min_height,
                            j : j + self.tile_sample_min_width,
                        ]
                        tile = self.encoder(tile, causal=causal)
                    else:
                        tile = None
                    row.append(tile)
                    count += 1
                rows.append(row)

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
            if rank == 0:
                result_rows = []
                for i, row in enumerate(rows):
                    result_row = []
                    for j, tile in enumerate(row):
                        if i > 0:
                            tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                        if j > 0:
                            tile = self.blend_h(row[j - 1], tile, blend_width)
                        result_row.append(
                            tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width]
                        )
                    result_rows.append(torch.cat(result_row, dim=4))

                enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
            else:
                enc = comm.recv_tensor(rank - 1, group, device=x.device, dtype=x.dtype)

            if rank < world_size - 1:
                comm.send_tensor(enc, rank + 1, group)

            comm.sync()
            return enc

        auto_encoder.tiled_encode = new_tiled_encode.__get__(auto_encoder)

        @functools.wraps(auto_encoder.__class__.tiled_decode)
        def new_tiled_decode(
            self: AutoencoderKLLTX2Video,
            z: torch.Tensor,
            temb: Optional[torch.Tensor],
            causal=None,
            return_dict: bool = True,
            *args,
            **kwargs,
        ):
            _, _, num_frames, height, width = z.shape
            sample_height = height * self.spatial_compression_ratio
            sample_width = width * self.spatial_compression_ratio

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

            count = 0
            rows = []
            for i in range(0, height, tile_latent_stride_height):
                row = []
                for j in range(0, width, tile_latent_stride_width):
                    if count % world_size == rank:
                        tile = z[
                            :, :, :, i : i + tile_latent_min_height, j : j + tile_latent_min_width
                        ]
                        decoded = self.decoder(tile, temb, causal=causal)
                    else:
                        decoded = None
                    row.append(decoded)
                    count += 1
                rows.append(row)

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
            if rank == 0:
                result_rows = []
                for i, row in enumerate(rows):
                    result_row = []
                    for j, tile in enumerate(row):
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
                    result_rows.append(torch.cat(result_row, dim=4))

                dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]
            else:
                dec = comm.recv_tensor(rank - 1, group, device=z.device, dtype=z.dtype)

            if rank < world_size - 1:
                comm.send_tensor(dec, rank + 1, group)

            comm.sync()
            if not return_dict:
                return (dec,)

            return DecoderOutput(sample=dec)

        auto_encoder.tiled_decode = new_tiled_decode.__get__(auto_encoder)

        return auto_encoder
