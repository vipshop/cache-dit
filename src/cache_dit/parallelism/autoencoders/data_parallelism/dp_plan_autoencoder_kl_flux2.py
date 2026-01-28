import functools

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLFlux2
from diffusers.models.autoencoders.vae import DecoderOutput

from cache_dit.logger import init_logger
from cache_dit.parallelism.config import ParallelismConfig
from .dp_plan_registers import (
    AutoEncoderDataParallelismPlanner,
    AutoEncoderDataParallelismPlannerRegister,
)
from .utils import TileBatchedP2PComm

logger = init_logger(__name__)


@AutoEncoderDataParallelismPlannerRegister.register("AutoencoderKLFlux2")
class AutoencoderKLFlux2DataParallelismPlanner(AutoEncoderDataParallelismPlanner):
    def apply(
        self,
        auto_encoder: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        assert isinstance(
            auto_encoder, AutoencoderKLFlux2
        ), "AutoencoderKLFlux2DataParallelismPlanner can only be applied to AutoencoderKLFlux2"
        dp_mesh = self.mesh(parallelism_config=parallelism_config)
        auto_encoder = self.parallelize_tiling(
            auto_encoder=auto_encoder,
            dp_mesh=dp_mesh,
        )

        return auto_encoder

    def parallelize_tiling(
        self,
        auto_encoder: AutoencoderKLFlux2,
        dp_mesh: dist.DeviceMesh,
    ):
        group = dp_mesh.get_group()
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)

        auto_encoder.enable_tiling()

        comm = TileBatchedP2PComm()
        comm.set_dims(4)  # [1, 3, H, W]

        @functools.wraps(auto_encoder.__class__._tiled_encode)
        def new_tiled_encode(
            self: AutoencoderKLFlux2,
            x: torch.Tensor,
            *args,
            **kwargs,
        ):
            overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
            blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
            row_limit = self.tile_latent_min_size - blend_extent

            # Split the image into 512x512 tiles and encode them separately.
            count = 0
            rows = []
            for i in range(0, x.shape[2], overlap_size):
                row = []
                for j in range(0, x.shape[3], overlap_size):
                    if count % world_size == rank:
                        tile = x[
                            :,
                            :,
                            i : i + self.tile_sample_min_size,
                            j : j + self.tile_sample_min_size,
                        ]
                        tile = self.encoder(tile)
                        if self.config.use_quant_conv:
                            tile = self.quant_conv(tile)
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
                        # blend the above tile and the left tile
                        # to the current tile and add the current tile to the result row
                        if i > 0:
                            tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                        if j > 0:
                            tile = self.blend_h(row[j - 1], tile, blend_extent)
                        result_row.append(tile[:, :, :row_limit, :row_limit])
                    result_rows.append(torch.cat(result_row, dim=3))

                enc = torch.cat(result_rows, dim=2)
            else:
                enc = comm.recv_tensor(rank - 1, group, device=x.device, dtype=x.dtype)
            if rank < world_size - 1:
                comm.send_tensor(enc, rank + 1, group)

            comm.sync()
            return enc

        auto_encoder._tiled_encode = new_tiled_encode.__get__(auto_encoder)

        @functools.wraps(auto_encoder.__class__.tiled_decode)
        def new_tiled_decode(
            self: AutoencoderKLFlux2,
            z: torch.Tensor,
            *args,
            return_dict: bool = False,
            **kwargs,
        ):
            overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
            blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
            row_limit = self.tile_sample_min_size - blend_extent

            # Split z into overlapping 64x64 tiles and decode them separately.
            # The tiles have an overlap to avoid seams between tiles.
            count = 0
            rows = []
            for i in range(0, z.shape[2], overlap_size):
                row = []
                for j in range(0, z.shape[3], overlap_size):
                    if count % world_size == rank:
                        tile = z[
                            :,
                            :,
                            i : i + self.tile_latent_min_size,
                            j : j + self.tile_latent_min_size,
                        ]
                        if self.config.use_post_quant_conv:
                            tile = self.post_quant_conv(tile)
                        decoded = self.decoder(tile)
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
                        # blend the above tile and the left tile
                        # to the current tile and add the current tile to the result row
                        if i > 0:
                            tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                        if j > 0:
                            tile = self.blend_h(row[j - 1], tile, blend_extent)
                        result_row.append(tile[:, :, :row_limit, :row_limit])
                    result_rows.append(torch.cat(result_row, dim=3))

                dec = torch.cat(result_rows, dim=2)
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
