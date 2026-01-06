# Adapted from: https://github.com/chengzeyi/ParaAttention.git
import functools

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLWan
from diffusers.models.autoencoders.vae import DecoderOutput

from cache_dit.logger import init_logger
from cache_dit.parallelism.config import ParallelismConfig
from .dp_plan_registers import (
    AutoEncoderDataParallelismPlanner,
    AutoEncoderDataParallelismPlannerRegister,
)
from .utils import send_tensor, recv_tensor

logger = init_logger(__name__)


@AutoEncoderDataParallelismPlannerRegister.register("AutoencoderKLWan")
class AutoencoderKLWanDataParallelismPlanner(AutoEncoderDataParallelismPlanner):
    def apply(
        self,
        auto_encoder: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        assert isinstance(
            auto_encoder, AutoencoderKLWan
        ), "AutoencoderKLWanDataParallelismPlanner can only be applied to AutoencoderKLWan"
        auto_encoder_world_size = parallelism_config.auto_encoder_world_size
        device_type = torch.accelerator.current_accelerator().type
        dp_mesh = dist.init_device_mesh(
            device_type=device_type,
            mesh_shape=[auto_encoder_world_size],
        )

        auto_encoder = self.parallelize_tiling(
            auto_encoder=auto_encoder,
            dp_mesh=dp_mesh,
        )
        return auto_encoder

    def parallelize_tiling(
        self,
        auto_encoder: AutoencoderKLWan,
        dp_mesh: dist.DeviceMesh,
    ):
        group = dp_mesh.get_group()
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)

        def blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
            blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
            for y in range(blend_extent):
                b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[
                    :, :, :, y, :
                ] * (y / blend_extent)
            return b

        def blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
            blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
            for x in range(blend_extent):
                b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[
                    :, :, :, :, x
                ] * (x / blend_extent)
            return b

        @functools.wraps(auto_encoder.__class__._encode)
        def new_encode(
            self: AutoencoderKLWan,
            x: torch.Tensor,
            *args,
            **kwargs,
        ):
            self.tile_sample_min_height = 256
            self.tile_sample_min_width = 256

            # The minimal distance between two spatial tiles
            self.tile_sample_stride_height = 192
            self.tile_sample_stride_width = 192
            self.spatial_compression_ratio = 8

            batch_size, num_channels, num_frames, height, width = x.shape
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

            if hasattr(self, "tile_sample_min_height"):
                tile_sample_min_height = self.tile_sample_min_height
            else:
                tile_sample_min_height = self.tile_sample_min_size

            if hasattr(self, "tile_sample_min_width"):
                tile_sample_min_width = self.tile_sample_min_width
            else:
                tile_sample_min_width = self.tile_sample_min_size

            # Split x into overlapping tiles and encode them separately.
            # The tiles have an overlap to avoid seams between tiles.
            count = 0
            rows = []
            for j in range(0, height, self.tile_sample_stride_height):
                row = []
                for k in range(0, width, self.tile_sample_stride_width):
                    if count % world_size == rank:
                        tile = x[
                            :, :, :, j : j + tile_sample_min_height, k : k + tile_sample_min_width
                        ]
                        self.clear_cache()
                        t = tile.shape[2]
                        iter_ = 1 + (t - 1) // 4
                        for i in range(iter_):
                            self._enc_conv_idx = [0]
                            if i == 0:
                                out = self.encoder(
                                    tile[:, :, :1, :, :],
                                    feat_cache=self._enc_feat_map,
                                    feat_idx=self._enc_conv_idx,
                                )
                            else:
                                out_ = self.encoder(
                                    tile[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                                    feat_cache=self._enc_feat_map,
                                    feat_idx=self._enc_conv_idx,
                                )
                                out = torch.cat([out, out_], 2)

                        enc = self.quant_conv(out)
                        # mu, logvar = enc[:, : self.z_dim, :, :, :], enc[:, self.z_dim :, :, :, :]
                        # enc = torch.cat([mu, logvar], dim=1)
                        self.clear_cache()
                        # tile = self.encoder(tile)
                    else:
                        enc = None
                    row.append(enc)
                    count += 1
                rows.append(row)

            if rank == 0:
                count = 0
                for i in range(len(rows)):
                    for j in range(len(rows[i])):
                        if count % world_size != rank:
                            rows[i][j] = recv_tensor(
                                count % world_size, group, device=x.device, dtype=x.dtype
                            )
                        count += 1
            else:
                for i in range(len(rows)):
                    for j in range(len(rows[i])):
                        tile = rows[i][j]
                        if tile is not None:
                            send_tensor(tile, 0, group)

            if rank == 0:
                result_rows = []
                for i, row in enumerate(rows):
                    result_row = []
                    for j, tile in enumerate(row):
                        # blend the above tile and the left tile
                        # to the current tile and add the current tile to the result row
                        if i > 0:
                            tile = blend_v(rows[i - 1][j], tile, blend_height)
                        if j > 0:
                            tile = blend_h(row[j - 1], tile, blend_width)
                        result_row.append(
                            tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width]
                        )
                    result_rows.append(torch.cat(result_row, dim=-1))

                enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
            else:
                enc = recv_tensor(rank - 1, group, device=x.device, dtype=x.dtype)
            if rank < world_size - 1:
                send_tensor(enc, rank + 1, group)
            return enc

        auto_encoder._encode = new_encode.__get__(auto_encoder)

        @functools.wraps(auto_encoder.__class__._decode)
        def new_decode(
            self: AutoencoderKLWan,
            z: torch.Tensor,
            *args,
            return_dict: bool = True,
            **kwargs,
        ):

            self.tile_sample_min_height = 256
            self.tile_sample_min_width = 256

            # The minimal distance between two spatial tiles
            self.tile_sample_stride_height = 192
            self.tile_sample_stride_width = 192
            self.spatial_compression_ratio = 8

            batch_size, num_channels, num_frames, height, width = z.shape
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

            # Split z into overlapping tiles and decode them separately.
            # The tiles have an overlap to avoid seams between tiles.
            count = 0
            rows = []
            for j in range(0, height, tile_latent_stride_height):
                row = []
                for k in range(0, width, tile_latent_stride_width):
                    if count % world_size == rank:
                        tile = z[
                            :, :, :, j : j + tile_latent_min_height, k : k + tile_latent_min_width
                        ]
                        tile = self.post_quant_conv(tile)
                        self.clear_cache()
                        iter_ = tile.shape[2]
                        for i in range(iter_):
                            self._conv_idx = [0]
                            if i == 0:
                                out = self.decoder(
                                    tile[:, :, i : i + 1, :, :],
                                    feat_cache=self._feat_map,
                                    feat_idx=self._conv_idx,
                                )
                            else:
                                out_ = self.decoder(
                                    tile[:, :, i : i + 1, :, :],
                                    feat_cache=self._feat_map,
                                    feat_idx=self._conv_idx,
                                )
                                out = torch.cat([out, out_], 2)
                        decoded = torch.clamp(out, min=-1.0, max=1.0)
                        self.clear_cache()
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
                            rows[i][j] = recv_tensor(
                                count % world_size, group, device=z.device, dtype=z.dtype
                            )
                        count += 1
            else:
                for i in range(len(rows)):
                    for j in range(len(rows[i])):
                        decoded = rows[i][j]
                        if decoded is not None:
                            send_tensor(decoded, 0, group)

            if rank == 0:
                result_rows = []
                for i, row in enumerate(rows):
                    result_row = []
                    for j, tile in enumerate(row):
                        # blend the above tile and the left tile
                        # to the current tile and add the current tile to the result row
                        if i > 0:
                            tile = blend_v(rows[i - 1][j], tile, blend_height)
                        if j > 0:
                            tile = blend_h(row[j - 1], tile, blend_width)
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
                dec = recv_tensor(rank - 1, group, device=z.device, dtype=z.dtype)
            if rank < world_size - 1:
                send_tensor(dec, rank + 1, group)

            if not return_dict:
                return (dec,)
            return DecoderOutput(dec, dec)

        auto_encoder._decode = new_decode.__get__(auto_encoder)

        return auto_encoder
