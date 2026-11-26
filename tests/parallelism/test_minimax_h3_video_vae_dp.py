"""Exercise MiniMax-H3 tile batching and failure propagation on a CPU VAE subgroup."""

from datetime import timedelta
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch import nn

from cache_dit.distributed.autoencoders import parallelize_autoencoder
from cache_dit.distributed.autoencoders.register import AutoEncoderDataParallelismPlanner
from cache_dit.distributed.config import ParallelismConfig

if TYPE_CHECKING:
  from diffusers import AutoencoderKLMiniMaxH3


class RecordingDecoder(nn.Module):

  def __init__(self) -> None:
    super().__init__()
    self.calls: list[list[float]] = []
    self.fail = False

  def forward(self, tile: torch.Tensor) -> torch.Tensor:
    if self.fail:
      raise RuntimeError("injected tile failure")
    self.calls.append(tile[:, 0, 0, 0, 0].tolist())
    return (tile + tile.mean(dim=(-1, -2), keepdim=True)).repeat_interleave(4, dim=2)


def _video_vae() -> "AutoencoderKLMiniMaxH3":
  from diffusers import AutoencoderKLMiniMaxH3

  vae = AutoencoderKLMiniMaxH3(latent_channels=1,
                               block_out_channels=(4, 4),
                               spatial_downsample_factors=(1, 1),
                               temporal_downsample_factors=(2, 2),
                               norm_num_groups=1,
                               decoder_num_layers=1,
                               decoder_num_attention_heads=1,
                               decoder_attention_head_dim=8)
  vae.enable_tiling(tile_sample_min_height=4,
                    tile_sample_min_width=4,
                    tile_sample_min_overlap_height=2,
                    tile_sample_min_overlap_width=2)
  nn.init.ones_(vae.post_quant_conv.weight)
  nn.init.zeros_(vae.post_quant_conv.bias)
  vae.decoder = RecordingDecoder()
  return vae


@torch.no_grad()
def _run_rank(world_rank: int, rendezvous: str) -> None:
  from diffusers import AutoencoderKL

  torch.set_num_threads(1)
  dist.init_process_group("gloo",
                          init_method=rendezvous,
                          rank=world_rank,
                          world_size=3,
                          timeout=timedelta(seconds=45))
  try:
    # Group ranks differ from global ranks, which send/recv expect in PyTorch.
    mesh = dist.DeviceMesh("cpu", [1, 2])
    singleton = dist.DeviceMesh("cpu", [[0], [1], [2]], mesh_dim_names=("replica", "vae"))["vae"]
    if world_rank == 0:
      return
    rank = dist.get_rank(mesh.get_group())
    reference, vae, other = _video_vae(), _video_vae(), _video_vae()
    # The planner must not overwrite a preference set before installation.
    other._h3_vae_batch_local_tiles = False
    config = ParallelismConfig(tp_size=2)
    # Install during inference mode, then exercise the planner under no_grad.
    with torch.inference_mode(), patch.object(AutoEncoderDataParallelismPlanner,
                                              "mesh",
                                              return_value=mesh):
      image = AutoencoderKL(block_out_channels=(4, ), norm_num_groups=1)
      assert parallelize_autoencoder(image, config).use_tiling
      parallelize_autoencoder(vae, config)
      parallelize_autoencoder(other, config)

    latent = torch.arange(37 * 10 * 16).view(1, 1, 37, 10, 16).float()
    expected = reference.decode(latent).sample
    assert expected.shape[2] == 124
    # Local tiles are batched by default: 14 tiles per rank, one forward per window.
    torch.testing.assert_close(vae.decode(latent).sample, expected, rtol=0, atol=0)
    assert [len(batch) for batch in vae.decoder.calls] == [14] * 7
    vae.decoder.calls.clear()

    # The dynamic instance attribute is the opt-out, no public setter needed.
    vae._h3_vae_batch_local_tiles = False
    torch.testing.assert_close(vae.decode(latent).sample, expected, rtol=0, atol=0)
    assert [len(batch) for batch in vae.decoder.calls] == [1] * 98
    vae.decoder.calls.clear()
    del vae._h3_vae_batch_local_tiles

    # `other` opted out before installation; the planner must keep that preference.
    torch.testing.assert_close(other.decode(latent).sample, expected, rtol=0, atol=0)
    assert [len(batch) for batch in other.decoder.calls] == [1] * 98
    other.decoder.calls.clear()
    del other._h3_vae_batch_local_tiles

    for batch, height, width in ((2, 10, 16), (2, 3, 8), (2, 4, 4), (0, 10, 16)):
      reference.decoder.calls.clear()
      vae.decoder.calls.clear()
      z = torch.arange(batch * 3 * height * width).view(batch, 1, 3, height, width).float()
      expected = reference._decode_clip(z)
      owned_tiles = reference.decoder.calls[rank::2]
      torch.testing.assert_close(vae._decode_clip(z), expected, rtol=0, atol=0)
      assert vae.decoder.calls == ([list(chain.from_iterable(owned_tiles))] if owned_tiles else [])

    z = latent[:, :, :3]
    vae.disable_tiling()
    reference.disable_tiling()
    torch.testing.assert_close(vae._decode_clip(z), reference._decode_clip(z), rtol=0, atol=0)
    vae.enable_tiling()
    reference.enable_tiling()
    expected = reference._decode_clip(z)
    vae.decoder.calls.clear()
    torch.testing.assert_close(vae._decode_clip(z), expected, rtol=0, atol=0)
    assert len(vae.decoder.calls) == 1

    vae.decoder.fail = rank == 1
    with pytest.raises(RuntimeError, match="group rank 1: RuntimeError: injected tile failure"):
      vae._decode_clip(z)
    vae.decoder.fail = False
    with (patch.object(vae, "_stitch_tiles", side_effect=RuntimeError("injected stitch failure")),
          pytest.raises(RuntimeError, match="group rank 0: RuntimeError: injected stitch failure")):
      vae._decode_clip(z)
    torch.testing.assert_close(vae._decode_clip(z), expected, rtol=0, atol=0)

    torch.testing.assert_close(other._decode_clip(z), expected, rtol=0, atol=0)
    assert [len(batch) for batch in other.decoder.calls] == [14]
    vae._h3_vae_batch_local_tiles = False
    vae.decoder.calls.clear()
    torch.testing.assert_close(vae._decode_clip(z), expected, rtol=0, atol=0)
    assert len(vae.decoder.calls) == 14

    with patch.object(AutoEncoderDataParallelismPlanner, "mesh", return_value=singleton):
      parallelize_autoencoder(reference, config)
    reference.decoder.calls.clear()
    torch.testing.assert_close(reference._decode_clip(z), expected, rtol=0, atol=0)
    assert len(reference.decoder.calls) == 28
  finally:
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires CPU/Gloo")
def test_h3_tile_ownership_stitching_and_failures(tmp_path: Path) -> None:
  torch.multiprocessing.spawn(_run_rank, args=((tmp_path / "group").as_uri(), ), nprocs=3)
