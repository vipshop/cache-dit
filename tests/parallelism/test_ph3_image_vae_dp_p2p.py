"""Two-rank regression for the P-H3-Image VAE planner (`--parallel-vae`) on a CPU/gloo build.

The single-round tests in `test_ph3_image_vae_dp.py` only prove the patch point and the pass-through.
This file exercises the plan itself: two real ranks, four tiles, real P2P, and a comparison against
the serial result. Three failure modes are covered that no shape check would catch:

1. **Tile ownership.** The round-robin has to give every tile to exactly one rank; a rank that also
   computes a tile it will receive leaves rank 0 stitching a duplicate.
2. **Stitch correctness.** Only rank 0 stitches (the blending of overlapping tiles is order
   dependent), then hands the result down the chain, so every rank returns the same tensor.
3. **Bit-exactness.** Each rank must return the serial `_decode_clip` / `_encode_clip` result to the
   last bit: distributing tiles is a partition of the work, not an approximation.
"""

import socket
import subprocess
import sys

import pytest
import torch

_NEEDS_DIST = not torch.distributed.is_available()
pytestmark = pytest.mark.skipif(_NEEDS_DIST, reason="needs torch.distributed")

# `latent * spatial_compression_ratio = 320` px over 256-px tiles means 2x2 tiles.
_LATENT = 160


def _free_port() -> int:
  """Pick a rendezvous port that is free right now.

  :returns: A TCP port number.
  """
  with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    return sock.getsockname()[1]


def _tiny_vae():
  from diffusers import AutoencoderKLPH3Image

  torch.manual_seed(0)
  return AutoencoderKLPH3Image(
    in_channels=3,
    out_channels=3,
    latent_channels=4,
    block_out_channels=(8, ),
    layers_per_block=1,
    spatial_downsample_factors=(2, ),
    temporal_downsample_factors=(1, ),
    norm_num_groups=4,
    decoder_num_layers=1,
    decoder_num_attention_heads=2,
    decoder_attention_head_dim=8,
    decoder_num_register_tokens=2,
  )


def _worker(rank: int, port: int) -> None:
  """Run one rank: patch a tiny VAE, decode, and compare with the serial result.

  :param rank: This process's rank.
  :param port: Rendezvous port.
  """
  import torch.distributed as dist

  from cache_dit.distributed.autoencoders.autoencoder_kl_p_h3_image import (
    AutoencoderKLPH3ImageDataParallelismPlanner, )

  dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
  try:
    vae = _tiny_vae()
    latent = torch.randn(1, 4, 1, _LATENT, _LATENT, generator=torch.Generator().manual_seed(7))
    pixels = torch.rand(1,
                        3,
                        1,
                        _LATENT * 2,
                        _LATENT * 2,
                        generator=torch.Generator().manual_seed(11))

    # The same weights and inputs on both ranks: the serial path is the reference on each of them.
    expected_decode = vae._decode_clip(latent)
    expected_encode = vae._encode_clip(pixels)

    tiles = vae._split_tiles(_LATENT * vae.spatial_compression_ratio, vae.tile_sample_min_height,
                             vae.tile_sample_min_overlap_height)
    assert len(tiles[0]) > 1, "the test needs more than one tile per axis to move any work"

    class _Mesh:

      def get_group(self):
        return dist.group.WORLD

    AutoencoderKLPH3ImageDataParallelismPlanner().parallelize_tiling(auto_encoder=vae,
                                                                     dp_mesh=_Mesh())

    torch.testing.assert_close(vae._decode_clip(latent), expected_decode, rtol=0, atol=0)
    torch.testing.assert_close(vae._encode_clip(pixels), expected_encode, rtol=0, atol=0)
    dist.barrier()
  finally:
    dist.destroy_process_group()


def test_two_ranks_reproduce_the_serial_decode_and_encode():
  port = _free_port()
  workers = [
    subprocess.Popen(
      [sys.executable, __file__, "--rank",
       str(rank), "--port", str(port)],
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True) for rank in (0, 1)
  ]
  outputs = []
  for worker in workers:
    try:
      outputs.append(worker.communicate(timeout=300)[0])
    except subprocess.TimeoutExpired:
      worker.kill()
      outputs.append(worker.communicate()[0] + "\n<timed out: the two ranks deadlocked>")
  for rank, (worker, output) in enumerate(zip(workers, outputs)):
    assert worker.returncode == 0, f"rank {rank} exited with {worker.returncode}:\n{output}"


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--rank", type=int, required=True)
  parser.add_argument("--port", type=int, required=True)
  arguments = parser.parse_args()
  _worker(arguments.rank, arguments.port)
