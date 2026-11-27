"""CPU-only guards for the P-H3-Image VAE data-parallel planner (`vae=true` in `--parallel-params`).

The planner is a tile distributor: `AutoencoderKLPH3Image` lays 256-pixel tiles with a 64-pixel
overlap and stitches them back, so a tile is the only independent unit of work. Three things can go
wrong without any shape error, and each one is covered here:

1. **Registration.** The auto-encoder registry matches the **full** class name, so a missing or
   misspelled `_safe_import` falls back to the generic `AutoencoderKL` planner, which patches
   `tiled_decode` / `blend_v` — neither exists on this VAE.
2. **The patch point.** This VAE tiles inside `_encode_clip` / `_decode_clip` (the MiniMax-H3-Image
   VAE, which the sibling planner targets, tiles inside `_encode` / `_decode`). Patching the wrong
   pair leaves the serial path running while every log line still says VAE-P is on.
3. **The single-rank pass-through.** `world_size == 1` must return the original result unchanged:
   that is what makes the plan bit-exact rather than "close enough".
"""

import torch
import torch.distributed as dist
import pytest

from cache_dit.distributed.autoencoders.autoencoder_kl_p_h3_image import (
  AutoencoderKLPH3ImageDataParallelismPlanner, )
from cache_dit.distributed.autoencoders.planners import _activate_auto_encoder_dp_planners
from cache_dit.distributed.autoencoders.register import AutoEncoderDataParallelismPlannerRegister


@pytest.fixture
def single_rank_group(tmp_path):
  """A one-process `gloo` group: `TileSafeP2PComm` reads the default backend at construction.

  :param tmp_path: Per-test directory for the file-store rendezvous.
  :yields: None once the group is up; torn down after the test.
  """
  if not dist.is_initialized():
    dist.init_process_group(
      backend="gloo",
      init_method=f"file://{tmp_path}/pg",
      rank=0,
      world_size=1,
    )
  yield
  if dist.is_initialized():
    dist.destroy_process_group()


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


class _SingleRankMesh:
  """The one thing `parallelize_tiling` needs from the device mesh."""

  def get_group(self):
    return None


def test_planner_is_reachable_through_the_activation_path():
  _activate_auto_encoder_dp_planners()

  assert "AutoencoderKLPH3Image" in AutoEncoderDataParallelismPlannerRegister.supported_planners(
  )[1]
  assert (AutoEncoderDataParallelismPlannerRegister.get_planner("AutoencoderKLPH3Image")
          is AutoencoderKLPH3ImageDataParallelismPlanner)


def test_planner_patches_the_tiled_clip_methods(single_rank_group):
  vae = _tiny_vae()
  latent = torch.randn(1, 4, 1, 8, 8)
  expected = vae._decode_clip(latent)
  pixels = torch.rand(1, 3, 1, 16, 16)
  expected_encode = vae._encode_clip(pixels)

  planner = AutoencoderKLPH3ImageDataParallelismPlanner()
  planner.parallelize_tiling(auto_encoder=vae, dp_mesh=_SingleRankMesh())

  assert vae._decode_clip.__name__ == "_decode_clip", "the patch must keep the method's identity"
  assert vae._decode_clip.__wrapped__ is not None, "the original has to stay reachable"
  # the serial path of a single rank is the untouched implementation
  torch.testing.assert_close(vae._decode_clip(latent), expected)
  torch.testing.assert_close(vae._encode_clip(pixels), expected_encode)


def test_decode_still_works_through_the_public_entry_point(single_rank_group):
  vae = _tiny_vae()
  latent = torch.randn(1, 4, 1, 8, 8)

  AutoencoderKLPH3ImageDataParallelismPlanner().parallelize_tiling(auto_encoder=vae,
                                                                   dp_mesh=_SingleRankMesh())

  decoded = vae.decode(latent, return_dict=False)[0]
  assert decoded.shape == (1, 3, 1, 16, 16)
