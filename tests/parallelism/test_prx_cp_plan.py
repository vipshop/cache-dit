"""CPU checks for the PRX CP plan and its joint-mask rebuild.

The Ulysses all-to-all lays the global sequence out as `cat_r([text_r, img_r])`, so the joint
text+image key mask has to be rebuilt in that exact order. A wrong order still renders a
plausible image, so the layout is asserted here directly instead of relying on an end-to-end
PSNR comparison.

Usage:
  python -m pytest tests/parallelism/test_prx_cp_plan.py -q
"""

import pytest
import torch

try:
  from diffusers.models.transformers.transformer_prx import PRXTransformer2DModel  # noqa: F401
except ImportError:
  pytest.skip("diffusers without PRX support", allow_module_level=True)

from cache_dit.distributed.config import ParallelismConfig
from cache_dit.distributed.transformers.prx import (
  PrxContextParallelismPlanner,
  _build_cp_joint_mask,
)


class _FakeContextParallelConfig:

  def __init__(self, world_size: int):
    self._world_size = world_size


class _FakeProcessor:

  def __init__(self, world_size: int):
    self._cp_config = _FakeContextParallelConfig(world_size)


def _tiny_transformer() -> PRXTransformer2DModel:
  torch.manual_seed(0)
  return PRXTransformer2DModel(
    in_channels=3,
    patch_size=16,
    context_in_dim=16,
    hidden_size=256,
    mlp_ratio=2.0,
    num_heads=2,
    depth=1,
    axes_dim=[64, 64],
    bottleneck_size=96,
    resolution_embeds=True,
  )


@pytest.mark.parametrize("world_size", [2, 4])
def test_joint_mask_follows_all_to_all_order(world_size):
  # Two real tokens followed by padding: the pad positions must stay at the end of every text
  # chunk, never bleed into an image chunk.
  local_txt = 4
  local_img = 3
  per_rank = [True, True, False, False]
  global_mask = torch.tensor([per_rank * world_size], dtype=torch.bool)

  joint = _build_cp_joint_mask(
    _FakeProcessor(world_size),
    global_mask,
    batch_size=1,
    local_txt=local_txt,
    local_img=local_img,
    device=torch.device("cpu"),
  )

  assert joint.shape == (1, 1, 1, world_size * (local_txt + local_img))
  flat = joint.view(-1)
  for rank in range(world_size):
    base = rank * (local_txt + local_img)
    text_chunk = flat[base:base + local_txt]
    image_chunk = flat[base + local_txt:base + local_txt + local_img]
    assert torch.equal(text_chunk, torch.tensor(per_rank)), f"rank {rank} text chunk order"
    assert image_chunk.all(), f"rank {rank} image chunk is not fully unmasked"


def test_joint_mask_casts_float_masks_to_bool():
  # SDPA reads a float mask as an additive bias, which would reward padded keys instead of
  # masking them. Upstream casts to bool; the CP rebuild has to do the same.
  joint = _build_cp_joint_mask(
    _FakeProcessor(2),
    torch.ones(1, 4),
    batch_size=1,
    local_txt=2,
    local_img=2,
    device=torch.device("cpu"),
  )
  assert joint.dtype == torch.bool


def test_joint_mask_none_passthrough():
  assert _build_cp_joint_mask(_FakeProcessor(2),
                              None,
                              batch_size=1,
                              local_txt=2,
                              local_img=2,
                              device=torch.device("cpu")) is None


def test_joint_mask_rejects_mismatched_text_length():
  with pytest.raises(ValueError, match="global text length"):
    _build_cp_joint_mask(
      _FakeProcessor(2),
      torch.ones(1, 6, dtype=torch.bool),
      batch_size=1,
      local_txt=2,
      local_img=2,
      device=torch.device("cpu"),
    )


def test_joint_mask_rejects_batch_mismatch():
  with pytest.raises(ValueError, match="batch"):
    _build_cp_joint_mask(
      _FakeProcessor(2),
      torch.ones(2, 8, dtype=torch.bool),
      batch_size=1,
      local_txt=4,
      local_img=2,
      device=torch.device("cpu"),
    )


def test_planner_rejects_ring_and_ulysses_anything():
  planner = PrxContextParallelismPlanner()
  transformer = _tiny_transformer()

  with pytest.raises(NotImplementedError, match="Ring"):
    planner._apply(
      transformer=transformer,
      parallelism_config=ParallelismConfig(ring_size=2),
    )

  with pytest.raises(NotImplementedError, match="ulysses-anything"):
    planner._apply(
      transformer=transformer,
      parallelism_config=ParallelismConfig(ulysses_size=2, ulysses_anything=True),
    )
