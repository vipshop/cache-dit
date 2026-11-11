"""CPU tests for the LongCat-Video-Avatar CP/TP plans and the CLI registration."""

import types

import pytest
import torch

from cache_dit.cli.registers import ExampleRegister
from cache_dit.distributed.config import ParallelismConfig
from cache_dit.distributed.transformers.longcat_video_avatar import (
  LongCatVideoAvatarContextParallelismPlanner, )
from cache_dit.models.pipelines.longcat_video_avatar.transformer import frame_range


def _config(**overrides):
  params = dict(ulysses_size=2, tp_size=None, ring_size=None)
  params.update(overrides)
  return ParallelismConfig(**params)


def _transformer(depth=2):
  return types.SimpleNamespace(
    config=types.SimpleNamespace(depth=depth),
    blocks=torch.nn.ModuleList([torch.nn.Identity() for _ in range(depth)]),
  )


def test_cp_plan_is_a_single_point_split_with_a_late_gather():
  planner = LongCatVideoAvatarContextParallelismPlanner()
  plan = planner._apply(transformer=_transformer(), parallelism_config=_config())

  assert set(plan) == {"blocks.0", "final_layer"}
  split = plan["blocks.0"]["hidden_states"]
  assert split.split_dim == 1
  assert split.expected_dims == 3
  # The gather happens on the 64-wide patch features, not on the 4096-wide residual stream.
  assert plan["final_layer"].gather_dim == 1


def test_cp_rejects_ring_and_uneven_ulysses():
  planner = LongCatVideoAvatarContextParallelismPlanner()

  with pytest.raises(NotImplementedError, match="Ring"):
    planner._apply(transformer=_transformer(), parallelism_config=_config(ring_size=2))

  with pytest.raises(NotImplementedError, match="ulysses-anything"):
    planner._apply(
      transformer=_transformer(),
      parallelism_config=_config(ulysses_anything=True),
    )


class _FakeMesh:
  """Minimal stand-in for the CP `DeviceMesh` (`size` / `get_local_rank` are all we read)."""

  def __init__(self, size, rank):
    self._size = size
    self._rank = rank

  def size(self):
    return self._size

  def get_local_rank(self):
    return self._rank


def _cp_config(mesh_size, mesh_rank, world=None, rank=0):
  """Build a CP config the way cache-dit does: global rank/size plus a CP-scoped mesh."""
  return types.SimpleNamespace(
    _rank=rank,
    _world_size=world if world is not None else mesh_size,
    _flattened_mesh=_FakeMesh(mesh_size, mesh_rank) if mesh_size > 1 else None,
  )


def test_frame_range_requires_whole_frames():
  assert frame_range(None, 12) == (0, 12)
  assert frame_range(_cp_config(1, 0), 12) == (0, 12)
  assert frame_range(_cp_config(2, 1), 12) == (6, 12)
  assert frame_range(_cp_config(2, 0), 12) == (0, 6)

  with pytest.raises(ValueError, match="divisible by the context-parallel size"):
    frame_range(_cp_config(2, 1), 9)


def test_frame_range_splits_by_context_parallelism_only():
  """`ulysses_tp` must split the frames by the CP mesh, not by the global world size.

  `_rank` / `_world_size` are the global values (world 4 for CP2xTP2), so reading them would cut
  the frame axis into quarters no rank owns while `blocks.0` cuts it in half.
  """
  assert frame_range(_cp_config(2, 0, world=4, rank=0), 12) == (0, 6)
  assert frame_range(_cp_config(2, 1, world=4, rank=3), 12) == (6, 12)


def test_cp_plan_uses_the_cp_mesh_for_a_hybrid_run():
  """The plan is built from the CP degree only, so CP2xTP2 still splits once in half.

  A real hybrid `ParallelismConfig` builds a device mesh, which needs an initialized process group,
  so the CP knobs are handed over as a plain namespace here.
  """
  planner = LongCatVideoAvatarContextParallelismPlanner()
  hybrid = types.SimpleNamespace(
    ulysses_size=2,
    ring_size=None,
    tp_size=2,
    ulysses_anything=False,
    ulysses_float8=False,
    ulysses_async=False,
  )
  plan = planner._apply(transformer=_transformer(depth=3), parallelism_config=hybrid)

  assert set(plan) == {"blocks.0", "final_layer"}
  assert plan["blocks.0"]["hidden_states"].split_dim == 1
  assert plan["final_layer"].gather_dim == 1


def test_cli_examples_are_registered():
  names = ExampleRegister.list_examples()
  for name in ("longcat_video_avatar_a2v", "longcat_video_avatar_ai2v"):
    assert name in names, f"{name} is not registered"


def test_cli_env_path_mapping():
  from cache_dit.cli.examples import _env_path_mapping

  assert _env_path_mapping["LONGCAT_VIDEO_AVATAR_DIR"] == "meituan-longcat/LongCat-Video-Avatar-1.5"
  # The avatar checkpoint leans on the base release for the prompt encoder / VAE / tokenizer.
  assert _env_path_mapping["LONGCAT_VIDEO_DIR"] == "meituan-longcat/LongCat-Video"
