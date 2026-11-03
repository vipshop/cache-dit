"""CPU tests for the LongCat-Video-Avatar CP/TP plans and the CLI registration."""

import types

import pytest
import torch

from cache_dit._utils.registers import ExampleRegister
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


def test_frame_range_requires_whole_frames():
  cp_config = types.SimpleNamespace(_cp_rank=1, _cp_world_size=2)

  assert frame_range(None, 12) == (0, 12)
  assert frame_range(types.SimpleNamespace(_cp_rank=0, _cp_world_size=1), 12) == (0, 12)
  assert frame_range(cp_config, 12) == (6, 12)
  assert frame_range(types.SimpleNamespace(_cp_rank=0, _cp_world_size=2), 12) == (0, 6)

  with pytest.raises(ValueError, match="divisible by the context-parallel size"):
    frame_range(cp_config, 9)


def test_frame_range_splits_by_context_parallelism_only():
  """`ulysses_tp` must split the frames by the CP mesh, not by the global world size."""
  cp2_tp2 = types.SimpleNamespace(_rank=2, _world_size=4, _cp_rank=0, _cp_world_size=2)

  assert frame_range(cp2_tp2, 12) == (0, 6)
  assert frame_range(types.SimpleNamespace(_rank=3, _world_size=4, _cp_rank=1, _cp_world_size=2),
                     12) == (6, 12)


def test_cli_examples_are_registered():
  names = ExampleRegister.list_examples()
  for name in ("longcat_video_avatar_a2v", "longcat_video_avatar_ai2v"):
    assert name in names, f"{name} is not registered"


def test_cli_env_path_mapping():
  from cache_dit._utils.examples import _env_path_mapping

  assert _env_path_mapping["LONGCAT_VIDEO_AVATAR_DIR"] == "meituan-longcat/LongCat-Video-Avatar-1.5"
  # The avatar checkpoint leans on the base release for the prompt encoder / VAE / tokenizer.
  assert _env_path_mapping["LONGCAT_VIDEO_DIR"] == "meituan-longcat/LongCat-Video"
