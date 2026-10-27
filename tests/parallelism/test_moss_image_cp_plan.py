"""CPU checks for the MOSS-Image CP plan and the TP layer plan.

MOSS-Image runs one joint `[image, text]` sequence per sample, so the split has to sit on the first
denoising block (cache-dit replays that hook to shard the Fn residual), the RoPE table needs its own
output-split marker because every block shares it, and the joint sequence must be gathered before
the forward slices out the image tokens. All three are string contracts a wrong plan still renders
a plausible image from, so they are asserted directly.

Usage:
  python -m pytest tests/parallelism/test_moss_image_cp_plan.py -q
"""

import inspect

import pytest
import torch

from cache_dit.distributed.config import ParallelismConfig
from cache_dit.distributed.text_encoders.planners import _activate_text_encoder_tp_planners
from cache_dit.distributed.text_encoders.qwen3_5 import Qwen3_5TensorParallelismPlanner
from cache_dit.distributed.text_encoders.register import (
  TextEncoderTensorParallelismPlannerRegister, )
from cache_dit.distributed.transformers.moss_image import (
  MOSSImageContextParallelismPlanner,
  MOSSImageTensorParallelismPlanner,
)
from cache_dit.distributed.transformers.planners import _activate_cp_planners, _activate_tp_planners
from cache_dit.distributed.transformers.register import (
  ContextParallelismPlannerRegister,
  TensorParallelismPlannerRegister,
)
from cache_dit.models.pipelines.moss_image import MOSSImageTransformer

_NUM_LAYERS = 2
_HEADS = 4


def _tiny_transformer() -> MOSSImageTransformer:
  torch.manual_seed(0)
  return MOSSImageTransformer(
    dim=32,
    ffn_dim=64,
    num_heads=_HEADS,
    num_layers=_NUM_LAYERS,
    text_dim=16,
    freq_dim=16,
    latent_channels=8,
  ).eval()


def _ulysses_config(**kwargs) -> ParallelismConfig:
  return ParallelismConfig(ulysses_size=2, **kwargs)


def test_cp_plan_splits_the_joint_sequence_at_the_first_block():
  transformer = _tiny_transformer()
  plan = MOSSImageContextParallelismPlanner()._apply(
    transformer=transformer,
    parallelism_config=_ulysses_config(),
  )

  assert set(plan) == {"rope_split", "layers.0", "sequence_gather"}

  # Output-split marker: the key is the output index, not a parameter name.
  rope_plan = plan["rope_split"][0]
  assert rope_plan.split_output is True
  assert rope_plan.split_dim == 1
  assert rope_plan.expected_dims == 3

  hidden_plan = plan["layers.0"]["hidden_states"]
  assert hidden_plan.split_output is False
  assert hidden_plan.split_dim == 1
  assert hidden_plan.expected_dims == 3

  gather_plan = plan["sequence_gather"]
  assert gather_plan.gather_dim == 1
  assert gather_plan.expected_dims == 3

  # The plan keys must resolve on the real module tree.
  assert isinstance(transformer.get_submodule("rope_split"), torch.nn.Identity)
  assert isinstance(transformer.get_submodule("sequence_gather"), torch.nn.Identity)
  assert transformer.get_submodule("layers.0") is transformer.layers[0]

  # The split targets `hidden_states` by name, so the block's first parameter has to carry it.
  parameters = list(inspect.signature(transformer.layers[0].forward).parameters)
  assert parameters[:4] == ["hidden_states", "rope", "modulation", "skip_layers"]


def test_cp_plan_rejects_ring():
  with pytest.raises(NotImplementedError, match="Ring"):
    MOSSImageContextParallelismPlanner()._apply(
      transformer=_tiny_transformer(),
      parallelism_config=ParallelismConfig(ring_size=2),
    )


class _FakeMesh:
  """Minimal stand-in for a device mesh: the planner only reads its size before sharding."""

  def __init__(self, size: int):
    self._size = size

  def size(self) -> int:
    return self._size


def test_tp_plan_rejects_a_partial_ffn_group():
  """`2 * ffn_dim` not divisible by `2 * tp_size` would leave zero rows in the rearranged weight."""
  with pytest.raises(ValueError, match="divisible by 2 \\* tp_size"):
    MOSSImageTensorParallelismPlanner().parallelize_transformer(_tiny_transformer(), _FakeMesh(3))


def test_tp_heads_sharding_follows_the_local_block():
  from cache_dit.distributed.utils import shard_div_attr

  block = _tiny_transformer().layers[0]
  assert shard_div_attr(block, "heads", 2) == _HEADS // 2
  with pytest.raises(ValueError, match="divisible by tp_size"):
    shard_div_attr(block, "heads", 3)


def test_planners_are_registered_under_the_model_names():
  _activate_cp_planners()
  _activate_tp_planners()
  _activate_text_encoder_tp_planners()

  assert ContextParallelismPlannerRegister.get_planner(
    "MOSSImageTransformer") is MOSSImageContextParallelismPlanner
  assert TensorParallelismPlannerRegister.get_planner(
    "MOSSImageTransformer") is MOSSImageTensorParallelismPlanner
  # The new Qwen3.5 planner must not shadow the plain Qwen3 one (or the other way round).
  assert TextEncoderTensorParallelismPlannerRegister.get_planner(
    "Qwen3_5Model") is Qwen3_5TensorParallelismPlanner
  assert TextEncoderTensorParallelismPlannerRegister.get_planner(
    "Qwen3Model").__name__ == "Qwen3TensorParallelismPlanner"
