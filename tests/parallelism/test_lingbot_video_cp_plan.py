"""CPU checks for the LingBot-Video CP plan and the TP layer plan.

LingBot-Video runs one joint `[video, text]` sequence per sample, so the split has to sit on the
first denoising block (cache-dit replays that hook to shard the Fn residual), the token-wise RoPE and
modulation tensors need their own output-split markers because every block consumes them, and the
projected velocity must be gathered before the forward slices out the video tokens. All of these are
string contracts a wrong plan still renders a plausible clip from, so they are asserted directly.

Usage:
  python -m pytest tests/parallelism/test_lingbot_video_cp_plan.py -q
"""

import inspect

import pytest
import torch

from cache_dit.distributed.config import ParallelismConfig
from cache_dit.distributed.transformers.lingbot_video import (
  LingBotVideoContextParallelismPlanner,
  LingBotVideoTensorParallelismPlanner,
)
from cache_dit.distributed.transformers.planners import _activate_cp_planners, _activate_tp_planners
from cache_dit.distributed.transformers.register import (
  ContextParallelismPlannerRegister,
  TensorParallelismPlannerRegister,
)
from cache_dit.models.pipelines.lingbot_video import LingBotVideoTransformer3DModel

_NUM_BLOCKS = 2
_HEADS = 4


def _tiny_transformer() -> LingBotVideoTransformer3DModel:
  torch.manual_seed(0)
  return LingBotVideoTransformer3DModel(
    patch_size=(1, 2, 2),
    in_channels=16,
    out_channels=16,
    hidden_size=64,
    num_attention_heads=_HEADS,
    depth=_NUM_BLOCKS,
    intermediate_size=128,
    text_dim=32,
    freq_dim=16,
    axes_dims=(4, 6, 6),
    axes_lens=(16, 8, 8),
  ).eval()


def _ulysses_config(**kwargs) -> ParallelismConfig:
  return ParallelismConfig(ulysses_size=2, **kwargs)


def test_cp_plan_splits_the_joint_sequence_at_the_first_block():
  transformer = _tiny_transformer()
  plan = LingBotVideoContextParallelismPlanner()._apply(
    transformer=transformer,
    parallelism_config=_ulysses_config(),
  )

  assert set(plan) == {"cp_rotary", "cp_temb_input", "cp_temb6", "blocks.0", "cp_out"}

  # Output-split markers: the key is the output index, not a parameter name.
  for name in ("cp_rotary", "cp_temb_input", "cp_temb6"):
    marker_plan = plan[name][0]
    assert marker_plan.split_output is True
    assert marker_plan.split_dim == 1
    assert marker_plan.expected_dims == 3

  hidden_plan = plan["blocks.0"]["hidden_states"]
  assert hidden_plan.split_output is False
  assert hidden_plan.split_dim == 1
  assert hidden_plan.expected_dims == 3

  gather_plan = plan["cp_out"]
  assert gather_plan.gather_dim == 1
  assert gather_plan.expected_dims == 3

  # The plan keys must resolve on the real module tree.
  for name in ("cp_rotary", "cp_temb_input", "cp_temb6", "cp_out"):
    assert isinstance(transformer.get_submodule(name), torch.nn.Identity)
  assert transformer.get_submodule("blocks.0") is transformer.blocks[0]

  # The split targets `hidden_states` by name, so the block's first parameter has to carry it.
  parameters = list(inspect.signature(transformer.blocks[0].forward).parameters)
  assert parameters[:3] == ["hidden_states", "temb6", "rotary_emb"]


def test_cp_plan_rejects_ring():
  with pytest.raises(NotImplementedError, match="Ring"):
    LingBotVideoContextParallelismPlanner()._apply(
      transformer=_tiny_transformer(),
      parallelism_config=ParallelismConfig(ring_size=2),
    )


class _FakeMesh:
  """Minimal stand-in for a device mesh: the planner only reads its size before sharding."""

  def __init__(self, size: int):
    self._size = size

  def size(self) -> int:
    return self._size


def test_tp_layer_plan_covers_attention_and_ffn(monkeypatch):
  import cache_dit.distributed.transformers.lingbot_video as planner_module

  captured = []
  monkeypatch.setattr(
    planner_module,
    "parallelize_module",
    lambda module, device_mesh, parallelize_plan: captured.append(parallelize_plan),
  )

  transformer = _tiny_transformer()
  planner = LingBotVideoTensorParallelismPlanner()
  _, layer_plans = planner.parallelize_transformer(transformer, _FakeMesh(2))

  assert len(layer_plans) == _NUM_BLOCKS
  assert set(layer_plans[0]) == {
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out",
    "ffn.gate_proj",
    "ffn.up_proj",
    "ffn.down_proj",
  }
  # Every block is sharded, and the head count follows the local shard.
  assert len(captured) == _NUM_BLOCKS
  assert transformer.blocks[0].attn.num_heads == _HEADS // 2
  # The modulation table must stay replicated: the block chunks it into six parts.
  assert transformer.blocks[0].scale_shift_table.shape == (1, 6 * 64)


def test_tp_rejects_an_indivisible_ffn_width():
  transformer = _tiny_transformer()
  # intermediate_size=128 is divisible by 2 and 4 but not by 3.
  with pytest.raises(ValueError, match="divisible by tp_size"):
    LingBotVideoTensorParallelismPlanner().parallelize_transformer(transformer, _FakeMesh(3))


def test_tp_heads_sharding_follows_the_local_block():
  from cache_dit.distributed.utils import shard_div_attr

  block = _tiny_transformer().blocks[0]
  assert shard_div_attr(block.attn, "num_heads", 2) == _HEADS // 2
  with pytest.raises(ValueError, match="divisible by tp_size"):
    shard_div_attr(block.attn, "num_heads", 3)


def test_planners_are_registered_under_the_model_names():
  _activate_cp_planners()
  _activate_tp_planners()

  # Look the planners up by the real transformer class name: the registry resolves a prefix, so this
  # also covers the case of a class rename breaking the registration.
  assert ContextParallelismPlannerRegister.get_planner(
    LingBotVideoTransformer3DModel.__name__) is LingBotVideoContextParallelismPlanner
  assert TensorParallelismPlannerRegister.get_planner(
    LingBotVideoTransformer3DModel.__name__) is LingBotVideoTensorParallelismPlanner
