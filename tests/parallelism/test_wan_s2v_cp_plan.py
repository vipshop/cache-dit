"""CPU checks for the Wan2.2-S2V CP plan and TP layer plan.

Both plans are string contracts: a wrong split point or a missing shard still produces tensors of
the right shape, so the split/gather markers and the layer plans are asserted directly.

Two S2V-specific points:

- The RoPE table has to be split at the first block, not on the output of `rope`: `inject_motion`
  concatenates the motion-token rotary embedding after `rope` returns, so an output split would be
  re-materialized at full length.
- The audio injector is a stack of cross-attention layers outside `blocks`, so the Wan TP plan
  would leave it replicated.

Usage:
  python -m pytest tests/parallelism/test_wan_s2v_cp_plan.py -q
"""

import inspect

import pytest
import torch

from cache_dit.distributed.config import ParallelismConfig
from cache_dit.distributed.transformers.planners import _activate_cp_planners, _activate_tp_planners
from cache_dit.distributed.transformers.register import (
  ContextParallelismPlannerRegister,
  TensorParallelismPlannerRegister,
)
from cache_dit.distributed.transformers.wan_s2v import (
  WanS2VContextParallelismPlanner,
  WanS2VTensorParallelismPlanner,
)
from cache_dit.models.pipelines.wan_s2v.transformer import WanS2VTransformer3DModel

_NUM_BLOCKS = 4
_NUM_INJECT_LAYERS = 2
_HEADS = 4


def _tiny_transformer() -> WanS2VTransformer3DModel:
  torch.manual_seed(0)
  return WanS2VTransformer3DModel(
    patch_size=(1, 2, 2),
    num_attention_heads=_HEADS,
    attention_head_dim=16,
    in_channels=16,
    out_channels=16,
    text_dim=32,
    freq_dim=32,
    audio_dim=1024,
    audio_inject_layers=(0, 2),
    enable_adain=True,
    adain_mode="attn_norm",
    pose_dim=16,
    ffn_dim=128,
    num_layers=_NUM_BLOCKS,
    num_weighted_avg_layers=25,
    cross_attn_norm=True,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
    added_kv_proj_dim=None,
    rope_max_seq_len=1024,
    enable_framepack=True,
    framepack_drop_mode="padd",
    add_last_motion=True,
    zero_timestep=True,
  ).eval()


def test_cp_plan_splits_sequence_and_rope_at_the_first_block():
  transformer = _tiny_transformer()
  plan = WanS2VContextParallelismPlanner()._apply(
    transformer=transformer,
    parallelism_config=ParallelismConfig(ulysses_size=2),
  )

  assert set(plan) == {"blocks.0", "proj_out"}

  hidden_plan = plan["blocks.0"]["hidden_states"]
  assert hidden_plan.split_output is False
  assert hidden_plan.split_dim == 1
  assert hidden_plan.expected_dims == 3

  gather_plan = plan["proj_out"]
  assert gather_plan.gather_dim == 1
  assert gather_plan.expected_dims == 3

  # The plan names real submodules and real block parameters.
  assert transformer.get_submodule("blocks.0") is transformer.blocks[0]
  assert isinstance(transformer.get_submodule("proj_out"), torch.nn.Linear)
  parameters = list(inspect.signature(transformer.blocks[0].forward).parameters)
  assert parameters[:5] == [
    "hidden_states", "encoder_hidden_states", "temb", "rotary_emb", "audio_ctx"
  ]


def test_rope_is_sharded_in_the_forward_not_by_a_hook():
  """One rotary table feeds every block, so a `blocks.0` hook would leave blocks 1..N-1
  unsharded."""
  source = inspect.getsource(WanS2VTransformer3DModel.forward)
  assert "rotary_emb = rotary_emb.chunk(cp_size, dim=1)[cp_rank]" in source


def test_cp_plan_rejects_ulysses_anything():
  with pytest.raises(NotImplementedError, match="ulysses-anything"):
    WanS2VContextParallelismPlanner()._apply(
      transformer=_tiny_transformer(),
      parallelism_config=ParallelismConfig(ulysses_size=2, ulysses_anything=True),
    )


class _FakeGroup:
  """Stand-in for a process group: the RMS-norm swap reads its size and the local rank."""

  def __init__(self, size: int):
    self._size = size

  def size(self) -> int:
    return self._size

  def rank(self) -> int:
    return 0


class _FakeMesh:
  """Minimal device-mesh stand-in: the planner only reads its size and group before sharding."""

  def __init__(self, size: int):
    self._size = size

  def size(self) -> int:
    return self._size

  def get_group(self) -> _FakeGroup:
    return _FakeGroup(self._size)


def test_tp_layer_plan_covers_blocks_and_audio_injector(monkeypatch):
  import cache_dit.distributed.transformers.wan_s2v as planner_module

  captured = []
  monkeypatch.setattr(
    planner_module,
    "parallelize_module",
    lambda module, device_mesh, parallelize_plan: captured.append(parallelize_plan),
  )

  transformer = _tiny_transformer()
  planner = WanS2VTensorParallelismPlanner()
  _, layer_plans = planner.parallelize_transformer(transformer, _FakeMesh(2))

  # One plan per block plus one per audio-injection layer.
  assert len(layer_plans) == _NUM_BLOCKS + _NUM_INJECT_LAYERS
  assert len(captured) == _NUM_BLOCKS + _NUM_INJECT_LAYERS

  block_plan = layer_plans[0]
  assert {
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "ffn.net.0.proj",
    "ffn.net.2",
  } <= set(block_plan)

  injector_plan = layer_plans[_NUM_BLOCKS]
  assert set(injector_plan) == {
    "audio_injector.injector.0.to_q",
    "audio_injector.injector.0.to_k",
    "audio_injector.injector.0.to_v",
    "audio_injector.injector.0.to_out.0",
  }
  # Head counts follow the local shard, both in the blocks and in the injector.
  assert transformer.blocks[0].attn1.heads == _HEADS // 2
  assert transformer.audio_injector.injector[0].heads == _HEADS // 2


def test_tp_rejects_an_indivisible_head_count():
  transformer = _tiny_transformer()
  with pytest.raises(ValueError, match="divisible by tp_size"):
    WanS2VTensorParallelismPlanner().parallelize_transformer(transformer, _FakeMesh(3))


def test_planners_are_registered_under_the_model_name():
  _activate_cp_planners()
  _activate_tp_planners()

  model_name = WanS2VTransformer3DModel.__name__
  assert ContextParallelismPlannerRegister.get_planner(
    model_name) is WanS2VContextParallelismPlanner
  assert TensorParallelismPlannerRegister.get_planner(model_name) is WanS2VTensorParallelismPlanner
  # The Wan planners must keep resolving for the Wan family itself.
  assert ContextParallelismPlannerRegister.get_planner(
    "WanTransformer3DModel").__name__ == "WanContextParallelismPlanner"
