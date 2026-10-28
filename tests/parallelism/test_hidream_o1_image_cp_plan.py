"""CPU checks for the HiDream-O1-Image CP plan and the TP layer plan.

HiDream-O1-Image keeps text, the timestep token and the image patches in one joint sequence, so the
split has to sit on the first denoising block (cache-dit replays that hook to shard the Fn residual),
the packed mRoPE table needs its own output-split marker because every block shares it, and the
joint sequence must be gathered before `norm` / `final_layer2` see it. All three are string
contracts a wrong plan still renders a plausible image from, so they are asserted directly.

Usage:
  python -m pytest tests/parallelism/test_hidream_o1_image_cp_plan.py -q
"""

import inspect

import pytest
import torch
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

from cache_dit.distributed.config import ParallelismConfig
from cache_dit.distributed.transformers.hidream_o1_image import (
  HiDreamO1ImageContextParallelismPlanner,
  HiDreamO1ImageTensorParallelismPlanner,
)
from cache_dit.distributed.transformers.planners import _activate_cp_planners, _activate_tp_planners
from cache_dit.distributed.transformers.register import (
  ContextParallelismPlannerRegister,
  TensorParallelismPlannerRegister,
)
from cache_dit.models.pipelines.hidream_o1_image import HiDreamO1ImageTransformer2DModel

_NUM_LAYERS = 2
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 16


def _tiny_transformer() -> HiDreamO1ImageTransformer2DModel:
  torch.manual_seed(0)
  return HiDreamO1ImageTransformer2DModel(
    hidden_size=64,
    intermediate_size=128,
    num_layers=_NUM_LAYERS,
    num_attention_heads=_HEADS,
    num_key_value_heads=_KV_HEADS,
    head_dim=_HEAD_DIM,
    vocab_size=128,
    patch_size=32,
    in_channels=3,
  ).eval()


def _ulysses_config(**kwargs) -> ParallelismConfig:
  return ParallelismConfig(ulysses_size=2, **kwargs)


def test_cp_plan_splits_the_joint_sequence_at_the_first_block():
  transformer = _tiny_transformer()
  plan = HiDreamO1ImageContextParallelismPlanner()._apply(
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
  assert hidden_plan.split_dim == 1
  assert hidden_plan.expected_dims == 3
  assert hidden_plan.split_output is False

  gather = plan["sequence_gather"]
  assert gather.gather_dim == 1
  assert gather.expected_dims == 3


def test_cp_plan_never_touches_the_attention_mask():
  # The dense causal/full mask is indexed by absolute position and Ulysses keeps the full key
  # sequence, so any split/permute hook on it would corrupt the output.
  plan = HiDreamO1ImageContextParallelismPlanner()._apply(
    transformer=_tiny_transformer(),
    parallelism_config=_ulysses_config(),
  )
  assert "attention_mask" not in plan["layers.0"]


def test_cp_plan_rejects_ring():
  transformer = _tiny_transformer()
  with pytest.raises(NotImplementedError, match="Ring"):
    HiDreamO1ImageContextParallelismPlanner()._apply(
      transformer=transformer,
      parallelism_config=ParallelismConfig(ring_size=2),
    )


def test_cp_plan_accepts_unset_ring_size():
  # `ring_size` stays None for a plain Ulysses config; a shape-only check would raise a TypeError.
  plan = HiDreamO1ImageContextParallelismPlanner()._apply(
    transformer=_tiny_transformer(),
    parallelism_config=_ulysses_config(ulysses_anything=True),
  )
  assert "layers.0" in plan


def test_both_planners_are_registered_for_the_transformer_class():
  _activate_cp_planners()
  _activate_tp_planners()
  name = "HiDreamO1ImageTransformer2DModel"
  assert ContextParallelismPlannerRegister.get_planner(
    name) is HiDreamO1ImageContextParallelismPlanner
  assert TensorParallelismPlannerRegister.get_planner(
    name) is HiDreamO1ImageTensorParallelismPlanner


def test_tp_layer_plan_shards_attention_and_ffn():
  plan = HiDreamO1ImageTensorParallelismPlanner.layer_plan()
  assert set(plan) == {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
  }
  # Q/K/V and gate/up feed a Rowwise partner, so they are column-sharded; the two output
  # projections are row-sharded and all-reduce.
  assert isinstance(plan["self_attn.q_proj"], ColwiseParallel)
  assert isinstance(plan["self_attn.o_proj"], RowwiseParallel)
  assert isinstance(plan["mlp.up_proj"], ColwiseParallel)
  assert isinstance(plan["mlp.down_proj"], RowwiseParallel)


def test_tp_planner_shards_head_counts_but_not_head_dim(monkeypatch):
  """Drive the real planner, not `shard_div_attr` directly: a dropped call must fail here."""
  from cache_dit.distributed.transformers import hidream_o1_image as planner_module

  class _FakeMesh:

    def size(self) -> int:
      return 2

  # The weight sharding itself needs a real process group; only the metadata side is under test.
  monkeypatch.setattr(planner_module, "parallelize_module", lambda **kwargs: None)

  transformer = _tiny_transformer()
  planner = HiDreamO1ImageTensorParallelismPlanner()
  transformer, layer_plans = planner.parallelize_transformer(transformer, _FakeMesh())

  assert len(layer_plans) == _NUM_LAYERS
  for block in transformer.layers:
    attention = block.self_attn
    assert attention.num_heads == _HEADS // 2
    assert attention.num_kv_heads == _KV_HEADS // 2
    # `head_dim` never crosses a shard boundary: the per-head norms operate on it.
    assert attention.head_dim == _HEAD_DIM
    # The GQA widening ratio must survive the division.
    assert attention.num_heads // attention.num_kv_heads == _HEADS // _KV_HEADS


def test_attention_and_block_signatures_keep_the_cache_contract():
  transformer = _tiny_transformer()
  block = transformer.layers[0]
  parameters = list(inspect.signature(block.forward).parameters)
  assert parameters[0] == "hidden_states"
  # The block returns a single tensor (Pattern_3) and carries rope/mask/cp_config as parameters.
  assert set(parameters[1:]) == {"rope", "attention_mask", "cp_config"}
  assert transformer._repeated_blocks == [type(block).__name__]
  assert transformer.config.num_layers == _NUM_LAYERS
