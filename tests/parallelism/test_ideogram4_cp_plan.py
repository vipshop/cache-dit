"""CPU checks for the Ideogram-4 CP plan and the TP layer plan.

Ideogram-4 packs text and image tokens into one sequence, so context parallelism can split
at the transformer entry instead of patching the forward. The plan therefore only has to
shard the per-token root arguments and gather on `final_layer`. `segment_ids` must stay
whole: the `(B, 1, L, L)` block-diagonal mask is built from it inside the forward and has
to describe the global key axis. Both properties are asserted here directly, because a
wrong split (or a mistakenly sharded `segment_ids`) still renders a plausible image.

The attention fork is only allowed to differ from upstream in the dispatch call, so the
patched `__call__` must stay bit-exact while no `_cp_config` is attached.

Usage:
  python -m pytest tests/parallelism/test_ideogram4_cp_plan.py -q
"""

import inspect

import pytest
import torch

try:
  from diffusers.models.transformers.transformer_ideogram4 import (
    Ideogram4AttnProcessor,
    Ideogram4Transformer2DModel,
  )
except ImportError:
  pytest.skip("diffusers without Ideogram-4 support", allow_module_level=True)

import cache_dit.distributed.transformers.ideogram4 as ideogram4
from cache_dit.distributed.config import ParallelismConfig
from cache_dit.distributed.transformers.ideogram4 import (
  Ideogram4ContextParallelismPlanner,
  Ideogram4TensorParallelismPlanner,
)
from cache_dit.distributed.transformers.register import (
  ContextParallelismPlannerRegister,
  TensorParallelismPlannerRegister,
)

_IN_CHANNELS = 8
_LLM_FEATURES = 16
_HEADS = 2
_HEAD_DIM = 16
_HIDDEN = _HEADS * _HEAD_DIM
_LAYERS = 2


def _tiny_transformer() -> Ideogram4Transformer2DModel:
  torch.manual_seed(0)
  return Ideogram4Transformer2DModel(
    in_channels=_IN_CHANNELS,
    num_layers=_LAYERS,
    attention_head_dim=_HEAD_DIM,
    num_attention_heads=_HEADS,
    intermediate_size=2 * _HEAD_DIM * _HEADS,
    adaln_dim=8,
    llm_features_dim=_LLM_FEATURES,
    # `mrope_section[i] * 3` must stay within `head_dim // 2` frequencies of the tiny head.
    mrope_section=(2, 2, 2),
  ).eval()


def _cp_config() -> ParallelismConfig:
  return ParallelismConfig(ulysses_size=2)


def test_cp_plan_splits_only_the_packed_sequence_inputs():
  plan = Ideogram4ContextParallelismPlanner()._apply(
    transformer=_tiny_transformer(),
    parallelism_config=_cp_config(),
  )

  root = plan[""]
  assert set(root) == {"hidden_states", "encoder_hidden_states", "position_ids", "indicator"}
  for name in ("hidden_states", "encoder_hidden_states", "position_ids"):
    assert root[name].split_dim == 1, name
    assert root[name].expected_dims == 3, name
    assert not root[name].split_output, name
  assert root["indicator"].split_dim == 1
  assert root["indicator"].expected_dims == 2
  assert not root["indicator"].split_output

  # The segment mask is built from the full `segment_ids`, so it must never be sharded.
  assert "segment_ids" not in root
  assert set(plan) == {"", "final_layer"}

  assert plan["final_layer"].gather_dim == 1
  assert plan["final_layer"].expected_dims == 3


def test_cp_plan_keys_resolve_on_the_real_model():
  # A renamed upstream argument would only produce a `warning_once` and a silently skipped
  # split at runtime, so the names are checked against the actual signature and module tree.
  transformer = _tiny_transformer()
  params = set(inspect.signature(transformer.forward).parameters)
  plan = Ideogram4ContextParallelismPlanner()._apply(
    transformer=transformer,
    parallelism_config=_cp_config(),
  )

  for name in plan[""]:
    assert name in params, f"CP plan input '{name}' is not a forward parameter"
  assert hasattr(transformer, "final_layer")


def test_planners_are_registered_under_the_exact_class_name():
  cp_planner = ContextParallelismPlannerRegister.get_planner("Ideogram4Transformer2DModel")
  assert cp_planner is Ideogram4ContextParallelismPlanner
  tp_planner = TensorParallelismPlannerRegister.get_planner("Ideogram4Transformer2DModel")
  assert tp_planner is Ideogram4TensorParallelismPlanner


def test_cp_planner_rejects_ring_and_ulysses_anything():
  planner = Ideogram4ContextParallelismPlanner()
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

  with pytest.raises(NotImplementedError, match="mask-capable native"):
    planner._apply(
      transformer=transformer,
      parallelism_config=ParallelismConfig(ulysses_size=2, attention_backend="flash"),
    )


class _FakeTpMesh:

  def __init__(self, size: int) -> None:
    self._size = size

  def size(self) -> int:
    return self._size


def test_tp_plan_shards_attention_and_swiglu(monkeypatch):
  transformer = _tiny_transformer()
  recorded = []

  def fake_parallelize_module(module, device_mesh, parallelize_plan):
    assert device_mesh.size() == 2
    recorded.append(parallelize_plan)
    return module

  monkeypatch.setattr(ideogram4, "parallelize_module", fake_parallelize_module)

  planner = Ideogram4TensorParallelismPlanner()
  _, layer_plans = planner.parallelize_transformer(
    transformer=transformer,
    tp_mesh=_FakeTpMesh(2),
  )

  assert len(layer_plans) == _LAYERS == len(recorded)
  for block, layer_plan in zip(transformer.layers, layer_plans):
    # `shard_div_attr` runs for real: the processor keeps `head_dim` and reads `num_heads`.
    assert block.attention.num_heads == _HEADS // 2
    assert block.attention.head_dim == _HEAD_DIM
    assert set(layer_plan) == {
      "attention.to_q",
      "attention.to_k",
      "attention.to_v",
      "attention.to_out.0",
      "feed_forward.w1",
      "feed_forward.w3",
      "feed_forward.w2",
    }
    for name in ("attention.to_q", "attention.to_k", "attention.to_v", "feed_forward.w1",
                 "feed_forward.w3"):
      assert layer_plan[name].__class__.__name__ == "ColwiseParallel", name
    for name in ("attention.to_out.0", "feed_forward.w2"):
      assert layer_plan[name].__class__.__name__ == "RowwiseParallel", name
    # `chunk(4, dim=-1)` boundaries must survive, and the entry/exit paths stay replicated.
    assert "adaln_modulation" not in layer_plan


def test_attention_fork_is_bit_exact_without_cp_config():
  # The fork replaces upstream `__call__` at class level; with no `_cp_config` armed it must
  # reproduce the upstream numbers exactly, so an upstream change (RoPE layout, mask
  # handling, ...) surfaces here instead of corrupting every CP run.
  ideogram4._patch_attention_processor_for_cp()
  current = Ideogram4AttnProcessor.__call__
  upstream = current.__wrapped__

  torch.manual_seed(7)
  transformer = _tiny_transformer()
  attn = transformer.layers[0].attention
  processor = attn.processor
  assert getattr(processor, "_cp_config", None) is None

  hidden_states = torch.randn(1, 6, _HIDDEN)
  cos = torch.randn(1, 6, _HEAD_DIM)
  sin = torch.randn(1, 6, _HEAD_DIM)
  attention_mask = torch.ones(1, 1, 6, 6, dtype=torch.bool)

  with torch.no_grad():
    ref = upstream(processor, attn, hidden_states, attention_mask, (cos, sin))
    got = current(processor, attn, hidden_states, attention_mask, (cos, sin))

  assert torch.equal(ref, got)
