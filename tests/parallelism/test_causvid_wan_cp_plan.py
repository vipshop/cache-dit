"""CPU checks for the CausVid-Wan CP plan and the TP layer plan.

CausVid's chunk-causal attention keeps a KV cache addressed by global token coordinates, so the CP
plan only splits tensors whose partition is rank-local: the RoPE tables of the current chunk, the
hidden states at the first block, and the gather after `proj_out`. The Ulysses exchange itself lives
inside the attention processor, which means a wrong plan still renders a plausible clip — hence the
string-level assertions here.

Usage:
  python -m pytest tests/parallelism/test_causvid_wan_cp_plan.py -q
"""

import pytest
import torch

from cache_dit.distributed.config import ParallelismConfig
from cache_dit.distributed.transformers.causvid import (
  CausVidContextParallelismPlanner,
  CausVidTensorParallelismPlanner,
)
from cache_dit.distributed.transformers.planners import _activate_cp_planners, _activate_tp_planners
from cache_dit.distributed.transformers.register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlannerRegister,
)
from cache_dit.models.pipelines.causvid.transformer import CausVidTransformer3DModel

_NUM_BLOCKS = 2
_HEADS = 4


def _tiny_transformer() -> CausVidTransformer3DModel:
  torch.manual_seed(0)
  return CausVidTransformer3DModel(
    patch_size=(1, 2, 2),
    num_attention_heads=_HEADS,
    attention_head_dim=8,
    in_channels=16,
    out_channels=16,
    text_dim=32,
    freq_dim=32,
    ffn_dim=64,
    num_layers=_NUM_BLOCKS,
    cross_attn_norm=False,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
    rope_max_seq_len=64,
  ).eval()


def _ulysses(**kwargs) -> ParallelismConfig:
  values = dict(ulysses_size=2)
  values.update(kwargs)
  return ParallelismConfig(**values)


def test_cp_plan_splits_rope_and_the_first_block_then_gathers_proj_out():
  plan = CausVidContextParallelismPlanner()._apply(transformer=_tiny_transformer(),
                                                   parallelism_config=_ulysses())

  assert set(plan) == {"rope", "blocks.0", "proj_out"}
  for slot in (0, 1):
    split = plan["rope"][slot]
    assert split.split_dim == 1
    assert split.expected_dims == 4
    assert split.split_output is True

  hidden_split = plan["blocks.0"]["hidden_states"]
  assert hidden_split.split_dim == 1
  assert hidden_split.expected_dims == 3
  assert hidden_split.split_output is False

  gather = plan["proj_out"]
  assert gather.gather_dim == 1
  assert gather.expected_dims == 3


def test_cp_plan_keys_resolve_on_the_real_module_tree():
  transformer = _tiny_transformer()
  plan = CausVidContextParallelismPlanner()._apply(transformer=transformer,
                                                   parallelism_config=_ulysses())

  for key in ("rope", "proj_out"):
    assert key in plan, f"CP plan lost the '{key}' marker"
    assert hasattr(transformer, key), f"CP plan key '{key}' is not a submodule"
  assert "blocks.0" in plan
  assert hasattr(transformer, "blocks") and len(transformer.blocks) >= 1


def test_cp_plan_refuses_ring_attention():
  with pytest.raises(NotImplementedError, match="Ring"):
    CausVidContextParallelismPlanner()._apply(
      transformer=_tiny_transformer(),
      parallelism_config=ParallelismConfig(ring_size=2),
    )


def test_cp_plan_refuses_ulysses_anything():
  with pytest.raises(NotImplementedError, match="ulysses-anything"):
    CausVidContextParallelismPlanner()._apply(
      transformer=_tiny_transformer(),
      parallelism_config=_ulysses(ulysses_anything=True),
    )


def test_cp_plan_requires_divisible_heads_after_tp():
  # `ParallelismConfig` validates the world size against the distributed group, so build a valid
  # config and then force the shapes the guard has to accept / reject.
  # 4 heads / tp 2 = 2 heads, which still divides by the Ulysses size.
  divides = _ulysses()
  divides.tp_size = 2
  CausVidContextParallelismPlanner()._apply(transformer=_tiny_transformer(),
                                            parallelism_config=divides)

  indivisible = _ulysses()
  indivisible.ulysses_size = 3
  with pytest.raises(ValueError, match="head"):
    CausVidContextParallelismPlanner()._apply(transformer=_tiny_transformer(),
                                              parallelism_config=indivisible)


def test_planners_are_registered_under_the_transformer_prefix():
  _activate_cp_planners()
  _activate_tp_planners()
  transformer = _tiny_transformer()

  cp_planner_cls = ContextParallelismPlannerRegister.get_planner(transformer)
  assert cp_planner_cls is CausVidContextParallelismPlanner
  assert issubclass(cp_planner_cls, ContextParallelismPlanner)

  tp_planner_cls = TensorParallelismPlannerRegister.get_planner(transformer)
  assert tp_planner_cls is CausVidTensorParallelismPlanner
