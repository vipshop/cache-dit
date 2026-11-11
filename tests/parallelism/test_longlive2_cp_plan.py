"""CPU checks for the LongLive 2.0 CP plan and TP planner registration.

LongLive 2.0's chunk-causal attention keeps a KV cache addressed by global token coordinates, so the
CP plan only splits tensors whose partition is rank-local: the RoPE tables of the current chunk, the
hidden states at the first block, and the per-token timestep. The Ulysses exchange itself lives inside
the attention processor, which means a wrong plan still renders a plausible clip - hence the
string-level assertions here.

Usage:
  python -m pytest tests/parallelism/test_longlive2_cp_plan.py -q
"""

import pytest
import torch

from cache_dit.distributed.config import ParallelismConfig
from cache_dit.distributed.parallel import _ContextParallelConfig
from cache_dit.distributed.transformers.longlive2 import (
  LongLive2ContextParallelismPlanner,
  LongLive2TensorParallelismPlanner,
)
from cache_dit.distributed.transformers.planners import _activate_cp_planners, _activate_tp_planners
from cache_dit.distributed.transformers.register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlannerRegister,
)
from cache_dit.models.pipelines.longlive2.transformer import (
  LongLive2CausalAttnProcessor,
  LongLive2Transformer3DModel,
)

_NUM_BLOCKS = 2
_HEADS = 4


def _tiny_transformer() -> LongLive2Transformer3DModel:
  torch.manual_seed(0)
  return LongLive2Transformer3DModel(
    patch_size=(1, 2, 2),
    num_attention_heads=_HEADS,
    attention_head_dim=8,
    in_channels=16,
    out_channels=16,
    text_dim=32,
    freq_dim=32,
    ffn_dim=64,
    num_layers=_NUM_BLOCKS,
    cross_attn_norm=True,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
    rope_max_seq_len=64,
  ).eval()


def _runtime_cp(degree: int = 2) -> _ContextParallelConfig:
  """The runtime object cache-dit attaches to `transformer._cp_config`."""
  return _ContextParallelConfig(ulysses_degree=degree, _world_size=degree, _rank=0)


def _ulysses(**kwargs) -> ParallelismConfig:
  values = dict(ulysses_size=2)
  values.update(kwargs)
  return ParallelismConfig(**values)


def test_cp_plan_splits_rope_and_the_first_block_then_gathers_proj_out():
  plan = LongLive2ContextParallelismPlanner()._apply(transformer=_tiny_transformer(),
                                                     parallelism_config=_ulysses())

  assert set(plan) == {"rope", "blocks.0", "proj_out", ""}
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

  # Wan2.2 TI2V conditioning passes one timestep per token; it must be split with the sequence.
  timestep_split = plan[""]["timestep"]
  assert timestep_split.split_dim == 1
  assert timestep_split.expected_dims == 2
  assert timestep_split.split_output is False


def test_cp_plan_keys_resolve_on_the_real_module_tree():
  transformer = _tiny_transformer()
  plan = LongLive2ContextParallelismPlanner()._apply(transformer=transformer,
                                                     parallelism_config=_ulysses())

  for key in ("rope", "proj_out"):
    assert key in plan, f"CP plan lost the '{key}' marker"
    assert hasattr(transformer, key), f"CP plan key '{key}' is not a submodule"
  assert "blocks.0" in plan
  assert hasattr(transformer, "blocks") and len(transformer.blocks) >= 1


def test_cp_plan_refuses_ring_attention():
  with pytest.raises(NotImplementedError, match="Ring"):
    LongLive2ContextParallelismPlanner()._apply(
      transformer=_tiny_transformer(),
      parallelism_config=ParallelismConfig(ring_size=2),
    )


def test_cp_plan_refuses_ulysses_anything():
  with pytest.raises(NotImplementedError, match="ulysses-anything"):
    LongLive2ContextParallelismPlanner()._apply(
      transformer=_tiny_transformer(),
      parallelism_config=_ulysses(ulysses_anything=True),
    )


def test_planners_are_registered_under_the_transformer_prefix():
  _activate_cp_planners()
  _activate_tp_planners()
  transformer = _tiny_transformer()

  cp_planner_cls = ContextParallelismPlannerRegister.get_planner(transformer)
  assert cp_planner_cls is LongLive2ContextParallelismPlanner
  assert issubclass(cp_planner_cls, ContextParallelismPlanner)

  tp_planner_cls = TensorParallelismPlannerRegister.get_planner(transformer)
  assert tp_planner_cls is LongLive2TensorParallelismPlanner


def test_cache_heads_follow_the_ulysses_shard(monkeypatch):
  # Building the real all-to-all communicator needs an initialized process group; the cache sizing
  # decision under test does not, so stub the communicator setup out.
  monkeypatch.setattr(LongLive2CausalAttnProcessor, "prepare_cp",
                      lambda self, cp_config: setattr(self, "_cp_config", cp_config))

  transformer = _tiny_transformer()
  shape = (1, 16, 2, 8, 8)
  reference = transformer.init_kv_cache(1, shape, num_frames=2)
  assert reference[0].heads == _HEADS

  transformer._cp_config = _runtime_cp()
  sharded = transformer.init_kv_cache(1, shape, num_frames=2)
  assert sharded[0].heads == _HEADS // 2


def test_kv_cache_rejects_a_chunk_that_does_not_divide():
  transformer = _tiny_transformer()
  transformer._cp_config = _runtime_cp()
  # 3 heads cannot be split by 2 Ulysses ranks.
  transformer.blocks[0].attn1.heads = 3
  with pytest.raises(ValueError, match="divide"):
    transformer.init_kv_cache(1, (1, 16, 2, 8, 8), num_frames=2)
