"""Regression tests for the SANA-Video 2.0 CP / TP plans.

The plans are keyed on the vendored transformer's markers (`sequence_split` / `rope_*_split` /
`sequence_gather`) rather than on `transformer.forward` arguments, because the Attention-Residual
depth buffers inside the block loop have to hold the local shard. These tests pin the marker names,
the split axes (tokens on dim 1, the two complex RoPE tables on dim 2) and the TP layer plan.

Usage:
  python -m pytest tests/parallelism/test_sana_video2_cp_plan.py -q
"""
from types import SimpleNamespace

import pytest
import torch

from cache_dit.distributed.transformers.sana_video2 import (
  SanaVideo2ContextParallelismPlanner,
  SanaVideo2TensorParallelismPlanner,
  sana_video2_block_plan,
)
from cache_dit.models.pipelines.sana_video2.transformer import SanaVideo2Transformer3DModel


def _tiny_transformer(depth: int = 4) -> SanaVideo2Transformer3DModel:
  torch.manual_seed(0)
  return SanaVideo2Transformer3DModel(
    in_channels=4,
    hidden_size=192,
    depth=depth,
    num_heads=2,
    caption_channels=16,
    model_max_length=4,
    linear_head_dim=64,
    softmax_head_dim=96,
    attn_res_block_size=2,
    input_size=2,
  ).eval()


def _config(ulysses: int = 2, ring: int = 1, anything: bool = False) -> SimpleNamespace:
  return SimpleNamespace(ulysses_size=ulysses, ring_size=ring, ulysses_anything=anything)


def test_cp_plan_markers_and_axes():
  transformer = _tiny_transformer()
  plan = SanaVideo2ContextParallelismPlanner()._apply(transformer=transformer,
                                                      parallelism_config=_config())
  assert set(plan) == {
    "sequence_split", "rope_linear_split", "rope_softmax_split", "sequence_gather"
  }
  assert plan["sequence_split"][0].split_dim == 1
  assert plan["sequence_split"][0].expected_dims == 3
  assert plan["sequence_split"][0].split_output is True
  for name in ("rope_linear_split", "rope_softmax_split"):
    assert plan[name][0].split_dim == 2
    assert plan[name][0].expected_dims == 4
  assert plan["sequence_gather"].gather_dim == 1


def test_cp_markers_exist_on_the_transformer():
  transformer = _tiny_transformer()
  for name in ("sequence_split", "rope_linear_split", "rope_softmax_split", "sequence_gather"):
    assert isinstance(getattr(transformer, name), torch.nn.Identity)


def test_rope_tables_are_complex_and_split_on_the_token_axis():
  transformer = _tiny_transformer()
  table = transformer.rope_softmax((2, 2, 2), torch.device("cpu"))
  assert tuple(table.shape) == (1, 1, 8, transformer.config.softmax_head_dim // 2)
  assert table.is_complex()
  shard = torch.tensor_split(table, 2, dim=2)[1]
  assert tuple(shard.shape) == (1, 1, 4, transformer.config.softmax_head_dim // 2)


def test_head_divisibility_is_checked_for_anchor_layers():
  transformer = _tiny_transformer()
  # 2 softmax heads per anchor layer: a degree of 4 cannot split them evenly.
  with pytest.raises(ValueError, match="ulysses-anything"):
    SanaVideo2ContextParallelismPlanner()._apply(transformer=transformer,
                                                 parallelism_config=_config(ulysses=4))
  # Degree 2 is fine.
  SanaVideo2ContextParallelismPlanner()._apply(transformer=transformer,
                                               parallelism_config=_config(ulysses=2))
  # ... and so is degree 4 with uneven Ulysses.
  SanaVideo2ContextParallelismPlanner()._apply(transformer=transformer,
                                               parallelism_config=_config(ulysses=4, anything=True))


def test_tp_plan_shards_only_the_ffn():
  block = _tiny_transformer().blocks[0]
  plan = sana_video2_block_plan(block)
  assert set(plan) == {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
  assert plan["mlp.down_proj"].__class__.__name__ == "RowwiseParallel"
  assert plan["mlp.gate_proj"].__class__.__name__ == "ColwiseParallel"


def test_tp_plan_keys_resolve_to_real_submodules():
  block = _tiny_transformer().blocks[0]
  for key in sana_video2_block_plan(block):
    assert block.get_submodule(key) is not torch.nn.Identity()


def test_cp_rejects_per_frame_timestep():
  """TI2V modulates per frame; a CP token shard cuts through frames, so it must be rejected."""
  transformer = _tiny_transformer()
  transformer._cp_config = _config(ulysses=2)
  latents = torch.randn(1, 4, 2, 2, 2)
  timestep = torch.full((1, 1, 2, 1, 1), 300.0)
  try:
    with pytest.raises(NotImplementedError, match="context parallelism"):
      transformer(
        hidden_states=latents,
        timestep=timestep,
        encoder_hidden_states=torch.randn(1, 4, 16),
        encoder_attention_mask=torch.ones(1, 4, dtype=torch.long),
        return_dict=False,
      )
  finally:
    del transformer._cp_config


def test_token_shard_is_not_frame_aligned():
  """Documents why the guard above exists: a 97-frame clip over 2 ranks cuts through frames.

  720p / 97 frames gives 13 latent frames of 920 tokens each. The per-frame branch reshapes the
  local shard by the *global* frame count, which only lines up when one "slot" of the reshape is
  exactly one frame -- here it is half a frame, and nothing raises.
  """
  latent_frames, plane = 13, 23 * 40
  tokens = latent_frames * plane
  for degree in (2, 4):
    shard = tokens // degree
    assert tokens % degree == 0 and shard % latent_frames == 0  # the reshape would succeed
    assert shard // latent_frames != plane  # ... but one slot is not one frame


def test_tp_planner_class_is_registered():
  from cache_dit.distributed.transformers.register import (
    TensorParallelismPlannerRegister, )

  name = "SanaVideo2Transformer3DModel"
  assert issubclass(
    TensorParallelismPlannerRegister.get_planner(name),
    SanaVideo2TensorParallelismPlanner,
  )
  from cache_dit.distributed.transformers.register import (
    ContextParallelismPlannerRegister, )

  assert issubclass(
    ContextParallelismPlannerRegister.get_planner(name),
    SanaVideo2ContextParallelismPlanner,
  )
