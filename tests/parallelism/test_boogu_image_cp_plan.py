"""CPU checks for the Boogu-Image CP plan and the TP layer plan.

The plan is asserted at the string level because a wrong entry does not raise:
the double-stream stack must be split once (both streams, first block only) and
keep every block shape invariant, the joint rope table must NOT be split (the
block stitches the local rope from the global `[instruct | image]` layout), and
no stage may gather inside a cached block stack.

Usage:
  python -m pytest tests/parallelism/test_boogu_image_cp_plan.py -q
"""

import pytest

from cache_dit.distributed.config import ParallelismConfig
from cache_dit.distributed.transformers.boogu_image import (
  BooguImageContextParallelismPlanner,
  BooguImageTensorParallelismPlanner,
)
from cache_dit.distributed.transformers.planners import _activate_cp_planners
from cache_dit.distributed.transformers.planners import _activate_tp_planners
from cache_dit.distributed.transformers.register import (
  ContextParallelismPlannerRegister,
  TensorParallelismPlannerRegister,
)
from cache_dit.models.pipelines.boogu_image.transformer_boogu import (
  BooguImageTransformer2DModel, )


def _tiny_transformer(_NUM_REFINERS: int = 2) -> BooguImageTransformer2DModel:
  return BooguImageTransformer2DModel(
    patch_size=2,
    in_channels=4,
    hidden_size=96,
    num_layers=5,
    num_double_stream_layers=2,
    num_refiner_layers=_NUM_REFINERS,
    num_attention_heads=4,
    num_kv_heads=2,
    multiple_of=32,
    axes_dim_rope=(8, 8, 8),
    axes_lens=(64, 64, 64),
    instruction_feature_configs=dict(instruction_feat_dim=32,
                                     reduce_type="mean",
                                     num_instruction_feat_layers=1),
  )


def test_boogu_cp_planner_is_registered() -> None:
  _activate_cp_planners()
  planner_cls = ContextParallelismPlannerRegister.get_planner("BooguImageTransformer")
  assert planner_cls is BooguImageContextParallelismPlanner


def test_boogu_cp_plan_splits_refiners_and_single_stream() -> None:
  transformer = _tiny_transformer()
  plan = BooguImageContextParallelismPlanner()._apply(
    transformer=transformer,
    parallelism_config=ParallelismConfig(ulysses_size=2),
  )

  for stack in ("noise_refiner", "ref_image_refiner", "single_stream_layers"):
    assert f"{stack}.0" in plan
    assert f"{stack}.*" in plan

  # The split covers both streams of the double-stream stack and nothing else.
  assert set(plan["double_stream_layers.0"]) == {"img_hidden_states", "instruct_hidden_states"}
  # The joint rope table is stitched inside the block, never split by a hook.
  assert "rotary_emb" not in plan["double_stream_layers.0"]


def test_boogu_cp_plan_gathers_outside_the_cached_stacks() -> None:
  transformer = _tiny_transformer()
  plan = BooguImageContextParallelismPlanner()._apply(
    transformer=transformer,
    parallelism_config=ParallelismConfig(ulysses_size=2),
  )

  last_single = len(transformer.single_stream_layers) - 1
  last_double = len(transformer.double_stream_layers) - 1
  # Cached stacks must stay shape invariant: no output hook on their last block.
  assert f"single_stream_layers.{last_single}" not in plan
  assert f"double_stream_layers.{last_double}" not in plan
  # The refiners are not cached, so they still gather at their last block.
  assert f"noise_refiner.{len(transformer.noise_refiner) - 1}" in plan


def test_boogu_cp_plan_arms_only_split_processors() -> None:
  transformer = _tiny_transformer()
  BooguImageContextParallelismPlanner()._apply(
    transformer=transformer,
    parallelism_config=ParallelismConfig(ulysses_size=2),
  )

  # Split stacks take the Ulysses path.
  assert transformer.single_stream_layers[0].attn.processor._boogu_cp_eligible
  assert transformer.noise_refiner[0].attn.processor._boogu_cp_eligible
  # The context refiner is replicated.
  assert not transformer.context_refiner[0].attn.processor._boogu_cp_eligible
  # The double-stream processors are toggled per forward by their block.
  assert not (transformer.double_stream_layers[0].img_self_attn.processor._boogu_cp_eligible)


def test_boogu_cp_plan_single_refiner_layer_keeps_split_and_gather() -> None:
  transformer = _tiny_transformer(_NUM_REFINERS=1)
  plan = BooguImageContextParallelismPlanner()._apply(
    transformer=transformer,
    parallelism_config=ParallelismConfig(ulysses_size=2),
  )

  for stack in ("noise_refiner", "ref_image_refiner"):
    # A one-layer stack must not lose its split to the gather entry: the
    # framework's list form carries both hooks on the same module.
    entry = plan[f"{stack}.0"]
    assert isinstance(entry, list) and len(entry) == 2
    assert set(entry[0]) == {"hidden_states", "image_rotary_emb"}


def test_boogu_cp_plan_rejects_async_ulysses() -> None:
  transformer = _tiny_transformer()
  config = ParallelismConfig(ulysses_size=2, ulysses_async=True)

  with pytest.raises(ValueError, match="async Ulysses"):
    BooguImageContextParallelismPlanner()._apply(
      transformer=transformer,
      parallelism_config=config,
    )


def test_boogu_tp_planner_is_registered_and_shards_ffn() -> None:
  _activate_tp_planners()
  planner_cls = TensorParallelismPlannerRegister.get_planner("BooguImageTransformer")
  assert planner_cls is BooguImageTensorParallelismPlanner

  transformer = _tiny_transformer()
  single_plan = BooguImageTensorParallelismPlanner._single_stream_layer_plan(
    transformer.single_stream_layers[0])
  assert set(single_plan) == {
    "feed_forward.linear_1", "feed_forward.linear_3", "feed_forward.linear_2", "norm1.linear"
  }

  double_plan = BooguImageTensorParallelismPlanner._double_stream_layer_plan(
    transformer.double_stream_layers[0])
  assert "img_feed_forward.linear_1" in double_plan
  assert "instruct_feed_forward.linear_2" in double_plan
  assert "img_norm1.linear" in double_plan
  # Attention projections stay replicated (num_kv_heads is not divisible by 2).
  assert not any(key.startswith(("img_self_attn", "img_instruct_attn")) for key in double_plan)
