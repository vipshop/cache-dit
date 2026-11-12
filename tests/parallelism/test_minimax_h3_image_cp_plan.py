"""CPU-only guards for the MiniMax-H3-Image CP / TP planner.

The image transformer differs from MiniMax-H3 in exactly one place that a plan can get wrong: it has
**one output head per sampling step** (`pdd_proj_out`, a `ModuleList`) instead of a single
`proj_out` / `audio_proj_out` pair. Two failure modes are covered here:

1. **Planner registration.** The planner registries are keyed by the **full** transformer class
   name, so the image entry is what decides the plan; a missing or misspelled `_safe_import` in
   `planners.py` would fall back to "no planner registered" instead. Resolving to the video planner
   would gather at `proj_out` (absent here) and leave the six per-step heads sharded, i.e. every rank
   would run its head on a partial sequence without any shape error.
2. **Plan order.** The cached path replays the first split hook of `transformer_blocks[0]` to
   re-derive the residual of a skipped step, so `hidden_states` has to be registered before
   `adaln_indices` on that block.
"""

import torch

from cache_dit.distributed.config import ParallelismConfig
from cache_dit.distributed.transformers.minimax_h3_image import (
  MiniMaxH3ImageContextParallelismPlanner,
  MiniMaxH3ImageTensorParallelismPlanner,
)
from cache_dit.distributed.transformers.register import (
  ContextParallelismPlannerRegister,
  TensorParallelismPlannerRegister,
)


def _tiny_transformer():
  from diffusers import MiniMaxH3ImageTransformer2DModel

  torch.manual_seed(0)
  return MiniMaxH3ImageTransformer2DModel(
    num_attention_heads=4,
    attention_head_dim=8,
    hidden_size=32,
    num_layers=1,
    num_refiner_layers=1,
    ffn_dim=64,
    in_channels=2,
    patch_size=(1, 2, 2),
    text_dim=8,
    freq_dim=8,
    time_embed_hidden_dim=32,
    time_embed_dim=16,
    rope_freq_dim=2,
  )


def test_planners_are_reachable_through_the_activation_path():
  """Go through the production activation path, not a direct import.

  Importing the planner module in this test would pass even if the `_safe_import` entry in
  `planners.py` were missing or pointed at the wrong module.
  """
  from cache_dit.distributed.transformers.dispatch import (
    _ensure_cp_planners_activated,
    _ensure_tp_planners_activated,
  )

  _ensure_cp_planners_activated()
  _ensure_tp_planners_activated()

  assert "MiniMaxH3ImageTransformer2DModel" in ContextParallelismPlannerRegister.supported_planners(
  )[1]
  assert "MiniMaxH3ImageTransformer2DModel" in TensorParallelismPlannerRegister.supported_planners(
  )[1]
  assert (ContextParallelismPlannerRegister.get_planner("MiniMaxH3ImageTransformer2DModel")
          is MiniMaxH3ImageContextParallelismPlanner)
  assert (TensorParallelismPlannerRegister.get_planner("MiniMaxH3ImageTransformer2DModel")
          is MiniMaxH3ImageTensorParallelismPlanner)
  # the video planners keep resolving for their own class
  assert (ContextParallelismPlannerRegister.get_planner("MiniMaxH3Transformer3DModel").__name__ ==
          "MiniMaxH3ContextParallelismPlanner")
  assert (TensorParallelismPlannerRegister.get_planner("MiniMaxH3Transformer3DModel").__name__ ==
          "MiniMaxH3TensorParallelismPlanner")


def test_cp_plan_splits_hidden_state_before_adaln_indices():
  transformer = _tiny_transformer()
  cp_plan = MiniMaxH3ImageContextParallelismPlanner()._apply(
    transformer, ParallelismConfig(ulysses_size=2, ulysses_anything=True))

  keys = list(cp_plan)
  assert keys[0] == "rope"
  assert keys[1] == "transformer_blocks.0"
  assert keys.index("transformer_blocks.0") < keys.index("transformer_blocks.*"), (
    "the cached path replays the first split hook of `transformer_blocks[0]`")
  assert "hidden_states" in cp_plan["transformer_blocks.0"]

  # `rope` returns a (cos, sin) pair: both outputs are split
  for index in (0, 1):
    assert cp_plan["rope"][index].split_output is True

  # `forward` scatters the streams with global row indices, so its own inputs stay whole
  assert "" not in cp_plan, "the packed buffer is built from global indices: no root split"
  for subplan in cp_plan.values():
    if isinstance(subplan, dict):
      assert not ({"position_ids", "token_tags"} & set(subplan))
  # `timestep_indices` is only split at the output norm, after `forward`'s validation
  assert "timestep_indices" in cp_plan["norm_out"]


def test_cp_plan_gathers_at_the_per_step_output_heads():
  transformer = _tiny_transformer()
  cp_plan = MiniMaxH3ImageContextParallelismPlanner()._apply(
    transformer, ParallelismConfig(ulysses_size=2, ulysses_anything=True))

  # the video plan's single-head gathers do not exist on this transformer
  assert "proj_out" not in cp_plan
  assert "audio_proj_out" not in cp_plan
  head_gather = cp_plan["pdd_proj_out.*"]
  assert (head_gather.gather_dim, head_gather.expected_dims) == (1, 3)
  # one `(B, S, patch_dim)` tensor per step, so gathering after the heads moves 56x less data than
  # gathering after the block stack
  assert transformer.config.hidden_size > transformer.config.in_channels
