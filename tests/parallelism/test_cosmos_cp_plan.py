"""CPU checks for the Cosmos Predict2 (Anima) CP plan and the processor fork parity.

The CP plan splits the latents once at `transformer_blocks.0`, re-splits
`encoder_hidden_states` at every block (the forward passes the full text from a
loop-invariant local variable), output-splits the `rope` tables and gathers on
`proj_out`. A wrong layout still renders a plausible image, so the plan shape is
asserted here directly. The `CosmosAttnProcessor2_0` fork is only allowed to
differ from upstream in the attention dispatch, so the patched `__call__` must
stay bit-identical to the saved upstream one while no `_cp_config` is attached.

Usage:
  python -m pytest tests/parallelism/test_cosmos_cp_plan.py -q
"""

import pytest
import torch

try:
  from diffusers.models.attention_processor import Attention
  from diffusers.models.transformers.transformer_cosmos import (
    CosmosAttnProcessor2_0,
    CosmosTransformer3DModel,
  )
except ImportError:
  pytest.skip("diffusers without Cosmos support", allow_module_level=True)

from cache_dit.distributed.transformers.cosmos import (
  CosmosContextParallelismPlanner,
  __patch__CosmosAttnProcessor2_0__call__,
)
from cache_dit.distributed.transformers.register import (
  ContextParallelismPlannerRegister,
  TensorParallelismPlannerRegister,
)

_HIDDEN = 64
_HEADS = 2
_HEAD_DIM = 32


def _tiny_transformer() -> CosmosTransformer3DModel:
  torch.manual_seed(0)
  return CosmosTransformer3DModel(
    in_channels=16,
    out_channels=16,
    num_attention_heads=_HEADS,
    attention_head_dim=_HEAD_DIM,
    num_layers=2,
    mlp_ratio=2.0,
    text_embed_dim=64,
    adaln_lora_dim=32,
    max_size=(8, 32, 32),
    rope_scale=(1.0, 1.0, 1.0),
    extra_pos_embed_type=None,  # the Anima checkpoint ships `null`
  ).eval()


def test_cp_plan_layout():
  plan = CosmosContextParallelismPlanner()._apply(transformer=_tiny_transformer())

  rope = plan["rope"]
  assert set(rope) == {0, 1}
  for spec in rope.values():
    assert spec.split_dim == 0 and spec.expected_dims == 2 and spec.split_output

  block0 = plan["transformer_blocks.0"]
  assert set(block0) == {"hidden_states"}
  assert block0["hidden_states"].split_dim == 1
  assert block0["hidden_states"].expected_dims == 3
  assert not block0["hidden_states"].split_output

  blocks = plan["transformer_blocks.*"]
  assert set(blocks) == {"encoder_hidden_states"}
  assert blocks["encoder_hidden_states"].split_dim == 1
  assert blocks["encoder_hidden_states"].expected_dims == 3
  assert not blocks["encoder_hidden_states"].split_output

  assert plan["proj_out"].gather_dim == 1 and plan["proj_out"].expected_dims == 3


def test_planners_are_registered_under_the_exact_class_name():
  # The prefix "Cosmos" would also match `Cosmos3OmniTransformer`, which has no planner:
  # registering the exact class name is what keeps Cosmos3 failing loudly instead of
  # silently receiving the wrong plan.
  cp_planner = ContextParallelismPlannerRegister.get_planner("CosmosTransformer3DModel")
  assert cp_planner is CosmosContextParallelismPlanner
  tp_planner = TensorParallelismPlannerRegister.get_planner("CosmosTransformer3DModel")
  assert tp_planner.__name__ == "CosmosTensorParallelismPlanner"

  with pytest.raises(ValueError, match="Cosmos3OmniTransformer"):
    ContextParallelismPlannerRegister.get_planner("Cosmos3OmniTransformer")
  with pytest.raises(ValueError, match="Cosmos3OmniTransformer"):
    TensorParallelismPlannerRegister.get_planner("Cosmos3OmniTransformer")


def _attention(cross_attention_dim: int) -> Attention:
  return Attention(
    query_dim=_HIDDEN,
    cross_attention_dim=cross_attention_dim,
    heads=_HEADS,
    dim_head=_HEAD_DIM,
    qk_norm="rms_norm",
    elementwise_affine=True,
    out_bias=False,
    processor=CosmosAttnProcessor2_0(),
  )


@pytest.mark.parametrize("with_context", [False, True])
def test_processor_fork_is_bit_exact_without_cp_config(with_context):
  # The fork copies the upstream `__call__` and only swaps the dispatch; with no
  # `_cp_config` attached it must reproduce the upstream numbers bit-exactly, so a
  # future upstream change (RoPE unbind dim, GQA handling, ...) surfaces here instead
  # of corrupting every CP run. Both parametrizations mirror the real block calls:
  # `attn1` (self attention with RoPE) and `attn2` (cross attention, no RoPE).
  current = CosmosAttnProcessor2_0.__call__
  upstream = current.__wrapped__ if current is __patch__CosmosAttnProcessor2_0__call__ else current

  torch.manual_seed(7)
  attn = _attention(64 if with_context else _HIDDEN)
  processor = attn.processor
  hidden_states = torch.randn(1, 8, _HIDDEN)
  encoder_hidden_states = torch.randn(1, 5, _HIDDEN) if with_context else None
  image_rotary_emb = None if with_context else (torch.randn(8, _HEAD_DIM),
                                                torch.randn(8, _HEAD_DIM))

  with torch.no_grad():
    ref = upstream(
      processor,
      attn,
      hidden_states,
      encoder_hidden_states=encoder_hidden_states,
      image_rotary_emb=image_rotary_emb,
    )
    got = __patch__CosmosAttnProcessor2_0__call__(
      processor,
      attn,
      hidden_states,
      encoder_hidden_states=encoder_hidden_states,
      image_rotary_emb=image_rotary_emb,
    )

  assert torch.equal(ref, got)
