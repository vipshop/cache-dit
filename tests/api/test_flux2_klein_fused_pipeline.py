"""Contract tests for cache-dit's ``_Flux2Klein`` optimisation pass.

CPU-only, no checkpoint. They pin the parts that fail silently: the registration names (a
missing adapter makes ``enable_cache`` return ``None`` and fail later, a missing planner makes
CP/TP fall back), the forward-pattern signature that ``check_forward_pattern=True`` inspects,
and the bit-exactness of the fused double block against the stock diffusers block (a swapped
img/txt stream would only show up as a slightly worse image).
"""

import inspect
import types

import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from diffusers.models.transformers.transformer_flux2 import Flux2TransformerBlock  # noqa: E402

from cache_dit.caching.block_adapters.block_registers import BlockAdapterRegister  # noqa: E402
from cache_dit.caching.forward_pattern import ForwardPattern  # noqa: E402
from cache_dit.distributed.transformers.register import (  # noqa: E402
  ContextParallelismPlannerRegister, TensorParallelismPlannerRegister,
)
from cache_dit.models.pipelines.flux2_klein.block import (  # noqa: E402
  fused_double_block_forward, fused_single_block_forward,
)

DIM = 16
HEADS = 2
HEAD_DIM = 8


def _tiny_double_block() -> Flux2TransformerBlock:
  return Flux2TransformerBlock(dim=DIM,
                               num_attention_heads=HEADS,
                               attention_head_dim=HEAD_DIM,
                               mlp_ratio=2.0)


def _tiny_transformer():
  from diffusers import Flux2Transformer2DModel

  return Flux2Transformer2DModel(patch_size=1,
                                 in_channels=4,
                                 out_channels=4,
                                 num_layers=1,
                                 num_single_layers=1,
                                 attention_head_dim=HEAD_DIM,
                                 num_attention_heads=HEADS,
                                 joint_attention_dim=DIM,
                                 mlp_ratio=2.0,
                                 axes_dims_rope=(2, 3, 3, 8))


def test_adapter_is_registered_for_the_underscore_pipeline():
  """`enable_cache` resolves adapters by class-name prefix, so the `_` name needs its own key."""

  names = list(BlockAdapterRegister._adapters)
  assert "_Flux2Klein" in names
  # No other registered prefix may capture `_Flux2KleinTransformer2DModel` first, or the
  # pipeline would be dispatched with the wrong block layout.
  earlier = [
    name for name in names[:names.index("_Flux2Klein")]
    if "_Flux2KleinTransformer2DModel".startswith(name)
  ]
  assert earlier == []


def test_adapter_keeps_two_block_lists_without_a_patch_functor():
  pipe = types.SimpleNamespace(transformer=_tiny_transformer())

  adapter = BlockAdapterRegister._adapters["_Flux2Klein"](pipe)

  assert len(adapter.blocks) == 2
  assert adapter.forward_pattern == [ForwardPattern.Pattern_1, ForwardPattern.Pattern_3]
  assert adapter.check_forward_pattern is True
  # No patch functor: FLUX.2 needs no forward rewrite, and one would re-introduce the
  # `_repeated_blocks` class-name dependency this pass avoids.
  assert adapter.patch_functor is None


def test_cp_and_tp_planners_resolve_for_the_subclass():
  from cache_dit.distributed.transformers.flux2 import (
    Flux2ContextParallelismPlanner,
    Flux2TensorParallelismPlanner,
  )
  from cache_dit.models.pipelines.flux2_klein.transformer_flux2_klein import (
    _Flux2KleinTransformer2DModel, )

  name = _Flux2KleinTransformer2DModel.__name__
  cp_planner = ContextParallelismPlannerRegister.get_planner(name)
  tp_planner = TensorParallelismPlannerRegister.get_planner(name)

  assert issubclass(cp_planner, Flux2ContextParallelismPlanner)
  assert issubclass(tp_planner, Flux2TensorParallelismPlanner)
  assert name in ContextParallelismPlannerRegister._cp_planner_registry
  assert name in TensorParallelismPlannerRegister._tp_planner_registry


def test_block_class_names_stay_untouched():
  """`_repeated_blocks` matches by class name, so the blocks may not be re-typed."""

  from cache_dit.models.pipelines.flux2_klein import block as fused_block
  from cache_dit.models.pipelines.flux2_klein.transformer_flux2_klein import (
    _Flux2KleinTransformer2DModel, )

  assert _Flux2KleinTransformer2DModel._repeated_blocks == [
    "Flux2TransformerBlock",
    "Flux2SingleTransformerBlock",
  ]
  # No replacement block classes are exported: the pass rebinds `forward` per instance.
  assert [name for name in dir(fused_block) if name.endswith("TransformerBlock")] == []


def test_fused_block_forwards_keep_the_forward_pattern_signature():
  """`check_forward_pattern=True` reads parameter names and the tensor count in the annotation."""

  double_params = set(inspect.signature(fused_double_block_forward).parameters)
  for required in ForwardPattern.Pattern_1.In:
    assert required in double_params, f"missing forward-pattern parameter: {required}"
  double_outputs = str(
    inspect.signature(fused_double_block_forward).return_annotation).count("torch.Tensor")
  assert double_outputs == len(ForwardPattern.Pattern_1.Out)

  single_params = set(inspect.signature(fused_single_block_forward).parameters)
  for required in ForwardPattern.Pattern_3.In:
    assert required in single_params, f"missing forward-pattern parameter: {required}"
  # `Any` is the documented way to skip the output-count check (the single block returns either
  # a tensor or a tuple depending on `split_hidden_states`); anything else must count correctly.
  single_outputs = str(
    inspect.signature(fused_single_block_forward).return_annotation).count("torch.Tensor")
  assert single_outputs in (0, len(ForwardPattern.Pattern_3.Out))


def test_fused_double_block_matches_the_stock_block_bit_exactly():
  torch.manual_seed(0)
  stock = _tiny_double_block().to(torch.bfloat16)
  fused = _tiny_double_block().to(torch.bfloat16)
  fused.load_state_dict(stock.state_dict())
  fused.forward = fused_double_block_forward.__get__(fused)

  hidden = torch.randn(1, 7, DIM, dtype=torch.bfloat16)
  encoder = torch.randn(1, 5, DIM, dtype=torch.bfloat16)
  temb_img = torch.randn(1, 1, 6 * DIM, dtype=torch.bfloat16)
  temb_txt = torch.randn(1, 1, 6 * DIM, dtype=torch.bfloat16)
  cos = torch.randn(12, HEAD_DIM)
  sin = torch.randn(12, HEAD_DIM)

  expected = stock(hidden, encoder, temb_img, temb_txt, image_rotary_emb=(cos, sin))
  got = fused(hidden, encoder, temb_img, temb_txt, image_rotary_emb=(cos, sin))

  assert len(got) == len(expected)
  for expected_tensor, got_tensor in zip(expected, got):
    assert torch.equal(expected_tensor, got_tensor)


def test_upgrade_transformer_turns_the_operators_on(monkeypatch):
  """`_Flux2KleinPipeline.from_pretrained` only calls `upgrade_transformer`, so that is the opt-in.

  `tests/api/test_env.py` pins the parsing and both directions of the switch; this pins that the
  FLUX.2 install is what enables the family, without the user setting anything.
  """

  from cache_dit.models.pipelines import env
  from cache_dit.models.pipelines.flux2_klein import fused_ops
  from cache_dit.models.pipelines.flux2_klein.transformer_flux2_klein import upgrade_transformer

  monkeypatch.delenv(env.FUSED_OPS_ENV, raising=False)
  monkeypatch.setattr(fused_ops, "FUSED_OPS_ENABLED", False)

  upgrade_transformer(_tiny_transformer())

  assert fused_ops.FUSED_OPS_ENABLED is True


def test_ops_fall_back_to_eager_without_the_extension(monkeypatch):
  """A missing extension has to degrade to the literal expression, not raise."""

  from cache_dit.models.pipelines.flux2_klein import fused_ops

  monkeypatch.setattr(fused_ops, "FUSED_OPS_ENABLED", True)
  monkeypatch.setattr(fused_ops, "ACTIVE_OPS", frozenset(fused_ops.ALL_OPS))
  # The flag is resolved once at import: calling the loader per block cost two extra dynamo
  # graphs in the block-level compile, so the switch is a module constant the test flips.
  monkeypatch.setattr(fused_ops, "EXTENSION_AVAILABLE", False)

  x = torch.randn(1, 5, DIM)
  one_plus = torch.randn(1, DIM)
  shift = torch.randn(1, DIM)
  assert torch.equal(fused_ops.apply_ln_modulate(x, one_plus, shift, 1e-6),
                     fused_ops._eager_ln_modulate(x, one_plus, shift, 1e-6))

  hidden = torch.randn(1, 5, DIM)
  gate = torch.randn(1, DIM)
  delta = torch.randn(1, 5, DIM)
  got_hidden, got_mod = fused_ops.apply_residual_ln_modulate(hidden, gate, delta, one_plus, shift,
                                                             1e-6)
  exp_hidden, exp_mod = fused_ops._eager_residual_ln_modulate(hidden, gate, delta, one_plus, shift,
                                                              1e-6)
  assert torch.equal(got_hidden, exp_hidden)
  assert torch.equal(got_mod, exp_mod)
