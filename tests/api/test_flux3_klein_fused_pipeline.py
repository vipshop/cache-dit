"""CPU-only structural tests for cache-dit's fused FLUX.3 pass.

FLUX.3 is a `ModularPipeline`, so there is no cache-dit pipeline subclass to assert on, and the
CUDA extension is not available in CI. What has to hold without a GPU:

1. The fused block forwards reproduce the stock blocks bit for bit whenever the operators fall
   back, so a precondition can change the speed but never the values.
2. `Flux3Transformer2DModel._repeated_blocks` matches **class names**, so the fused pass may only
   rebind instance attributes: the block classes and instances stay exactly the ones `--compile`
   and the offload range discovery walk - otherwise `--compile` degrades to a whole-module
   compile without saying so.
3. The block adapter and the parallelism planners resolve for the fused transformer.

`tests/kernels/test_flux3_klein_model_ops.py` covers the operators themselves and needs a GPU.
"""

import inspect

import pytest
import torch

from cache_dit.caching.block_adapters import BlockAdapterRegister
from cache_dit.models.pipelines.flux3_klein import _Flux3KleinTransformer2DModel
from cache_dit.models.pipelines.flux3_klein import block as fused_block
from cache_dit.models.pipelines.flux3_klein import fused_ops

diffusers = pytest.importorskip("diffusers")

BLOCK_MODULE = "diffusers.models.transformers.transformer_flux3"


def _blocks():
  module = __import__(BLOCK_MODULE, fromlist=["Flux3ModeTransformerBlock"])
  return module.Flux3ModeTransformerBlock, module.Flux3SingleTransformerBlock


def _tiny_transformer():
  from diffusers import Flux3Transformer2DModel

  torch.manual_seed(0)
  return Flux3Transformer2DModel(
    in_channels=8,
    cond_in_channels=8,
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=16,
    num_attention_heads=1,
    joint_attention_dim=8,
    timestep_channels=8,
    flow_map=True,
  ).eval()


def _no_kernels(monkeypatch):
  """Force the fallback branch, which is what CI (no `_C_ops`) does anyway.

  The family starts disabled, so `_ops_ready` alone would not be what the assertions below
  measure: the switch would short-circuit first and `apply_*` would equal `_eager_*` for a reason
  that has nothing to do with the extension. Turn the family on so the entry-point check is the
  only thing standing between the call and the kernel.
  """

  monkeypatch.setattr(fused_ops, "FUSED_OPS_ENABLED", True)
  monkeypatch.setattr(fused_ops, "_ops_ready", False)


def test_block_forwards_are_bit_exact_against_the_stock_blocks(monkeypatch):
  _no_kernels(monkeypatch)
  mode_cls, single_cls = _blocks()
  torch.manual_seed(0)
  dim, heads, head_dim = 16, 1, 16

  single = single_cls(dim=dim, num_attention_heads=heads, attention_head_dim=head_dim)
  mode = mode_cls(dim=dim, num_attention_heads=heads, attention_head_dim=head_dim)
  fused_single = single_cls(dim=dim, num_attention_heads=heads, attention_head_dim=head_dim)
  fused_mode = mode_cls(dim=dim, num_attention_heads=heads, attention_head_dim=head_dim)
  fused_single.load_state_dict(single.state_dict())
  fused_mode.load_state_dict(mode.state_dict())
  fused_single.forward = fused_block.fused_single_block_forward.__get__(fused_single)
  fused_mode.forward = fused_block.fused_mode_block_forward.__get__(fused_mode)

  seq, encoder_seq = 7, 3
  # fp32 blocks and inputs: the operators reject anything that is not bf16, so this exercises the
  # fallback branch, which is the branch that has to be bit-exact.
  hidden = torch.randn(1, seq, dim)
  encoder = torch.randn(1, encoder_seq, dim)
  temb = torch.randn(1, seq, 3 * dim) * 0.1
  context_temb = torch.randn(1, encoder_seq, 3 * dim) * 0.1
  joint = torch.randn(encoder_seq + seq, heads * head_dim) * 0.1
  # `use_real_unbind_dim=-1`: the tables are full-width cos/sin, not interleaved pairs.
  cos, sin = joint.cos(), joint.sin()
  rotary = (cos, sin)
  single_rotary = (cos[encoder_seq:], sin[encoder_seq:])

  with torch.no_grad():
    reference = mode(hidden, temb, single_rotary)
    actual = fused_mode(hidden, temb, single_rotary)
    assert torch.equal(actual, reference)

    reference = single(hidden, encoder, temb, context_temb, rotary)
    actual = fused_single(hidden, encoder, temb, context_temb, rotary)
  assert torch.equal(actual[0], reference[0])
  assert torch.equal(actual[1], reference[1])


def test_fused_forwards_keep_the_upstream_signatures():
  _, single_cls = _blocks()
  for name, fused_fn in (("fused_mode_block_forward", fused_block.fused_mode_block_forward),
                         ("fused_single_block_forward", fused_block.fused_single_block_forward)):
    stock = (single_cls.forward if name.endswith("single_block_forward") else _blocks()[0].forward)
    assert list(inspect.signature(fused_fn).parameters) == list(inspect.signature(stock).parameters)


def test_fused_pass_keeps_the_class_names_repeated_blocks_matches():
  transformer = _tiny_transformer()
  repeated = list(transformer._repeated_blocks)
  joint_class = type(transformer.single_transformer_blocks[0]).__name__

  transformer.__class__ = _Flux3KleinTransformer2DModel
  transformer.apply_fused_ops()

  assert list(transformer._repeated_blocks) == repeated
  assert joint_class in repeated
  # Instance rebinding only: the class, and therefore name-based discovery, is untouched.
  assert type(transformer.single_transformer_blocks[0]).__name__ == joint_class
  for block in transformer.single_transformer_blocks:
    assert block.__dict__["forward"].__func__ is fused_block.fused_single_block_forward


def test_remove_fused_ops_removes_the_instance_overrides():
  transformer = _tiny_transformer()
  transformer.__class__ = _Flux3KleinTransformer2DModel
  transformer.apply_fused_ops()
  transformer.remove_fused_ops()

  for block in transformer.single_transformer_blocks:
    assert "forward" not in block.__dict__
  assert not getattr(transformer, "_flux3_klein_fused_installed", False)


def test_operators_fall_back_to_the_stock_expression_without_the_extension(monkeypatch):
  _no_kernels(monkeypatch)
  dim = 16
  x = torch.randn(1, 4, dim, dtype=torch.bfloat16)
  scale = torch.randn(1, 4, dim, dtype=torch.bfloat16) * 0.1
  shift = torch.randn(1, 4, dim, dtype=torch.bfloat16) * 0.1
  fused_ops.reset_fused_ops_stats()

  assert torch.equal(fused_ops.apply_ln_modulate(x, scale, shift, 1e-6),
                     fused_ops._eager_ln_modulate(x, scale, shift, 1e-6))
  hidden = torch.randn(1, 4, dim, dtype=torch.bfloat16)
  attn = torch.randn(1, 4, dim, dtype=torch.bfloat16)
  ff = torch.randn(1, 4, dim, dtype=torch.bfloat16)
  gate = torch.randn(1, 4, dim, dtype=torch.bfloat16) * 0.1
  assert torch.equal(fused_ops.apply_gated_residual(hidden, gate, attn, ff),
                     fused_ops._eager_gated_residual(hidden, gate, attn, ff))

  stats = fused_ops.fused_ops_stats()
  assert stats["hits"]["ln_modulate"] == 0
  assert stats["fallbacks"]["ln_modulate"] == 1 and stats["fallbacks"]["gated_residual"] == 1


def test_install_turns_the_operators_on_except_when_the_switch_is_off(monkeypatch):
  """FLUX.3 has no cache-dit pipeline class, so `apply_fused_ops` is the whole opt-in.

  `tests/api/test_env.py` pins the parsing; this pins that the install is what enables the family,
  and that `CACHE_DIT_FUSED_OPS=0` survives it.
  """

  from cache_dit.models.pipelines import env

  monkeypatch.delenv(env.FUSED_OPS_ENV, raising=False)
  monkeypatch.setattr(fused_ops, "FUSED_OPS_ENABLED", False)
  transformer = _tiny_transformer()
  transformer.__class__ = _Flux3KleinTransformer2DModel
  transformer.apply_fused_ops()
  assert fused_ops.FUSED_OPS_ENABLED is True

  monkeypatch.setenv(env.FUSED_OPS_ENV, "0")
  monkeypatch.setattr(fused_ops, "FUSED_OPS_ENABLED", False)
  disabled = _tiny_transformer()
  disabled.__class__ = _Flux3KleinTransformer2DModel
  disabled.apply_fused_ops()
  assert fused_ops.FUSED_OPS_ENABLED is False


def test_the_allow_list_is_a_code_constant_not_a_variable():
  """`CACHE_DIT_FUSED_OPS` is the only variable; the shipped subset is the measured winner set.

  The dispatch side of the same claim (a narrowed allow-list really skips the kernel) needs a GPU
  and lives in `tests/kernels/test_flux3_klein_model_ops.py`.
  """

  assert fused_ops.ACTIVE_OPS == frozenset(fused_ops.ALL_OPS)
  assert set(fused_ops.ALL_OPS) == {"qk_norm_rope", "ln_modulate", "gated_residual"}


def test_adapter_and_planners_resolve_for_the_fused_transformer():
  transformer = _tiny_transformer()
  transformer.__class__ = _Flux3KleinTransformer2DModel

  # Transformer-only path: this is the reason the "_Flux3Klein" entry has to exist, and it has to
  # win over the "Flux3" prefix.
  adapter = BlockAdapterRegister.get_adapter(transformer)
  assert adapter is not None
  assert adapter.blocks_name == "single_transformer_blocks"

  import cache_dit.distributed.transformers.flux3  # noqa: F401 - registers the planners
  from cache_dit.distributed.transformers.register import (ContextParallelismPlannerRegister,
                                                           TensorParallelismPlannerRegister)

  for register in (ContextParallelismPlannerRegister, TensorParallelismPlannerRegister):
    # The registry matches class names by prefix; without these entries parallelism raises.
    assert register.get_planner("_Flux3KleinTransformer2DModel") is not None
