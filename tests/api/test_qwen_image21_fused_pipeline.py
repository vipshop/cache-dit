"""Contract tests for cache-dit's ``_qwen_image21`` optimisation pass.

The tests are CPU-only and need no checkpoint: they lock the registration names, the
functor wiring and the bit-exactness of the fused block against the stock diffusers block,
which is where the silent failure modes live (a skipped patch leaves every layer reading
layer 0's KV cache and only shows up as a noisy image).
"""

import inspect

import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from diffusers import QwenImage21Transformer2DModel  # noqa: E402
from diffusers.models.transformers.transformer_qwenimage21 import (  # noqa: E402
  QwenImage21TransformerBlock, )

from cache_dit.caching.block_adapters.block_registers import BlockAdapterRegister  # noqa: E402
from cache_dit.caching.forward_pattern import ForwardPattern  # noqa: E402
from cache_dit.caching.patch_functors import QwenImage21PatchFunctor  # noqa: E402
from cache_dit.distributed.transformers.register import (  # noqa: E402
  ContextParallelismPlannerRegister, TensorParallelismPlannerRegister,
)


def _tiny_transformer() -> QwenImage21Transformer2DModel:
  return QwenImage21Transformer2DModel(
    patch_size=1,
    in_channels=4,
    out_channels=4,
    num_layers=2,
    attention_head_dim=8,
    num_attention_heads=2,
    context_in_dim=16,
    mlp_ratio=2,
    axes_dims_rope=(2, 3, 3),
  )


def test_adapter_is_registered_for_the_underscore_pipeline():
  """`enable_cache` matches pipelines by class-name prefix, so the `_` name needs its own key."""

  names = list(BlockAdapterRegister._adapters)
  assert "_QwenImage21" in names
  # The stock key must still be reachable and still be dispatched before the generic one.
  assert names.index("QwenImage21") < names.index("QwenImage")


def test_cp_and_tp_planners_are_registered_for_the_underscore_names():
  from cache_dit.distributed.transformers.qwen_image21 import (
    QwenImage21ContextParallelismPlanner,
    QwenImage21TensorParallelismPlanner,
  )
  from cache_dit.models.pipelines.qwen_image21 import _QwenImage21Transformer2DModel
  from cache_dit.models.pipelines.qwen_image21.functor import _QwenImage21PatchFunctor

  cp_planner = ContextParallelismPlannerRegister.get_planner("_QwenImage21Transformer2DModel")
  tp_planner = TensorParallelismPlannerRegister.get_planner(_QwenImage21Transformer2DModel.__name__)

  assert issubclass(cp_planner, QwenImage21ContextParallelismPlanner)
  assert tp_planner is QwenImage21TensorParallelismPlanner
  # CP has to go through the fused functor, otherwise the CP split/gather is never installed.
  assert isinstance(cp_planner().patch_functor(), _QwenImage21PatchFunctor)
  assert "_QwenImage21" in ContextParallelismPlannerRegister._cp_planner_registry
  assert "_QwenImage21Transformer2DModel" in TensorParallelismPlannerRegister._tp_planner_registry


def test_stock_functor_would_skip_the_subclass_and_fused_functor_does_not():
  """The reason `_QwenImage21PatchFunctor` overrides `is_from_diffusers`.

  The base check looks at `transformer.__module__`, which is cache-dit's for the subclass,
  so the stock functor returns without patching — silently.
  """

  from cache_dit.models.pipelines.qwen_image21 import _QwenImage21Transformer2DModel
  from cache_dit.models.pipelines.qwen_image21.functor import _QwenImage21PatchFunctor

  transformer = _tiny_transformer()
  transformer.__class__ = _QwenImage21Transformer2DModel

  assert QwenImage21PatchFunctor.is_from_diffusers(transformer) is False
  assert _QwenImage21PatchFunctor.is_from_diffusers(transformer) is True

  QwenImage21PatchFunctor().apply(transformer)
  assert getattr(transformer, "_is_patched", False) is False

  _QwenImage21PatchFunctor().apply(transformer)
  assert getattr(transformer, "_is_patched", False) is True
  assert transformer.transformer_blocks[0].forward.__name__ == "fused_block_forward"
  assert transformer.transformer_blocks[1]._idx == 1


def test_apply_fused_ops_is_idempotent_and_leaves_the_cache_patch_alone():
  from cache_dit.models.pipelines.qwen_image21 import _QwenImage21Transformer2DModel
  from cache_dit.models.pipelines.qwen_image21.transformer_qwen_image21 import upgrade_transformer

  transformer = _tiny_transformer()
  upgraded = upgrade_transformer(transformer)

  assert isinstance(upgraded, _QwenImage21Transformer2DModel)
  assert upgraded is transformer, "upgrade must be in place so no weights are re-created"
  assert upgraded._qwen21_fused_installed is True
  assert getattr(upgraded, "_is_patched", False) is False

  before = upgraded.transformer_blocks[0].forward
  upgraded.apply_fused_ops()
  assert upgraded.transformer_blocks[0].forward == before


def test_fused_block_forward_keeps_the_forward_pattern_signature():
  """`_check_forward_pattern` inspects names and the return-annotation tensor count."""

  from cache_dit.models.pipelines.qwen_image21.block import fused_block_forward

  params = set(inspect.signature(fused_block_forward).parameters)
  for required in ForwardPattern.Pattern_3.In:
    assert required in params, f"missing forward-pattern parameter: {required}"
  assert str(inspect.signature(fused_block_forward).return_annotation).count("torch.Tensor") == len(
    ForwardPattern.Pattern_3.Out)


def test_fused_block_matches_stock_block_bit_exactly():
  from cache_dit.models.pipelines.qwen_image21.block import fused_block_forward

  torch.manual_seed(0)
  block = QwenImage21TransformerBlock(dim=16,
                                      num_attention_heads=2,
                                      attention_head_dim=8,
                                      mlp_ratio=2).to(torch.bfloat16)
  fused = QwenImage21TransformerBlock(dim=16,
                                      num_attention_heads=2,
                                      attention_head_dim=8,
                                      mlp_ratio=2).to(torch.bfloat16)
  fused.load_state_dict(block.state_dict())
  fused._idx = 0
  fused.forward = fused_block_forward.__get__(fused)

  hidden = torch.randn(1, 7, 16, dtype=torch.bfloat16)
  modulation = torch.randn(2, 64, dtype=torch.bfloat16)
  mask = torch.zeros(7, dtype=torch.bool)
  mask[:3] = True
  rotary = torch.polar(torch.ones(7, 4), torch.randn(7, 4))

  expected = block(hidden, modulation, rotary_emb=rotary, target_token_mask=mask)
  got = fused(hidden, modulation, rotary_emb=rotary, target_token_mask=mask)
  assert torch.equal(expected, got)


def test_fused_block_resolves_a_layer_cache_container():
  """On the cached path the sixth argument is the container, not a single layer."""

  from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

  from cache_dit.models.pipelines.qwen_image21.block import fused_block_forward

  torch.manual_seed(0)
  block = QwenImage21TransformerBlock(dim=16,
                                      num_attention_heads=2,
                                      attention_head_dim=8,
                                      mlp_ratio=2).to(torch.bfloat16)
  block._idx = 1
  block.forward = fused_block_forward.__get__(block)

  hidden = torch.randn(1, 7, 16, dtype=torch.bfloat16)
  modulation = torch.randn(2, 64, dtype=torch.bfloat16)
  rotary = torch.polar(torch.ones(7, 4), torch.randn(7, 4))
  caches = QwenImage21KVCache(2)

  # `extract` stores layer 1's prefix; a wrong resolution would write layer 0.
  block(
    hidden,
    modulation,
    rotary_emb=rotary,
    kv_caches=caches,
    kv_cache_mode="extract",
    cache_write_slice=slice(0, 3),
  )
  assert caches.get_layer(0).k is None
  assert caches.get_layer(1).k is not None
  assert caches.get_layer(1).k.shape[1] == 3


def test_switch_only_turns_off_on_explicit_falsey_values(monkeypatch):
  from cache_dit.models.pipelines.qwen_image21 import fused_ops

  for value, expected in (("0", False), ("false", False), ("off", False), ("no", False),
                          ("1", True), ("true", True), ("", True), ("yes", True)):
    monkeypatch.setenv("CACHE_DIT_QWEN_IMAGE21_FUSED_OPS", value)
    assert fused_ops._switch_from_env() is expected

  monkeypatch.delenv("CACHE_DIT_QWEN_IMAGE21_FUSED_OPS")
  assert fused_ops._switch_from_env() is True


def test_ops_fall_back_to_eager_without_the_extension(monkeypatch):
  """A missing CUDA extension must degrade to the literal expression, not raise."""

  from cache_dit.models.pipelines.qwen_image21 import fused_ops

  monkeypatch.setattr(fused_ops, "FUSED_OPS_ENABLED", True)
  monkeypatch.setattr(fused_ops, "fused_ops_extension_available", lambda: False)

  x = torch.randn(1, 5, 3, 8)
  weight = torch.randn(8)
  rotary = torch.polar(torch.ones(5, 4), torch.randn(5, 4))
  got = fused_ops.apply_qk_norm_rope(x, weight, rotary, 1e-6)
  assert torch.equal(got, fused_ops._eager_qk_norm_rope(x, weight, rotary, 1e-6))

  hidden = torch.randn(1, 5, 8)
  params = torch.randn(2, 16)
  modulated, gate = fused_ops.apply_ln_modulate(hidden, params, None, 1e-6)
  expected, expected_gate = fused_ops._eager_ln_modulate(hidden, params, None, 1e-6)
  assert torch.equal(modulated, expected)
  # `gate` is the folded `(R, D)` row, not an expanded `(B, S, D)` tensor.
  assert torch.equal(gate, expected_gate)


def test_qkv_seam_is_patched_for_stock_flex_and_context_parallel():
  """Fusion must be installed on the function every processor variant calls, not on a class.

  Subclassing the processor was the alternative and it silently disabled fusion under context
  parallelism, because cache-dit replaces `QwenImage21AttnProcessor.__call__` wholesale.
  """

  import importlib

  from cache_dit.models.pipelines.qwen_image21 import attention_processor as ap

  diffusers_module = importlib.import_module(
    "diffusers.models.transformers.transformer_qwenimage21")
  cp_module = importlib.import_module("cache_dit.distributed.transformers.qwen_image21")

  # Start from a clean state: another test in this file may have installed the patch already.
  ap.uninstall_qkv_patch()
  original = diffusers_module._qwenimage21_prepare_qkv
  assert original is not ap._fused_prepare_qkv
  try:
    ap.install_qkv_patch()
    assert diffusers_module._qwenimage21_prepare_qkv is ap._fused_prepare_qkv
    # The CP path imported the symbol into its own module namespace, so it needs the same patch.
    if hasattr(cp_module, "_qwenimage21_prepare_qkv"):
      assert cp_module._qwenimage21_prepare_qkv is ap._fused_prepare_qkv
    assert ap.qkv_patch_is_installed()

    ap.uninstall_qkv_patch()
    assert diffusers_module._qwenimage21_prepare_qkv is original
    assert not ap.qkv_patch_is_installed()
  finally:
    ap.uninstall_qkv_patch()
    diffusers_module._qwenimage21_prepare_qkv = original
