"""Regression tests for the Qwen-Image 2.1 cache integration.

Qwen-Image 2.1 keeps a prefix KV cache indexed by the transformer block, and cache-dit replaces
`transformer.transformer_blocks` with a single-element wrapper while the cache is active. Without
the `QwenImage21PatchFunctor` the block loop therefore runs once with `index_block == 0` and every
layer reuses layer 0's K/V: no crash, but the image collapses to color noise. These CPU tests lock
the patch (per-layer cache, preserved forward signature, unchanged numerics) as well as the
`Pattern_3` adapter wiring.
"""
import copy
import inspect

import pytest
import torch

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.caching.block_adapters import BlockAdapterRegister
from cache_dit.caching.forward_pattern import ForwardPattern
from cache_dit.caching.patch_functors import QwenImage21PatchFunctor

diffusers = pytest.importorskip("diffusers")

if not hasattr(diffusers, "QwenImage21Transformer2DModel"):
  pytest.skip("Qwen-Image 2.1 requires a newer diffusers", allow_module_level=True)

TEXT_LEN = 5
GRID_H = GRID_W = 8
TARGET_TOKENS = GRID_H * GRID_W
IMAGE_SLOTS = TARGET_TOKENS // 4

REQUIRED_INPUTS = (
  "hidden_states",
  "encoder_hidden_states",
  "timestep",
  "img_shapes",
  "img_mask",
  "encoder_hidden_states_mask",
  "attention_kwargs",
  "kv_cache",
  "kv_cache_mode",
)


def _tiny_transformer():
  from diffusers import QwenImage21Transformer2DModel

  return QwenImage21Transformer2DModel(
    in_channels=4,
    context_in_dim=16,
    num_layers=2,
    num_attention_heads=2,
    attention_head_dim=8,
    axes_dims_rope=(2, 2, 4),
    mlp_ratio=2,
  ).eval()


def _inputs():
  generator = torch.Generator(device="cpu").manual_seed(42)
  img_mask = torch.zeros(1, TEXT_LEN + IMAGE_SLOTS, dtype=torch.bool)
  img_mask[:, TEXT_LEN:] = True
  return dict(
    hidden_states=torch.randn(1, TARGET_TOKENS, 4, generator=generator),
    encoder_hidden_states=torch.randn(1, TEXT_LEN, 16, generator=generator),
    img_shapes=[[(1, GRID_H, GRID_W)]],
    img_mask=img_mask,
  )


def _run_steps(model, inputs, num_steps=4, use_kv_cache=True):
  from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

  kv_cache = QwenImage21KVCache(len(model.transformer_blocks)) if use_kv_cache else None
  outputs = []
  for step in range(num_steps):
    mode = None
    if kv_cache is not None:
      mode = "extract" if step == 0 else "cached"
    outputs.append(
      model(
        **inputs,
        timestep=torch.tensor([1.0 - step / num_steps]),
        kv_cache=kv_cache,
        kv_cache_mode=mode,
        return_dict=False,
      )[0])
  return outputs


def test_qwenimage21_adapter_is_registered_before_qwenimage():
  # QwenImage21* also matches the "QwenImage" prefix; the registry keeps insertion order.
  names = list(BlockAdapterRegister._adapters)
  assert names.index("QwenImage21") < names.index("QwenImage")

  transformer = _tiny_transformer()
  adapter = BlockAdapterRegister.get_adapter(transformer)
  assert adapter.forward_pattern == ForwardPattern.Pattern_3
  assert adapter.patch_functor is not None
  assert adapter.blocks is transformer.transformer_blocks


def test_per_layer_prefix_cache_survives_the_cached_wrapper():
  from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

  transformer = _tiny_transformer()
  cache_dit.enable_cache(
    transformer,
    cache_config=DBCacheConfig(
      num_inference_steps=4,
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=4,
      residual_diff_threshold=0.0,
      enable_separate_cfg=False,
    ),
  )

  kv_cache = QwenImage21KVCache(2)
  transformer(**_inputs(),
              timestep=torch.tensor([1.0]),
              kv_cache=kv_cache,
              kv_cache_mode="extract",
              return_dict=False)

  layer0, layer1 = kv_cache.get_layer(0), kv_cache.get_layer(1)
  assert layer0.k is not None and layer1.k is not None
  assert layer0.k.shape[1] == TEXT_LEN
  assert not torch.equal(layer0.k, layer1.k), "every block shares layer 0's prefix KV cache"


def test_patched_forward_matches_reference():
  reference = _tiny_transformer()
  patched = copy.deepcopy(reference)
  adapter = BlockAdapterRegister.get_adapter(patched)
  assert adapter is not None

  signature = inspect.signature(patched.forward)
  for name in REQUIRED_INPUTS:
    assert name in signature.parameters, f"'{name}' disappeared from the patched forward signature"

  for use_kv_cache in (False, True):
    inputs = _inputs()
    ref = _run_steps(reference, inputs, use_kv_cache=use_kv_cache)
    got = _run_steps(patched, inputs, use_kv_cache=use_kv_cache)
    for step, (expected, actual) in enumerate(zip(ref, got)):
      torch.testing.assert_close(
        actual,
        expected,
        atol=0,
        rtol=0,
        msg=(f"patched forward diverged at step {step} (use_kv_cache={use_kv_cache})"))


def _cp_planner():
  from cache_dit.distributed.transformers.qwen_image21 import QwenImage21ContextParallelismPlanner

  return QwenImage21ContextParallelismPlanner()


def test_cp_planner_rejects_ring_and_ulysses_anything():
  from cache_dit import ParallelismConfig

  transformer = _tiny_transformer()

  with pytest.raises(NotImplementedError, match="Ring"):
    _cp_planner()._apply(transformer, ParallelismConfig(ring_size=2))

  with pytest.raises(NotImplementedError, match="ulysses_anything"):
    _cp_planner()._apply(transformer, ParallelismConfig(ulysses_size=2, ulysses_anything=True))


def test_cp_planner_rejects_flex_attention_processor():
  from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21FlexAttnProcessor

  from cache_dit import ParallelismConfig

  transformer = _tiny_transformer()
  for block in transformer.transformer_blocks:
    block.attn.set_processor(QwenImage21FlexAttnProcessor())

  with pytest.raises(ValueError, match="FlexAttnProcessor|flex"):
    _cp_planner()._apply(transformer, ParallelismConfig(ulysses_size=2))


def test_cp_planner_accepts_ulysses_and_patches_forward(monkeypatch):
  from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor

  import cache_dit.distributed.transformers.qwen_image21 as cp_module
  from cache_dit import ParallelismConfig

  # the processor patch is process wide, undo it for the rest of the suite
  original_call = QwenImage21AttnProcessor.__call__
  monkeypatch.setattr(QwenImage21AttnProcessor, "__call__", original_call)
  monkeypatch.setattr(cp_module, "_ATTN_PROCESSOR_PATCHED", False)

  transformer = _tiny_transformer()
  plan = _cp_planner()._apply(transformer, ParallelismConfig(ulysses_size=2))

  # the split lives in the patch functor, the hook plan stays empty
  assert plan == {}
  assert getattr(transformer, "_is_patched", False)
  assert [block._idx for block in transformer.transformer_blocks] == [0, 1]
  assert cp_module._ATTN_PROCESSOR_PATCHED
  assert QwenImage21AttnProcessor.__call__.__name__ == "cp_call"


def test_functor_keeps_state_dict_keys_unchanged():
  transformer = _tiny_transformer()
  before = set(transformer.state_dict())

  QwenImage21PatchFunctor().apply(transformer)

  assert set(transformer.state_dict()) == before, "patched blocks were registered a second time"


def test_cp_processor_matches_stock_processor_when_not_sharded(monkeypatch):
  """With a single rank the CP path must be numerically identical to the stock processor."""
  from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor

  import cache_dit.distributed.transformers.qwen_image21 as cp_module

  original_call = QwenImage21AttnProcessor.__call__
  monkeypatch.setattr(QwenImage21AttnProcessor, "__call__", original_call)
  monkeypatch.setattr(cp_module, "_ATTN_PROCESSOR_PATCHED", False)
  dispatch_calls = []
  original_dispatch = cp_module._dispatch_attention_fn

  def tracked_dispatch(*args, **kwargs):
    dispatch_calls.append(kwargs)
    return original_dispatch(*args, **kwargs)

  monkeypatch.setattr(cp_module, "_dispatch_attention_fn", tracked_dispatch)

  class _IdentityHandle:

    def __init__(self, x):
      self.x = x

    def wait(self):
      return self.x

  class _IdentityComm:

    def init_meta(self, x):
      return self

    def send_q(self, x):
      return _IdentityHandle(x)

    def send_k(self, x):
      return _IdentityHandle(x)

    def send_v(self, x):
      return _IdentityHandle(x)

    def send_o(self, x):
      return _IdentityHandle(x)

    def all_gather_tensor_dim(self, x, dim):
      return x

  transformer = _tiny_transformer()
  QwenImage21PatchFunctor().apply(transformer)
  cp_module._patch_attn_processor_for_cp()

  attn = transformer.transformer_blocks[0].attn
  hidden_states = torch.randn(1, TEXT_LEN + TARGET_TOKENS, attn.inner_dim)
  segments = [(0, TEXT_LEN, True)]

  processor = QwenImage21AttnProcessor()
  expected = original_call(processor, attn, hidden_states, segments=segments)
  actual = QwenImage21AttnProcessor()(
    attn,
    hidden_states,
    segments=segments,
    cp_meta=(_IdentityComm(), 0, 1),
  )

  torch.testing.assert_close(actual, expected, atol=0, rtol=0)
  assert len(dispatch_calls) == 2
  assert all(call["parallel_config"] is None for call in dispatch_calls)
  with pytest.raises(NotImplementedError, match="local attention head count"):
    QwenImage21AttnProcessor()(
      attn,
      hidden_states,
      segments=segments,
      cp_meta=(_IdentityComm(), 0, 3),
    )


def test_cached_warmup_is_bit_exact():
  reference = _tiny_transformer()
  cached = copy.deepcopy(reference)
  cache_dit.enable_cache(
    cached,
    cache_config=DBCacheConfig(
      num_inference_steps=4,
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=4,
      residual_diff_threshold=0.0,
      enable_separate_cfg=False,
    ),
  )

  inputs = _inputs()
  ref = _run_steps(reference, inputs)
  got = _run_steps(cached, inputs)
  for step, (expected, actual) in enumerate(zip(ref, got)):
    torch.testing.assert_close(actual,
                               expected,
                               atol=0,
                               rtol=0,
                               msg=f"warmup diverged at step {step}")

  stats = cache_dit.summary(cached)
  assert stats[0].accumulated_executed_steps == 4
