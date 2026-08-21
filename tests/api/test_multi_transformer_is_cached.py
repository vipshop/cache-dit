"""Regression test for the class-vs-instance `_is_cached` bug.

`CachedAdapter.create_context` sets `_is_cached` on `block_adapter.pipe.__class__`
but `_context_manager` on the instance. Transformer-only usage (no real pipeline)
wraps each transformer in its own `FakeDiffusionPipeline()` instance, but all such
instances share one class -- exactly what happens for multi-DiT pipelines like
MiniMax-H3, which build one BlockAdapter per DiT module. Enabling cache on a second
transformer used to see the first transformer's class-level `_is_cached` flag and
skip creating its own context manager, crashing with an AttributeError.
"""
import gc

import cache_dit
from cache_dit import BlockAdapter, DBCacheConfig, ForwardPattern
from utils import RandTransformer2DModel_Pattern_0_1_2


def test_enable_cache_on_two_independent_transformers():
  gc.collect()

  transformer_a = RandTransformer2DModel_Pattern_0_1_2(pattern=ForwardPattern.Pattern_0)
  transformer_b = RandTransformer2DModel_Pattern_0_1_2(pattern=ForwardPattern.Pattern_0)

  adapter_a = cache_dit.enable_cache(
    BlockAdapter(
      transformer=transformer_a,
      blocks=transformer_a.transformer_blocks,
      forward_pattern=ForwardPattern.Pattern_0,
    ),
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      residual_diff_threshold=0.05,
    ),
  )

  # Second, independent transformer: its BlockAdapter gets its own fresh
  # FakeDiffusionPipeline() instance, but that instance shares a class with
  # transformer_a's. This used to crash with:
  #   AttributeError: 'FakeDiffusionPipeline' object has no attribute '_context_manager'
  adapter_b = cache_dit.enable_cache(
    BlockAdapter(
      transformer=transformer_b,
      blocks=transformer_b.transformer_blocks,
      forward_pattern=ForwardPattern.Pattern_0,
    ),
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      residual_diff_threshold=0.05,
    ),
  )

  assert hasattr(transformer_b, "_context_manager")
  assert transformer_a._context_manager is not transformer_b._context_manager

  cache_dit.disable_cache(adapter_a)
  cache_dit.disable_cache(adapter_b)

  del transformer_a, transformer_b, adapter_a, adapter_b
  gc.collect()
