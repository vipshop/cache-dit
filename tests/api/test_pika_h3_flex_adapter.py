"""Regression tests for the Pika H3 Flex adapter wiring (no weights, no GPU).

Pika H3 Flex runs the MiniMax-H3 transformer, but its pipeline class is vendored, so every dispatch
that keys off a class name has to be pointed at it. The silent failure modes covered here:

1. `get_adapter` matches the *pipeline* class-name prefix. A vendored `PikaH3FlexModularPipeline`
   does not start with `MiniMaxH3`, so without its own registration `enable_cache` finds no adapter
   and caching silently does nothing.
2. A modular pipeline outside `_verified_modular_pipelines` is dispatched, but cache-dit logs a
   "not verified" warning and nothing else guards the dispatch.
3. `t2va` / `fl2va` load `transformer/` and `ref2va` loads `transformer_ref/`, so the adapter must
   resolve whichever partition the workflow pruned in.
"""
from cache_dit.caching.block_adapters.block_registers import BlockAdapterRegister
from cache_dit.models.pipelines.pika_h3_flex import PikaH3FlexBlocks


def test_adapter_registered_under_the_pipeline_prefix():
  assert "PikaH3Flex" in BlockAdapterRegister._adapters


def test_modular_pipeline_is_whitelisted():
  assert "PikaH3Flex" in BlockAdapterRegister._verified_modular_pipelines


def test_workflow_selects_its_own_transformer_partition():
  blocks = PikaH3FlexBlocks()
  assert set(blocks.available_workflows) == {"t2va", "fl2va", "ref2va"}

  for workflow in ("t2va", "fl2va"):
    components = {spec.name for spec in blocks.get_workflow(workflow).expected_components}
    assert "transformer" in components
    assert "transformer_ref" not in components

  components = {spec.name for spec in blocks.get_workflow("ref2va").expected_components}
  assert "transformer_ref" in components
  assert "transformer" not in components
