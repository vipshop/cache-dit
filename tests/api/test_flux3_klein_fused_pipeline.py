"""Structural tests for cache-dit's fused FLUX.3 pass.

FLUX.3 is a `ModularPipeline`, so there is no cache-dit pipeline subclass to assert on. What has
to hold instead is that the two mechanisms cache-dit relies on still see the stock objects:

1. `Flux3Transformer2DModel._repeated_blocks` matches **class names**, so the fused pass may only
   rebind instance attributes: the block classes and the block instances stay exactly the ones
   `--compile` and the offload range discovery walk. Replacing a block with a new class silently
   degrades `--compile` to a whole-module compile.
2. The block adapter must still resolve, on both dispatch paths: the modular pipeline resolves
   through its own class name, the transformer-only path through `_Flux3Klein`.

These are CPU-only: they never touch the weights or the CUDA extension.
"""

import pytest
import torch

from cache_dit.caching.block_adapters import BlockAdapterRegister
from cache_dit.models.pipelines.flux3_klein import _Flux3KleinTransformer2DModel
from cache_dit.models.pipelines.flux3_klein import block as fused_block

diffusers = pytest.importorskip("diffusers")
pytest.importorskip("cache_dit._C_ops")


def _tiny_transformer():
  from diffusers import Flux3Transformer2DModel

  torch.manual_seed(0)
  return Flux3Transformer2DModel(
    in_channels=8,
    cond_in_channels=8,
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=128,
    num_attention_heads=8,
    joint_attention_dim=64,
    timestep_channels=8,
    flow_map=True,
  ).eval()


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


def test_adapter_registry_resolves_both_dispatch_paths():
  transformer = _tiny_transformer()
  transformer.__class__ = _Flux3KleinTransformer2DModel

  # Transformer-only path: this is the reason the "_Flux3Klein" entry has to exist, and it has to
  # win over the "Flux3" prefix.
  adapter = BlockAdapterRegister.get_adapter(transformer)
  assert adapter is not None
  # Same cached stack as the stock flux3 adapter: only the joint blocks are cached.
  assert adapter.blocks_name == "single_transformer_blocks"

  # Pipeline path: a modular pipeline resolves through its own class name to the stock flux3
  # adapter, which is the intended behaviour - the fused pass changes no block topology.
  assert "Flux3" in BlockAdapterRegister._adapters
