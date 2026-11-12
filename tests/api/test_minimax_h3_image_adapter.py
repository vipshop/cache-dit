"""CPU-only guards for the MiniMax-H3-Image block adapter.

Two silent failures are covered:

1. **Registry order.** `BlockAdapterRegister` returns the first class-name prefix that matches and
   `MiniMaxH3ImageModularPipeline` also starts with "MiniMaxH3". If the image entry sits after the
   video one, dispatch lands on `minimax_h3_adapter`, whose `_relaxed_assert(transformer,
   MiniMaxH3Transformer3DModel)` is a hard assertion for a diffusers transformer.
2. **Block contract.** The transformer runs one packed sequence through `transformer_blocks` and each
   block returns `hidden_states` only, which is `Pattern_3`. The per-step output heads sit outside
   the stack, so they must not be part of the cached block list.
"""

from types import SimpleNamespace

import torch

from cache_dit.caching.block_adapters.adapters import minimax_h3_image_adapter
from cache_dit.caching.block_adapters.block_registers import BlockAdapterRegister
from cache_dit.caching.forward_pattern import ForwardPattern


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


def test_adapter_is_registered_before_minimax_h3():
  names = BlockAdapterRegister.supported_pipelines()[1]
  assert "MiniMaxH3Image" in names
  assert names.index("MiniMaxH3Image") < names.index("MiniMaxH3"), (
    "the registry matches the first class-name prefix, so the image entry has to precede the video "
    "one: `MiniMaxH3ImageModularPipeline` also starts with 'MiniMaxH3'")


def test_adapter_caches_the_block_stack_only():
  transformer = _tiny_transformer()
  adapter = minimax_h3_image_adapter(SimpleNamespace(transformer=transformer))

  assert adapter.transformer is transformer
  assert adapter.blocks is transformer.transformer_blocks
  assert adapter.forward_pattern is ForwardPattern.Pattern_3
  # guidance is distilled into the weights: one forward pass per step
  assert adapter.has_separate_cfg is False
  assert adapter.check_forward_pattern is True


def test_per_step_output_heads_are_outside_the_cached_stack():
  transformer = _tiny_transformer()
  assert len(transformer.pdd_proj_out) == transformer.config.num_pdd_heads
  cached = {id(block) for block in transformer.transformer_blocks}
  for head in transformer.pdd_proj_out:
    assert id(head) not in cached
