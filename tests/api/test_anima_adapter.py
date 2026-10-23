"""Regression tests for the Anima cache integration.

Anima is a diffusers modular pipeline that reuses the Cosmos Predict2 DiT
(`CosmosTransformer3DModel`), so it caches through the shared Cosmos `Pattern_2`
branch. The adapter is registered under the `"Anima"` prefix (the registry matches
pipeline class-name prefixes, and `AnimaModularPipeline` matches nothing else), and
`has_separate_cfg` follows the modular guider's condition count instead of a fixed
True. These CPU tests lock the adapter wiring, the repeated-blocks declaration, and
the cache-context bit-exactness on a tiny transformer.
"""
import inspect
import types

import pytest
import torch

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.caching.block_adapters import BlockAdapterRegister
from cache_dit.caching.forward_pattern import ForwardPattern
from cache_dit.utils import declare_repeated_blocks

diffusers = pytest.importorskip("diffusers")

if not hasattr(diffusers, "CosmosTransformer3DModel"):
  pytest.skip("Cosmos requires a newer diffusers", allow_module_level=True)

TEXT_LEN = 8
GRID = 16  # latent 32x32 with patch 2x2 -> 16x16 tokens


def _tiny_transformer():
  from diffusers import CosmosTransformer3DModel

  return CosmosTransformer3DModel(
    in_channels=16,
    out_channels=16,
    num_attention_heads=2,
    attention_head_dim=32,
    num_layers=2,
    mlp_ratio=2.0,
    text_embed_dim=64,
    adaln_lora_dim=32,
    max_size=(8, 32, 32),
    rope_scale=(1.0, 1.0, 1.0),
    extra_pos_embed_type=None,  # the Anima checkpoint ships `null`
  ).eval()


class _AnimaLikePipe:
  """Minimal stand-in for `AnimaModularPipeline`: transformer + modular guider."""

  def __init__(self, transformer, num_conditions: int):
    self.transformer = transformer
    self.guider = types.SimpleNamespace(num_conditions=num_conditions)

  def __call__(self, **kwargs):  # noqa: ARG002 - the cache adapter wraps this
    return (kwargs["hidden_states"], )


def test_anima_prefix_is_the_unique_match():
  prefixes = BlockAdapterRegister.supported_pipelines()[1]
  assert "Anima" in prefixes
  matches = [name for name in prefixes if "AnimaModularPipeline".startswith(name)]
  assert matches == ["Anima"], f"ambiguous adapter prefixes for Anima: {matches}"


@pytest.mark.parametrize(("num_conditions", "expected"), [(2, True), (1, False)])
def test_has_separate_cfg_follows_guider(num_conditions, expected):
  from cache_dit.caching.block_adapters import anima_adapter

  pipe = _AnimaLikePipe(_tiny_transformer(), num_conditions)
  adapter = anima_adapter(pipe)
  assert adapter.has_separate_cfg is expected
  assert adapter.blocks is pipe.transformer.transformer_blocks
  assert adapter.forward_pattern is ForwardPattern.Pattern_2


def test_cosmos_adapter_keeps_separate_cfg_fixed():
  from cache_dit.caching.block_adapters import cosmos_adapter

  # The regular Cosmos pipelines have no modular guider; their adapter must keep the
  # historical hardcoded True instead of deriving `False` from a missing guider.
  pipe = types.SimpleNamespace(transformer=_tiny_transformer())
  adapter = cosmos_adapter(pipe)
  assert adapter.has_separate_cfg is True


def test_declare_repeated_blocks():
  transformer = _tiny_transformer()
  declare_repeated_blocks(transformer)
  assert transformer._repeated_blocks == ["CosmosTransformerBlock"]


def test_cache_run_is_bit_exact_when_no_step_is_cached():
  # max_cached_steps=0 disables caching entirely; the run must stay bit-exact and the
  # patched forward must keep the original signature.
  transformer = _tiny_transformer()
  pipe = _AnimaLikePipe(transformer, num_conditions=2)
  adapter = cache_dit.caching.block_adapters.anima_adapter(pipe)

  generator = torch.Generator().manual_seed(42)
  inputs = dict(
    hidden_states=torch.randn(1, 16, 1, 2 * GRID, 2 * GRID, generator=generator),
    timestep=torch.tensor([0.5]),
    encoder_hidden_states=torch.randn(1, TEXT_LEN, 64, generator=generator),
    padding_mask=torch.zeros(1, 1, 2 * GRID, 2 * GRID),
  )

  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=0,
      max_cached_steps=0,
      num_inference_steps=4,
      enable_separate_cfg=True,
      residual_diff_threshold=0.0,
    ),
  )
  signature = str(inspect.signature(transformer.forward))
  with torch.no_grad():
    first = transformer(**inputs, return_dict=False)[0]
    second = transformer(**inputs, return_dict=False)[0]
  assert torch.equal(first, second)
  assert signature == str(inspect.signature(transformer.forward))
