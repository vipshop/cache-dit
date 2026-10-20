"""Regression tests for the diffusers modular pipeline (FLUX.3) integration.

Two silent failure modes are covered, both CPU-only:

1. `CachedAdapter.mock_transformer` must keep the original `transformer.forward` signature: a
   modular denoise step forwards only the kwargs that appear in `inspect.signature(forward)`, so a
   `(*args, **kwargs)` replacement silently drops `img_ids` / `txt_ids` (and `image_cond_*`).
2. `Flux3TensorParallelismPlanner.rearrange_ffn_weight` must keep the fused SwiGLU gate/up halves
   aligned per TP chunk, otherwise each local SwiGLU mixes gate with gate and the output is
   numerically wrong without any shape error.
"""
import inspect

import pytest
import torch

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.distributed.transformers.flux3 import Flux3TensorParallelismPlanner

REQUIRED_FLUX3_INPUTS = ("hidden_states", "encoder_hidden_states", "timestep", "img_ids", "txt_ids")


def _tiny_flux3_transformer():
  from diffusers import Flux3Transformer2DModel

  return Flux3Transformer2DModel(
    in_channels=8,
    cond_in_channels=8,
    num_layers=1,
    num_single_layers=2,
    attention_head_dim=8,
    num_attention_heads=2,
    joint_attention_dim=16,
    timestep_channels=8,
    mlp_ratio=2.0,
    axes_dims_rope=(2, 2, 2, 2),
  )


def test_flux3_forward_signature_preserved_after_enable_cache():
  transformer = _tiny_flux3_transformer()
  signature_before = inspect.signature(transformer.forward)

  cache_dit.enable_cache(
    transformer,
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      residual_diff_threshold=0.05,
    ),
  )

  signature_after = inspect.signature(transformer.forward)
  assert tuple(signature_after.parameters) == tuple(signature_before.parameters)
  for name in REQUIRED_FLUX3_INPUTS:
    assert name in signature_after.parameters, (f"'{name}' disappeared from the patched "
                                                "transformer.forward signature")

  # a modular denoise step forwards exactly the matching fields, all as keyword arguments
  batch, image_seq_len, text_seq_len = 1, 4, 2
  kwargs = dict(
    hidden_states=torch.randn(batch, image_seq_len, 8),
    encoder_hidden_states=torch.randn(batch, text_seq_len, 16),
    timestep=torch.rand(batch),
    img_ids=torch.zeros(image_seq_len, 4),
    txt_ids=torch.zeros(text_seq_len, 4),
  )
  # img_ids/txt_ids start at 0: RoPE ids must be plain [S, 4] position tables
  kwargs["img_ids"][:, 2] = torch.arange(image_seq_len)
  kwargs["txt_ids"][:, 3] = torch.arange(text_seq_len)

  output = transformer(**kwargs, return_dict=False)
  assert output[0].shape == (batch, image_seq_len, 8)

  cache_dit.disable_cache(transformer)


def test_flux3_ffn_weight_rearrange_keeps_swiglu_aligned():
  from diffusers.models.transformers.transformer_flux3 import Flux3FeedForward

  torch.manual_seed(0)
  dim, mult, tp_size = 8, 2.0, 2
  ff = Flux3FeedForward(dim=dim, dim_out=dim, mult=mult)
  block = torch.nn.Module()
  block.ff = ff

  x = torch.randn(2, 3, dim)
  reference = ff(x)

  original_weight = ff.linear_in.weight.data.clone()
  original_out_weight = ff.linear_out.weight.data.clone()

  planner = Flux3TensorParallelismPlanner()
  planner.rearrange_ffn_weight(block, tp_size)

  # simulate ColwiseParallel(linear_in): each rank holds a contiguous chunk of the output rows
  in_shards = ff.linear_in.weight.data.chunk(tp_size, dim=0)
  # simulate RowwiseParallel(linear_out): each rank holds a contiguous chunk of the input columns
  out_shards = original_out_weight.chunk(tp_size, dim=1)

  partial = []
  for in_shard, out_shard in zip(in_shards, out_shards):
    local = torch.nn.functional.linear(x, in_shard)  # [B, S, 2 * inner / tp]
    half = local.shape[-1] // 2
    local = torch.nn.functional.silu(local[..., :half]) * local[..., half:]
    partial.append(torch.nn.functional.linear(local, out_shard))
  # the row-wise shards are all-reduced (summed) into the full output
  sharded = sum(partial)

  assert torch.allclose(
    sharded, reference,
    atol=1e-6), ("SwiGLU gate/up halves are misaligned after the TP weight rearrangement: "
                 f"max abs diff {(sharded - reference).abs().max().item():.6e}")

  # the rearrangement must be a pure row permutation of the original weight
  assert torch.equal(
    ff.linear_in.weight.data.sort(dim=0).values,
    original_weight.sort(dim=0).values,
  )


QWENIMAGE21_DENOISER_INPUTS = (
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


def test_qwenimage21_modular_denoise_kwargs_survive_enable_cache():
  """A modular denoise step forwards only the kwargs found in `inspect.signature(forward)`."""
  try:
    from diffusers import QwenImage21Transformer2DModel
  except ImportError:
    pytest.skip("Qwen-Image 2.1 requires a newer diffusers")

  transformer = QwenImage21Transformer2DModel(
    in_channels=4,
    context_in_dim=16,
    num_layers=2,
    num_attention_heads=2,
    attention_head_dim=8,
    axes_dims_rope=(2, 2, 4),
    mlp_ratio=2,
  ).eval()

  cache_dit.enable_cache(
    transformer,
    cache_config=DBCacheConfig(
      num_inference_steps=2,
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=2,
      residual_diff_threshold=0.0,
      enable_separate_cfg=False,
    ),
  )

  # this is exactly how `QwenImage21LoopDenoiser` builds `additional_cond_kwargs`
  available = set(inspect.signature(transformer.forward).parameters)
  missing = [name for name in QWENIMAGE21_DENOISER_INPUTS if name not in available]
  assert not missing, f"modular denoise would silently drop {missing}"

  text_len, grid = 3, 4
  img_mask = torch.zeros(1, text_len + grid * grid // 4, dtype=torch.bool)
  img_mask[:, text_len:] = True
  inputs = dict(
    hidden_states=torch.randn(1, grid * grid, 4),
    encoder_hidden_states=torch.randn(1, text_len, 16),
    timestep=torch.tensor([0.5]),
    img_shapes=[[(1, grid, grid)]],
    img_mask=img_mask,
  )
  output = transformer(**{k: v for k, v in inputs.items() if k in available}, return_dict=False)
  # the transformer returns the whole joint sequence, condition prefix included
  assert output[0].shape == (1, text_len + grid * grid, transformer.config.out_channels)

  cache_dit.disable_cache(transformer)


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
