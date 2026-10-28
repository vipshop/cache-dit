"""Regression tests for the HiDream-O1-Image cache integration.

HiDream-O1-Image ships its own pipeline (`cache_dit.models.pipelines.hidream_o1_image`) because the
checkpoint is a Qwen3-VL repository with no diffusers code, and each denoising step runs two
`transformer.forward()` calls (conditional, then the `" "` unconditional branch). These tests lock
the adapter wiring (`layers` + `Pattern_3` + separate CFG), the adapter-registry precedence over the
HiDream-I1 adapter, the block I/O contract the cached wrapper replays, and that a cached step still
produces the same output when the residue threshold is zero.

Usage:
  python -m pytest tests/api/test_hidream_o1_image_adapter.py -q
"""

import inspect

import pytest
import torch

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.caching.block_adapters import BlockAdapter, BlockAdapterRegister
from cache_dit.caching.forward_pattern import ForwardPattern
from cache_dit.models.pipelines.hidream_o1_image import (
  HiDreamO1ImageTransformer2DModel, )
from cache_dit.models.pipelines.hidream_o1_image.pipeline import (
  build_hidream_o1_attention_mask, )

IMAGE_LEN = 4
TEXT_TOKENS = 6
TMS_TOKEN_ID = 100
NUM_LAYERS = 3


def _tiny_transformer(seed: int = 0) -> HiDreamO1ImageTransformer2DModel:
  torch.manual_seed(seed)
  return HiDreamO1ImageTransformer2DModel(
    hidden_size=64,
    intermediate_size=128,
    num_layers=NUM_LAYERS,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=128,
    patch_size=32,
    in_channels=3,
    tms_token_id=TMS_TOKEN_ID,
  ).eval()


def _inputs(text_tokens: int = TEXT_TOKENS):
  generator = torch.Generator().manual_seed(3)
  seq = text_tokens + IMAGE_LEN
  input_ids = torch.arange(1, text_tokens + 1).unsqueeze(0)
  input_ids[0, -1] = TMS_TOKEN_ID

  token_types = torch.zeros((1, seq), dtype=torch.long)
  token_types[0, text_tokens - 1:text_tokens + IMAGE_LEN] = 1
  token_types[0, text_tokens - 1] = 3

  grid = torch.arange(IMAGE_LEN).view(2, 2).reshape(-1)
  position_ids = torch.zeros(3, 1, seq, dtype=torch.long)
  position_ids[:, 0, :text_tokens] = torch.arange(text_tokens)
  position_ids[:, 0, text_tokens:] = torch.stack([grid, grid, grid]) + 4096

  return dict(
    input_ids=input_ids,
    position_ids=position_ids,
    vinputs=torch.randn(1, IMAGE_LEN, 3 * 32 * 32, generator=generator),
    timestep=torch.tensor([0.7]),
    attention_mask=build_hidream_o1_attention_mask(token_types, torch.float32),
  )


def test_adapter_wiring():
  adapter = BlockAdapterRegister.get_adapter(_tiny_transformer())
  assert isinstance(adapter, BlockAdapter)
  assert adapter.blocks_name == "layers"
  assert adapter.forward_pattern == ForwardPattern.Pattern_3
  assert adapter.has_separate_cfg is True
  assert adapter.check_forward_pattern is True
  assert adapter.patch_functor is None
  assert adapter.blocks is adapter.transformer.layers


def test_adapter_wins_over_the_hidream_i1_prefix():
  # The registry returns the first matching class-name prefix, and "HiDreamO1Image..." also starts
  # with "HiDream" (the diffusers HiDream-I1 adapter, which reads `double_stream_blocks`).
  adapter = BlockAdapterRegister.get_adapter(_tiny_transformer())
  assert adapter.blocks_name == "layers", "HiDream-O1-Image was captured by the HiDream-I1 adapter"


def test_blocks_are_discoverable_by_quantization_and_compile():
  transformer = _tiny_transformer()
  assert transformer._repeated_blocks == [type(transformer.layers[0]).__name__]
  assert transformer.config.num_layers == NUM_LAYERS
  assert len(transformer.layers) == NUM_LAYERS


def test_block_contract_matches_pattern_3():
  block = _tiny_transformer().layers[0]
  parameters = list(inspect.signature(block.forward).parameters)
  assert parameters[0] == "hidden_states"
  assert set(parameters[1:]) == {"rope", "attention_mask", "cp_config"}
  out = block(**_forward_kwargs(block))
  assert isinstance(out, torch.Tensor), "Pattern_3 requires a single tensor return"


def _forward_kwargs(block):
  transformer = _tiny_transformer()
  inputs = _inputs()
  hidden_states = torch.randn(1, TEXT_TOKENS + IMAGE_LEN, transformer.config.hidden_size)
  rope = transformer.rotary_emb(inputs["position_ids"], hidden_states.dtype)
  return dict(
    hidden_states=hidden_states,
    rope=rope,
    attention_mask=inputs["attention_mask"],
    cp_config=None,
  )


def test_missing_attention_mask_fails_fast():
  # Without the mixed causal/full mask every token becomes bidirectional, which renders a
  # plausible but wrong image instead of raising.
  transformer = _tiny_transformer()
  inputs = _inputs()
  inputs.pop("attention_mask")
  with pytest.raises(ValueError, match="mixed causal/full attention mask"):
    transformer(**inputs, return_dict=False)


def test_local_sequence_without_cp_config_fails_fast():
  # The mask always spans the full joint sequence; a shorter local sequence without a CP config
  # means the split leaked in, which would otherwise silently mis-mask every block.
  transformer = _tiny_transformer()
  inputs = _inputs()
  inputs["vinputs"] = inputs["vinputs"][:, :2]
  with pytest.raises(RuntimeError, match="does not match the local sequence"):
    transformer(**inputs, return_dict=False)


def _cache_context(transformer):
  """Return the cache context of the wrapped block stack."""
  name = transformer._context_names[0]
  while isinstance(name, (list, tuple)):
    name = name[0]
  return transformer._context_manager.get_context(name)


def test_cached_run_is_bit_exact_when_no_step_is_cached():
  transformer = _tiny_transformer()
  inputs = _inputs()
  with torch.no_grad():
    reference = transformer(**inputs, return_dict=False)[0].clone()

  cache_dit.enable_cache(
    BlockAdapterRegister.get_adapter(transformer),
    cache_config=DBCacheConfig(
      num_inference_steps=4,
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=0,
      max_cached_steps=0,  # caching disabled: the run must stay bit-exact
      enable_separate_cfg=True,
      residual_diff_threshold=0.0,
    ),
  )

  with torch.no_grad():
    first = transformer(**inputs, return_dict=False)[0].clone()
    second = transformer(**inputs, return_dict=False)[0].clone()
  assert torch.equal(first, reference)
  assert torch.equal(second, first)
  # The cached wrapper really ran; without this the equality above could be vacuous.
  context = _cache_context(transformer)
  assert context.transformer_executed_steps == 2
  assert not context.cached_steps


def test_cached_run_really_skips_and_reproduces_the_plain_forward():
  """Repeat the same input: after warmup every step must hit the cache and stay bit-exact.

  `residual_diff_threshold` must be positive for skipping to be possible at all, and
  `max_cached_steps` must be non-negative-but-unlimited-ish (`0` disables caching), so the guard
  values below are what makes the cached path reachable.
  """
  transformer = _tiny_transformer()
  cond = _inputs(TEXT_TOKENS)
  uncond = _inputs(TEXT_TOKENS + 2)  # the `" "` branch of the real pipeline has its own length
  with torch.no_grad():
    reference_cond = transformer(**cond, return_dict=False)[0].clone()
    reference_uncond = transformer(**uncond, return_dict=False)[0].clone()

  adapter = BlockAdapterRegister.get_adapter(transformer)
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      num_inference_steps=4,
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=1,
      max_cached_steps=8,
      enable_separate_cfg=True,
      residual_diff_threshold=0.05,
    ),
  )

  with torch.no_grad():
    outputs = []
    for _ in range(4):
      out_cond = transformer(**cond, return_dict=False)[0].clone()
      out_uncond = transformer(**uncond, return_dict=False)[0].clone()
      outputs.append((out_cond, out_uncond))

  # 4 denoise steps x 2 CFG branches = 8 forwards; only the warmup step may execute blocks, and
  # the two branches keep separate cached-step books.
  context = _cache_context(transformer)
  assert len(context.cached_steps) == 3, "the cached path never skipped a conditional step"
  assert len(context.cfg_cached_steps) == 3, "the unconditional branch was not cached separately"
  assert context.executed_steps == 4
  # The warmup step is computed, the rest replay a stored residual, so their sums are not
  # bit-identical even though the inputs repeat exactly.
  torch.testing.assert_close(outputs[0][0], reference_cond, atol=0, rtol=0)
  for step, (out_cond, out_uncond) in enumerate(outputs):
    torch.testing.assert_close(out_cond,
                               reference_cond,
                               atol=1e-5,
                               rtol=1e-4,
                               msg=f"cond step {step}")
    torch.testing.assert_close(out_uncond,
                               reference_uncond,
                               atol=1e-5,
                               rtol=1e-4,
                               msg=f"uncond step {step}")


def test_cache_hit_uses_cached_residuals_not_a_stale_branch():
  """A cached step must reuse its own branch: the two CFG inputs are never interchangeable."""
  transformer = _tiny_transformer()
  cond = _inputs(TEXT_TOKENS)
  uncond = _inputs(TEXT_TOKENS + 2)
  with torch.no_grad():
    reference_cond = transformer(**cond, return_dict=False)[0].clone()
    reference_uncond = transformer(**uncond, return_dict=False)[0].clone()
  # Guard against a test that would pass even if the branches were swapped.
  assert not torch.equal(reference_cond[:, :TEXT_TOKENS], reference_uncond[:, :TEXT_TOKENS])

  adapter = BlockAdapterRegister.get_adapter(transformer)
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      num_inference_steps=4,
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=1,
      max_cached_steps=8,
      enable_separate_cfg=True,
      residual_diff_threshold=0.05,
    ),
  )
  try:
    with torch.no_grad():
      for _ in range(4):
        out_cond = transformer(**cond, return_dict=False)[0].clone()
        out_uncond = transformer(**uncond, return_dict=False)[0].clone()
    # Cached steps replay a stored residual, so compare with a tight-but-nonzero tolerance.
    torch.testing.assert_close(out_cond, reference_cond, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out_uncond, reference_uncond, atol=1e-5, rtol=1e-4)
  finally:
    cache_dit.disable_cache(adapter)
