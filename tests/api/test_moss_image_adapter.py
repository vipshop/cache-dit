"""Regression tests for the MOSS-Image cache integration.

MOSS-Image ships its own pipeline (`cache_dit.models.pipelines.moss_image`) because diffusers does
not carry it, and each denoising step runs two `transformer.forward()` calls (cond, then uncond with
`uncond_skip_layers`). Those tests lock the adapter wiring (`layers` + `Pattern_3` + separate CFG),
the block-level `skip_layers` behavior under cached blocks, and the per-branch cache bookkeeping.
"""
import copy

import diffusers
import pytest
import torch

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.caching.block_adapters import BlockAdapter, BlockAdapterRegister
from cache_dit.caching.forward_pattern import ForwardPattern
from cache_dit.models.pipelines.moss_image import (
  MOSSImagePipeline,
  MOSSImageScheduler,
  MOSSImageTransformer,
)

LATENT_CHANNELS = 8
TEXT_DIM = 16
DIM = 32
FFN_DIM = 64
HEADS = 4
NUM_LAYERS = 3
TEXT_TOKENS = 5
HEIGHT = 2
WIDTH = 2


def _tiny_transformer(seed: int = 0):
  torch.manual_seed(seed)
  return MOSSImageTransformer(
    dim=DIM,
    ffn_dim=FFN_DIM,
    num_heads=HEADS,
    num_layers=NUM_LAYERS,
    text_dim=TEXT_DIM,
    freq_dim=16,
    latent_channels=LATENT_CHANNELS,
  ).eval()


def _tiny_pipeline(transformer=None) -> MOSSImagePipeline:
  return MOSSImagePipeline(
    transformer=_tiny_transformer() if transformer is None else transformer,
    scheduler=MOSSImageScheduler(),
  )


def _test_pipeline(transformer=None) -> "MOSSImageTestPipeline":
  return MOSSImageTestPipeline(
    transformer=_tiny_transformer() if transformer is None else transformer,
    scheduler=MOSSImageScheduler(),
  )


def _step_inputs(generator):
  return dict(
    hidden_states=torch.randn(1, LATENT_CHANNELS, HEIGHT, WIDTH, generator=generator),
    timestep=torch.tensor([0.5]),
    encoder_hidden_states=torch.randn(1, TEXT_TOKENS, TEXT_DIM, generator=generator),
    encoder_attention_mask=torch.ones(1, TEXT_TOKENS, dtype=torch.long),
    return_dict=False,
  )


class MOSSImageTestPipeline(MOSSImagePipeline):
  """Drive the transformer the way `MOSSImagePipeline.__call__` does, without text encoder / VAE.

  Two forwards per step (cond, then uncond with the CFG skip), which is the shape cache-dit's
  separate-CFG bookkeeping has to pair.
  """

  def __call__(self, cond_inputs, uncond_inputs, num_steps: int = 1, uncond_skip_layers=(1, )):
    cond_outs, uncond_outs = [], []
    for _ in range(num_steps):
      cond_outs.append(self.transformer(**cond_inputs)[0])
      uncond_outs.append(self.transformer(**uncond_inputs, skip_layers=uncond_skip_layers)[0])
    return cond_outs, uncond_outs


def test_moss_image_prefix_is_the_unique_match():
  prefixes = BlockAdapterRegister.supported_pipelines()[1]
  assert "MOSSImage" in prefixes
  matches = [name for name in prefixes if "MOSSImagePipeline".startswith(name)]
  assert matches == ["MOSSImage"], f"ambiguous adapter prefixes for MOSS-Image: {matches}"


def test_adapter_wiring():
  from cache_dit.caching.block_adapters import moss_image_adapter

  transformer = _tiny_transformer()
  adapter = moss_image_adapter(_test_pipeline(transformer))

  assert adapter.blocks_name == "layers"
  assert adapter.forward_pattern == ForwardPattern.Pattern_3
  assert adapter.check_forward_pattern is True
  # cond + uncond are two separate forwards per step
  assert adapter.has_separate_cfg is True
  # skipping lives inside the block, so no structural patch is needed
  assert adapter.patch_functor is None


def test_registered_adapter_supports_transformer_only_usage():
  transformer = _tiny_transformer()
  adapter = BlockAdapterRegister.get_adapter(transformer)

  # The registry returns the raw factory result (normalization happens in `cachify`), so the
  # transformer/block stack are the singular fields of the dataclass here.
  assert adapter is not None
  assert adapter.transformer is transformer
  assert adapter.blocks is transformer.layers
  assert adapter.has_separate_cfg is True


def test_builtin_transformer_declares_repeated_blocks_and_pipeline_type():
  pipe = _tiny_pipeline()
  assert cache_dit.utils.is_pipeline(pipe)
  assert isinstance(pipe.transformer, diffusers.ModelMixin)
  assert pipe.transformer._repeated_blocks == ["_Block"]
  # `layers` must stay a plain attribute of the transformer: cache-dit patches it by name.
  assert isinstance(pipe.transformer.layers, torch.nn.ModuleList)
  assert [block.index for block in pipe.transformer.layers] == list(range(NUM_LAYERS))


class _Passthrough(torch.nn.Module):
  """Stand-in for a skipped block; the stack calls blocks with extra positional args."""

  def forward(self, hidden_states, *args, **kwargs):
    return hidden_states


def _replace_blocks(transformer, indices):
  """Return a copy of `transformer` whose blocks at `indices` are exact passthroughs."""
  reference = copy.deepcopy(transformer)
  for index in indices:
    reference.layers[index] = _Passthrough()
  return reference


@pytest.mark.parametrize("skip_layers", [(0, 1, 2), (1, ), (0, )])
def test_skipped_blocks_are_exact_passthroughs(skip_layers):
  """`skip_layers` must reproduce a model whose skipped blocks do nothing, bit for bit."""
  transformer = _tiny_transformer()
  reference = _replace_blocks(transformer, skip_layers)
  generator = torch.Generator().manual_seed(3)
  inputs = _step_inputs(generator)

  with torch.no_grad():
    skipped = transformer(**inputs, skip_layers=skip_layers)[0]
    expected = reference(**inputs)[0]
    full = transformer(**inputs)[0]

  torch.testing.assert_close(skipped, expected, atol=0, rtol=0)
  # And a partial skip really changes the result, so the assertion above is not vacuous.
  assert not torch.equal(skipped, full)


def test_cached_run_is_bit_exact_when_no_step_is_cached():
  reference_transformer = _tiny_transformer()
  transformer = copy.deepcopy(reference_transformer)

  pipe = _test_pipeline(transformer)
  adapter = BlockAdapterRegister.get_adapter(pipe)
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      num_inference_steps=3,
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=0,
      max_cached_steps=0,  # caching disabled: the run must stay bit-exact
      residual_diff_threshold=0.0,
    ),
  )

  context_manager = pipe._context_manager
  generator = torch.Generator().manual_seed(7)
  cond_inputs = _step_inputs(generator)
  uncond_inputs = _step_inputs(generator)
  try:
    with torch.no_grad():
      cond_outs, uncond_outs = pipe(cond_inputs, uncond_inputs, num_steps=3)
      for step, (cond_out, uncond_out) in enumerate(zip(cond_outs, uncond_outs)):
        cond_ref = reference_transformer(**cond_inputs)[0]
        uncond_ref = reference_transformer(**uncond_inputs, skip_layers=(1, ))[0]
        torch.testing.assert_close(cond_out,
                                   cond_ref,
                                   atol=0,
                                   rtol=0,
                                   msg=f"conditional branch diverged at step {step}")
        torch.testing.assert_close(uncond_out,
                                   uncond_ref,
                                   atol=0,
                                   rtol=0,
                                   msg=f"unconditional branch diverged at step {step}")

    # The cached wrapper really ran; a missing context would make the check above vacuous.
    context = context_manager.get_context(adapter.unique_blocks_name[0][0])
    assert context.transformer_executed_steps == 6  # 3 steps x 2 branches
  finally:
    cache_dit.disable_cache(adapter)


def test_cfg_skip_is_preserved_under_cached_blocks():
  """The cached-block wrapper must not silently drop `uncond_skip_layers`.

  Without block-level skipping the uncond branch would run all `NUM_LAYERS` blocks, so the cached
  result would differ from an uncached run that really skips block 1.
  """
  reference_transformer = _tiny_transformer()
  transformer = copy.deepcopy(reference_transformer)

  pipe = _test_pipeline(transformer)
  adapter = BlockAdapterRegister.get_adapter(pipe)
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      num_inference_steps=2,
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=0,
      max_cached_steps=0,
      residual_diff_threshold=0.0,
    ),
  )

  generator = torch.Generator().manual_seed(11)
  cond_inputs = _step_inputs(generator)
  uncond_inputs = _step_inputs(generator)
  try:
    with torch.no_grad():
      _, uncond_outs = pipe(cond_inputs, uncond_inputs, num_steps=2)
      skipped_ref = reference_transformer(**uncond_inputs, skip_layers=(1, ))[0]
      no_skip_ref = reference_transformer(**uncond_inputs)[0]
    for step, uncond_out in enumerate(uncond_outs):
      torch.testing.assert_close(uncond_out,
                                 skipped_ref,
                                 atol=0,
                                 rtol=0,
                                 msg=f"cached uncond diverged from the skipped reference at {step}")
    assert not torch.equal(skipped_ref, no_skip_ref), "the skip must change the uncond prediction"
  finally:
    cache_dit.disable_cache(adapter)


def test_separate_cfg_counts_one_step_per_denoising_step():
  pipe = _test_pipeline()
  cache_config = DBCacheConfig(
    num_inference_steps=4,
    Fn_compute_blocks=1,
    Bn_compute_blocks=0,
    max_warmup_steps=1,
    max_cached_steps=4,
    residual_diff_threshold=0.05,
  )
  cache_dit.enable_cache(pipe, cache_config=cache_config)

  assert cache_config.enable_separate_cfg is True

  generator = torch.Generator().manual_seed(13)
  cond_inputs = _step_inputs(generator)
  uncond_inputs = _step_inputs(generator)
  try:
    with torch.no_grad():
      pipe(cond_inputs, uncond_inputs, num_steps=4)
    context = pipe._context_manager.get_context(pipe.transformer._context_names[0])
    # 8 forwards (4 steps x cond/uncond) but only 4 counted denoising steps.
    assert context.transformer_executed_steps == 8
    assert context.executed_steps == 4
  finally:
    cache_dit.disable_cache(pipe)

  assert len(pipe.transformer.layers) == NUM_LAYERS


def test_pipeline_summary_and_disable():
  pipe = _test_pipeline()
  cache_dit.enable_cache(
    pipe,
    cache_config=DBCacheConfig(
      num_inference_steps=2,
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=2,
      residual_diff_threshold=0.05,
    ),
  )
  generator = torch.Generator().manual_seed(17)
  with torch.no_grad():
    pipe(_step_inputs(generator), _step_inputs(generator), num_steps=2)

  # One stat for the transformer and one for its `layers` cache context.
  stats = cache_dit.summary(pipe, logging=False)
  assert len(stats) == 2
  assert all(stat.cache_options for stat in stats)

  cache_dit.disable_cache(pipe)
  assert not BlockAdapter.is_cached(pipe.transformer)
  assert not hasattr(pipe.transformer, "_context_manager")
