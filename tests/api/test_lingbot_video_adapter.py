"""Regression tests for the LingBot-Video cache integration.

LingBot-Video ships its own pipeline (`cache_dit.models.pipelines.lingbot_video`) because diffusers
does not carry it, and each denoising step runs two `transformer.forward()` calls (cond, then
uncond). These tests lock the adapter wiring (`blocks` + `Pattern_3` + separate CFG), the
`_repeated_blocks` declaration the quantizer and `torch.compile` rely on, and the per-branch cache
bookkeeping.

Usage:
  python -m pytest tests/api/test_lingbot_video_adapter.py -q
"""

import copy

import diffusers
import torch

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.caching.block_adapters import BlockAdapterRegister
from cache_dit.caching.forward_pattern import ForwardPattern
from cache_dit.models.pipelines.lingbot_video import (
  FlowUniPCMultistepScheduler,
  LingBotVideoPipeline,
  LingBotVideoTransformer3DModel,
)

LATENT_CHANNELS = 16
TEXT_DIM = 32
TEXT_TOKENS = 6
HEIGHT = 2
WIDTH = 2
NUM_LAYERS = 3


def _tiny_transformer(seed: int = 0) -> LingBotVideoTransformer3DModel:
  torch.manual_seed(seed)
  return LingBotVideoTransformer3DModel(
    patch_size=(1, 2, 2),
    in_channels=LATENT_CHANNELS,
    out_channels=LATENT_CHANNELS,
    hidden_size=64,
    num_attention_heads=4,
    depth=NUM_LAYERS,
    intermediate_size=128,
    text_dim=TEXT_DIM,
    freq_dim=16,
    axes_dims=(4, 6, 6),
    axes_lens=(16, 8, 8),
  ).eval()


def _tiny_pipeline(transformer=None) -> LingBotVideoPipeline:
  return LingBotVideoPipeline(
    transformer=_tiny_transformer() if transformer is None else transformer,
    vae=None,
    text_encoder=None,
    processor=None,
    scheduler=FlowUniPCMultistepScheduler(num_train_timesteps=1000),
  )


def _step_inputs(generator, tokens: int = TEXT_TOKENS):
  return dict(
    hidden_states=torch.randn(1, LATENT_CHANNELS, 1, HEIGHT, WIDTH, generator=generator),
    timestep=torch.tensor([750.0]),
    encoder_hidden_states=torch.randn(1, tokens, TEXT_DIM, generator=generator),
    return_dict=False,
  )


class LingBotVideoTestPipeline(LingBotVideoPipeline):
  """Drive the transformer the way `__call__` does, without text encoder / VAE."""

  def __call__(self, cond_inputs, uncond_inputs, num_steps: int = 1):
    cond_outs, uncond_outs = [], []
    for _ in range(num_steps):
      cond_outs.append(self.transformer(**cond_inputs)[0])
      uncond_outs.append(self.transformer(**uncond_inputs)[0])
    return cond_outs, uncond_outs


def test_lingbot_video_prefix_is_the_unique_match():
  prefixes = BlockAdapterRegister.supported_pipelines()[1]
  assert "LingBotVideo" in prefixes
  matches = [name for name in prefixes if "LingBotVideoPipeline".startswith(name)]
  assert matches == ["LingBotVideo"], f"ambiguous adapter prefixes for LingBot-Video: {matches}"


def test_adapter_wiring():
  from cache_dit.caching.block_adapters import lingbot_video_adapter

  transformer = _tiny_transformer()
  adapter = lingbot_video_adapter(
    LingBotVideoTestPipeline(
      transformer=transformer,
      vae=None,
      text_encoder=None,
      processor=None,
      scheduler=FlowUniPCMultistepScheduler(),
    ))

  assert adapter.blocks_name == "blocks"
  assert adapter.forward_pattern == ForwardPattern.Pattern_3
  assert adapter.check_forward_pattern is True
  # cond + uncond are two separate forwards per step
  assert adapter.has_separate_cfg is True
  # the token-wise modulation is computed outside the loop, so no structural patch is needed
  assert adapter.patch_functor is None


def test_registered_adapter_supports_transformer_only_usage():
  transformer = _tiny_transformer()
  adapter = BlockAdapterRegister.get_adapter(transformer)

  assert adapter is not None
  assert adapter.transformer is transformer
  assert adapter.blocks is transformer.blocks
  assert adapter.has_separate_cfg is True


def test_builtin_transformer_declares_repeated_blocks_and_pipeline_type():
  pipe = _tiny_pipeline()
  assert cache_dit.utils.is_pipeline(pipe)
  assert isinstance(pipe.transformer, diffusers.ModelMixin)
  assert pipe.transformer._repeated_blocks == ["LingBotVideoBlock"]
  # `blocks` must stay a plain attribute of the transformer: cache-dit patches it by name.
  assert isinstance(pipe.transformer.blocks, torch.nn.ModuleList)
  assert [block.layer_idx for block in pipe.transformer.blocks] == list(range(NUM_LAYERS))


def test_cached_run_is_bit_exact_when_no_step_is_cached():
  reference_transformer = _tiny_transformer()
  transformer = copy.deepcopy(reference_transformer)

  pipe = LingBotVideoTestPipeline(
    transformer=transformer,
    vae=None,
    text_encoder=None,
    processor=None,
    scheduler=FlowUniPCMultistepScheduler(),
  )
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

  generator = torch.Generator().manual_seed(7)
  cond_inputs = _step_inputs(generator)
  uncond_inputs = _step_inputs(generator)
  try:
    with torch.no_grad():
      cond_outs, uncond_outs = pipe(cond_inputs, uncond_inputs, num_steps=3)
      for step, (cond_out, uncond_out) in enumerate(zip(cond_outs, uncond_outs)):
        cond_ref = reference_transformer(**cond_inputs)[0]
        uncond_ref = reference_transformer(**uncond_inputs)[0]
        torch.testing.assert_close(
          cond_out,
          cond_ref,
          atol=0,
          rtol=0,
          msg=f"conditional branch diverged at step {step}",
        )
        torch.testing.assert_close(
          uncond_out,
          uncond_ref,
          atol=0,
          rtol=0,
          msg=f"unconditional branch diverged at step {step}",
        )
  finally:
    cache_dit.disable_cache(pipe)
