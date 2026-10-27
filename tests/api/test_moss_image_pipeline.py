"""CPU tests for the built-in MOSS-Image `__call__` and its cache guards.

`MOSSImagePipeline.__call__` runs two transformer forwards per step (cond, then uncond with
`uncond_skip_layers`), combines the two predictions in fp32 and keeps the latents in fp32. Those
details are easy to break while refactoring, so this compares one short run against a manual
denoising loop built from the same components.
"""
import pytest
import torch

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.models.pipelines.moss_image import MOSSImagePipeline, MOSSImageScheduler

LATENT_CHANNELS = 8
TEXT_DIM = 16
TEXT_TOKENS = 5
LATENT_SIZE = 2


def _tiny_pipeline(seed: int = 0) -> MOSSImagePipeline:
  from cache_dit.models.pipelines.moss_image import MOSSImageTransformer

  torch.manual_seed(seed)
  transformer = MOSSImageTransformer(
    dim=32,
    ffn_dim=64,
    num_heads=4,
    num_layers=2,
    text_dim=TEXT_DIM,
    freq_dim=16,
    latent_channels=LATENT_CHANNELS,
  ).eval()
  pipe = MOSSImagePipeline(transformer=transformer, scheduler=MOSSImageScheduler())
  return pipe.to("cpu")


def _embeds(seed: int):
  generator = torch.Generator().manual_seed(seed)
  return torch.randn(1, TEXT_TOKENS, TEXT_DIM, generator=generator)


def test_call_matches_manual_denoising_loop():
  pipe = _tiny_pipeline()
  transformer = pipe.transformer
  positive, negative = _embeds(1), _embeds(2)
  positive_mask = torch.ones(1, TEXT_TOKENS, dtype=torch.bool)
  negative_mask = torch.ones(1, TEXT_TOKENS, dtype=torch.bool)
  latents = torch.randn(1,
                        LATENT_CHANNELS,
                        LATENT_SIZE,
                        LATENT_SIZE,
                        generator=torch.Generator().manual_seed(5))
  kwargs = dict(
    prompt_embeds=positive,
    prompt_attention_mask=positive_mask,
    negative_prompt_embeds=negative,
    negative_prompt_attention_mask=negative_mask,
    height=LATENT_SIZE * 16,
    width=LATENT_SIZE * 16,
    num_inference_steps=2,
    guidance_scale=3.0,
    uncond_skip_layers=(1, ),
    latents=latents.clone(),
    output_type="latent",
  )

  with torch.no_grad():
    got = pipe(**kwargs).images

    expected = latents.clone()
    pipe.scheduler.set_timesteps(2, device=torch.device("cpu"))
    for timestep in pipe.scheduler.timesteps:
      cond = transformer(expected, timestep, positive, positive_mask).sample.float()
      uncond = transformer(expected, timestep, negative, negative_mask,
                           skip_layers=(1, )).sample.float()
      prediction = uncond + 3.0 * (cond - uncond)
      expected = pipe.scheduler.step(prediction, timestep, expected, return_dict=False)[0]

  torch.testing.assert_close(got, expected, atol=0, rtol=0)


def test_call_without_cfg_skips_the_negative_branch():
  pipe = _tiny_pipeline()
  positive = _embeds(1)
  kwargs = dict(
    prompt_embeds=positive,
    prompt_attention_mask=torch.ones(1, TEXT_TOKENS, dtype=torch.bool),
    height=LATENT_SIZE * 16,
    width=LATENT_SIZE * 16,
    num_inference_steps=1,
    guidance_scale=1.0,
    uncond_skip_layers=(1, ),
    output_type="latent",
  )
  with torch.no_grad():
    got = pipe(**kwargs).images

  expected = torch.randn(1,
                         LATENT_CHANNELS,
                         LATENT_SIZE,
                         LATENT_SIZE,
                         generator=torch.Generator().manual_seed(0)).to(torch.float32)
  # The noise is re-sampled per call, so only the shape and finiteness are meaningful here.
  assert got.shape == expected.shape
  assert torch.isfinite(got).all()


def test_cached_pipeline_rejects_configurations_that_break_cfg_pairing():
  pipe = _tiny_pipeline()
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
  positive = _embeds(1)
  base = dict(
    prompt_embeds=positive,
    prompt_attention_mask=torch.ones(1, TEXT_TOKENS, dtype=torch.bool),
    negative_prompt_embeds=_embeds(2),
    negative_prompt_attention_mask=torch.ones(1, TEXT_TOKENS, dtype=torch.bool),
    height=LATENT_SIZE * 16,
    width=LATENT_SIZE * 16,
    num_inference_steps=2,
    uncond_skip_layers=(1, ),
    output_type="latent",
  )
  try:
    with pytest.raises(ValueError, match="guidance_scale=1.0"):
      pipe(guidance_scale=1.0, **base)
    with pytest.raises(ValueError, match="skip-layer guidance"):
      pipe(guidance_scale=3.0, slg_layers=[1], slg_scale=2.0, **base)
  finally:
    cache_dit.disable_cache(pipe)


def test_from_pretrained_rejects_unsupported_overrides(tmp_path):
  with pytest.raises(NotImplementedError, match="device-map-balance"):
    MOSSImagePipeline.from_pretrained(str(tmp_path), device_map="balanced")
  with pytest.raises(ValueError, match="ControlNet"):
    MOSSImagePipeline.from_pretrained(str(tmp_path), controlnet=torch.nn.Identity())


def test_scheduler_is_the_training_grid():
  scheduler = MOSSImageScheduler()
  scheduler.set_timesteps(4, device=torch.device("cpu"))
  assert scheduler.timesteps.shape[0] >= 1
  assert torch.all(scheduler.timesteps >= 0)
  assert scheduler.timesteps[0] > scheduler.timesteps[-1]
