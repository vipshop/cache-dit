"""CPU tests for the built-in LingBot-Video `__call__` and its cache guard.

`LingBotVideoPipeline.__call__` runs two transformer forwards per step (cond, then uncond), keeps the
latents in fp32 and lets the scheduler own the sigma schedule. Those details are easy to break while
refactoring, so this compares one short run against a manual denoising loop built from the same
components.

Usage:
  python -m pytest tests/api/test_lingbot_video_pipeline.py -q
"""

import pytest
import torch

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.models.pipelines.lingbot_video import (
  FlowUniPCMultistepScheduler,
  LingBotVideoPipeline,
  LingBotVideoTransformer3DModel,
)

LATENT_CHANNELS = 16
TEXT_DIM = 32
TEXT_TOKENS = 6
HEIGHT = 32
WIDTH = 32
FRAMES = 1


def _tiny_transformer(seed: int = 0) -> LingBotVideoTransformer3DModel:
  torch.manual_seed(seed)
  return LingBotVideoTransformer3DModel(
    patch_size=(1, 2, 2),
    in_channels=LATENT_CHANNELS,
    out_channels=LATENT_CHANNELS,
    hidden_size=64,
    num_attention_heads=4,
    depth=2,
    intermediate_size=128,
    text_dim=TEXT_DIM,
    freq_dim=16,
    # head_dim == sum(axes_dims) is enforced by the transformer.
    axes_dims=(4, 6, 6),
    axes_lens=(16, 8, 8),
  ).eval()


def _tiny_pipeline(seed: int = 0) -> LingBotVideoPipeline:
  return LingBotVideoPipeline(
    transformer=_tiny_transformer(seed),
    vae=None,
    text_encoder=None,
    processor=None,
    scheduler=FlowUniPCMultistepScheduler(num_train_timesteps=1000),
  ).to("cpu")


def _embeds(seed: int, tokens: int = TEXT_TOKENS):
  generator = torch.Generator().manual_seed(seed)
  return torch.randn(1, tokens, TEXT_DIM, generator=generator)


def _latents(seed: int):
  return torch.randn(
    1,
    LATENT_CHANNELS,
    (FRAMES - 1) // 4 + 1,
    HEIGHT // 8,
    WIDTH // 8,
    generator=torch.Generator().manual_seed(seed),
  )


def test_call_matches_manual_denoising_loop():
  pipe = _tiny_pipeline()
  transformer = pipe.transformer
  positive, negative = _embeds(1), _embeds(2)
  mask = torch.ones(1, TEXT_TOKENS, dtype=torch.long)

  with torch.no_grad():
    got = pipe(
      prompt="",
      prompt_embeds=positive,
      prompt_mask=mask,
      negative_prompt_embeds=negative,
      negative_prompt_mask=mask,
      height=HEIGHT,
      width=WIDTH,
      num_frames=FRAMES,
      num_inference_steps=2,
      guidance_scale=3.0,
      latents=_latents(5),
      output_type="latent",
    ).frames

    expected = _latents(5)
    pipe.scheduler.set_timesteps(2, device=torch.device("cpu"), shift=3.0)
    for timestep in pipe.scheduler.timesteps:
      timestep_batch = timestep.float().reshape(1)
      cond = transformer(expected, timestep_batch, positive, mask, return_dict=False)[0].float()
      uncond = transformer(expected, timestep_batch, negative, mask, return_dict=False)[0].float()
      prediction = uncond + 3.0 * (cond - uncond)
      expected = pipe.scheduler.step(prediction, timestep, expected, return_dict=False)[0]

  torch.testing.assert_close(got, expected, atol=0, rtol=0)


def test_transformer_rejects_padded_text():
  """The joint sequence and the RoPE table are built from the text length, so padding is
  rejected."""
  transformer = _tiny_transformer()
  hidden = torch.zeros(1, LATENT_CHANNELS, 1, HEIGHT // 8, WIDTH // 8)
  text = torch.zeros(1, TEXT_TOKENS, TEXT_DIM)
  mask = torch.ones(1, TEXT_TOKENS, dtype=torch.long)
  mask[0, -1] = 0
  with pytest.raises(NotImplementedError, match="truncated to the valid length"):
    transformer(hidden, torch.zeros(1), text, encoder_attention_mask=mask, return_dict=False)


def test_call_truncates_user_provided_embeddings():
  """Pre-computed embeddings must follow the same valid-length rule as `encode_prompt`."""
  pipe = _tiny_pipeline()
  mask = torch.ones(1, TEXT_TOKENS, dtype=torch.long)
  mask[0, -2:] = 0
  base = dict(
    prompt="",
    prompt_mask=mask,
    negative_prompt_embeds=_embeds(2),
    negative_prompt_mask=torch.ones(1, TEXT_TOKENS, dtype=torch.long),
    height=HEIGHT,
    width=WIDTH,
    num_frames=FRAMES,
    num_inference_steps=1,
    guidance_scale=3.0,
    latents=_latents(5),
    output_type="latent",
  )
  truncated_kwargs = dict(base)
  truncated_kwargs["prompt_mask"] = mask[:, :-2]
  with torch.no_grad():
    padded = pipe(prompt_embeds=_embeds(1), **base).frames
    truncated = pipe(prompt_embeds=_embeds(1)[:, :-2], **truncated_kwargs).frames
  torch.testing.assert_close(padded, truncated, atol=0, rtol=0)


def test_transformer_rejects_a_batched_forward():
  transformer = _tiny_transformer()
  hidden = torch.zeros(2, LATENT_CHANNELS, 1, HEIGHT // 8, WIDTH // 8)
  text = torch.zeros(2, TEXT_TOKENS, TEXT_DIM)
  with pytest.raises(NotImplementedError, match="one sample per forward"):
    transformer(hidden, torch.zeros(2), text, return_dict=False)


def test_transformer_rejects_moe_configs():
  with pytest.raises(NotImplementedError, match="MoE"):
    LingBotVideoTransformer3DModel(
      hidden_size=64,
      num_attention_heads=4,
      depth=1,
      intermediate_size=128,
      text_dim=TEXT_DIM,
      axes_dims=(4, 6, 6),
      num_experts=8,
    )


def test_cached_pipeline_rejects_a_single_forward_per_step():
  pipe = _tiny_pipeline()
  cache_dit.enable_cache(
    pipe,
    cache_config=DBCacheConfig(
      num_inference_steps=2,
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=1,
      residual_diff_threshold=0.05,
    ),
  )
  mask = torch.ones(1, TEXT_TOKENS, dtype=torch.long)
  try:
    with pytest.raises(ValueError, match="guidance_scale=1.0"):
      pipe(
        prompt="",
        prompt_embeds=_embeds(1),
        prompt_mask=mask,
        height=HEIGHT,
        width=WIDTH,
        num_frames=FRAMES,
        num_inference_steps=2,
        guidance_scale=1.0,
        latents=_latents(5),
        output_type="latent",
      )
  finally:
    cache_dit.disable_cache(pipe)


def test_from_pretrained_rejects_unsupported_overrides(tmp_path):
  with pytest.raises(NotImplementedError, match="device-map-balance"):
    LingBotVideoPipeline.from_pretrained(str(tmp_path), device_map="balanced")
  with pytest.raises(ValueError, match="ControlNet"):
    LingBotVideoPipeline.from_pretrained(str(tmp_path), controlnet=torch.nn.Identity())


def test_check_inputs_rejects_bad_frame_and_size():
  LingBotVideoPipeline.check_inputs(16, 16, 1)
  LingBotVideoPipeline.check_inputs(16, 16, 9)
  with pytest.raises(ValueError, match="4n\\+1"):
    LingBotVideoPipeline.check_inputs(16, 16, 7)
  with pytest.raises(ValueError, match="multiples of 16"):
    LingBotVideoPipeline.check_inputs(24, 16, 1)


def test_scheduler_builds_a_descending_sigma_schedule():
  scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000)
  scheduler.set_timesteps(4, device=torch.device("cpu"), shift=3.0)
  assert scheduler.timesteps.shape[0] == 4
  assert torch.all(scheduler.timesteps[:-1] > scheduler.timesteps[1:])
  assert torch.all(scheduler.sigmas[:-1] > scheduler.sigmas[1:])
