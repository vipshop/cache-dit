"""CPU tests for the vendored LongCat-Video-Avatar integration.

No weights and no GPU: a tiny configuration of the real classes pins the contracts that the
integration depends on.
"""

import types

import pytest
import torch

from cache_dit.caching.block_adapters.adapters import longcat_video_avatar_adapter
from cache_dit.caching.forward_pattern import ForwardPattern
from cache_dit.models.pipelines.longcat_video_avatar import (
  LongCatAvatarSingleStreamBlock,
  LongCatVideoAvatarTransformer3DModel,
  LongCatVideoAvatarPipeline,
)
from cache_dit.models.pipelines.longcat_video_avatar.pipeline import merge_lora_state_dict

_TINY = dict(
  in_channels=8,
  out_channels=8,
  hidden_size=16,
  depth=2,
  num_heads=2,
  caption_channels=12,
  mlp_ratio=2,
  adaln_tembed_dim=8,
  frequency_embedding_size=8,
  patch_size=(1, 2, 2),
  audio_window=5,
  audio_block=5,
  audio_channel=4,
  intermediate_dim=8,
  output_dim=8,
  context_tokens=3,
  vae_scale=4,
)


def _transformer(**overrides):
  torch.manual_seed(0)
  return LongCatVideoAvatarTransformer3DModel(**{**_TINY, **overrides}).eval()


def _inputs(frames=4, latent_hw=4, batch=1, seed=0):
  torch.manual_seed(seed)
  latents = torch.randn(batch, 8, frames, latent_hw, latent_hw)
  text = torch.randn(batch, 1, 6, 12)
  mask = torch.ones(batch, 6, dtype=torch.long)
  audio = torch.randn(batch, frames * 4 + 1, 5, 5, 4)
  return latents, text, mask, audio


def test_adapter_wiring():
  transformer = _transformer()
  pipe = types.SimpleNamespace(transformer=transformer, do_classifier_free_guidance=True)

  adapter = longcat_video_avatar_adapter(pipe)

  assert adapter.blocks_name == "blocks"
  assert isinstance(adapter.blocks[0], LongCatAvatarSingleStreamBlock)
  assert adapter.forward_pattern == ForwardPattern.Pattern_3
  assert adapter.has_separate_cfg is True


def test_adapter_honours_pipeline_recipe_flag():
  transformer = _transformer()
  pipe = types.SimpleNamespace(transformer=transformer,
                               do_classifier_free_guidance=True,
                               cache_has_separate_cfg=False)

  adapter = longcat_video_avatar_adapter(pipe)

  # The distilled recipe runs a single transformer call per step; the guide says two.
  assert adapter.has_separate_cfg is False


def test_adapter_rejects_foreign_transformer():
  block = LongCatAvatarSingleStreamBlock(hidden_size=16,
                                         num_heads=2,
                                         mlp_ratio=2,
                                         adaln_tembed_dim=8,
                                         output_dim=8)
  pipe = types.SimpleNamespace(transformer=torch.nn.Module(), do_classifier_free_guidance=True)

  with pytest.raises(AssertionError, match="LongCatVideoAvatarTransformer3DModel"):
    longcat_video_avatar_adapter(pipe)

  pipe.transformer.blocks = torch.nn.ModuleList([block])
  # The block itself is fine, but the transformer type is not the vendored one.
  with pytest.raises(AssertionError):
    longcat_video_avatar_adapter(pipe)


def test_audio_conditioning_reaches_every_block():
  """The audio tokens are a block argument: changing them must change the output."""
  transformer = _transformer()
  latents, text, mask, audio = _inputs()

  with torch.no_grad():
    out_audio = transformer(latents, torch.rand(1), text, mask, audio_embs=audio)
    out_zeros = transformer(latents, torch.rand(1), text, mask, audio_embs=torch.zeros_like(audio))

  assert out_audio.shape == latents.shape
  assert not torch.allclose(out_audio, out_zeros), "the audio conditioning did not reach the blocks"


def test_audio_conditioning_survives_skipped_blocks():
  """With the block loop replayed by one block (cached shape), audio still drives the output."""
  transformer = _transformer()
  latents, text, mask, audio = _inputs()

  block = transformer.blocks[0]
  transformer.blocks = torch.nn.ModuleList([block])
  transformer.config.depth = 1

  with torch.no_grad():
    out_a = transformer(latents, torch.rand(1), text, mask, audio_embs=audio)
    out_b = transformer(latents, torch.rand(1), text, mask, audio_embs=torch.zeros_like(audio))

  assert out_a.shape == latents.shape
  assert not torch.allclose(out_a, out_b)


def test_unpatchify_round_trip_shape():
  transformer = _transformer()
  tokens = torch.zeros(1, 4 * 2 * 2, 2 * 2 * 8)

  out = transformer.unpatchify(tokens, 4, 2, 2)

  assert out.shape == (1, 8, 4, 4, 4)


def test_lora_merge_matches_runtime_patch():
  """The merged weight must reproduce `W x + up(down(x)) * multiplier * alpha_scale`."""
  torch.manual_seed(0)
  linear = torch.nn.Linear(6, 4, bias=False)
  rank, alpha, multiplier = 3, 6.0, 1.0
  down = torch.randn(rank, 6) * 0.1
  up = torch.randn(4, rank) * 0.1
  alpha_scale = alpha / rank
  x = torch.randn(5, 6)

  container = torch.nn.Module()
  container.proj = linear
  state_dict = {
    "lora___lorahyphen___proj.lora_down.weight": down,
    "lora___lorahyphen___proj.lora_up.weight": up,
    "lora___lorahyphen___proj.alpha_scale": torch.tensor(alpha_scale),
  }

  with torch.no_grad():
    expected = linear(x) + torch.nn.functional.linear(torch.nn.functional.linear(x, down),
                                                      up) * (multiplier * alpha_scale)

  merged = merge_lora_state_dict(container, state_dict, multiplier=multiplier)

  assert merged == 1
  assert torch.allclose(container.proj(x), expected, atol=1e-5)


def test_lora_merge_handles_packed_up_blocks():
  """A packed projection splits the rank axis into parallel up-blocks (fused QKV)."""
  torch.manual_seed(0)
  linear = torch.nn.Linear(6, 6, bias=False)
  rank, n_blocks, alpha = 2, 3, 6.0
  down = torch.randn(n_blocks * rank, 6) * 0.1
  ups = [torch.randn(2, rank) * 0.1 for _ in range(n_blocks)]
  alpha_scale = alpha / rank
  x = torch.randn(5, 6)

  container = torch.nn.Module()
  container.proj = linear
  state_dict = {
    "lora___lorahyphen___proj.lora_down.weight": down,
    "lora___lorahyphen___proj.alpha_scale": torch.tensor(alpha_scale),
  }
  for index, up in enumerate(ups):
    state_dict[f"lora___lorahyphen___proj.lora_up.blocks.{index}.weight"] = up

  chunks = torch.nn.functional.linear(x, down).chunk(n_blocks, dim=-1)
  patched = torch.cat([torch.nn.functional.linear(chunk, up) for chunk, up in zip(chunks, ups)],
                      dim=-1)
  with torch.no_grad():
    expected = linear(x) + patched * alpha_scale

  merged = merge_lora_state_dict(container, state_dict)

  assert merged == 1
  assert torch.allclose(container.proj(x), expected, atol=1e-5)


def _pipeline(transformer, cached=False):
  transformer._is_cached = cached
  pipe = LongCatVideoAvatarPipeline(
    tokenizer=None,
    text_encoder=None,
    vae=None,
    scheduler=None,
    transformer=transformer,
  )
  return pipe


def test_cache_guard_rejects_mismatched_call_count():
  transformer = _transformer()
  pipe = _pipeline(transformer, cached=True)
  pipe._distill_merged = True
  audio = torch.zeros(16000, dtype=torch.float32)

  # The distilled recipe is a single call per step, but the adapter was built for two.
  pipe.cache_has_separate_cfg = True
  with pytest.raises(ValueError, match="transformer call"):
    pipe(audio=audio,
         num_frames=13,
         num_inference_steps=8,
         use_distill=True,
         text_guidance_scale=1.0,
         audio_guidance_scale=1.0,
         height=32,
         width=32)

  pipe.cache_has_separate_cfg = False
  # Matching recipe: the guard passes and the run fails later, on the missing tokenizer.
  with pytest.raises(Exception) as error:
    pipe(audio=audio,
         num_frames=13,
         num_inference_steps=8,
         use_distill=True,
         text_guidance_scale=1.0,
         audio_guidance_scale=1.0,
         height=32,
         width=32)
  assert "transformer call" not in str(error.value)


def test_distill_requires_merged_lora():
  transformer = _transformer()
  pipe = _pipeline(transformer, cached=False)
  audio = torch.zeros(16000, dtype=torch.float32)

  with pytest.raises(ValueError, match="DMD LoRA"):
    pipe(audio=audio,
         num_frames=13,
         num_inference_steps=8,
         use_distill=True,
         text_guidance_scale=1.0,
         audio_guidance_scale=1.0,
         height=32,
         width=32)


def test_short_clip_is_rejected():
  transformer = _transformer()
  pipe = _pipeline(transformer)
  audio = torch.zeros(16000, dtype=torch.float32)

  with pytest.raises(ValueError, match="13 pixel frames"):
    pipe(audio=audio,
         num_frames=9,
         num_inference_steps=8,
         use_distill=False,
         text_guidance_scale=4.0,
         audio_guidance_scale=4.0,
         height=32,
         width=32)


def test_distill_schedule_length_is_fixed():
  transformer = _transformer()
  pipe = _pipeline(transformer)

  with pytest.raises(ValueError, match="exactly 8"):
    pipe.get_timesteps_sigmas(4, use_distill=True)

  assert pipe.get_timesteps_sigmas(8, use_distill=True).shape == (8, )
  assert pipe.get_timesteps_sigmas(12, use_distill=False).shape == (12, )
