"""End-to-end shape/flow test for the vendored Wan2.2-S2V pipeline (tiny random weights).

The chunk loop is the part that no unit test covers: the first chunk drops 3 warm-up frames, every
later chunk folds the last 73 decoded frames back in as motion latents, and the cache bookkeeping has
to restart per chunk. A wrong frame count or a stale motion latent still returns a video tensor, so
this test pins the shapes and the reset call.

Text and audio encoders are passed as `None` and the pre-computed embeddings are handed in: the
encoders are diffusers/transformers components with nothing model-specific in them.

Usage:
  python -m pytest tests/api/test_wan_s2v_pipeline_tiny.py -q
"""

import pytest
import torch

from cache_dit.models.pipelines.wan_s2v.pipeline import (
  WanSpeechToVideoPipeline,
  resize_min_center_crop,
)
from cache_dit.models.pipelines.wan_s2v.transformer import WanS2VTransformer3DModel

dim = 32
_HEIGHT = 64
_WIDTH = 64
_CHUNK_FRAMES = 8  # 8 - 3 warm-up frames = 5 visible frames per chunk
_TEXT_TOKENS = 8
_TEXT_DIM = 32


def _tiny_vae():
  from diffusers import AutoencoderKLWan

  torch.manual_seed(0)
  return AutoencoderKLWan(
    base_dim=8,
    z_dim=16,
    dim_mult=[1, 2, 4, 4],
    num_res_blocks=1,
    temperal_downsample=[False, True, True],
    latents_mean=[0.0] * 16,
    latents_std=[1.0] * 16,
  ).eval()


def _tiny_transformer():
  torch.manual_seed(0)
  return WanS2VTransformer3DModel(
    patch_size=(1, 2, 2),
    num_attention_heads=4,
    attention_head_dim=8,
    in_channels=16,
    out_channels=16,
    text_dim=_TEXT_DIM,
    freq_dim=32,
    audio_dim=1024,
    audio_inject_layers=(0, 1),
    enable_adain=True,
    adain_mode="attn_norm",
    pose_dim=16,
    ffn_dim=64,
    num_layers=2,
    num_weighted_avg_layers=25,
    cross_attn_norm=True,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
    added_kv_proj_dim=None,
    rope_max_seq_len=1024,
    enable_framepack=True,
    framepack_drop_mode="padd",
    add_last_motion=True,
    zero_timestep=True,
  ).eval()


def _tiny_pipeline() -> WanSpeechToVideoPipeline:
  from diffusers import UniPCMultistepScheduler

  pipe = WanSpeechToVideoPipeline(
    tokenizer=None,
    text_encoder=None,
    vae=_tiny_vae(),
    scheduler=UniPCMultistepScheduler(flow_shift=5.0),
    transformer=_tiny_transformer(),
    audio_encoder=None,
    audio_processor=None,
  )
  return pipe.to("cpu")


def _call(pipe, num_chunks=1):
  # One audio frame per latent frame: the causal encoder turns `_CHUNK_FRAMES` bucket frames into
  # `_CHUNK_FRAMES // 4 - latent_motion_frames + latent_motion_frames` frames, so mirror the shape
  # the real encoder produces: [1, layers, dim, num_frames_per_chunk].
  audio_embeds = torch.zeros(1, 25, 1024, num_chunks * _CHUNK_FRAMES, dtype=torch.float32)
  prompt_embeds = torch.zeros(1, 512, _TEXT_DIM, dtype=torch.float32)
  image = torch.zeros(1, 3, _HEIGHT, _WIDTH)
  return pipe(
    image=image,
    audio=None,
    sampling_rate=16000,
    prompt=None,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=torch.zeros(1, 512, _TEXT_DIM, dtype=torch.float32),
    audio_embeds=audio_embeds,
    height=_HEIGHT,
    width=_WIDTH,
    num_frames_per_chunk=_CHUNK_FRAMES,
    num_inference_steps=2,
    guidance_scale=1.0,
    output_type="np",
    return_dict=True,
  )


def test_single_chunk_frame_count():
  pipe = _tiny_pipeline()
  output = _call(pipe, num_chunks=1)
  frames = output.frames
  assert len(frames) == 1
  # 8 chunk frames minus the 3 warm-up frames dropped by the first chunk.
  assert len(frames[0]) == _CHUNK_FRAMES - 3


def test_chunk_loop_restarts_the_cache_context(monkeypatch):
  pipe = _tiny_pipeline()
  calls = []
  monkeypatch.setattr(pipe, "_reset_cache_context", lambda: calls.append(1))
  output = _call(pipe, num_chunks=2)
  # Only the second chunk restarts the bookkeeping; the first chunk is a fresh inference already.
  assert calls == [1]
  assert len(output.frames[0]) == 2 * _CHUNK_FRAMES - 3


def test_resize_min_center_crop_matches_the_official_preprocessing():
  from PIL import Image

  # Landscape 96x48 into 32x32: the short side becomes 32 and the excess width is centre-cropped.
  image = Image.new("RGB", (96, 48), color=(255, 0, 0))
  resized = resize_min_center_crop(image, 32, 32)
  assert resized.size == (32, 32)

  # Square source into a non-square target pads on the短 side with black, like torchvision.
  square = Image.new("RGB", (64, 64), color=(0, 255, 0))
  padded = resize_min_center_crop(square, 32, 16)
  assert padded.size == (16, 32)


def test_pipeline_rejects_a_missing_waveform():
  pipe = _tiny_pipeline()
  with pytest.raises(ValueError, match="audio"):
    pipe(
      image=torch.zeros(1, 3, _HEIGHT, _WIDTH),
      audio=None,
      sampling_rate=16000,
      prompt="x",
      height=_HEIGHT,
      width=_WIDTH,
      num_frames_per_chunk=_CHUNK_FRAMES,
      num_inference_steps=1,
    )


def test_from_pretrained_rejects_device_map(tmp_path):
  with pytest.raises(NotImplementedError, match="device_map"):
    WanSpeechToVideoPipeline.from_pretrained(str(tmp_path), device_map="balanced")
