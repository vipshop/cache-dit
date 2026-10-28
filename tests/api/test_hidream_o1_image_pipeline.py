"""CPU regression tests for the built-in HiDream-O1-Image pipeline.

The pipeline is vendored (`cache_dit.models.pipelines.hidream_o1_image`) because the checkpoint is a
Qwen3-VL repository: no `model_index.json`, no VAE, no text encoder, FP32 safetensors named
`model.language_model.*`. These tests lock the request construction (chat template + `<|boi_token|>`
+ `<|tms_token|>` + image slots, interleaved mRoPE with the `fix_point` offset), the mixed
causal/full mask, the CFG bookkeeping, the cache guard, and the strict weight mapping.

Usage:
  python -m pytest tests/api/test_hidream_o1_image_pipeline.py -q
"""

import os

import pytest
import torch
from safetensors.torch import save_file

from cache_dit.models.pipelines.hidream_o1_image import (
  HiDreamO1ImagePipeline,
  HiDreamO1ImageScheduler,
  HiDreamO1ImageTransformer2DModel,
)
from cache_dit.models.pipelines.hidream_o1_image.pipeline import (
  find_closest_resolution,
  load_hidream_o1_state_dict,
  patchify,
  unpatchify,
)

PATCH_SIZE = 32
TMS_TOKEN_ID = 100
BOI_TOKEN_ID = 99
HEIGHT = 64
WIDTH = 64
IMAGE_LEN = (HEIGHT // PATCH_SIZE) * (WIDTH // PATCH_SIZE)


class _FakeTokenizer:
  """Minimal tokenizer: maps the two special tokens to single ids and every other char to `3..`."""

  chat_template = "fake"
  boi_token = "<|boi_token|>"
  tms_token = "<|tms_token|>"
  bor_token = "<|bor_token|>"
  eor_token = "<|eor_token|>"
  bot_token = "<|bot_token|>"

  def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
    return f"<user>{messages[0]['content']}</user>"

  def encode(self, text, return_tensors=None, add_special_tokens=False):
    ids = []
    index = 0
    while index < len(text):
      if text.startswith(self.boi_token, index):
        ids.append(BOI_TOKEN_ID)
        index += len(self.boi_token)
      elif text.startswith(self.tms_token, index):
        ids.append(TMS_TOKEN_ID)
        index += len(self.tms_token)
      else:
        if not text[index].isspace():
          ids.append(3 + (ord(text[index]) % 60))
        index += 1
    if return_tensors == "pt":
      return torch.tensor([ids], dtype=torch.long)
    return ids


def _tiny_transformer(seed: int = 0) -> HiDreamO1ImageTransformer2DModel:
  torch.manual_seed(seed)
  return HiDreamO1ImageTransformer2DModel(
    hidden_size=64,
    intermediate_size=128,
    num_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=128,
    patch_size=PATCH_SIZE,
    in_channels=3,
    tms_token_id=TMS_TOKEN_ID,
  ).eval()


def _tiny_pipeline(**kwargs) -> HiDreamO1ImagePipeline:
  return HiDreamO1ImagePipeline(
    transformer=_tiny_transformer(),
    scheduler=HiDreamO1ImageScheduler(shift=3.0),
    tokenizer=_FakeTokenizer(),
    **kwargs,
  )


def _count_forwards(pipeline) -> list:
  calls = []
  original = pipeline.transformer.forward

  def counting(*args, **kwargs):
    calls.append(1)
    return original(*args, **kwargs)

  pipeline.transformer.forward = counting
  return calls


def test_text_sample_builds_the_joint_sequence():
  pipeline = _tiny_pipeline()
  sample = pipeline.build_text_sample("a cat", HEIGHT, WIDTH)

  assert sample["image_length"] == IMAGE_LEN
  # Text tokens end with <|boi_token|> and the timestep token, then the image slots.
  assert sample["input_ids"].shape[1] == sample["text_length"]
  assert sample["position_ids"].shape == (3, 1, sample["text_length"] + IMAGE_LEN)
  assert sample["attention_mask"].shape == (1, 1, sample["text_length"] + IMAGE_LEN,
                                            sample["text_length"] + IMAGE_LEN)
  # The image block is shifted to the release's `fix_point` offset instead of continuing the text
  # counter.
  image_positions = sample["position_ids"][:, 0, sample["text_length"]:]
  assert int(image_positions.min()) >= 4096


def test_attention_mask_is_causal_on_text_rows_only():
  pipeline = _tiny_pipeline()
  sample = pipeline.build_text_sample("a cat", HEIGHT, WIDTH)
  text_length = sample["text_length"]
  mask = sample["attention_mask"][0, 0]
  blocked = mask < -1e30

  # Text rows attend only up to themselves.
  for row in range(text_length - 1):
    assert blocked[row, row + 1:].all()
    assert not blocked[row, :row + 1].any()
  # The timestep row and every image row attend the whole sequence.
  assert not blocked[text_length - 1].any()
  assert not blocked[-1].any()


def test_call_runs_two_forwards_per_step_with_cfg():
  pipeline = _tiny_pipeline()
  calls = _count_forwards(pipeline)
  image = pipeline("a cat",
                   height=HEIGHT,
                   width=WIDTH,
                   num_inference_steps=2,
                   guidance_scale=5.0,
                   snap_resolution=False).images[0]
  assert image.size == (WIDTH, HEIGHT)
  assert len(calls) == 4, f"expected cond+uncond per step, got {len(calls)} forwards"


def test_call_runs_one_forward_without_cfg():
  pipeline = _tiny_pipeline()
  calls = _count_forwards(pipeline)
  pipeline("a cat",
           height=HEIGHT,
           width=WIDTH,
           num_inference_steps=2,
           guidance_scale=1.0,
           snap_resolution=False)
  assert len(calls) == 2


def test_cache_guard_rejects_missing_cfg():
  pipeline = _tiny_pipeline()
  pipeline.transformer._is_cached = True
  with pytest.raises(ValueError, match="one conditional and one unconditional forward"):
    pipeline("a cat",
             height=HEIGHT,
             width=WIDTH,
             num_inference_steps=1,
             guidance_scale=1.0,
             snap_resolution=False)


def test_output_types():
  pipeline = _tiny_pipeline()
  tensor = pipeline("a cat",
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=1,
                    guidance_scale=5.0,
                    snap_resolution=False,
                    output_type="pt").images
  assert tensor.shape == (1, 3, HEIGHT, WIDTH)
  assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0

  latent = pipeline("a cat",
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=1,
                    guidance_scale=5.0,
                    snap_resolution=False,
                    output_type="latent").images
  assert latent.shape == (1, IMAGE_LEN, 3 * PATCH_SIZE * PATCH_SIZE)


def test_resolution_snapping_and_patch_multiple():
  assert find_closest_resolution(1000, 1000) == (2048, 2048)
  assert find_closest_resolution(1920, 1080) == (2560, 1440)
  pipeline = _tiny_pipeline()
  with pytest.raises(ValueError, match="multiples of 32"):
    pipeline("a cat", height=HEIGHT + 1, width=WIDTH, num_inference_steps=1, snap_resolution=False)


def test_patchify_round_trip():
  pixels = torch.randn(1, 3, HEIGHT, WIDTH)
  patches = patchify(pixels)
  assert patches.shape == (1, IMAGE_LEN, 3 * PATCH_SIZE * PATCH_SIZE)
  assert torch.equal(unpatchify(patches, HEIGHT, WIDTH), pixels)


def _checkpoint_keys(transformer) -> dict:
  state = {}
  for name, tensor in transformer.state_dict().items():
    if name.startswith(("embed_tokens.", "layers.", "norm.")):
      state[f"model.language_model.{name}"] = tensor
    else:
      state[f"model.{name}"] = tensor
  return state


def test_weight_loader_maps_and_validates_strictly(tmp_path):
  source = _tiny_transformer(seed=1)
  state = _checkpoint_keys(source)
  # Extra tensors the release ships and cache-dit must drop.
  state["lm_head.weight"] = torch.zeros(4, 4)
  state["model.visual.blocks.0.attn.qkv.weight"] = torch.zeros(4, 4)
  save_file(state, os.path.join(tmp_path, "model-00001-of-00001.safetensors"))

  target = _tiny_transformer(seed=2)
  load_hidream_o1_state_dict(target, str(tmp_path), dtype=torch.float32)
  for name, tensor in source.state_dict().items():
    assert torch.equal(target.state_dict()[name], tensor), f"{name} did not load"


def test_weight_loader_rejects_a_missing_tensor(tmp_path):
  source = _tiny_transformer(seed=1)
  state = _checkpoint_keys(source)
  state.pop("model.t_embedder1.mlp.0.weight")
  save_file(state, os.path.join(tmp_path, "model-00001-of-00001.safetensors"))
  with pytest.raises(ValueError, match="missing"):
    load_hidream_o1_state_dict(_tiny_transformer(seed=2), str(tmp_path), dtype=torch.float32)


def test_transformer_config_is_flat():
  # `register_to_config` must receive scalars: housing a `Qwen3VLConfig` here would break
  # `save_pretrained` and hide the real depth from cache-dit.
  transformer = _tiny_transformer()
  config = dict(transformer.config)
  assert config["num_layers"] == 2
  assert "text_config" not in config
  for key, value in config.items():
    assert not hasattr(value, "to_dict"), f"{key} holds a nested config object"
