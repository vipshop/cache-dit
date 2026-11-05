"""Tests for the HY-Nomas request contracts (segment planning, captions, sigmas).

These are the parts of the conditioning pipeline that silently change the model's input
distribution: the event→latent boundary conversion, the JSON key order of the caption, the crop of
the chat-template prefix and the action windows derived from timestamps.  They need no weights.
"""
import json

import pytest
import torch

from cache_dit.models.pipelines.hy_nomas.sampling import (
  NOMAS_SYSTEM_PROMPT,
  action_metadata,
  action_video_windows,
  build_text_inputs,
  cache_bytes_per_token,
  canonical_camera,
  canonical_caption,
  check_cache_budget,
  nomas_sigmas,
  plan_segments,
  postprocess_text,
  segment_camera_embedding,
  to_unit_range,
)


class _FakeTokenizer:
  """Minimal Qwen-style tokenizer: one token per character plus the user marker."""

  padding_side = "right"
  marker = "<|im_start|>user\n"

  def encode(self, text, add_special_tokens=False):
    return [ord(char) for char in text]

  def apply_chat_template(self, conversations, tokenize=True, add_generation_prompt=True,
                          return_dict=False, truncation=False, max_length=None, padding=None,
                          return_tensors=None):
    if isinstance(conversations[0], dict):
      conversations = [conversations]
    rows = []
    for conversation in conversations:
      prefix = f"{NOMAS_SYSTEM_PROMPT}{self.marker}"
      ids = [ord(char) for char in prefix]
      for message in conversation[1:]:
        ids += [ord(char) for char in message["content"]]
      rows.append(ids)
    crop = len(list(f"{NOMAS_SYSTEM_PROMPT}{self.marker}"))
    width = max_length or max(len(row) for row in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    mask = torch.zeros(len(rows), width, dtype=torch.long)
    for index, row in enumerate(rows):
      row = row[:width]
      ids[index, :len(row)] = torch.tensor(row, dtype=torch.long)
      mask[index, :len(row)] = 1
    if not return_dict:
      return ids[0].tolist()
    return {"input_ids": ids, "attention_mask": mask, "crop": crop}


def _event(caption_fields=None, actions=None):
  event = {
    "Style": "Cinematic",
    "segments": [{
      "visual_description_long": "A fox walks."
    }],
  }
  if caption_fields:
    event.update(caption_fields)
  if actions is not None:
    event["segments"][0]["contained_actions"] = actions
  return json.dumps(event)


def test_plan_segments_uniform_and_explicit():
  uniform = plan_segments(40, "a fox", None, None, segment_size=10)
  assert [(s.start, s.end) for s in uniform] == [(0, 10)]
  many = plan_segments(240, "a fox", None, ["a", "b", "c", "d", "e"], segment_size=10)
  # 240 pixel frames -> 60 latents -> six segments of ten, one continuation per later segment.
  assert [(s.start, s.end) for s in many] == [(i, i + 10) for i in range(0, 60, 10)]
  assert [s.caption for s in many][:3] == ["a fox", "a", "b"]
  with pytest.raises(ValueError, match="one caption per segment"):
    plan_segments(240, "a fox", None, ["a"], segment_size=10)
  explicit = plan_segments(240, "a fox", [{"frames": 120, "caption": "a"},
                                          {"frames": 120, "caption": "b"}], None)
  assert [(s.start, s.end) for s in explicit] == [(0, 30), (30, 60)]
  with pytest.raises(ValueError, match="must equal num_frames"):
    plan_segments(240, "x", [{"frames": 120, "caption": "a"}], None)
  with pytest.raises(ValueError, match="cannot be combined"):
    plan_segments(240, "x", [{"frames": 240, "caption": "a"}], ["b"])


def test_canonical_caption_keeps_actions_last():
  raw = _event(actions=[{"start_time": "00:00.000", "end_time": "00:01.000",
                         "action_description": "walks"}])
  out = canonical_caption(raw, "json")
  event = json.loads(out)["segments"][0]
  assert list(event)[-1] == "contained_actions"
  assert canonical_caption("free text", "plain") == "free text"
  with pytest.raises(ValueError, match="exactly one object"):
    canonical_caption(json.dumps({"segments": [{}, {}]}), "json")
  with pytest.raises(ValueError, match="must be json or plain"):
    canonical_caption("x", "yaml")


def test_action_windows_tile_the_segment():
  actions = [
    {"start_time": "00:00.000", "end_time": "00:01.000"},
    {"start_time": "00:01.000", "end_time": "00:02.000"},
  ]
  # fps 24: frame 1 s -> int(24)//4 = 6 latent frames.
  windows = action_video_windows(actions, frames=12, fps=24)
  assert windows is not None
  assert windows[0][0] == 0 and windows[-1][1] == 12
  assert all(windows[i][1] == windows[i + 1][0] for i in range(len(windows) - 1))
  # Overlapping actions degrade to full attention instead of failing.
  assert action_video_windows([{"start_time": "0", "end_time": "2"},
                               {"start_time": "1", "end_time": "3"}], 12, 24) is None
  assert action_video_windows([{"start_time": "0"}], 12, 24) is None


def test_action_metadata_offsets_follow_the_common_prefix():
  actions = [
    {"start_time": "00:00.000", "end_time": "00:01.000", "action_description": "walks"},
    {"start_time": "00:01.000", "end_time": "00:02.000", "action_description": "sits"},
  ]
  caption = canonical_caption(_event(actions=actions), "json")
  tokenizer = _FakeTokenizer()
  inputs = build_text_inputs(tokenizer, [caption], max_length=64)
  mask = inputs["attention_mask"].bool()
  start, offsets, windows = action_metadata(caption, tokenizer, mask, frames=12, fps=24)
  assert offsets is not None and len(offsets) == len(actions) + 1
  assert offsets == tuple(sorted(offsets))
  assert offsets[-1] == int(mask.sum())
  assert start == offsets[0]
  assert windows is not None


def test_build_text_inputs_crops_the_system_prompt():
  tokenizer = _FakeTokenizer()
  inputs = build_text_inputs(tokenizer, ["a fox"], max_length=16)
  crop = inputs["crop"]
  assert crop == len(f"{NOMAS_SYSTEM_PROMPT}{_FakeTokenizer.marker}")
  assert inputs["input_ids"].shape == (1, crop + 16)
  # `postprocess_text` takes `hidden_states[-3]`, so the list must have >= 3 entries.
  hidden = [torch.zeros(1, crop + 16, 3584)] * 3
  states, mask = postprocess_text(hidden, inputs)
  assert states.shape == (1, 16, 3584)
  assert mask.shape == (1, 16)
  with pytest.raises(ValueError, match="right-padded"):
    broken = _FakeTokenizer()
    broken.padding_side = "left"
    build_text_inputs(broken, ["a"], max_length=8)


def test_build_text_inputs_rejects_a_tokenizer_without_the_marker():
  """The crop is found by locating the Qwen user marker; without it the crop is undefined."""

  class _NoMarkerTokenizer(_FakeTokenizer):
    def apply_chat_template(self, conversations, **kwargs):
      result = super().apply_chat_template(conversations, **kwargs)
      if isinstance(result, dict):
        result = dict(result)
        result.pop("crop", None)
        result["input_ids"] = result["input_ids"] + 7  # no code point matches the marker
        return result
      # `return_dict=False` path: a plain id list that must not contain the marker either.
      return [token + 7 for token in result]

  with pytest.raises(ValueError, match="Qwen user chat marker"):
    build_text_inputs(_NoMarkerTokenizer(), ["a fox"], max_length=8)


def test_sigmas_end_at_zero_and_are_shifted():
  sigmas = nomas_sigmas(8, shift=5.0)
  assert sigmas.shape == (9, )
  assert sigmas[0] == pytest.approx(1.0)
  assert sigmas[-1] == 0.0
  assert torch.all(sigmas[:-1] > sigmas[1:])
  with pytest.raises(ValueError):
    nomas_sigmas(0)


def test_cache_budget_math():
  # 2 * layers * hidden * 2 bytes, divided by the head shards.
  assert cache_bytes_per_token(54, 2048) == 2 * 54 * 2048 * 2
  assert cache_bytes_per_token(54, 2048, head_shards=4) == 2 * 54 * 2048 * 2 // 4
  required = check_cache_budget(1000, 54, 2048, 1, budget_gib=32.0)
  assert required == 1000 * 2 * 54 * 2048 * 2
  with pytest.raises(ValueError, match="reduce resolution"):
    check_cache_budget(10**7, 54, 2048, 1, budget_gib=1.0)


def test_camera_timeline_and_segment_slice():
  with pytest.raises(ValueError, match="canonical"):
    canonical_camera(torch.zeros(4, 4, 4), latent_frames=3)
  camera = canonical_camera(torch.zeros((3 - 1) * 4 + 1, 4, 4), latent_frames=3)
  assert camera.shape == (9, 4, 4)
  embedding = torch.arange(1 * 3 * 64, dtype=torch.float32).reshape(1, 3, 64)
  assert torch.equal(segment_camera_embedding(embedding, 1, 3), embedding[:, 1:3])
  assert segment_camera_embedding(None, 0, 1) is None


def test_unit_range_matches_upstream_formula():
  video = torch.tensor([[[[-2.0, 0.0, 2.0]]]])
  out = to_unit_range(video)
  assert torch.allclose(out, torch.tensor([[[[0.0, 0.5, 1.0]]]]))
