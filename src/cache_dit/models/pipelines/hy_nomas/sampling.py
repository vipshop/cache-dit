"""Request-side HY-Nomas contracts: segment planning, caption canonicalisation, action windows.

Copied from ``sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas``
(request validation) and ``configs/pipeline_configs/nomas.py`` (tokenisation + sigmas).  The event
JSON schema, the crop of the chat-template prefix and the action-window derivation are part of the
checkpoint's conditioning contract, so they are reproduced verbatim; ``msgspec`` structs became
dataclasses.

Nothing here touches a model: these are pure functions over strings, tokenizers and frame counts,
which is what makes them testable without weights.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from ....logger import init_logger

logger = init_logger(__name__)

# Preserve upstream whitespace: this text is part of the conditioning contract.
NOMAS_SYSTEM_PROMPT = (
  "You are a helpful assistant. Describe the video by detailing the following aspects: "
  "        1. The main content and theme of the video. "
  "        2. The color, shape, size, texture, quantity, text, and spatial relationships of the "
  "objects. "
  "        3. Actions, events, behaviors temporal relationships, physical movement changes of the "
  "objects. "
  "        4. background environment, light, style and atmosphere. "
  "        5. camera angles, movements, and transitions used in the video."
)

DEFAULT_TEXT_LEN = 1280
DEFAULT_SEGMENT_LATENT_FRAMES = 10
DEFAULT_FLOW_SHIFT = 5.0
DEFAULT_CACHE_NOISE_SIGMA = 0.206897


@dataclass(frozen=True)
class NomasSegment:
  """One event segment: a latent-frame range and its caption.

  :param start: First latent frame (inclusive).
  :param end: Last latent frame (exclusive).
  :param caption: Canonicalised caption of this event.
  """

  start: int
  end: int
  caption: str


def plan_segments(
  num_frames: int,
  prompt: str,
  segments: Optional[Sequence[dict]],
  continuations: Optional[Sequence[str]],
  segment_size: int = DEFAULT_SEGMENT_LATENT_FRAMES,
) -> List[NomasSegment]:
  """Split a request into event segments.

  :param num_frames: Requested pixel frame count.
  :param prompt: Caption of the first event.
  :param segments: Optional explicit ``[{"frames": int, "caption": str}, ...]`` list.
  :param continuations: Optional captions for the events after the first.
  :param segment_size: Latent frames per event when ``segments`` is not given.
  :returns: The planned segments.
  """
  if type(num_frames) is not int or num_frames < 1:
    raise ValueError("num_frames must be a positive integer")
  latent_frames = (num_frames - 1) // 4 + 1
  if segments is None:
    starts = list(range(0, latent_frames, segment_size))
    captions = [prompt] * len(starts)
    if continuations is not None:
      if len(continuations) != len(starts) - 1:
        raise ValueError("continuation_prompts must contain one caption per segment after the "
                         "first")
      captions[1:] = continuations
    boundaries = starts + [latent_frames]
  else:
    if continuations is not None or not segments:
      raise ValueError("segments must be nonempty and cannot be combined with continuation_prompts")
    pixel_end = 0
    boundaries, captions = [0], []
    for segment in segments:
      if not isinstance(segment, dict) or set(segment) != {"frames", "caption"}:
        raise ValueError("each segment must contain exactly frames and caption; camera uses the "
                         "global norm_extrinsics timeline")
      frames = segment["frames"]
      if type(frames) is not int or frames < 1:
        raise ValueError("segment frames must be a positive integer")
      pixel_end += frames
      # Match causal_ar's pixel-boundary conversion, including min-one-frame.
      boundaries.append(max(boundaries[-1] + 1, min(pixel_end // 4, latent_frames)))
      captions.append(segment["caption"])
    if pixel_end != num_frames:
      raise ValueError(f"sum of segment frames ({pixel_end}) must equal num_frames ({num_frames})")
    boundaries[-1] = latent_frames
  if not all(isinstance(caption, str) and caption.strip() for caption in captions):
    raise ValueError("each Nomas caption must be a nonempty string")
  if any(a >= b for a, b in zip(boundaries, boundaries[1:])):
    raise ValueError("event boundaries produce empty latent segments; combine short events")
  return [NomasSegment(a, b, c) for a, b, c in zip(boundaries, boundaries[1:], captions)]


def canonical_caption(caption: str, prompt_format: str) -> str:
  """Validate the caption against the requested format and normalise the JSON key order.

  :param caption: Raw caption; JSON in ``json`` mode.
  :param prompt_format: ``json`` (official schema) or ``plain`` (smoke tests only).
  :returns: The caption to encode.
  """
  if prompt_format == "plain":
    return caption
  if prompt_format != "json":
    raise ValueError("prompt_format must be json or plain")
  try:
    obj = json.loads(caption)
  except json.JSONDecodeError as exc:
    raise ValueError("json mode requires an upstream segment JSON caption") from exc
  if (not isinstance(obj, dict) or not isinstance(obj.get("segments"), list)
      or len(obj["segments"]) != 1 or not isinstance(obj["segments"][0], dict)):
    raise ValueError("each caption must contain exactly one object in segments")
  event = obj["segments"][0]
  actions = event.pop("contained_actions", None)
  if actions is not None:
    if not isinstance(actions, (list, dict)):
      raise ValueError("contained_actions must be a list or ordered object")
    event["contained_actions"] = actions  # Must be last, as in upstream prompt builder.
  return json.dumps(obj, ensure_ascii=False)


def _seconds(value) -> float:
  if value is None:
    return 0.0
  parts = str(value).split(":")
  if not 1 <= len(parts) <= 3:
    raise ValueError("action timestamp must be seconds, MM:SS or HH:MM:SS")
  seconds = sum(float(part) * 60**i for i, part in enumerate(reversed(parts)))
  if not math.isfinite(seconds):
    raise ValueError("action timestamps must be finite")
  return seconds


def action_video_windows(actions: Sequence[dict], frames: int, fps: int):
  """Return ``None`` (full intra-segment attention) or the action window tiling.

  :param actions: ``contained_actions`` of the event.
  :param frames: Latent frames of the segment.
  :param fps: Request frame rate.
  :returns: ``[(start_frame, end_frame, action_index), ...]`` or ``None``.
  """
  spans = []
  for index, action in enumerate(actions):
    if not isinstance(action, dict) or not {"start_time", "end_time"} <= action.keys():
      logger.warning("Nomas action %d has no start_time/end_time; action mask disabled", index)
      return None
    start, end = (_seconds(action[key]) for key in ("start_time", "end_time"))
    a, b = (min(frames, int(max(t, 0.0) * fps) // 4) for t in (start, end))
    if b > a:
      spans.append((a, b, index))
  if not spans:
    return None
  windows, cursor = [], 0
  for a, b, index in sorted(spans):
    if a < cursor:
      logger.warning("Nomas action windows overlap; action mask disabled")
      return None
    if a > cursor:
      windows.append((cursor, a, -1))
    windows.append((a, b, index))
    cursor = b
  if cursor < frames:
    windows.append((cursor, frames, -1))
  if len(windows) == 1 and windows[0][2] == 0 and len(actions) == 1:
    return None
  return tuple(windows)


def action_metadata(caption: str, tokenizer, text_mask: torch.Tensor, frames: int, fps: int,
                    text_len: int = DEFAULT_TEXT_LEN):
  """Derive the head/action text offsets and the action windows of one event.

  :param caption: Canonical JSON caption.
  :param tokenizer: The text tokenizer.
  :param text_mask: ``[1, S]`` boolean mask of the encoded caption.
  :param frames: Latent frames of the segment.
  :param fps: Request frame rate.
  :param text_len: Text token budget.
  :returns: ``(action_start, text_offsets, video_windows)``.
  """
  obj = json.loads(caption)
  event = obj["segments"][0]
  actions = event.get("contained_actions", [])
  values = list(actions.values()) if isinstance(actions, dict) else actions
  if not values:
    return -1, None, None
  length = text_mask.shape[1]
  full = build_text_inputs(tokenizer, [caption], max_length=length)
  crop = int(full["crop"])
  ids = full["input_ids"][0].tolist()
  valid = int(text_mask.sum())
  offsets = []
  for k in range(len(values)):
    variant = dict(event)
    if k == 0:
      variant.pop("contained_actions")
    else:
      variant["contained_actions"] = (dict(list(actions.items())[:k])
                                      if isinstance(actions, dict) else actions[:k])
    text = json.dumps(obj | {"segments": [variant]}, ensure_ascii=False)
    other = build_text_inputs(tokenizer, [text], max_length=length)["input_ids"][0].tolist()
    common = next((i for i, (a, b) in enumerate(zip(ids, other)) if a != b), min(len(ids),
                                                                                len(other)))
    offset = max(0, min(common - crop, length - 1, valid))
    offsets.append(max(offset, offsets[-1] if offsets else 0))
  offsets.append(valid)
  windows = action_video_windows(values, frames, fps)
  if windows is None:
    return offsets[0], None, None
  return offsets[0], tuple(offsets), windows


def build_text_inputs(tokenizer, prompts: Sequence[str], max_length: int = DEFAULT_TEXT_LEN):
  """Chat-template tokenisation with the system-prompt prefix length returned as ``crop``.

  :param tokenizer: Qwen2.5-VL tokenizer with a right padding side.
  :param prompts: Event captions.
  :param max_length: Token budget **after** the crop.
  :returns: ``{"input_ids", "attention_mask", "crop"}``.
  """
  if tokenizer.padding_side != "right":
    raise ValueError("Nomas requires right-padded text")
  conversations = [
    [
      {"role": "system", "content": NOMAS_SYSTEM_PROMPT},
      {"role": "user", "content": text or " "},
    ] for text in prompts
  ]
  marker = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
  ids = tokenizer.apply_chat_template(conversations[0], tokenize=True, add_generation_prompt=True,
                                      return_dict=False)
  crop = next((i + len(marker) for i in range(len(ids) - len(marker) + 1)
               if ids[i:i + len(marker)] == marker), None)
  if crop is None:
    raise ValueError("Tokenizer must provide the Qwen user chat marker")
  inputs = tokenizer.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True,
                                         return_dict=True, truncation=True,
                                         max_length=int(max_length) + crop, padding="max_length",
                                         return_tensors="pt")
  inputs["crop"] = crop
  return inputs


def postprocess_text(hidden_states, text_inputs):
  """Take the third-from-last hidden state and drop the chat-template prefix.

  :param hidden_states: Tuple/list of per-layer text hidden states.
  :param text_inputs: Output of :func:`build_text_inputs`.
  :returns: ``(text_states [1, S, 3584], attention_mask [1, S])``.
  """
  crop = int(text_inputs["crop"])
  # The tokenizer returns a 0/1 long mask; the cache and the attention plan need booleans.
  return (hidden_states[-3][:, crop:], text_inputs["attention_mask"][:, crop:].bool())


def nomas_sigmas(steps: int = 8, shift: float = DEFAULT_FLOW_SHIFT) -> torch.Tensor:
  """Flow-match sigma schedule of the distilled checkpoint (``steps + 1`` values ending at 0).

  :param steps: Euler steps.
  :param shift: Flow shift; the released checkpoint uses 5.
  :returns: The sigma schedule.
  """
  if steps < 1 or not math.isfinite(shift) or shift <= 0:
    raise ValueError("steps and shift must be positive and finite")
  sigmas = torch.linspace(1, 0, steps + 1, dtype=torch.float32)
  return shift * sigmas / (1 + (shift - 1) * sigmas)


def cache_capacity(shape: Sequence[int], text_len: int) -> int:
  """Token capacity needed by the committed segments of one request.

  :param shape: ``(frames, height, width)`` of the latent segment.
  :param text_len: Text tokens per segment.
  :returns: Token count to allocate.
  """
  return math.prod(shape) + text_len


def cache_bytes_per_token(num_layers: int, hidden_size: int, head_shards: int = 1) -> int:
  """Per-rank K/V bytes of one cached token (bf16).

  :param num_layers: Transformer depth.
  :param hidden_size: Residual width.
  :param head_shards: ``tp_size * sp_size``.
  :returns: Bytes per cached token.
  """
  return 2 * num_layers * hidden_size * 2 // head_shards


def check_cache_budget(capacity: int, num_layers: int, hidden_size: int, head_shards: int,
                       budget_gib: float, free_bytes: Optional[int] = None,
                       reserve_gib: float = 8.0) -> int:
  """Bound the persistent K/V before allocating it.

  :param capacity: Token capacity.
  :param num_layers: Transformer depth.
  :param hidden_size: Residual width.
  :param head_shards: ``tp_size * sp_size``.
  :param budget_gib: Configured budget in GiB.
  :param free_bytes: Free device memory, if known.
  :param reserve_gib: Memory to leave free on the device.
  :returns: The required bytes.
  :raises ValueError: When the request needs more than the budget.
  """
  required = capacity * cache_bytes_per_token(num_layers, hidden_size, head_shards)
  limit = budget_gib * 2**30
  if free_bytes is not None:
    limit = min(limit, max(0, free_bytes - reserve_gib * 2**30))
  if required > limit:
    raise ValueError(f"Nomas KV requires {required / 2**30:.2f} GiB, budget "
                     f"{limit / 2**30:.2f} GiB; reduce resolution, event count or text length.")
  return required


def canonical_camera(norm_extrinsics, latent_frames: int, device="cpu"):
  """Validate the canonical normalized extrinsics timeline ``(L-1)*4+1``.

  :param norm_extrinsics: Array-like ``[F, 4, 4]`` or ``None``.
  :param latent_frames: Latent frames of the whole clip.
  :param device: Target device.
  :returns: The validated tensor or ``None``.
  """
  if norm_extrinsics is None:
    return None
  camera = torch.as_tensor(norm_extrinsics, device=device, dtype=torch.float32)
  expected = (latent_frames - 1) * 4 + 1
  if camera.shape != (expected, 4, 4) or not torch.isfinite(camera).all():
    raise ValueError(f"norm_extrinsics must be finite [{expected},4,4] on the canonical "
                     f"(L-1)*4+1 timeline")
  return camera


def segment_camera_embedding(camera_embedding, start: int, end: int):
  """Slice the whole-clip camera tokens for one segment.

  :param camera_embedding: ``[1, L, 64]`` embedding of the whole clip, or ``None``.
  :param start: First latent frame of the segment.
  :param end: Last latent frame (exclusive).
  :returns: ``[1, end - start, 64]`` or ``None``.
  """
  if camera_embedding is None:
    return None
  return camera_embedding[:, start:end]


def to_unit_range(video: torch.Tensor) -> torch.Tensor:
  """Map decoder pixels to ``[0, 1]`` floats (upstream ``image / 2 + 0.5``).

  :param video: Decoder output.
  :returns: The clamped unit-range tensor.
  """
  return (video.float() / 2 + 0.5).clamp(0, 1)
