"""Built-in HiDream-O1-Image text-to-image pipeline.

HiDream-O1-Image ships no diffusers pipeline: its checkpoint is a Qwen3-VL repository
(`architectures=["Qwen3VLForConditionalGeneration"]`, `model_type="qwen3_vl"`, FP32 safetensors,
no `model_index.json`), and the release code lives in the model repo. This module vendors the
inference path so cache-dit can run the model without remote code.

Note:
  Keep the constraints below intact, cache-dit's dispatch depends on them:

  * :class:`HiDreamO1ImagePipeline` must inherit `DiffusionPipeline` and
    :class:`HiDreamO1ImageTransformer2DModel` must inherit `ModelMixin` + `ConfigMixin`.
    cache-dit resolves pipelines through `is_pipeline()` and patches `pipe.__class__.__call__` to
    reset the cache context once per inference.
  * The block loop must call one block per iteration and nothing else, so the cached-block wrapper
    reproduces the exact computation.
  * There is no VAE and no text encoder: the model embeds its own token ids and predicts raw RGB
    patches. `--parallel-vae` and `--parallel-text` are therefore rejected by the CLI example
    instead of being silently skipped.
"""
from __future__ import annotations

import glob
import json
import os
from typing import List, Optional, Tuple

import torch
from diffusers import DiffusionPipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from PIL import Image
from safetensors import safe_open
from transformers import AutoTokenizer

from ....logger import init_logger
from .scheduler import HiDreamO1ImageScheduler
from .transformer import HiDreamO1ImageTransformer2DModel

logger = init_logger(__name__)

_INFERENCE_DTYPE = torch.bfloat16

# Sampling constants, propagated from the release sampler through vllm-omni
# (`hidream_o1_image/pipeline_hidream_o1_image.py` L36-39, tagged upstream `@21bcd30471ac`):
#   PATCH_SIZE = 32, TIMESTEP_TOKEN_NUM = 1, NOISE_SCALE = 8.0, T_EPS = 0.001
# and `FIX_POINT = 4096` from `models/utils.py` (`get_rope_index_fix_point(fix_point=4096)`).
PATCH_SIZE = 32
TIMESTEP_TOKEN_NUM = 1
NOISE_SCALE = 8.0
T_EPS = 0.001
FIX_POINT = 4096

# Resolutions the checkpoint was trained on; a request is snapped to the closest **aspect ratio**
# and then uses that entry's full size (1024x1024 has ratio 1.0, so it renders at 2048x2048).
# Pass `snap_resolution=False` to keep an arbitrary multiple of `PATCH_SIZE`.
PREDEFINED_RESOLUTIONS: Tuple[Tuple[int, int], ...] = (
  (2048, 2048),
  (2304, 1728),
  (1728, 2304),
  (2560, 1440),
  (1440, 2560),
  (2496, 1664),
  (1664, 2496),
  (3104, 1312),
  (1312, 3104),
  (2304, 1792),
  (1792, 2304),
)

_SPECIAL_TOKENS = ("boi_token", "bor_token", "eor_token", "bot_token", "tms_token")
_SKIPPED_WEIGHT_PREFIXES = ("model.visual.", )
_SKIPPED_WEIGHT_KEYS = ("lm_head.weight", )
# Vision placeholders of the HiDream-O1-Image checkpoint, used when `config.json` omits them.
_DEFAULT_TOKEN_IDS = {
  "image_token_id": 151655,
  "video_token_id": 151656,
  "vision_start_token_id": 151652,
}


def find_closest_resolution(width: int, height: int) -> Tuple[int, int]:
  """Snap a requested size to the closest trained aspect ratio.

  :param width: Requested width.
  :param height: Requested height.
  :returns:`(width, height)` of the closest predefined resolution.
  """
  img_ratio = width / height
  best_res = None
  min_diff = float("inf")
  for w, h in PREDEFINED_RESOLUTIONS:
    diff = abs(w / h - img_ratio)
    if diff < min_diff:
      min_diff = diff
      best_res = (w, h)
  return best_res


def get_rope_index_fix_point(
  *,
  spatial_merge_size: int,
  image_token_id: int,
  video_token_id: int,
  vision_start_token_id: int,
  input_ids: torch.Tensor,
  image_grid_thw: torch.Tensor,
  video_grid_thw: Optional[torch.Tensor],
  attention_mask: Optional[torch.Tensor],
  skip_vision_start_token: List[int],
  fix_point: int = FIX_POINT,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Build interleaved mRoPE position ids with the reference's `fix_point` offset.

  Image tokens are placed at a fixed offset (`fix_point`) instead of continuing the text counter,
  which is what the checkpoint was trained with. Ported verbatim from the release sampler (via
  vllm-omni) so the table stays bit-identical.

  :param spatial_merge_size: Spatial merge size of the vision tower; `1` for this model.
  :param image_token_id: Token id reserved for image placeholders.
  :param video_token_id: Token id reserved for video placeholders.
  :param vision_start_token_id: Token id marking the first image placeholder.
  :param input_ids: Text token ids followed by the image placeholders, `[B, S]`.
  :param image_grid_thw: `[num_images, 3]` grids of the images.
  :param video_grid_thw: `[num_videos, 3]` grids of the videos, or `None`.
  :param attention_mask: Optional `[B, S]` padding mask.
  :param skip_vision_start_token: Per-image flags that drop the vision-start token from the text
    counter.
  :param fix_point: Offset applied to the image position ids.
  :returns: `(position_ids [3, B, S], mrope_position_deltas [B, 1])`.
  """
  if video_grid_thw is not None:
    video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
    video_grid_thw[:, 0] = 1

  mrope_position_deltas = []
  total_input_ids = input_ids
  if attention_mask is None:
    attention_mask = torch.ones_like(total_input_ids)
  position_ids = torch.ones(
    3,
    input_ids.shape[0],
    input_ids.shape[1],
    dtype=input_ids.dtype,
    device=input_ids.device,
  )
  image_index, video_index = 0, 0
  attention_mask = attention_mask.to(total_input_ids.device)
  for i, input_ids_row in enumerate(total_input_ids):
    input_ids_row = input_ids_row[attention_mask[i] == 1]
    vision_start_indices = torch.argwhere(input_ids_row == vision_start_token_id).squeeze(1)
    vision_tokens = input_ids_row[vision_start_indices + 1]
    image_nums = int((vision_tokens == image_token_id).sum())
    video_nums = int((vision_tokens == video_token_id).sum())
    input_tokens = input_ids_row.tolist()
    llm_pos_ids_list: List[torch.Tensor] = []
    st = 0
    remain_images, remain_videos = image_nums, video_nums
    for _ in range(image_nums + video_nums):
      if image_token_id in input_tokens and remain_images > 0:
        ed_image = input_tokens.index(image_token_id, st)
      else:
        ed_image = len(input_tokens) + 1
      if video_token_id in input_tokens and remain_videos > 0:
        ed_video = input_tokens.index(video_token_id, st)
      else:
        ed_video = len(input_tokens) + 1
      if ed_image < ed_video:
        t, h, w = (
          image_grid_thw[image_index][0],
          image_grid_thw[image_index][1],
          image_grid_thw[image_index][2],
        )
        image_index += 1
        remain_images -= 1
        ed = ed_image
      else:
        t, h, w = (
          video_grid_thw[video_index][0],
          video_grid_thw[video_index][1],
          video_grid_thw[video_index][2],
        )
        video_index += 1
        remain_videos -= 1
        ed = ed_video
      llm_grid_t, llm_grid_h, llm_grid_w = (
        t.item(),
        h.item() // spatial_merge_size,
        w.item() // spatial_merge_size,
      )
      text_len = ed - st
      text_len -= skip_vision_start_token[image_index - 1]
      text_len = max(0, text_len)

      st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
      llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

      t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
      h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
      w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()

      if skip_vision_start_token[image_index - 1]:
        if fix_point > 0:
          fix_point = fix_point - st_idx
          llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + fix_point + st_idx)
          fix_point = 0
      else:
        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
      st = ed + llm_grid_t * llm_grid_h * llm_grid_w

    if st < len(input_tokens):
      st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
      text_len = len(input_tokens) - st
      llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

    llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
    mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
  mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
  return position_ids, mrope_position_deltas


def build_hidream_o1_attention_mask(token_types: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
  """Build the mixed causal/full mask once per request.

  Text rows stay causal; the timestep row and every image row get full attention. The release uses
  exactly this dense form whenever the backend has no piecewise-span support.

  :param token_types: `[B, S]` tensor, `0` for text rows and `> 0` for full rows.
  :param dtype: Mask dtype; matches the activation dtype.
  :returns: Additive mask `[B, 1, S, S]`.
  """
  if token_types.ndim == 1:
    token_types = token_types.unsqueeze(0)
  batch_size, seq_len = token_types.shape
  min_value = torch.finfo(dtype).min
  attention_mask = torch.triu(
    torch.full((batch_size, seq_len, seq_len), min_value, dtype=dtype, device=token_types.device),
    diagonal=1,
  )
  attention_mask[token_types.bool()] = 0
  return attention_mask.unsqueeze(1)


def patchify(pixels: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
  """Flatten an image into `patch_size x patch_size` RGB patches.

  :param pixels: Image tensor `[B, 3, H, W]`.
  :param patch_size: Spatial size of one patch.
  :returns: Patches `[B, (H/p) * (W/p), 3 * p * p]`.
  """
  batch, channels, height, width = pixels.shape
  h, w = height // patch_size, width // patch_size
  return pixels.view(batch, channels, h, patch_size, w,
                     patch_size).permute(0, 2, 4, 1, 3,
                                         5).reshape(batch, h * w,
                                                    channels * patch_size * patch_size)


def unpatchify(patches: torch.Tensor,
               height: int,
               width: int,
               patch_size: int = PATCH_SIZE) -> torch.Tensor:
  """Restore an image from flattened patches.

  :param patches: Patches `[B, (H/p) * (W/p), 3 * p * p]`.
  :param height: Output image height.
  :param width: Output image width.
  :param patch_size: Spatial size of one patch.
  :returns: Image tensor `[B, 3, H, W]`.
  """
  batch = patches.shape[0]
  h, w = height // patch_size, width // patch_size
  return patches.view(batch, h, w, 3, patch_size,
                      patch_size).permute(0, 3, 1, 4, 2, 5).reshape(batch, 3, height, width)


def _load_chat_template(tokenizer, root: str) -> None:
  """Attach the release's chat template when the tokenizer did not read it itself.

  The shipped `tokenizer_config.json` already carries an inline `chat_template`, so this is a no-op
  for the released checkpoint; it stays as a fallback for exports that only ship the sibling
  `chat_template.json` / `chat_template.jinja` files.

  :param tokenizer: Loaded tokenizer, modified in place.
  :param root: Checkpoint directory.
  """
  if getattr(tokenizer, "chat_template", None) is not None:
    return
  for name in ("chat_template.json", "chat_template.jinja"):
    path = os.path.join(root, name)
    if not os.path.isfile(path):
      continue
    if name.endswith(".json"):
      with open(path, encoding="utf-8") as file:
        payload = json.load(file)
      tokenizer.chat_template = payload.get("chat_template") if isinstance(payload, dict) else None
    else:
      with open(path, encoding="utf-8") as file:
        tokenizer.chat_template = file.read()
    if tokenizer.chat_template is not None:
      logger.debug(f"Attached chat template from {path}.")
      return


def _map_weight_key(key: str) -> Optional[str]:
  """Map a checkpoint key onto this module's parameter name.

  :param key: Raw checkpoint key.
  :returns: Local parameter name, or `None` when the key is not part of the denoiser.
  """
  if key.startswith("model.language_model."):
    return key[len("model.language_model."):]
  if key.startswith("model."):
    return key[len("model."):]
  return None


def _resolve_shards(root: str) -> List[str]:
  """List the safetensors shards of a checkpoint root.

  :param root: Checkpoint directory.
  :returns: Sorted absolute shard paths.
  """
  index_path = os.path.join(root, "model.safetensors.index.json")
  if os.path.isfile(index_path):
    with open(index_path, encoding="utf-8") as file:
      weight_map = json.load(file).get("weight_map") or {}
    names = sorted(set(weight_map.values()))
    if names:
      return [os.path.join(root, name) for name in names]
  return sorted(glob.glob(os.path.join(root, "*.safetensors")))


def load_hidream_o1_state_dict(
  transformer: HiDreamO1ImageTransformer2DModel,
  root: str,
  dtype: torch.dtype = _INFERENCE_DTYPE,
) -> None:
  """Stream the FP32 checkpoint into the transformer, one tensor at a time.

  The shard list comes from `model.safetensors.index.json` when present and is globbed otherwise, so
  a partial download (only some shards) fails loudly instead of reporting the missing tensors as
  missing weights. Tensors are cast as they are copied, which keeps the peak host memory near the
  bf16 model size instead of the FP32 checkpoint size.

  :param transformer: Destination module, on CPU.
  :param root: Checkpoint directory holding the safetensors shards.
  :param dtype: Target dtype.
  """
  shards = _resolve_shards(root)
  if not shards:
    raise FileNotFoundError(f"No safetensors shard found in {root}.")

  params = dict(transformer.named_parameters())
  loaded = set()
  unexpected = []
  for shard in shards:
    with safe_open(shard, framework="pt", device="cpu") as handle:
      for key in handle.keys():
        if key.startswith(_SKIPPED_WEIGHT_PREFIXES) or key in _SKIPPED_WEIGHT_KEYS:
          continue
        name = _map_weight_key(key)
        if name is None or name not in params:
          unexpected.append(key)
          continue
        tensor = handle.get_tensor(key)
        param = params[name]
        if tuple(tensor.shape) != tuple(param.shape):
          raise ValueError(f"Shape mismatch for {key}: checkpoint {tuple(tensor.shape)} vs "
                           f"module {tuple(param.shape)}.")
        param.data.copy_(tensor.to(dtype))
        loaded.add(name)

  missing = sorted(set(params) - loaded)
  if missing or unexpected:
    raise ValueError(f"HiDream-O1-Image weight mismatch: {len(missing)} missing "
                     f"(e.g. {missing[:5]}), {len(unexpected)} unexpected (e.g. {unexpected[:5]}).")
  logger.info(f"Loaded {len(loaded)} tensors from {len(shards)} HiDream-O1-Image shards.")


def _build_transformer_config(root: str) -> dict:
  """Derive the transformer hyper-parameters from the checkpoint's Qwen3-VL config.

  :param root: Checkpoint directory.
  :returns: Keyword arguments for :class:`HiDreamO1ImageTransformer2DModel`.
  """
  with open(os.path.join(root, "config.json"), encoding="utf-8") as file:
    payload = json.load(file)
  text = payload.get("text_config") or payload
  rope_parameters = text.get("rope_parameters") or text.get("rope_scaling") or {}
  return dict(
    hidden_size=text["hidden_size"],
    intermediate_size=text["intermediate_size"],
    num_layers=text["num_hidden_layers"],
    num_attention_heads=text["num_attention_heads"],
    num_key_value_heads=text["num_key_value_heads"],
    head_dim=text.get("head_dim", text["hidden_size"] // text["num_attention_heads"]),
    vocab_size=text["vocab_size"],
    rms_norm_eps=text["rms_norm_eps"],
    hidden_act=text.get("hidden_act", "silu"),
    rope_theta=rope_parameters.get("rope_theta", text.get("rope_theta", 10000.0)),
    mrope_section=tuple(rope_parameters.get("mrope_section", (24, 20, 20))),
  )


def _read_token_ids(root: str) -> dict:
  """Read the checkpoint's special token ids used by the joint sequence.

  :param root: Checkpoint directory.
  :returns: Mapping with `image_token_id`, `video_token_id` and `vision_start_token_id`.
  """
  with open(os.path.join(root, "config.json"), encoding="utf-8") as file:
    payload = json.load(file)
  token_ids = dict(_DEFAULT_TOKEN_IDS)
  for key in token_ids:
    if key in payload:
      token_ids[key] = int(payload[key])
  return token_ids


class HiDreamO1ImagePipeline(DiffusionPipeline):
  """Text-to-image pipeline for HiDream-O1-Image.

  Components are assembled directly from the Qwen3-VL checkpoint layout (`config.json` plus the
  safetensors shards at the repository root), so `from_pretrained(<root>)` needs neither
  `model_index.json` nor remote code.

  Note:
    cache-dit caching assumes one `transformer.forward()` per CFG branch, i.e. a guidance scale
    above 1 and `num_images_per_prompt=1`.
  """

  model_cpu_offload_seq = "transformer"
  _optional_components = ["tokenizer"]

  def __init__(
    self,
    transformer: HiDreamO1ImageTransformer2DModel,
    scheduler: HiDreamO1ImageScheduler,
    tokenizer=None,
    token_ids=None,
  ):
    super().__init__()
    self.register_modules(transformer=transformer, scheduler=scheduler, tokenizer=tokenizer)
    # Vision placeholder ids are checkpoint data, not architecture; `from_pretrained` reads them
    # from `config.json` and falls back to the shipped values.
    self.token_ids = dict(_DEFAULT_TOKEN_IDS) if token_ids is None else dict(token_ids)

  @classmethod
  def from_pretrained(
    cls,
    pretrained_model_name_or_path,
    *args,
    torch_dtype=None,
    dtype=None,
    quantization_config=None,
    device_map=None,
    trust_remote_code=False,
    scheduler=None,
    transformer=None,
    vae=None,
    text_encoder=None,
    tokenizer=None,
    controlnet=None,
    local_files_only=False,
    **kwargs,
  ) -> "HiDreamO1ImagePipeline":
    """Load HiDream-O1-Image components from a checkpoint root.

    :param pretrained_model_name_or_path: Checkpoint root (the Qwen3-VL repository layout).
    :param torch_dtype: Inference dtype; only bf16 is supported by the release.
    :param dtype: Alias of `torch_dtype`.
    :param quantization_config: Ignored; quantize through cache-dit's own flags instead.
    :param device_map: Unsupported, raises. The CLI skips `pipe.to(device)` when
      `--device-map-balance` is set, which would leave the pipeline on CPU.
    :param trust_remote_code: Ignored; the model code is built into cache-dit.
    :param scheduler: Pre-built scheduler, replaces the built-in one.
    :param transformer: Pre-built transformer, skips the checkpoint weight load.
    :param vae: Unsupported, the model is pixel-space.
    :param text_encoder: Unsupported, the DiT embeds its own token ids.
    :param tokenizer: Pre-built tokenizer.
    :param controlnet: Unsupported.
    :param local_files_only: Whether to forbid network access while resolving files.
    :param kwargs: Additional diffusers arguments, ignored with a warning.
    :returns: The assembled pipeline.
    """
    if args:
      raise ValueError("HiDreamO1ImagePipeline.from_pretrained takes keyword arguments only.")
    if controlnet is not None:
      raise ValueError("HiDream-O1-Image does not support ControlNet.")
    if vae is not None:
      raise ValueError("HiDream-O1-Image is pixel-space and has no VAE.")
    if text_encoder is not None:
      raise ValueError("HiDream-O1-Image embeds its own token ids and has no text encoder.")

    resolved_dtype = torch_dtype if torch_dtype is not None else dtype
    if resolved_dtype is None:
      resolved_dtype = _INFERENCE_DTYPE
    if isinstance(resolved_dtype, dict):
      raise ValueError("HiDream-O1-Image takes a single dtype; per-component mappings are not "
                       "supported.")
    if resolved_dtype != _INFERENCE_DTYPE:
      raise ValueError(f"HiDream-O1-Image only supports bfloat16, got {resolved_dtype}.")
    if quantization_config is not None:
      logger.warning("Ignoring quantization_config: quantize HiDream-O1-Image through cache-dit's "
                     "`--quantize-type` flags so the DiT blocks are discovered.")
    if device_map is not None:
      raise NotImplementedError(
        "HiDream-O1-Image does not support `--device-map-balance`: the CLI skips `pipe.to(device)` "
        "for that flag, which would leave the whole pipeline on CPU. Drop the flag or use "
        "cache-dit offloading.")
    if trust_remote_code:
      logger.debug(
        "trust_remote_code is not needed: HiDream-O1-Image code is built into cache-dit.")
    if kwargs:
      logger.warning(f"Ignoring unsupported from_pretrained kwargs: {sorted(kwargs)}.")

    root = str(pretrained_model_name_or_path)
    token_ids = _read_token_ids(root)
    if transformer is None:
      config = _build_transformer_config(root)
      config.update(patch_size=PATCH_SIZE, in_channels=3)
      # Build the parameters directly in the inference dtype: the checkpoint is FP32, so an fp32
      # module would double the resident weight memory and defeat fp8 / svdq quantization, which
      # only accept bf16 linear weights.
      previous_dtype = torch.get_default_dtype()
      torch.set_default_dtype(resolved_dtype)
      try:
        transformer = HiDreamO1ImageTransformer2DModel(**config)
      finally:
        torch.set_default_dtype(previous_dtype)
      load_hidream_o1_state_dict(transformer, root, dtype=resolved_dtype)
    transformer.eval()

    if scheduler is None:
      scheduler = HiDreamO1ImageScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        prediction_type="flow_prediction",
        use_dynamic_shifting=False,
      )

    if tokenizer is None:
      tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=local_files_only)
    _load_chat_template(tokenizer, root)
    for attribute in _SPECIAL_TOKENS:
      setattr(tokenizer, attribute, f"<|{attribute}|>")

    return cls(
      transformer=transformer,
      scheduler=scheduler,
      tokenizer=tokenizer,
      token_ids=token_ids,
    )

  def _validate_special_tokens(self) -> None:
    """Fail fast when a special token does not map to exactly one id.

    A multi-token `<|tms_token|>` would silently shift the whole joint sequence; a multi-token
    `<|boi_token|>` would shift the image block by one position.
    """
    for attribute in ("tms_token", "boi_token"):
      token = getattr(self.tokenizer, attribute)
      ids = self.tokenizer.encode(token, add_special_tokens=False)
      if len(ids) != 1:
        raise RuntimeError(f"{token} encodes to {ids}; expected exactly one token id.")
    expected = self.transformer.config.tms_token_id
    actual = self.tokenizer.encode(self.tokenizer.tms_token, add_special_tokens=False)[0]
    if actual != expected:
      raise RuntimeError(f"tms_token maps to {actual}, model expects [{expected}]")

  def build_text_sample(self, prompt: str, height: int, width: int) -> dict:
    """Tokenize a prompt and build its mask, position ids and image slot.

    :param prompt: Text prompt.
    :param height: Target image height, divisible by `patch_size`.
    :param width: Target image width, divisible by `patch_size`.
    :returns: Dict with `input_ids`, `position_ids`, `attention_mask`, `text_length` and
      `image_length`.
    """
    image_token_id = self.token_ids["image_token_id"]
    video_token_id = self.token_ids["video_token_id"]
    vision_start_token_id = self.token_ids["vision_start_token_id"]
    image_length = (height // PATCH_SIZE) * (width // PATCH_SIZE)

    messages = [{"role": "user", "content": prompt}]
    template_caption = (
      self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) +
      self.tokenizer.boi_token + self.tokenizer.tms_token * TIMESTEP_TOKEN_NUM)
    input_ids = self.tokenizer.encode(
      template_caption,
      return_tensors="pt",
      add_special_tokens=False,
    )

    image_grid_thw = torch.tensor(
      [1, height // PATCH_SIZE, width // PATCH_SIZE],
      dtype=torch.int64,
    ).unsqueeze(0)
    vision_tokens = torch.zeros((1, image_length), dtype=input_ids.dtype) + image_token_id
    vision_tokens[0, 0] = vision_start_token_id
    input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

    position_ids, _ = get_rope_index_fix_point(
      spatial_merge_size=1,
      image_token_id=image_token_id,
      video_token_id=video_token_id,
      vision_start_token_id=vision_start_token_id,
      input_ids=input_ids_pad,
      image_grid_thw=image_grid_thw,
      video_grid_thw=None,
      attention_mask=None,
      skip_vision_start_token=[1],
    )

    text_length = int(input_ids.shape[-1])
    all_seq_len = int(position_ids.shape[-1])
    token_types = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
    begin = text_length - TIMESTEP_TOKEN_NUM
    token_types[0, begin:begin + image_length + TIMESTEP_TOKEN_NUM] = 1
    token_types[0, text_length - TIMESTEP_TOKEN_NUM:text_length] = 3

    return {
      "input_ids": input_ids,
      "position_ids": position_ids,
      "attention_mask": build_hidream_o1_attention_mask(token_types, self._dtype),
      "text_length": text_length,
      "image_length": image_length,
    }

  @property
  def _dtype(self) -> torch.dtype:
    """Activation dtype, taken from the loaded transformer.

    The release runs in bf16, but deriving it here keeps `pipe.to(torch.float32)` and CPU tests
    consistent: an additive mask must match the attention dtype.

    :returns: The transformer parameter dtype.
    """
    return next(self.transformer.parameters()).dtype

  @staticmethod
  def _to_device(sample: dict, device: torch.device) -> dict:
    """Move the tensor entries of a text sample onto the execution device once per call.

    :param sample: Sample built by :meth:`build_text_sample`.
    :param device: Target device.
    :returns: The sample with tensors on `device`.
    """
    return {
      key: value.to(device) if torch.is_tensor(value) else value
      for key, value in sample.items()
    }

  def _prepare_noise_and_patchify(
    self,
    height: int,
    width: int,
    seed: int,
    device: torch.device,
    noise_scale: float = NOISE_SCALE,
  ) -> torch.Tensor:
    """Sample the initial noise exactly like the reference and flatten it into patches.

    :param height: Target image height.
    :param width: Target image width.
    :param seed: Base seed; the reference samples the noise with `seed + 1`.
    :param device: Device the patches are placed on.
    :param noise_scale: Standard deviation of the noise.
    :returns: Patches `[1, (H/p) * (W/p), 3 * p * p]`.
    """
    # Always CPU-sampled with `seed + 1` so a run is reproducible across hardware.
    generator = torch.Generator("cpu").manual_seed(seed + 1)
    noise = noise_scale * torch.randn((1, 3, height, width), generator=generator)
    return patchify(noise.to(device, self._dtype))

  def _forward_once(self, sample: dict, z: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    """One transformer forward, returning only the image rows.

    :param sample: Text sample built by :meth:`build_text_sample` (on device).
    :param z: Noisy patches `[1, L, 3 * p * p]`.
    :param timestep: Denoising timestep in `[0, 1]`.
    :returns: Predicted clean patches `[1, L, 3 * p * p]`.
    """
    # No `torch.autocast` here: the weights are already created in the inference dtype, and an
    # ambient autocast region turns the fp32 matmuls inside SVDQuant's activation calibration back
    # into bf16, which breaks `torch.linalg.qr` ("geqrf_cuda not implemented for 'BFloat16'").
    device = z.device
    outputs = self.transformer(
      input_ids=sample["input_ids"],
      position_ids=sample["position_ids"],
      vinputs=z,
      timestep=timestep.reshape(-1).to(device),
      attention_mask=sample["attention_mask"],
    )
    text_length = sample["text_length"]
    image_length = sample["image_length"]
    return outputs.sample[:, text_length:text_length + image_length]

  @torch.no_grad()
  def __call__(
    self,
    prompt: Optional[str] = None,
    negative_prompt: str = " ",
    height: int = 2048,
    width: int = 2048,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    num_images_per_prompt: int = 1,
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.Tensor] = None,
    output_type: str = "pil",
    return_dict: bool = True,
    snap_resolution: bool = True,
    noise_scale: float = NOISE_SCALE,
  ):
    """Generate an image.

    :param prompt: Text prompt.
    :param negative_prompt: Prompt of the unconditional branch; the reference uses a single space.
    :param height: Output height. With `snap_resolution` the **whole** closest trained resolution
      is used, so a 1024x1024 request renders at 2048x2048 (the trained 1:1 entry).
    :param width: Output width, same snapping rule.
    :param num_inference_steps: Denoising steps.
    :param guidance_scale: CFG scale; `1.0` runs a single forward per step.
    :param num_images_per_prompt: Must stay 1.
    :param generator: Source of the initial-noise seed; only `initial_seed()` is read and the noise
      is always sampled on CPU with `seed + 1`, matching the reference exactly.
    :param latents: Optional pre-sampled patches `[1, L, 3 * p * p]`.
    :param output_type: `pil`, `np`, `pt` or `latent`.
    :param return_dict: Return an `ImagePipelineOutput` instead of a tuple.
    :param snap_resolution: Whether to snap the requested size to the trained resolution grid.
    :param noise_scale: Standard deviation of the initial noise. The vendored reference
      (`@21bcd30471ac`) uses 8.0; the model card's `--noise_scale_start/--noise_scale_end` default
      to 7.5, so the knob is exposed rather than silently picking one.
    :returns: Generated images.
    """
    if prompt is None or not isinstance(prompt, str):
      raise ValueError("`prompt` must be a string.")
    if num_images_per_prompt != 1:
      raise ValueError("HiDream-O1-Image only renders one image per call.")
    if num_inference_steps <= 0:
      raise ValueError("`num_inference_steps` must be positive.")
    if height <= 0 or width <= 0 or height % PATCH_SIZE or width % PATCH_SIZE:
      raise ValueError(f"height/width must be positive multiples of {PATCH_SIZE}.")
    if output_type not in ("pil", "np", "pt", "latent"):
      raise ValueError("output_type must be pil, np, pt or latent.")
    if snap_resolution:
      width, height = find_closest_resolution(width, height)

    device = self._execution_device
    self._validate_special_tokens()

    do_cfg = guidance_scale != 1.0
    if getattr(self.transformer, "_is_cached", False) and not do_cfg:
      # cache-dit's separate-CFG bookkeeping pairs exactly two forward passes per denoising step.
      raise ValueError(
        "cache-dit caching assumes one conditional and one unconditional forward per step; "
        "`guidance_scale=1.0` runs a single forward and would desynchronize the step counters. "
        "Use a guidance scale above 1, or disable separate CFG in the cache config.")

    conditional = self._to_device(self.build_text_sample(prompt, height, width), device)
    unconditional = None
    if do_cfg:
      unconditional = self._to_device(self.build_text_sample(negative_prompt, height, width),
                                      device)

    seed = 0 if generator is None else int(generator.initial_seed())
    if latents is None:
      z = self._prepare_noise_and_patchify(height, width, seed, device, noise_scale)
    else:
      z = latents.to(device=device, dtype=self._dtype)

    self.scheduler.set_timesteps(num_inference_steps, device=device)
    for step_t in self.progress_bar(self.scheduler.timesteps):
      step_t_f32 = step_t.float()
      # The release feeds the model the *complement* of sigma (`t_pixeldit = 1 - t/1000`) and then
      # converts the predicted `x0` back to a velocity with `sigma = max(t/1000, T_EPS)`.
      timestep = 1.0 - step_t_f32 / 1000.0
      sigma = (step_t_f32 / 1000.0).to(dtype=torch.float32).clamp_min(T_EPS)

      x_pred = self._forward_once(conditional, z, timestep)
      velocity = (x_pred.to(torch.float32) - z.to(torch.float32)) / sigma
      if do_cfg:
        x_pred_uncond = self._forward_once(unconditional, z, timestep)
        velocity_uncond = (x_pred_uncond.to(torch.float32) - z.to(torch.float32)) / sigma
        velocity = velocity_uncond + guidance_scale * (velocity - velocity_uncond)

      z = self.scheduler.step(-velocity, step_t_f32, z.float(),
                              return_dict=False)[0].to(self._dtype)

    if output_type == "latent":
      images = z
    else:
      pixels = ((unpatchify(z.float(), height, width) + 1) / 2).clamp(0, 1)
      if output_type == "pt":
        images = pixels
      elif output_type == "np":
        images = pixels.cpu().permute(0, 2, 3, 1).numpy()
      else:
        array = (pixels.cpu().permute(0, 2, 3, 1).numpy() * 255).round().astype("uint8")
        images = [Image.fromarray(sample).convert("RGB") for sample in array]

    self.maybe_free_model_hooks()
    if not return_dict:
      return (images, )
    return ImagePipelineOutput(images=images)
