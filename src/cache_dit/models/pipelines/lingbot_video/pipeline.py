"""Built-in LingBot-Video Dense text-to-video / text-to-image pipeline.

Vendored from the official `lingbot_video/pipeline_lingbot_video.py`
(https://github.com/Robbyant/lingbot-video, Apache-2.0, Copyright Ant Group Co., Ltd.) because
diffusers does not ship it: the
checkpoint's `model_index.json` points `transformer` and `scheduler` at the repo's own
`lingbot_video` module, so a plain `DiffusionPipeline.from_pretrained` cannot import them. Keeping
one copy here lets cache-dit run `robbyant/lingbot-video-dense-1.3b` without `trust_remote_code` or
a remote `custom_pipeline`.

Upstream defaults: bf16 DiT and text-encoder weights, fp32 norms / timestep MLP / per-block
`scale_shift_table` (the transformer keeps those in fp32 through `_keep_in_fp32_modules`), fp32
initial noise and latent updates, fp32 VAE decoding, flow-matching shift 3.0, 40 sampling steps.
Standard CFG runs one `transformer.forward()` per branch, i.e. two forwards per sampling step and
batch size 1 — the batched-CFG and CFG-parallel paths of the upstream pipeline are not vendored.

Note:
  Keep the constraints below intact, cache-dit's dispatch depends on them.
  * `register_modules` keeps the component names `transformer` / `vae` / `text_encoder` /
    `processor` / `scheduler`; `resolve_transformer()`, `_is_text_encoder()` and
    `_is_auto_encoder()` dispatch on those attributes and on the component module paths.
  * The DiT is called once per CFG branch with batch size 1, so the block adapter declares
    `has_separate_cfg=True`; caching is only valid while `guidance_scale > 1`.
  * The text encoder stays a `transformers` model and the VAE stays `diffusers.AutoencoderKLWan` so
    `--parallel-text` / `--parallel-vae` keep working.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from ....logger import init_logger
from .scheduler import FlowUniPCMultistepScheduler
from .utils import LOW_NOISE_TAIL_V1_DEFAULT_STEPS, compute_refiner_sigmas

logger = init_logger(__name__)

TOKEN_LENGTH = 37698
HIDDEN_STATE_SKIP_LAYER = 0

_INFERENCE_DTYPE = torch.bfloat16
_VAE_DTYPE = torch.float32

PROMPT_TEMPLATE = (
  "<|im_start|>system\nGiven a user input that may include a text prompt alone, "
  "a text prompt with an image reference, or a text prompt with a video reference "
  "or a video reference alone, generate an \"Enhanced prompt\" that provides detailed "
  "visual descriptions suitable for video generation. Evaluate the level of detail "
  "in the user's input: if it is simple, enrich it by adding specifics about colors, "
  "shapes, sizes, textures, lighting, motion dynamics, camera movement, temporal "
  "progression, and spatial relationships to create vivid, concrete, and temporally "
  "coherent scenes to create vivid and concrete scenes. Please generate only the "
  "enhanced description for the prompt below and avoid including any additional "
  "commentary or evaluations:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
  "<|im_start|>assistant\n")
IMG_PROMPT_TEMPLATE = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PROMPT_TEMPLATE = "<|vision_start|><|video_pad|><|vision_end|>"

DEFAULT_NEGATIVE_PROMPT = (
  '{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", '
  '"pixelated", "jpeg artifacts", "low resolution", "unstable color", "color flicker", '
  '"underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], '
  '"artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", "cgi", '
  '"sketch", "digital art"], "composition_and_content": ["text", "watermark", "signature", '
  '"logo", "subtitles", "pillarboxed", "side bars", "portrait image in landscape frame"], '
  '"temporal_and_motion_stability": ["flickering", "jittery", "motion blur", "temporal '
  'inconsistency", "warping", "morphing", "incoherent motion", "unnatural movement", "static '
  'object with sudden jump", "frame-to-frame inconsistency"], "material_and_structure": '
  '["plastic-like glass", "unrealistic texture", "deformed bottle", "liquid freezing improperly", '
  '"distorted reflections"]}}')

DEFAULT_NEGATIVE_PROMPT_IMAGE = (
  '{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", '
  '"pixelated", "jpeg artifacts", "low resolution", "underexposed", "overexposed", "invisible '
  'subject", "subject hidden in darkness"], "artistic_style": ["painting", "illustration", '
  '"drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], "composition_and_content": '
  '["text", "watermark", "signature", "logo", "pillarboxed", "side bars", "portrait image in '
  'landscape frame"], "material_and_structure": ["plastic-like glass", "unrealistic texture", '
  '"deformed bottle", "liquid freezing improperly", "distorted reflections"]}}')


@dataclass
class LingBotVideoPipelineOutput(BaseOutput):
  """Frames as one `(T, H, W, C)` float array per batch item, or the raw latents."""

  frames: Union[List[np.ndarray], torch.Tensor]


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
  try:
    return next(module.parameters()).dtype
  except StopIteration:
    return torch.float32


def _module_device(module: torch.nn.Module) -> torch.device:
  try:
    return next(module.parameters()).device
  except StopIteration:
    return torch.device("cpu")


def _transformer_timestep(timestep: torch.Tensor, transformer_dtype: torch.dtype) -> torch.Tensor:
  sigma = timestep.float() / 1000.0
  if transformer_dtype in {torch.bfloat16, torch.float16}:
    sigma = sigma.to(transformer_dtype)
  return (sigma * 1000.0).float()


def _transformer_autocast(device: torch.device, transformer_dtype: torch.dtype):
  from contextlib import nullcontext

  if device.type != "cuda" or transformer_dtype not in {torch.bfloat16, torch.float16}:
    return nullcontext()
  return torch.autocast(device_type="cuda", dtype=transformer_dtype)


def _disable_unused_vlm_lm_head(text_encoder: torch.nn.Module) -> bool:
  lm_head = getattr(text_encoder, "lm_head", None)
  if not isinstance(lm_head, torch.nn.Module) or isinstance(lm_head, torch.nn.Identity):
    return False
  text_encoder.lm_head = torch.nn.Identity()
  return True


def _run_vlm_encoder(text_encoder: torch.nn.Module, **inputs: Any) -> Any:
  encoder = getattr(text_encoder, "model", None)
  if not isinstance(encoder, torch.nn.Module):
    raise TypeError(
      f"{type(text_encoder).__name__} does not expose a torch.nn.Module encoder core.")
  return encoder(**inputs)


def _resolve_dtypes(dtype_spec) -> Tuple[torch.dtype, torch.dtype]:
  """Split a pipeline dtype spec into the inference dtype and the VAE dtype.

  :param dtype_spec: `None`, a single `torch.dtype`, or a per-component mapping such as
    `{"default": torch.bfloat16, "vae": torch.float32}`.
  :returns: `(inference_dtype, vae_dtype)`.
  """
  if dtype_spec is None:
    return _INFERENCE_DTYPE, _VAE_DTYPE
  if isinstance(dtype_spec, dict):
    inference_dtype = dtype_spec.get("default") or dtype_spec.get("transformer") or _INFERENCE_DTYPE
    vae_dtype = dtype_spec.get("vae") or _VAE_DTYPE
    return inference_dtype, vae_dtype
  return dtype_spec, _VAE_DTYPE


def _to_valid_length(prompt_embeds: torch.Tensor, prompt_mask: torch.Tensor):
  """Drop right padding so the DiT sees the same text length as `encode_prompt` produces.

  :param prompt_embeds: `(1, L, text_dim)` text features.
  :param prompt_mask: `(1, L)` mask, `1` = valid token.
  :returns: The embeddings and mask sliced to the valid length.
  """
  true_len = int(prompt_mask[0].sum().item())
  return prompt_embeds[:, :true_len], prompt_mask[:, :true_len]


def _subdir(root: str, name: str) -> str:
  """Return `root/name` when it exists, else `root` itself (component-style checkpoints)."""
  import os

  candidate = os.path.join(root, name)
  return candidate if os.path.isdir(candidate) else root


class LingBotVideoPipeline(DiffusionPipeline):
  """Text-to-video / text-to-image pipeline for LingBot-Video Dense 1.3B.

  Components live in their own subdirectories (`transformer/`, `text_encoder/`, `processor/`,
  `vae/`, `scheduler/`), so `from_pretrained(<root>)` assembles them directly and no remote
  pipeline code is required.

  Note:
    cache-dit caching assumes one `transformer.forward()` per CFG branch, i.e. batch size 1 and
    `guidance_scale > 1`. The batched-CFG and CFG-parallel branches of the upstream pipeline are
    not vendored.
  """

  model_cpu_offload_seq = "text_encoder->transformer->vae"
  _optional_components = ["text_encoder", "processor", "vae"]
  _callback_tensor_inputs = ["latents"]
  # Native frame rate of the released checkpoint (81 frames ~= 3.4 s).
  fps = 24

  def __init__(
    self,
    transformer,
    vae,
    text_encoder=None,
    processor=None,
    scheduler=None,
  ):
    super().__init__()
    self.register_modules(
      transformer=transformer,
      vae=vae,
      text_encoder=text_encoder,
      processor=processor,
      scheduler=scheduler,
    )
    if self.text_encoder is not None:
      _disable_unused_vlm_lm_head(self.text_encoder)
    self.vae_scale_factor_temporal = 4
    self.vae_scale_factor_spatial = 8
    self.token_length = TOKEN_LENGTH
    self.hidden_state_skip_layer = HIDDEN_STATE_SKIP_LAYER
    self.prompt_template = PROMPT_TEMPLATE
    self.img_prompt_template = IMG_PROMPT_TEMPLATE
    self.video_prompt_template = VIDEO_PROMPT_TEMPLATE
    self._crop_start: Optional[int] = None

  @property
  def device(self) -> torch.device:
    return self.transformer.device

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
    processor=None,
    controlnet=None,
    local_files_only=False,
    **kwargs,
  ) -> "LingBotVideoPipeline":
    """Load LingBot-Video components from a checkpoint root.

    :param pretrained_model_name_or_path: Checkpoint root holding the component subdirectories.
    :param torch_dtype: dtype for the DiT and text encoder, or a per-component mapping such as
      `{"default": torch.bfloat16, "vae": torch.float32}`. `dtype` is accepted as an alias.
    :param quantization_config: Ignored; quantize through cache-dit's own flags instead.
    :param device_map: Unsupported; call `.to(device)` (or cache-dit offloading) explicitly.
    :param trust_remote_code: Ignored; all components here are loaded locally.
    :param scheduler: Pre-built scheduler, replaces the checkpoint's.
    :param transformer: Pre-built transformer, replaces the checkpoint's.
    :param vae: Pre-built VAE, replaces the checkpoint's.
    :param text_encoder: Pre-built text encoder, replaces the checkpoint's.
    :param processor: Pre-built processor, replaces the checkpoint's.
    :param controlnet: Unsupported, must stay `None`.
    :param local_files_only: Whether to forbid network access while resolving files.
    :param kwargs: Additional diffusers arguments, ignored.
    :returns: The assembled pipeline.
    """
    if args:
      raise ValueError("LingBotVideoPipeline.from_pretrained takes keyword arguments only.")
    if controlnet is not None:
      raise ValueError("LingBot-Video does not support ControlNet.")

    inference_dtype, vae_dtype = _resolve_dtypes(torch_dtype if torch_dtype is not None else dtype)
    if quantization_config is not None:
      logger.warning("Ignoring quantization_config: quantize LingBot-Video through cache-dit's "
                     "`--quantize-type` flags so the DiT blocks are discovered.")
    if device_map is not None:
      raise NotImplementedError(
        "LingBot-Video does not support `--device-map-balance`: the CLI skips `pipe.to(device)` for "
        "that flag, which would leave the whole pipeline on CPU. Drop the flag or use cache-dit "
        "offloading.")
    if trust_remote_code:
      logger.debug("trust_remote_code is not needed: LingBot-Video code is built into cache-dit.")
    if kwargs:
      logger.warning(f"Ignoring unsupported from_pretrained kwargs: {sorted(kwargs)}.")

    root = str(pretrained_model_name_or_path)
    local_only = dict(local_files_only=local_files_only)

    if transformer is None:
      from .transformer import LingBotVideoTransformer3DModel

      transformer = LingBotVideoTransformer3DModel.from_pretrained(
        _subdir(root, "transformer"),
        torch_dtype=inference_dtype,
        **local_only,
      )
    if text_encoder is None:
      from transformers import Qwen3VLForConditionalGeneration

      text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        _subdir(root, "text_encoder"),
        dtype=inference_dtype,
        **local_only,
      )
    if processor is None:
      from transformers import Qwen3VLProcessor

      processor = Qwen3VLProcessor.from_pretrained(_subdir(root, "processor"), **local_only)
    if scheduler is None:
      scheduler = FlowUniPCMultistepScheduler.from_pretrained(
        _subdir(root, "scheduler"),
        **local_only,
      )
    if vae is None:
      # Must stay a diffusers module: `_is_auto_encoder()` dispatches VAE parallelism by module path.
      from diffusers import AutoencoderKLWan

      vae = AutoencoderKLWan.from_pretrained(
        _subdir(root, "vae"),
        torch_dtype=vae_dtype,
        **local_only,
      )

    for module in (transformer, text_encoder, vae):
      if module is not None:
        module.eval()

    return cls(
      transformer=transformer,
      vae=vae,
      text_encoder=text_encoder,
      processor=processor,
      scheduler=scheduler,
    )

  @staticmethod
  def check_inputs(height: int, width: int, num_frames: int) -> None:
    if num_frames != 1 and (num_frames - 1) % 4 != 0:
      raise ValueError(f"`num_frames` must be 1 or 4n+1, got {num_frames}.")
    if height % 16 != 0 or width % 16 != 0:
      raise ValueError(f"`height` and `width` must be multiples of 16, got {height}x{width}.")

  @staticmethod
  def _apply_inpainting(latents: torch.Tensor, cond_latent: torch.Tensor) -> torch.Tensor:
    cond_t = cond_latent.shape[2]
    latents[:, :, :cond_t] = cond_latent.float()
    return latents

  @staticmethod
  def apply_text_to_template(text: str, template: str = PROMPT_TEMPLATE) -> str:
    return template.format(text)

  def _compute_crop_start(self) -> int:
    if self._crop_start is None:
      marker = "<|USER_INPUT_MARKER|>"
      marked = self.prompt_template.format(marker)
      marker_pos = marked.find(marker)
      if marker_pos < 0:
        self._crop_start = 0
      else:
        prefix = self.processor(
          text=marked[:marker_pos],
          images=None,
          videos=None,
          return_tensors="pt",
        )
        self._crop_start = int(prefix["input_ids"].shape[1])
    return self._crop_start

  def _build_prompt_inputs(
    self,
    prompt: str,
    images: Optional[Any] = None,
    videos: Optional[Any] = None,
    video_metadata: Optional[Any] = None,
    video_kwargs: Optional[Dict[str, Any]] = None,
  ):
    visual_template = ""
    if images is not None:
      visual_template = self.img_prompt_template
    elif videos is not None:
      visual_template = self.video_prompt_template

    text = self.apply_text_to_template(visual_template + prompt, self.prompt_template)
    kwargs = dict(video_kwargs or {})
    return self.processor(
      text=[text],
      images=images,
      videos=videos,
      video_metadata=video_metadata,
      do_resize=False,
      truncation=True,
      max_length=self.token_length,
      padding="longest",
      return_tensors="pt",
      **kwargs,
    )

  @torch.no_grad()
  def encode_prompt(
    self,
    prompt: str,
    *,
    images: Optional[Any] = None,
    videos: Optional[Any] = None,
    video_metadata: Optional[Any] = None,
    video_kwargs: Optional[Dict[str, Any]] = None,
    device: Optional[Union[str, torch.device]] = None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode one prompt with the Qwen3-VL text encoder.

    :param prompt: User prompt; LingBot-Video expects the structured JSON caption form.
    :param images: Optional reference image(s) for the vision template.
    :param videos: Optional reference video(s) for the vision template.
    :param video_metadata: Optional metadata forwarded to the processor.
    :param video_kwargs: Extra processor kwargs.
    :param device: Encoding device, defaults to the pipeline execution device.
    :returns:`(prompt_embeds, prompt_mask)` truncated to the valid text length.
    """
    if self.text_encoder is None or self.processor is None:
      raise ValueError("`text_encoder` and `processor` are required for encode_prompt().")

    device = torch.device(device) if device is not None else self._execution_device
    inputs = self._build_prompt_inputs(
      prompt,
      images=images,
      videos=videos,
      video_metadata=video_metadata,
      video_kwargs=video_kwargs,
    )
    inputs = inputs.to(device)
    outputs = _run_vlm_encoder(
      self.text_encoder,
      **inputs,
      output_hidden_states=bool(self.hidden_state_skip_layer),
    )
    prompt_embeds = outputs.last_hidden_state
    if self.hidden_state_skip_layer not in (None, 0):
      hidden_states = getattr(outputs, "hidden_states", None)
      if hidden_states is None:
        raise RuntimeError("The VLM encoder did not return hidden-state history for "
                           f"hidden_state_skip_layer={self.hidden_state_skip_layer}.")
      prompt_embeds = hidden_states[-(self.hidden_state_skip_layer + 1)]

    prompt_mask = inputs["attention_mask"]
    crop_start = self._compute_crop_start()
    if crop_start > 0:
      prompt_embeds = prompt_embeds[:, crop_start:]
      prompt_mask = prompt_mask[:, crop_start:]

    # A single sample drops right padding before DiT inference, which also keeps the joint sequence
    # contiguous for context parallelism.
    return _to_valid_length(prompt_embeds, prompt_mask)

  def prepare_latents(
    self,
    num_frames: int,
    height: int,
    width: int,
    generator: Optional[torch.Generator],
    latents: Optional[torch.Tensor],
    device: torch.device,
  ) -> torch.Tensor:
    latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
    latent_height = height // self.vae_scale_factor_spatial
    latent_width = width // self.vae_scale_factor_spatial
    shape = (
      1,
      self.transformer.config.in_channels,
      latent_frames,
      latent_height,
      latent_width,
    )
    if latents is None:
      return randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
    if tuple(latents.shape) != shape:
      raise ValueError(f"`latents` shape must be {shape}, got {tuple(latents.shape)}.")
    # Cloned: the sampling loop writes into the latent in place (the cond-latent prefix), and the
    # caller's tensor must not be touched when it already has the right dtype and device.
    return latents.to(device=device, dtype=torch.float32).clone()

  def _dit_latent_to_vae(self, latents: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=torch.float32)
    std = torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=torch.float32)
    mean = mean.view(1, -1, 1, 1, 1)
    std = std.view(1, -1, 1, 1, 1)
    return latents.float() * std + mean

  def _vae_latent_to_dit(self, latents: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=torch.float32)
    std = torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=torch.float32)
    mean = mean.view(1, -1, 1, 1, 1)
    std = std.view(1, -1, 1, 1, 1)
    return (latents.float() - mean) / std

  @torch.no_grad()
  def _decode_latents(self, latents: torch.Tensor, output_type: str = "np") -> List[Any]:
    vae_device = _module_device(self.vae)
    vae_dtype = _module_dtype(self.vae)
    vae_latents = self._dit_latent_to_vae(latents).to(device=vae_device, dtype=torch.float32)
    if vae_latents.ndim == 5:
      vae_latents = vae_latents.contiguous(memory_format=torch.channels_last_3d)
    autocast_dtype = (vae_dtype if vae_device.type == "cuda"
                      and vae_dtype in {torch.bfloat16, torch.float16} else None)
    with torch.autocast(
        "cuda",
        dtype=autocast_dtype or torch.bfloat16,
        enabled=autocast_dtype is not None,
    ):
      decoded = self.vae.decode(vae_latents)
    frames = decoded[0] if isinstance(decoded, tuple) else decoded.sample
    frames = frames.float().clamp_(-1, 1)
    frames = (frames + 1.0) / 2.0
    frames = frames.permute(0, 2, 3, 4, 1).cpu().numpy()
    if output_type == "pil":
      # diffusers' video pipelines return one list of PIL frames per sample; the CLI writer and the
      # metrics CLI both consume that shape.
      from PIL import Image

      return [[Image.fromarray((frame * 255).round().astype("uint8")) for frame in video]
              for video in frames]
    return [video for video in frames]

  @torch.no_grad()
  def __call__(
    self,
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 40,
    guidance_scale: float = 3.0,
    shift: float = 3.0,
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.Tensor] = None,
    cond_latent: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_mask: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_mask: Optional[torch.Tensor] = None,
    output_type: str = "np",
    t_thresh: Optional[float] = None,
    refiner_sigma_tail_steps: int = LOW_NOISE_TAIL_V1_DEFAULT_STEPS,
    offload_vae_during_denoise: bool = False,
    return_dict: bool = True,
  ) -> Union[LingBotVideoPipelineOutput, Tuple[Union[List[np.ndarray], torch.Tensor]]]:
    """Generate one clip (or a single frame when `num_frames == 1`).

    :param prompt: Prompt; LingBot-Video is trained on the structured JSON caption form.
    :param negative_prompt: Unconditional prompt used by the CFG branch.
    :param height: Output height, a multiple of 16.
    :param width: Output width, a multiple of 16.
    :param num_frames: `1` (image) or `4n+1` frames.
    :param num_inference_steps: Sampling steps.
    :param guidance_scale: CFG scale; must be `> 1` for cache-dit caching to be valid.
    :param shift: Flow-matching shift.
    :param generator: Torch generator for the initial noise.
    :param latents: Optional pre-sampled latents.
    :param cond_latent: Optional clean temporal-prefix condition (TI2V / refiner).
    :param prompt_embeds: Pre-computed positive embeddings.
    :param prompt_mask: Mask matching `prompt_embeds`.
    :param negative_prompt_embeds: Pre-computed negative embeddings.
    :param negative_prompt_mask: Mask matching `negative_prompt_embeds`.
    :param output_type: `"np"` for one `(T, H, W, C)` array per sample, `"pil"` for one
      list of `PIL.Image` frames per sample, `"latent"` for the raw latents.
    :param t_thresh: Start sigma of the low-noise refiner pass; `None` disables it.
    :param refiner_sigma_tail_steps: Extra low-noise steps appended to the refiner schedule.
    :param offload_vae_during_denoise: Move the VAE to CPU while sampling.
    :param return_dict: Return a `LingBotVideoPipelineOutput` instead of a tuple.
    :returns: Decoded frames `(T, H, W, C)` per batch item, or the latents.
    """
    self.check_inputs(height, width, num_frames)
    if self.transformer is None or self.scheduler is None:
      raise ValueError("`transformer` and `scheduler` are required for generation.")

    device = self._execution_device
    do_cfg = guidance_scale > 1.0
    if getattr(self.transformer, "_is_cached", False) and not do_cfg:
      # cache-dit's separate-CFG bookkeeping pairs exactly two forward passes per denoising step.
      raise ValueError(
        "cache-dit caching assumes one conditional and one unconditional forward per step; "
        "`guidance_scale=1.0` runs a single forward and would desynchronize the step counters. "
        "Use a guidance scale above 1, or disable separate CFG in the cache config.")

    if prompt_embeds is not None:
      if prompt_mask is None:
        raise ValueError("`prompt_mask` is required when `prompt_embeds` is provided.")
      prompt_embeds, prompt_mask = _to_valid_length(
        prompt_embeds.to(device=device),
        prompt_mask.to(device=device),
      )
    else:
      prompt_embeds, prompt_mask = self.encode_prompt(prompt, device=device)
    if do_cfg:
      if negative_prompt_embeds is not None:
        if negative_prompt_mask is None:
          raise ValueError(
            "`negative_prompt_mask` is required when `negative_prompt_embeds` is provided.")
        negative_embeds, negative_mask = _to_valid_length(
          negative_prompt_embeds.to(device=device),
          negative_prompt_mask.to(device=device),
        )
      else:
        negative_embeds, negative_mask = self.encode_prompt(negative_prompt, device=device)

    latents = self.prepare_latents(num_frames, height, width, generator, latents, device)
    # Clean temporal-prefix condition (e.g. the ti2v refiner's first-frame latent): written into the
    # latent before sampling and after every scheduler step, so the fixed frames stay clean while
    # the rest denoise against them through attention.
    if cond_latent is not None:
      cond_latent = cond_latent.to(device=device, dtype=torch.float32)
      latents = self._apply_inpainting(latents, cond_latent)
    sigmas = compute_refiner_sigmas(
      sigma_max=float(self.scheduler.sigma_max),
      sigma_min=float(self.scheduler.sigma_min),
      num_inference_steps=num_inference_steps,
      shift=shift,
      t_thresh=t_thresh,
      tail_steps=refiner_sigma_tail_steps,
    )
    if sigmas is None:
      self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
    else:
      self.scheduler.set_timesteps(
        int(sigmas.shape[0]),
        device=device,
        sigmas=sigmas,
        shift=1.0,
      )
    transformer_dtype = _module_dtype(self.transformer)
    vae_restore_device: Optional[torch.device] = None
    vae_offloaded = False
    if offload_vae_during_denoise and output_type in {"np", "pil"} and self.vae is not None:
      vae_device = _module_device(self.vae)
      if vae_device.type == "cuda":
        self.vae.to("cpu")
        torch.cuda.empty_cache()
        vae_restore_device = vae_device
        vae_offloaded = True

    prompt_model_input = prompt_embeds.to(transformer_dtype)
    if do_cfg:
      negative_model_input = negative_embeds.to(transformer_dtype)

    for timestep in self.progress_bar(self.scheduler.timesteps):
      timestep_batch = _transformer_timestep(timestep, transformer_dtype).expand(1).to(device)
      latent_model_input = latents
      with _transformer_autocast(device, transformer_dtype):
        noise_pred = self.transformer(
          latent_model_input,
          timestep_batch,
          prompt_model_input,
          encoder_attention_mask=prompt_mask,
          return_dict=False,
        )[0].float()
      if do_cfg:
        with _transformer_autocast(device, transformer_dtype):
          noise_pred_uncond = self.transformer(
            latent_model_input,
            timestep_batch,
            negative_model_input,
            encoder_attention_mask=negative_mask,
            return_dict=False,
          )[0].float()
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

      latents = self.scheduler.step(
        noise_pred,
        timestep,
        latents,
        return_dict=False,
        generator=generator,
      )[0]
      if cond_latent is not None:
        latents = self._apply_inpainting(latents, cond_latent)

    if output_type == "latent":
      frames = latents
    elif output_type in {"np", "pil"}:
      if vae_offloaded and vae_restore_device is not None:
        self.vae.to(device=vae_restore_device)
        torch.cuda.empty_cache()
      frames = self._decode_latents(latents, output_type)
    else:
      raise ValueError(f"Unsupported output_type: {output_type}")

    self.maybe_free_model_hooks()
    if not return_dict:
      return (frames, )
    return LingBotVideoPipelineOutput(frames=frames)
