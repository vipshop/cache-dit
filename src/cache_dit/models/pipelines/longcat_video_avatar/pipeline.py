"""LongCat-Video-Avatar inference pipeline (audio-driven human video, A2V / AI2V).

Copied from `meituan-longcat/LongCat-Video` (main, 2026-11-03),
`longcat_video/pipeline_longcat_video_avatar.py`, reduced to the two single-clip entry points:

* `image=None` renders audio + text to video (**A2V** / `at2v` upstream);
* `image=<PIL image>` renders image + audio + text to video (**AI2V** / `ai2v` upstream), where the
  first latent frame is the encoded reference image and receives no audio update.

Intentional differences from the upstream copy:

* a `DiffusionPipeline` subclass with `register_modules`, so `cache-dit` can resolve components,
  offload them (`model_cpu_offload_seq`) and hand them to the TP / VAE-P planners;
* the upstream `at2v` / `ai2v` entry points are merged into one `__call__` (the only difference is
  the conditioning image), and the long-video autoregressive continuation (`generate_avc`, kv-cache,
  `num_segments`) is not ported;
* the distributed bookkeeping (`context_parallel_util.cp_broadcast` for text / latents / audio) is
  gone: cache-dit owns the parallel runtime, and every rank encodes the same prompt;
* the DMD distillation LoRA is **merged into the weights** at load time instead of patching
  `Linear.forward`. Both are numerically equivalent for inference (no dropout, no training), and a
  merged weight survives layerwise offload, quantization and block-level caching, all of which
  replace or re-wrap the modules the runtime patch hangs on;
* the loudness normalization (LUFS −23) is applied through `pyloudnorm` when it is installed and
  skipped with a warning otherwise;
* a cache guard rejects the configurations cache-dit cannot represent (see `__call__`).
"""

import os
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.video_processor import VideoProcessor
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoTokenizer, UMT5EncoderModel, WhisperModel

from ....logger import init_logger
from .transformer import LongCatVideoAvatarTransformer3DModel

logger = init_logger(__name__)

_LORA_HYPHEN = "___lorahyphen___"
_LORA_PREFIX = "lora___lorahyphen___"

# Whisper-Large-v3 expects 16 kHz mono audio, and the lip-sync features are extracted at the encoder
# frame rate then interpolated to the video frame rate.
_AUDIO_SAMPLE_RATE = 16000
_MEL_CHUNK = 750 * 640
_ENC_CHUNK = 3000
_ENC_FPS = 50


def _resample_waveform(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
  """Resample a mono waveform with linear interpolation.

  :param audio: mono waveform.
  :param source_rate: sample rate of `audio`.
  :param target_rate: target sample rate.
  :returns: the resampled waveform as float32.
  """

  if source_rate == target_rate:
    return audio.astype(np.float32)
  duration = audio.shape[0] / float(source_rate)
  target = int(round(duration * target_rate))
  source_positions = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
  target_positions = np.linspace(0.0, duration, num=target, endpoint=False)
  return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _load_waveform(path: str, sample_rate: int = _AUDIO_SAMPLE_RATE) -> np.ndarray:
  """Read a wav file as a mono float32 waveform at the encoder sample rate.

  :param path: wav file path.
  :param sample_rate: target sample rate.
  :returns: mono waveform.
  """

  import soundfile as sf

  audio, sr = sf.read(path, dtype="float32", always_2d=True)
  return _resample_waveform(audio.mean(axis=1), sr, sample_rate)


def merge_lora_state_dict(transformer: torch.nn.Module, state_dict, multiplier: float = 1.0) -> int:
  """Merge a LongCat LoRA state dict into the transformer weights.

  The release ships a kohya-style LoRA whose keys encode the module path with
  `___lorahyphen___` separators and carry their own `alpha_scale`. For inference the runtime patch
  `W x + up(down(x)) * multiplier * alpha_scale` is exactly `W + up @ down * multiplier *
  alpha_scale`, so the weights can be folded once. A packed projection (the fused QKV) stores
  `n_separate` parallel up-blocks: block `i` reads its own rank slice of `down` and writes its own
  output slice, so the dense delta is that block matrix.

  :param transformer: the vendored avatar transformer.
  :param state_dict: LoRA tensors, as loaded from the safetensors file.
  :param multiplier: extra scale on top of each module's `alpha_scale`.
  :returns: number of merged modules.
  """

  module_names = sorted(
    {key.split(".lora_down.weight")[0]
     for key in state_dict if key.endswith(".lora_down.weight")})
  merged = 0
  for lora_name in module_names:
    module_name = lora_name.replace(_LORA_PREFIX, "").replace(_LORA_HYPHEN, ".")
    target = transformer
    for part in module_name.split("."):
      target = getattr(target, part, None)
      if target is None:
        break
    if not isinstance(target, torch.nn.Linear):
      logger.warning(f"Skipping `{module_name}`: not a Linear module of the transformer")
      continue
    down = state_dict[f"{lora_name}.lora_down.weight"].float()
    up_key = f"{lora_name}.lora_up.weight"
    if up_key in state_dict:
      delta = state_dict[up_key].float() @ down
    else:
      parts = sorted((key for key in state_dict if key.startswith(f"{lora_name}.lora_up.blocks.")),
                     key=lambda key: int(key.split(".")[-2]))
      if not parts:
        logger.warning(f"Skipping `{module_name}`: no LoRA up-projection in the checkpoint")
        continue
      n_blocks = len(parts)
      rank = down.shape[0] // n_blocks
      out_per_block = target.weight.shape[0] // n_blocks
      delta = torch.zeros(target.weight.shape, dtype=torch.float32)
      for block_index, key in enumerate(parts):
        rows = slice(block_index * out_per_block, (block_index + 1) * out_per_block)
        rank_slice = slice(block_index * rank, (block_index + 1) * rank)
        delta[rows, :] = state_dict[key].float() @ down[rank_slice, :]
    delta = delta * (multiplier * float(state_dict[f"{lora_name}.alpha_scale"].item()))
    target.weight.data = (target.weight.data.float() + delta).to(target.weight.dtype)
    merged += 1
  return merged


class LongCatVideoAvatarPipeline(DiffusionPipeline):
  """Audio (+ optional image) driven human video generation.

  :param tokenizer: UMT5 tokenizer.
  :param text_encoder: `UMT5EncoderModel`, the prompt encoder.
  :param vae: `AutoencoderKLWan` (16 channels, 8x8x4 downsampling).
  :param scheduler: `FlowMatchEulerDiscreteScheduler` (shift 7 for this checkpoint).
  :param transformer: :class:`LongCatVideoAvatarTransformer3DModel`.
  :param audio_encoder: `WhisperModel`, the waveform encoder (Whisper-Large-v3).
  :param audio_feature_extractor: the matching Whisper feature extractor.
  :param model_type: `avatar-v1.5` (Whisper audio encoder, 8-step distillation) or `avatar-v1.0`.
  """

  model_cpu_offload_seq = "text_encoder->audio_encoder->transformer->vae"
  _exclude_from_cpu_offload = ["audio_feature_extractor", "tokenizer"]
  _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

  def __init__(
    self,
    tokenizer: AutoTokenizer,
    text_encoder: UMT5EncoderModel,
    vae: AutoencoderKLWan,
    scheduler: FlowMatchEulerDiscreteScheduler,
    transformer: LongCatVideoAvatarTransformer3DModel,
    audio_encoder: Optional[WhisperModel] = None,
    audio_feature_extractor=None,
    model_type: str = "avatar-v1.5",
  ):
    super().__init__()

    self.register_modules(
      vae=vae,
      text_encoder=text_encoder,
      tokenizer=tokenizer,
      scheduler=scheduler,
      transformer=transformer,
      audio_encoder=audio_encoder,
      audio_feature_extractor=audio_feature_extractor,
    )

    self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(
      self, "vae", None) else 4
    self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(
      self, "vae", None) else 8
    self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    self.model_type = model_type
    self._num_timesteps = 1000
    self._num_distill_sample_steps = 8 if model_type == "avatar-v1.5" else 50
    self._text_guidance_scale = 4.0
    self._audio_guidance_scale = 4.0
    # cache-dit's `has_separate_cfg`: `None` means "infer from the guidance defaults", i.e. the
    # undialed 50-step recipe runs a conditional and an unconditional call per step. The distilled
    # recipe runs at guidance 1.0 (a single call per step) and overrides this in
    # `load_distill_lora`.
    self.cache_has_separate_cfg = None
    self._distill_merged = False

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
    """Load the released LongCat-Video-Avatar checkpoint.

    The checkpoint is not a `diffusers` repository: `model_index.json` only records a model name, and
    the components live in sub-directories (`base_model/`, `lora/`, `whisper-large-v3/`,
    `scheduler/`). The text encoder, tokenizer and VAE come from the base `LongCat-Video` release,
    which this checkpoint does not duplicate.

    :param pretrained_model_name_or_path: the `LongCat-Video-Avatar-1.5` directory.
    :param kwargs: `torch_dtype`, `base_model_path` (the `LongCat-Video` directory, also resolved
      from `LONGCAT_VIDEO_DIR`),       `text_encoder_path` (also `LONGCAT_VIDEO_TEXT_ENCODER_DIR`),
      `use_distill` (default True — the release's 8-step recipe; `LONGCAT_VIDEO_AVATAR_NO_DISTILL=1`
      opts out), `local_files_only`, `quantization_config`, plus the components the caller wants to
      inject (`transformer`, `vae`, `text_encoder`, `tokenizer`, `scheduler`, `audio_encoder`,
      `audio_feature_extractor`).
    :returns: a :class:`LongCatVideoAvatarPipeline`.
    """

    torch_dtype = kwargs.pop("torch_dtype", None) or torch.bfloat16
    local_files_only = kwargs.pop("local_files_only", True)
    quantization_config = kwargs.pop("quantization_config", None)
    use_distill = kwargs.pop("use_distill", None)
    if use_distill is None:
      # The release recommends the distilled 8-step recipe, so it is the default; the LoRA has to be
      # merged at load time (not at call time), because merging after TP / quantization / offload has
      # already rewritten the modules is not equivalent.
      use_distill = os.environ.get("LONGCAT_VIDEO_AVATAR_NO_DISTILL", "0").lower() != "1"
    base_model_path = kwargs.pop("base_model_path", None)
    text_encoder_path = kwargs.pop("text_encoder_path", None)
    device_map = kwargs.pop("device_map", None)
    kwargs.pop("trust_remote_code", None)
    if device_map is not None:
      raise NotImplementedError(
        "The vendored LongCat-Video-Avatar pipeline does not support `device_map`.")

    injected = {
      name: kwargs.pop(name, None)
      for name in ("scheduler", "transformer", "vae", "text_encoder", "tokenizer", "audio_encoder",
                   "audio_feature_extractor")
    }
    if kwargs:
      logger.warning(f"Ignoring unsupported `from_pretrained` kwargs: {sorted(kwargs)}")

    model_path = str(pretrained_model_name_or_path)
    if base_model_path is None:
      base_model_path = os.environ.get("LONGCAT_VIDEO_DIR", None)
    if base_model_path is None:
      sibling = os.path.join(os.path.dirname(os.path.abspath(model_path)), "LongCat-Video")
      base_model_path = sibling if os.path.isdir(sibling) else None
    if text_encoder_path is None:
      # `LONGCAT_VIDEO_TEXT_ENCODER_DIR` lets a setup point the prompt encoder at a shared UMT5-XXL
      # export (any `Wan*-Diffusers` checkpoint ships one) instead of duplicating 22.7 GB.
      text_encoder_path = os.environ.get("LONGCAT_VIDEO_TEXT_ENCODER_DIR", None) or base_model_path
    if base_model_path is None or text_encoder_path is None:
      raise FileNotFoundError(
        "LongCat-Video-Avatar needs the base `LongCat-Video` release for the text encoder, tokenizer "
        "and VAE. Pass `base_model_path=...` or set `LONGCAT_VIDEO_DIR`.")

    if quantization_config is not None:
      logger.warning(
        "`quantization_config` is dropped: the checkpoint's own quantizer cannot see the denoising "
        "stack, so quantize through cache-dit's flags instead.")
      quantization_config = None

    transformer = injected.get("transformer")
    if transformer is None:
      transformer = LongCatVideoAvatarTransformer3DModel.from_pretrained(
        model_path,
        subfolder="base_model",
        torch_dtype=torch_dtype,
        local_files_only=local_files_only)

    vae = injected.get("vae")
    if vae is None:
      vae = AutoencoderKLWan.from_pretrained(base_model_path,
                                             subfolder="vae",
                                             torch_dtype=torch_dtype,
                                             local_files_only=local_files_only)

    text_encoder = injected.get("text_encoder")
    if text_encoder is None:
      text_encoder = UMT5EncoderModel.from_pretrained(text_encoder_path,
                                                      subfolder="text_encoder",
                                                      torch_dtype=torch_dtype,
                                                      local_files_only=local_files_only)

    tokenizer = injected.get("tokenizer")
    if tokenizer is None:
      tokenizer = AutoTokenizer.from_pretrained(base_model_path,
                                                subfolder="tokenizer",
                                                local_files_only=local_files_only)

    scheduler = injected.get("scheduler")
    if scheduler is None:
      scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path,
                                                                  subfolder="scheduler",
                                                                  local_files_only=local_files_only)

    audio_encoder = injected.get("audio_encoder")
    audio_feature_extractor = injected.get("audio_feature_extractor")
    if audio_encoder is None or audio_feature_extractor is None:
      whisper_path = os.path.join(model_path, "whisper-large-v3")
      if not os.path.isdir(whisper_path):
        raise FileNotFoundError(f"Missing the Whisper audio encoder directory: {whisper_path}")
      if audio_encoder is None:
        audio_encoder = WhisperModel.from_pretrained(whisper_path, torch_dtype=torch.float32)
      if audio_feature_extractor is None:
        from transformers import AutoFeatureExtractor

        audio_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_path)

    pipe = cls(
      tokenizer=tokenizer,
      text_encoder=text_encoder,
      vae=vae,
      scheduler=scheduler,
      transformer=transformer,
      audio_encoder=audio_encoder.eval(),
      audio_feature_extractor=audio_feature_extractor,
      model_type="avatar-v1.5",
    )
    if use_distill:
      pipe.load_distill_lora(os.path.join(model_path, "lora", "dmd_lora.safetensors"))
    return pipe

  def load_distill_lora(self, lora_path: str, multiplier: float = 1.0) -> int:
    """Merge the step-distillation LoRA into the transformer weights.

    Upstream installs a runtime `forward` patch on every targeted `Linear`; merging is the same math
    for inference and survives offload / quantization / block-level caching, all of which replace or
    re-wrap those modules.

    :param lora_path: `lora/dmd_lora.safetensors` of the release.
    :param multiplier: extra LoRA scale on top of the stored `alpha_scale`.
    :returns: number of merged modules.
    """

    merged = merge_lora_state_dict(
      self.transformer,
      load_file(lora_path, device="cpu"),
      multiplier=multiplier,
    )
    logger.info(f"Merged the distillation LoRA into {merged} modules")
    # The distilled recipe samples at guidance 1.0, so every denoising step is a single transformer
    # call; tell the caching layer so the step bookkeeping is not halved.
    self.cache_has_separate_cfg = False
    self._distill_merged = True
    return merged

  @property
  def do_classifier_free_guidance(self) -> bool:
    """Whether the last (or default) run uses classifier-free guidance.

    :returns: True when either guidance scale is above 1, i.e. two transformer calls per step.
    """

    return self._text_guidance_scale > 1.0 or self._audio_guidance_scale > 1.0

  def _transformer_dtype(self) -> torch.dtype:
    """Dtype the transformer activations use.

    :returns: the patch-embedding weight dtype, which quantization leaves in bf16 (the projection is
      a 3-D convolution, not a quantizable `Linear`).
    """

    return self.transformer.x_embedder.proj.weight.dtype

  def get_timesteps_sigmas(self, sampling_steps: int, use_distill: bool = False) -> torch.Tensor:
    """Build the sigma schedule.

    :param sampling_steps: number of denoising steps.
    :param use_distill: use the distilled 8-step schedule instead of a linear ramp.
    :returns:`[sampling_steps]` float32 sigmas, descending.
    """

    if use_distill:
      if sampling_steps != self._num_distill_sample_steps:
        raise ValueError(
          f"The distilled schedule of this checkpoint has exactly {self._num_distill_sample_steps} "
          f"steps (one sigma per student step); got num_inference_steps={sampling_steps}. Pass "
          f"`--steps {self._num_distill_sample_steps}`, or unset `use_distill` for the linear ramp."
        )
      distill_indices = torch.arange(1, self._num_distill_sample_steps + 1, dtype=torch.float32)
      distill_indices = (distill_indices *
                         (self._num_timesteps // self._num_distill_sample_steps)).round().long()
      distill_indices = self._num_timesteps - distill_indices
      sigmas = torch.flip(torch.linspace(0, 1, self._num_timesteps), [0])
      sigmas = torch.flip(sigmas[distill_indices], [0]).float()
    else:
      sigmas = torch.linspace(1, 0.001, sampling_steps)
    return sigmas.to(torch.float32)

  def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
    """Normalize latents with the VAE statistics.

    :param latents:`[B, C, T, H, W]` raw VAE latents.
    :returns: normalized latents.
    """

    mean, std = self._latents_stats(latents)
    return (latents - mean) * std

  def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
    """Undo :meth:`normalize_latents`.

    :param latents:`[B, C, T, H, W]` normalized latents.
    :returns: raw VAE latents.
    """

    mean, std = self._latents_stats(latents)
    return latents / std + mean

  def _latents_stats(self, latents: torch.Tensor):
    """Broadcast the VAE latent statistics to the latent shape.

    :param latents: `[B, C, T, H, W]` latents.
    :returns: `(mean, 1 / std)` both broadcastable to `latents`.
    """

    mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1,
                                                           1).to(latents.device, latents.dtype)
    std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1,
                                                               1).to(latents.device, latents.dtype)
    return mean, std

  def _retrieve_latents(self, encoder_output, generator=None) -> torch.Tensor:
    """Take the deterministic latent of a VAE encode.

    :param encoder_output: `AutoencoderKLOutput` of the VAE encoder.
    :param generator: unused, kept for parity with the upstream helper.
    :returns: `[B, C, T, H, W]` latents (the mode of the diagonal Gaussian).
    """

    if hasattr(encoder_output, "latent_dist"):
      return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
      return encoder_output.latents
    raise AttributeError("Could not access latents of the VAE encoder output")

  def prepare_latents(
    self,
    image: Optional[torch.Tensor] = None,
    batch_size: int = 1,
    num_channels_latents: int = 16,
    height: int = 480,
    width: int = 832,
    num_frames: int = 93,
    num_cond_frames: int = 0,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Sample the initial latents and encode the conditioning image.

    :param image: `[B, C, H, W]` conditioning image, or `None` for the A2V task.
    :param batch_size: batch size.
    :param num_channels_latents: latent channels.
    :param height: pixel height.
    :param width: pixel width.
    :param num_frames: number of pixel frames.
    :param num_cond_frames: number of conditioning pixel frames (1 for an image).
    :param dtype: latent dtype.
    :param device: latent device.
    :param generator: torch generator for the initial noise.
    :param latents: pre-computed initial latents.
    :returns: `[B, C, T, H, W]` initial latents with the conditioning frames overwritten.
    """

    if latents is not None:
      return latents.to(device=device, dtype=dtype)

    num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
    shape = (
      batch_size,
      num_channels_latents,
      num_latent_frames,
      int(height) // self.vae_scale_factor_spatial,
      int(width) // self.vae_scale_factor_spatial,
    )
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

    if image is not None:
      num_cond_latents = 1 + (num_cond_frames - 1) // self.vae_scale_factor_temporal
      cond_latents = []
      for i in range(batch_size):
        encoded_input = image[i].unsqueeze(0).unsqueeze(2)
        assert encoded_input.shape[
          2] == num_cond_frames, "the conditioning input must be num_cond_frames long"
        latent = self._retrieve_latents(self.vae.encode(encoded_input))
        cond_latents.append(latent)
      cond_latents = self.normalize_latents(torch.cat(cond_latents, dim=0).to(dtype))
      latents[:, :, :num_cond_latents] = cond_latents

    return latents

  def _get_t5_prompt_embeds(
    self,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
  ):
    """Encode a prompt with the UMT5 encoder.

    :param prompt: prompt(s).
    :param num_videos_per_prompt: number of samples per prompt.
    :param max_sequence_length: padding length of the tokenizer.
    :param device: target device.
    :param dtype: embedding dtype.
    :returns:`(prompt_embeds [B, 1, L, C], attention_mask [B, L])`.
    """

    dtype = dtype or self.text_encoder.dtype
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = self.tokenizer(
      prompt,
      padding="max_length",
      max_length=max_sequence_length,
      truncation=True,
      add_special_tokens=True,
      return_attention_mask=True,
      return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    mask = mask.to(device=device)
    mask = torch.cat([mask] * num_videos_per_prompt, dim=0)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, 1, seq_len, -1)
    return prompt_embeds, mask

  def encode_prompt(
    self,
    prompt: Union[str, List[str]],
    negative_prompt: Optional[Union[str, List[str]]] = None,
    do_classifier_free_guidance: bool = True,
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
  ):
    """Encode the positive and (when guiding) negative prompts.

    :param prompt: prompt(s).
    :param negative_prompt: negative prompt(s).
    :param do_classifier_free_guidance: whether the negative branch is needed.
    :param num_videos_per_prompt: number of samples per prompt.
    :param max_sequence_length: padding length of the tokenizer.
    :param device: target device.
    :param dtype: embedding dtype.
    :returns:`(prompt_embeds, prompt_mask, negative_embeds, negative_mask)`.
    """

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds, prompt_mask = self._get_t5_prompt_embeds(prompt, num_videos_per_prompt,
                                                            max_sequence_length, device, dtype)

    if not do_classifier_free_guidance:
      return prompt_embeds, prompt_mask, None, None

    negative_prompt = negative_prompt or ""
    negative_prompt = [negative_prompt] * len(prompt) if isinstance(negative_prompt,
                                                                    str) else negative_prompt
    negative_embeds, negative_mask = self._get_t5_prompt_embeds(negative_prompt,
                                                                num_videos_per_prompt,
                                                                max_sequence_length, device, dtype)
    return prompt_embeds, prompt_mask, negative_embeds, negative_mask

  def _loudness_norm(self, audio_array: np.ndarray, sample_rate: int = _AUDIO_SAMPLE_RATE):
    """Normalize the loudness of a waveform to −23 LUFS.

    :param audio_array: mono waveform.
    :param sample_rate: sample rate of the waveform.
    :returns: the normalized waveform, or the input when `pyloudnorm` is unavailable.
    """

    try:
      import pyloudnorm as pyln
    except ImportError:
      logger.warning(
        "`pyloudnorm` is not installed: skipping the LUFS -23 normalization, which changes how "
        "strongly the audio drives the lips. Install `pyloudnorm` for a faithful run.")
      return audio_array
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
      return audio_array
    return pyln.normalize.loudness(audio_array, loudness, -23)

  @torch.no_grad()
  def get_audio_embedding(
    self,
    speech_array: np.ndarray,
    fps: int = 25,
    device=None,
    sample_rate: int = _AUDIO_SAMPLE_RATE,
  ) -> torch.Tensor:
    """Encode a waveform into per-frame audio features with the Whisper encoder.

    The encoder layer outputs are grouped (`0:8`, `8:16`, `16:24`, `24:32`, and the final hidden
    state) and each group is interpolated from the 50 Hz encoder rate to the video frame rate, which
    gives five feature vectors per frame — the `audio_window` of the transformer config.

    :param speech_array: mono waveform at `sample_rate`.
    :param fps: video frame rate.
    :param device: device of the audio encoder.
    :param sample_rate: sample rate of `speech_array`.
    :returns: `[T, audio_window, D]` audio features, one group per video frame.
    """

    def linear_interpolation_fps(features: torch.Tensor, input_fps: int, output_fps: int,
                                 output_len: int) -> torch.Tensor:
      features = features.transpose(1, 2)
      output_features = F.interpolate(features, size=output_len, align_corners=True, mode="linear")
      return output_features.transpose(1, 2)

    audio_duration = len(speech_array) / sample_rate
    video_length = int(audio_duration * fps)
    speech_array = self._loudness_norm(speech_array, sample_rate)

    mel_chunks = []
    for i in range(0, len(speech_array), _MEL_CHUNK):
      mel = self.audio_feature_extractor(speech_array[i:i + _MEL_CHUNK],
                                         sampling_rate=sample_rate,
                                         return_tensors="pt").input_features
      mel_chunks.append(mel)
    audio_features = torch.cat(mel_chunks, dim=-1)

    # The waveform encoder is driven by `pipe.to(device)`, but a caller can keep it on CPU on purpose
    # (it is the cheapest component to offload); follow wherever it actually lives.
    enc_device = next(self.audio_encoder.parameters()).device
    audio_features = audio_features.to(device=enc_device, dtype=self.audio_encoder.dtype)

    enc_chunks = []
    for i in range(0, audio_features.shape[-1], _ENC_CHUNK):
      chunk_hs = self.audio_encoder.encoder(audio_features[:, :, i:i + _ENC_CHUNK],
                                            output_hidden_states=True).hidden_states
      enc_chunks.append(torch.stack(chunk_hs, dim=2))
    audio_prompts = torch.cat(enc_chunks, dim=1)[:, :video_length * 2]

    feat0 = linear_interpolation_fps(audio_prompts[:, :, 0:8].mean(dim=2), _ENC_FPS, fps,
                                     video_length)
    feat1 = linear_interpolation_fps(audio_prompts[:, :, 8:16].mean(dim=2), _ENC_FPS, fps,
                                     video_length)
    feat2 = linear_interpolation_fps(audio_prompts[:, :, 16:24].mean(dim=2), _ENC_FPS, fps,
                                     video_length)
    feat3 = linear_interpolation_fps(audio_prompts[:, :, 24:32].mean(dim=2), _ENC_FPS, fps,
                                     video_length)
    feat4 = linear_interpolation_fps(audio_prompts[:, :, 32], _ENC_FPS, fps, video_length)
    return torch.stack([feat0, feat1, feat2, feat3, feat4], dim=2)[0]

  def _audio_center_indices(self, num_frames: int, stride: int) -> torch.Tensor:
    """Frame indices of the audio window around each video frame.

    :param num_frames: number of video frames.
    :param stride: audio stride, i.e. how many encoder frames one video frame advances.
    :returns:`[num_frames, audio_window]` indices, clamped to the valid range.
    """

    window = 2 * 2 + 1
    indices = torch.arange(window) - 2
    center = torch.arange(0, stride * num_frames, stride).unsqueeze(1) + indices.unsqueeze(0)
    return center.clamp(min=0)

  def _prepare_audio_embs(self, audio: np.ndarray, num_frames: int, fps: int,
                          device) -> torch.Tensor:
    """Build the audio conditioning tensor of one clip.

    :param audio: mono waveform at 16 kHz.
    :param num_frames: number of pixel frames of the clip.
    :param fps: video frame rate.
    :param device: device of the transformer.
    :returns:`[1, num_frames, audio_window, audio_block, audio_channel]` audio windows.
    """

    full = self.get_audio_embedding(audio, fps=fps, device=device, sample_rate=_AUDIO_SAMPLE_RATE)
    if torch.isnan(full).any():
      raise ValueError("The audio embedding contains NaN: check the driving waveform.")
    indices = self._audio_center_indices(num_frames, 1)
    indices = indices.clamp(max=full.shape[0] - 1)
    return full[indices][None, ...].to(device)

  @torch.no_grad()
  def __call__(
    self,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    image: Optional[Union[Image.Image, torch.Tensor]] = None,
    audio: Optional[np.ndarray] = None,
    audio_path: Optional[str] = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 93,
    num_inference_steps: int = 50,
    use_distill: bool = False,
    text_guidance_scale: float = 4.0,
    audio_guidance_scale: float = 4.0,
    num_videos_per_prompt: int = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    output_type: str = "np",
    max_sequence_length: int = 512,
    sampling_fps: int = 25,
    sampling_rate: int = _AUDIO_SAMPLE_RATE,
  ):
    """Render a talking clip (A2V without `image`, AI2V with one).

    :param prompt: text prompt.
    :param negative_prompt: negative prompt used by the guidance.
    :param image: reference image of the AI2V task; `None` renders the A2V task.
    :param audio: driving waveform at 16 kHz, mono.
    :param audio_path: wav file to read when `audio` is not given.
    :param height: pixel height, divisible by 16.
    :param width: pixel width, divisible by 16.
    :param num_frames: number of pixel frames; `(num_frames - 1)` must be divisible by 4.
    :param num_inference_steps: denoising steps.
    :param use_distill: use the 8-step distilled schedule (needs the merged DMD LoRA).
    :param text_guidance_scale: text classifier-free guidance scale.
    :param audio_guidance_scale: audio classifier-free guidance scale.
    :param num_videos_per_prompt: samples per prompt.
    :param generator: torch generator of the initial noise.
    :param latents: pre-computed initial latents.
    :param output_type: `np`, `latent` or `both`.
    :param max_sequence_length: padding length of the tokenizer.
    :param sampling_fps: frame rate the audio features are sampled at.
    :param sampling_rate: sample rate of the passed waveform; the encoder runs at 16 kHz.
    :returns: frames `(B, T, H, W, C)` for `np`, latents for `latent`, `(frames, latents)` for
      `both`.
    """

    if audio is None:
      if audio_path is None:
        raise ValueError(
          "LongCat-Video-Avatar needs a driving waveform: pass `audio` or `audio_path`.")
      audio = _load_waveform(audio_path, sample_rate=sampling_rate)
    elif sampling_rate != _AUDIO_SAMPLE_RATE:
      audio = _resample_waveform(np.asarray(audio, dtype=np.float32), sampling_rate,
                                 _AUDIO_SAMPLE_RATE)

    if num_frames % self.vae_scale_factor_temporal != 1:
      num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
      logger.warning(f"`num_frames` rounded to {num_frames} (the VAE temporal factor needs 4k+1)")
    num_frames = max(num_frames, 1)
    if num_frames < 13:
      raise ValueError(
        f"LongCat-Video-Avatar needs at least 13 pixel frames (4 latent frames) so the per-frame "
        f"audio window can be stitched; got num_frames={num_frames}.")

    if use_distill and not self._distill_merged:
      raise ValueError(
        "`use_distill=True` needs the DMD LoRA merged into the transformer, which happens at load "
        "time (`from_pretrained(use_distill=True)`, the default). Load the pipeline with "
        "`LONGCAT_VIDEO_AVATAR_NO_DISTILL=0`, or run the undistilled schedule with "
        "`use_distill=False`.")

    self._text_guidance_scale = text_guidance_scale
    self._audio_guidance_scale = audio_guidance_scale
    do_classifier_free_guidance = text_guidance_scale > 1.0 or audio_guidance_scale > 1.0

    if getattr(self.transformer, "_is_cached", False):
      expected_separate_cfg = (True if self.cache_has_separate_cfg is None else
                               self.cache_has_separate_cfg)
      if do_classifier_free_guidance != expected_separate_cfg:
        expected_calls = 2 if expected_separate_cfg else 1
        raise ValueError(
          f"cache-dit was applied for {expected_calls} transformer call(s) per denoising step, but "
          f"this run makes {2 if do_classifier_free_guidance else 1}: the guidance scales "
          f"({text_guidance_scale} text / {audio_guidance_scale} audio) do not match the recipe the "
          f"adapter was built for. The distilled 8-step recipe samples at guidance 1.0; the "
          f"undistilled recipe needs a scale above 1. Use the matching example, or disable caching."
        )

    device = self._execution_device
    dit_dtype = self._transformer_dtype()

    prompt_embeds, prompt_mask, negative_embeds, negative_mask = self.encode_prompt(
      prompt=prompt,
      negative_prompt=negative_prompt,
      do_classifier_free_guidance=do_classifier_free_guidance,
      num_videos_per_prompt=num_videos_per_prompt,
      max_sequence_length=max_sequence_length,
      dtype=dit_dtype,
      device=device,
    )
    if isinstance(prompt, str):
      batch_size = 1
    else:
      batch_size = len(prompt)

    audio_embs = self._prepare_audio_embs(audio, num_frames, sampling_fps, device)
    audio_cond_embs = torch.cat([audio_embs] * num_videos_per_prompt, dim=0)
    if do_classifier_free_guidance:
      prompt_embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)
      prompt_mask = torch.cat([negative_mask, prompt_mask], dim=0)
      audio_uncond_embs = torch.zeros_like(audio_cond_embs)
      audio_cond_embs = torch.cat([audio_cond_embs, audio_cond_embs], dim=0)

    num_cond_latents = 0
    cond_image = None
    if image is not None:
      if isinstance(image, Image.Image):
        cond_image = self.video_processor.preprocess(image,
                                                     height=height,
                                                     width=width,
                                                     resize_mode="crop").to(device=device,
                                                                            dtype=dit_dtype)
        if cond_image.ndim == 4:
          cond_image = cond_image.squeeze(0)
      else:
        cond_image = image.to(device=device, dtype=dit_dtype)
      num_cond_latents = 1
      cp_world = getattr(getattr(self.transformer, "_cp_config", None), "_world_size", 1) or 1
      if cp_world > 1:
        raise NotImplementedError(
          "The image-conditioned task (AI2V) cannot run under context parallelism: the conditioning "
          "frame is only owned by the rank its frame range covers, so the per-frame audio "
          "cross-attention and the conditioned self-attention would need different numbers of "
          "collectives per rank. Measured on 480x832 x 93 frames, forcing it through the uniform "
          "attention path drops the clip to 21.9 dB / SSIM 0.76. Use `--parallel tp` for AI2V, or "
          "the audio+text task (A2V) which parallelizes cleanly.")

    sigmas = self.get_timesteps_sigmas(num_inference_steps, use_distill=use_distill)
    self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, device=device)
    timesteps = self.scheduler.timesteps

    latents = self.prepare_latents(
      image=cond_image.unsqueeze(0) if cond_image is not None else None,
      batch_size=batch_size * num_videos_per_prompt,
      num_channels_latents=self.transformer.config.in_channels,
      height=height,
      width=width,
      num_frames=num_frames,
      num_cond_frames=1 if cond_image is not None else 0,
      dtype=torch.float32,
      device=device,
      generator=generator,
      latents=latents,
    )

    common_kwargs = dict(
      encoder_attention_mask=prompt_mask,
      num_cond_latents=num_cond_latents,
    )

    for t in timesteps:
      latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
      latent_model_input = latent_model_input.to(dit_dtype)

      timestep = t.expand(latent_model_input.shape[0]).to(dit_dtype)
      if num_cond_latents:
        timestep = timestep.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
        timestep[:, :1] = 0

      noise_pred = self.transformer(
        hidden_states=latent_model_input,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        audio_embs=audio_cond_embs,
        **common_kwargs,
      )

      if do_classifier_free_guidance:
        timestep_uncond = t.expand(latents.shape[0]).to(dit_dtype)
        if num_cond_latents:
          timestep_uncond = timestep_uncond.unsqueeze(-1).repeat(1, latents.shape[2])
          timestep_uncond[:, :1] = 0
        noise_pred_uncond = self.transformer(
          hidden_states=latents,
          timestep=timestep_uncond,
          encoder_hidden_states=negative_embeds,
          encoder_attention_mask=negative_mask,
          num_cond_latents=num_cond_latents,
          audio_embs=audio_uncond_embs,
        )
        noise_pred_uncond_text, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = (noise_pred_uncond + text_guidance_scale *
                      (noise_pred_cond - noise_pred_uncond_text) + audio_guidance_scale *
                      (noise_pred_uncond_text - noise_pred_uncond))

      noise_pred = -noise_pred
      if num_cond_latents:
        latents[:, :, num_cond_latents:] = self.scheduler.step(noise_pred[:, :, num_cond_latents:],
                                                               t,
                                                               latents[:, :, num_cond_latents:],
                                                               return_dict=False)[0]
      else:
        latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    if output_type == "latent":
      return latents

    latents_for_vae = self.denormalize_latents(latents.to(self.vae.dtype))
    video = self.vae.decode(latents_for_vae, return_dict=False)[0]
    video = self.video_processor.postprocess_video(video, output_type="np")

    if output_type == "both":
      return video, latents
    return video
