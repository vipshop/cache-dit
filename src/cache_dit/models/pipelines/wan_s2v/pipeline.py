"""Wan2.2-S2V speech-to-video pipeline (vendored).

Ported from the unmerged upstream PR `huggingface/diffusers#12258
<https://github.com/huggingface/diffusers/pull/12258>`_ (Apache-2.0), which is not part of any
diffusers release. cache-dit owns the code so that the model can be cached, sharded and
quantized without patching a third-party library.

Differences from the upstream PR, all of them required by cache-dit:

- No pose (driving-video) conditioning: ``pose_latents`` is always zero, matching the vllm-omni
  port of the same model.
- ``resize_min_center_crop`` is implemented locally instead of being added to diffusers'
  ``VaeImageProcessor``.
- The audio encoder is a plain ``Wav2Vec2Model`` plus a ``Wav2Vec2FeatureExtractor`` (the PR used
  ``Wav2Vec2ForCTC`` / ``Wav2Vec2Processor``, which drags in the CTC decoding stack).
- The cache bookkeeping is reset between chunks, because one ``__call__`` runs several complete
  denoising passes when the audio is longer than one chunk.
"""

import html
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKLWan
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from PIL import Image
from transformers import AutoTokenizer, UMT5EncoderModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model

from ....logger import init_logger
from .audio_processor import PipelineAudioInput, is_valid_audio
from .transformer import WanS2VTransformer3DModel

logger = init_logger(__name__)


def basic_clean(text: str) -> str:
  text = html.unescape(html.unescape(text))
  return text.strip()


def whitespace_clean(text: str) -> str:
  text = re.sub(r"\s+", " ", text)
  return text.strip()


def prompt_clean(text: str) -> str:
  return whitespace_clean(basic_clean(text))


def get_sample_indices(original_fps, total_frames, target_fps, num_sample, fixed_start=None):
  required_duration = num_sample / target_fps
  required_origin_frames = int(np.ceil(required_duration * original_fps))
  if required_duration > total_frames / original_fps:
    raise ValueError("required_duration must be less than video length")

  if fixed_start is not None and fixed_start >= 0:
    start_frame = fixed_start
  else:
    max_start = total_frames - required_origin_frames
    if max_start < 0:
      raise ValueError("video length is too short")
    start_frame = np.random.randint(0, max_start + 1)
  start_time = start_frame / original_fps

  end_time = start_time + required_duration
  time_points = np.linspace(start_time, end_time, num_sample, endpoint=False)

  frame_indices = np.round(np.array(time_points) * original_fps).astype(int)
  frame_indices = np.clip(frame_indices, 0, total_frames - 1)
  return frame_indices


def linear_interpolation(features: torch.Tensor, input_fps: int, output_fps: int) -> torch.Tensor:
  """Resample `[1, T, C]` audio features from `input_fps` to `output_fps` by linear
  interpolation."""
  features = features.transpose(1, 2)  # [1, C, T]
  seq_len = features.shape[2] / float(input_fps)
  output_len = int(seq_len * output_fps)
  output_features = F.interpolate(features, size=output_len, align_corners=True, mode="linear")
  return output_features.transpose(1, 2)  # [1, output_len, C]


def retrieve_latents(encoder_output: torch.Tensor,
                     generator: Optional[torch.Generator] = None,
                     sample_mode: str = "sample"):
  if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
    return encoder_output.latent_dist.sample(generator)
  elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
    return encoder_output.latent_dist.mode()
  elif hasattr(encoder_output, "latents"):
    return encoder_output.latents
  else:
    raise AttributeError("Could not access latents of provided encoder_output")


def resize_min_center_crop(image: Image.Image, height: int, width: int) -> Image.Image:
  """Resize so the shorter side becomes ``min(height, width)``, then center crop.

  This reproduces Wan2.2-S2V's preprocessing (``resize_type="min_dimension"`` +
  ``crop_type="center_crop"``); diffusers' ``VaeImageProcessor`` does not ship that mode.

  :param image: input PIL image.
  :param height: target height.
  :param width: target width.
  :returns: the preprocessed PIL image of exactly `(width, height)`.
  """
  short_side = min(image.height, image.width)
  scale = min(height, width) / short_side
  new_height = max(1, int(round(image.height * scale)))
  new_width = max(1, int(round(image.width * scale)))
  image = image.resize((new_width, new_height), resample=Image.BILINEAR)
  left = (new_width - width) // 2
  top = (new_height - height) // 2
  return image.crop((left, top, left + width, top + height))


class WanSpeechToVideoPipeline(DiffusionPipeline):
  """Prompt + reference image + audio -> video (Wan2.2-S2V).

  :param tokenizer: umT5 tokenizer.
  :param text_encoder: `UMT5EncoderModel`.
  :param vae: `AutoencoderKLWan` (Wan2.1 16-channel VAE).
  :param scheduler: `UniPCMultistepScheduler` with `flow_shift=5.0`.
  :param transformer: :class:`WanS2VTransformer3DModel`.
  :param audio_encoder: `Wav2Vec2Model` (`wav2vec2-large-xlsr-53`).
  :param audio_processor: `Wav2Vec2FeatureExtractor`.
  """

  model_cpu_offload_seq = "text_encoder->audio_encoder->transformer->vae"
  _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

  def __init__(
    self,
    tokenizer: AutoTokenizer,
    text_encoder: UMT5EncoderModel,
    vae: AutoencoderKLWan,
    scheduler: UniPCMultistepScheduler,
    transformer: WanS2VTransformer3DModel,
    audio_encoder: Wav2Vec2Model,
    audio_processor: Wav2Vec2FeatureExtractor,
  ):
    super().__init__()

    self.register_modules(
      vae=vae,
      text_encoder=text_encoder,
      tokenizer=tokenizer,
      transformer=transformer,
      scheduler=scheduler,
      audio_encoder=audio_encoder,
      audio_processor=audio_processor,
    )

    self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(
      self, "vae", None) else 4
    self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(
      self, "vae", None) else 8
    self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial,
                                          resample="bilinear")
    self.motion_frames = 73

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
    """Load the components from a converted Wan2.2-S2V directory.

    The checkpoint does not go through `DiffusionPipeline.from_pretrained`: its
    `model_index.json` names pipeline and transformer classes that diffusers cannot import.

    :param pretrained_model_name_or_path: directory produced by
      `tools/convert_wan_s2v_to_diffusers.py`.
    :param kwargs: `torch_dtype`, `local_files_only` and `quantization_config` are honoured;
      unknown keywords are reported and dropped.
    :returns: a :class:`WanSpeechToVideoPipeline`.
    """
    torch_dtype = kwargs.pop("torch_dtype", None)
    local_files_only = kwargs.pop("local_files_only", True)
    quantization_config = kwargs.pop("quantization_config", None)
    device_map = kwargs.pop("device_map", None)
    kwargs.pop("trust_remote_code", None)
    if device_map is not None:
      raise NotImplementedError(
        "`device_map` is not supported by the vendored Wan2.2-S2V pipeline.")
    # The CLI hands over components it wants injected (a caller-provided scheduler, a quantized
    # transformer, ...); anything left in `kwargs` is not understood and is reported.
    injected = {
      name: kwargs.pop(name, None)
      for name in ("scheduler", "transformer", "vae", "text_encoder", "tokenizer", "audio_encoder",
                   "audio_processor")
    }
    if kwargs:
      logger.warning(f"Ignoring unsupported `from_pretrained` kwargs: {sorted(kwargs)}")

    model_path = str(pretrained_model_name_or_path)

    def _load_component(folder: str, loader, weighted: bool = True, dtype=None):
      if injected.get(folder) is not None:
        return injected[folder]
      load_kwargs = {"subfolder": folder, "local_files_only": local_files_only}
      if weighted:
        load_kwargs["torch_dtype"] = dtype if dtype is not None else torch_dtype
      return loader.from_pretrained(model_path, **load_kwargs)

    transformer_kwargs = {}
    if quantization_config is not None:
      transformer_kwargs["quantization_config"] = quantization_config
    if injected.get("transformer") is not None:
      transformer = injected["transformer"]
    else:
      transformer = WanS2VTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
        **transformer_kwargs,
      )

    return cls(
      tokenizer=_load_component("tokenizer", AutoTokenizer, weighted=False),
      text_encoder=_load_component("text_encoder", UMT5EncoderModel),
      vae=_load_component("vae", AutoencoderKLWan),
      scheduler=_load_component("scheduler", UniPCMultistepScheduler, weighted=False),
      transformer=transformer,
      # The waveform encoder stays float32: its conv stack runs on the raw 16 kHz signal and the
      # released weights are float32, so a bf16 cast both loses input precision (the features are
      # cast to float32 before the causal encoder) and mismatches the conv bias dtype.
      audio_encoder=_load_component("audio_encoder", Wav2Vec2Model, dtype=torch.float32),
      audio_processor=_load_component("audio_processor", Wav2Vec2FeatureExtractor, weighted=False),
    )

  def _transformer_dtype(self) -> torch.dtype:
    """Dtype the transformer expects for its activations.

    Not `transformer.dtype` and not "the first parameter": the first parameter of this model is
    `scale_shift_table`, which `_keep_in_fp32_modules` pins to float32, and a quantized (fp8 / int4)
    run leaves non-floating weights behind. fp8 stays out of the menu because it cannot hold the
    activations.

    :returns: the first float16-and-wider parameter dtype, `torch.bfloat16` when there is none.
    """
    dtype = getattr(self.transformer, "dtype", None)
    if isinstance(dtype, torch.dtype) and dtype.is_floating_point and dtype.itemsize >= 2:
      return dtype
    for parameter in self.transformer.parameters():
      if parameter.is_floating_point() and parameter.dtype.itemsize >= 2:
        return parameter.dtype
    return torch.bfloat16

  def _reset_cache_context(self) -> None:
    """Restart the cache bookkeeping of the current inference.

    A chunk is one complete denoising run, so the cache state of the previous chunk (residuals, step
    counters, collected stats) must not leak into the next one.
    """
    context_manager = getattr(self, "_context_manager", None)
    context_kwargs = getattr(self, "_context_kwargs", None)
    if context_manager is None or context_kwargs is None:
      return
    current_context = context_manager.get_context()
    if current_context is None:
      return
    context_manager.reset_context(current_context, **context_kwargs)

  def _get_t5_prompt_embeds(
    self,
    prompt: Union[str, List[str]] = None,
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
  ) -> torch.Tensor:
    device = device or self._execution_device
    dtype = dtype or self.text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
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
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack([
      torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds
    ],
                                dim=0)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
    return prompt_embeds

  def encode_prompt(
    self,
    prompt: Union[str, List[str]],
    negative_prompt: Optional[Union[str, List[str]]] = None,
    do_classifier_free_guidance: bool = True,
    num_videos_per_prompt: int = 1,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
  ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Encode the prompt and the negative prompt with umT5.

    :returns:`(prompt_embeds, negative_prompt_embeds)`.
    """
    device = device or self._execution_device

    prompt = [prompt] if isinstance(prompt, str) else prompt
    if prompt is not None:
      batch_size = len(prompt)
    else:
      batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
      prompt_embeds = self._get_t5_prompt_embeds(
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
      )

    if do_classifier_free_guidance and negative_prompt_embeds is None:
      negative_prompt = negative_prompt or ""
      negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt,
                                                                     str) else negative_prompt

      if prompt is not None and type(prompt) is not type(negative_prompt):
        raise TypeError(
          f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
          f" {type(prompt)}.")
      elif batch_size != len(negative_prompt):
        raise ValueError(
          f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
          f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
          " the batch size of `prompt`.")

      negative_prompt_embeds = self._get_t5_prompt_embeds(
        prompt=negative_prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
      )

    return prompt_embeds, negative_prompt_embeds

  def encode_audio(
    self,
    audio: PipelineAudioInput,
    sampling_rate: int,
    num_frames: int,
    fps: int = 16,
    device: Optional[torch.device] = None,
  ) -> Tuple[torch.Tensor, int]:
    """Encode the waveform into per-frame audio features.

    wav2vec2 runs at 50 fps; its hidden states are interpolated to 30 fps and then bucketed at the
    video frame rate. Frames past the end of the audio are silent (zero) buckets.

    :param audio: mono waveform, sample rate `sampling_rate`.
    :param sampling_rate: sample rate of `audio`.
    :param num_frames: number of frames per chunk, i.e. the number of frames per bucket.
    :param fps: video frame rate.
    :param device: execution device.
    :returns: `(audio_embeds [1, layers, dim, buckets], num_chunks)`.
    """
    device = device or self._execution_device
    video_rate = 30
    audio_sample_m = 0

    input_values = self.audio_processor(audio, sampling_rate=sampling_rate,
                                        return_tensors="pt").input_values

    res = self.audio_encoder(input_values.to(device), output_hidden_states=True)
    feat = torch.cat(res.hidden_states)

    feat = linear_interpolation(feat, input_fps=50, output_fps=video_rate)

    audio_embed = feat.to(torch.float32)  # encoding for the motion

    num_layers, audio_frame_num, audio_dim = audio_embed.shape

    if num_layers > 1:
      return_all_layers = True
    else:
      return_all_layers = False

    scale = video_rate / fps

    num_repeat = int(audio_frame_num / (num_frames * scale)) + 1

    bucket_num = num_repeat * num_frames
    padd_audio_num = math.ceil(num_repeat * num_frames / fps * video_rate) - audio_frame_num
    batch_idx = get_sample_indices(
      original_fps=video_rate,
      total_frames=audio_frame_num + padd_audio_num,
      target_fps=fps,
      num_sample=bucket_num,
      fixed_start=0,
    )
    batch_audio_eb = []
    audio_sample_stride = int(video_rate / fps)
    for bi in batch_idx:
      if bi < audio_frame_num:
        chosen_idx = list(
          range(
            bi - audio_sample_m * audio_sample_stride,
            bi + (audio_sample_m + 1) * audio_sample_stride,
            audio_sample_stride,
          ))
        chosen_idx = [0 if c < 0 else c for c in chosen_idx]
        chosen_idx = [audio_frame_num - 1 if c >= audio_frame_num else c for c in chosen_idx]

        if return_all_layers:
          frame_audio_embed = audio_embed[:, chosen_idx].flatten(start_dim=-2, end_dim=-1)
        else:
          frame_audio_embed = audio_embed[0][chosen_idx].flatten()
      else:
        frame_audio_embed = (torch.zeros([audio_dim * (2 * audio_sample_m + 1)],
                                         device=audio_embed.device) if not return_all_layers else
                             torch.zeros([num_layers, audio_dim * (2 * audio_sample_m + 1)],
                                         device=audio_embed.device))
      batch_audio_eb.append(frame_audio_embed)
    audio_embed_bucket = torch.cat([c.unsqueeze(0) for c in batch_audio_eb], dim=0)

    audio_embed_bucket = audio_embed_bucket.to(device)
    audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
    if len(audio_embed_bucket.shape) == 3:
      audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
    elif len(audio_embed_bucket.shape) == 4:
      audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
    return audio_embed_bucket, num_repeat

  def check_inputs(
    self,
    prompt,
    negative_prompt,
    image,
    height,
    width,
    prompt_embeds=None,
    negative_prompt_embeds=None,
    image_embeds=None,
    audio=None,
    audio_embeds=None,
  ):
    if image is not None and image_embeds is not None:
      raise ValueError(
        f"Cannot forward both `image`: {image} and `image_embeds`: {image_embeds}. Please make sure to"
        " only forward one of the two.")
    if image is None and image_embeds is None:
      raise ValueError(
        "Provide either `image` or `image_embeds`. Cannot leave both `image` and `image_embeds` undefined."
      )
    if image is not None and not isinstance(image, torch.Tensor) and not isinstance(
        image, Image.Image):
      raise ValueError(
        f"`image` has to be of type `torch.Tensor` or `PIL.Image.Image` but is {type(image)}")
    if image_embeds is not None:
      # The reference branch needs the pixels: they are VAE-encoded into the condition latents and
      # the first video frame. Accepting pre-encoded embeddings would silently produce a video that
      # does not start from the reference image.
      raise NotImplementedError("`image_embeds` is not supported; pass `image` (the pixels).")
    if height % 16 != 0 or width % 16 != 0:
      raise ValueError(
        f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

    if prompt is not None and prompt_embeds is not None:
      raise ValueError(
        f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
        " only forward one of the two.")
    elif negative_prompt is not None and negative_prompt_embeds is not None:
      raise ValueError(
        f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
        " only forward one of the two.")
    elif prompt is None and prompt_embeds is None:
      raise ValueError(
        "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
      )
    elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
      raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    elif negative_prompt is not None and (not isinstance(negative_prompt, str)
                                          and not isinstance(negative_prompt, list)):
      raise ValueError(
        f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")
    if audio is not None and audio_embeds is not None:
      raise ValueError(
        f"Cannot forward both `audio`: {audio} and `audio_embeds`: {audio_embeds}. Please make sure to"
        " only forward one of the two.")
    elif audio is None and audio_embeds is None:
      raise ValueError(
        "Provide either `audio` or `audio_embeds`. Cannot leave both `audio` and `audio_embeds` undefined."
      )
    elif audio is not None and not is_valid_audio(audio):
      raise ValueError(
        f"`audio` has to be a mono waveform (np.ndarray / torch.Tensor / list) but is "
        f"{type(audio)}")

  def prepare_latents(
    self,
    image: PipelineImageInput,
    batch_size: int,
    latent_motion_frames: int,
    num_channels_latents: int = 16,
    height: int = 480,
    width: int = 832,
    num_frames_per_chunk: int = 80,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    init_first_frame: bool = False,
  ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Sample the noisy latents and encode the reference image and the motion conditioning."""
    num_latent_frames = (num_frames_per_chunk + 3 + self.motion_frames
                         ) // self.vae_scale_factor_temporal - latent_motion_frames
    latent_height = height // self.vae_scale_factor_spatial
    latent_width = width // self.vae_scale_factor_spatial

    shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
    if isinstance(generator, list) and len(generator) != batch_size:
      raise ValueError(
        f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
        f" size of {batch_size}. Make sure the batch size matches the length of the generators.")

    if latents is None:
      latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
      latents = latents.to(device=device, dtype=dtype)

    if image is None:
      return latents

    image = image.unsqueeze(2)  # [batch_size, channels, 1, height, width]
    video_condition = image.to(device=device, dtype=self.vae.dtype)

    latents_mean = (torch.tensor(self.vae.config.latents_mean).view(
      1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype))
    latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
      1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)

    if isinstance(generator, list):
      latent_condition = [
        retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax") for _ in generator
      ]
      latent_condition = torch.cat(latent_condition)
    else:
      latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
      latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

    latent_condition = latent_condition.to(dtype)
    latent_condition = (latent_condition - latents_mean) * latents_std

    motion_pixels = torch.zeros([1, 3, self.motion_frames, height, width],
                                dtype=self.vae.dtype,
                                device=device)
    videos_last_pixels = motion_pixels.detach()
    if init_first_frame:
      motion_pixels[:, :, -6:] = video_condition
    motion_latents = retrieve_latents(self.vae.encode(motion_pixels), sample_mode="argmax")
    motion_latents = (motion_latents - latents_mean) * latents_std

    return latents, latent_condition, videos_last_pixels, motion_latents

  @property
  def guidance_scale(self):
    return self._guidance_scale

  @property
  def do_classifier_free_guidance(self):
    return self._guidance_scale > 1

  @property
  def num_timesteps(self):
    return self._num_timesteps

  @property
  def current_timestep(self):
    return self._current_timestep

  @property
  def interrupt(self):
    return self._interrupt

  @torch.no_grad()
  def __call__(
    self,
    image: PipelineImageInput,
    audio: PipelineAudioInput,
    sampling_rate: int,
    prompt: Union[str, List[str]],
    negative_prompt: Union[str, List[str]] = None,
    height: int = 480,
    width: int = 832,
    num_frames_per_chunk: int = 80,
    num_inference_steps: int = 40,
    guidance_scale: float = 4.5,
    num_videos_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    image_embeds: Optional[torch.Tensor] = None,
    audio_embeds: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "np",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    max_sequence_length: int = 512,
    init_first_frame: bool = False,
    sampling_fps: int = 16,
    num_chunks: Optional[int] = None,
  ):
    """Generate a talking video from a reference image and a waveform.

    :param image: reference (identity) image.
    :param audio: mono waveform; `Wav2Vec2FeatureExtractor` input conventions.
    :param sampling_rate: sample rate of `audio` (16 kHz is what the encoder was trained on).
    :param prompt: text prompt.
    :param negative_prompt: negative prompt for CFG.
    :param height: output height, must be divisible by 16.
    :param width: output width, must be divisible by 16.
    :param num_frames_per_chunk: frames generated per chunk, a multiple of 4. The first chunk
      drops 3 warm-up frames, so the visible length is `num_chunks * num_frames_per_chunk - 3`.
    :param num_inference_steps: denoising steps per chunk.
    :param guidance_scale: CFG scale (the unconditional branch zeroes the audio conditioning).
    :param num_videos_per_prompt: number of samples per prompt.
    :param generator: torch generator for reproducible latents.
    :param latents: pre-sampled latents of the first chunk.
    :param prompt_embeds: pre-computed text embeddings.
    :param negative_prompt_embeds: pre-computed negative text embeddings.
    :param image_embeds: unused, kept for API symmetry with the upstream PR.
    :param audio_embeds: pre-computed audio embeddings `[1, layers, dim, buckets]`.
    :param output_type: `"np"`, `"pil"` or `"latent"`.
    :param return_dict: return a `WanPipelineOutput` when true.
    :param attention_kwargs: forwarded to the transformer.
    :param max_sequence_length: umT5 sequence length.
    :param init_first_frame: use the reference image as the first video frame.
    :param sampling_fps: output frame rate.
    :param num_chunks: number of audio chunks to render, `None` renders the whole audio.
    :returns: `WanPipelineOutput` with `frames`, or a tuple when `return_dict=False`.
    """
    self.check_inputs(
      prompt,
      negative_prompt,
      image,
      height,
      width,
      prompt_embeds,
      negative_prompt_embeds,
      image_embeds,
      audio,
      audio_embeds,
    )

    if num_frames_per_chunk % self.vae_scale_factor_temporal != 0:
      num_frames_per_chunk = (num_frames_per_chunk // self.vae_scale_factor_temporal *
                              self.vae_scale_factor_temporal)
      logger.warning(
        f"`num_frames_per_chunk` had to be divisible by {self.vae_scale_factor_temporal}. "
        f"Rounding to the nearest number: {num_frames_per_chunk}")
    num_frames_per_chunk = max(num_frames_per_chunk, 1)

    self._guidance_scale = guidance_scale
    self._current_timestep = None
    self._interrupt = False

    device = self._execution_device

    if prompt is not None and isinstance(prompt, str):
      batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
      batch_size = len(prompt)
    else:
      batch_size = prompt_embeds.shape[0]

    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
      prompt=prompt,
      negative_prompt=negative_prompt,
      do_classifier_free_guidance=self.do_classifier_free_guidance,
      num_videos_per_prompt=num_videos_per_prompt,
      prompt_embeds=prompt_embeds,
      negative_prompt_embeds=negative_prompt_embeds,
      max_sequence_length=max_sequence_length,
      device=device,
    )

    transformer_dtype = self._transformer_dtype()
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
      negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    if audio_embeds is None:
      audio_embeds, num_chunks_audio = self.encode_audio(audio, sampling_rate, num_frames_per_chunk,
                                                         sampling_fps, device)
    else:
      # Pre-computed embeddings: the number of chunks follows from the bucketed frame count.
      num_chunks_audio = max(1, audio_embeds.shape[-1] // num_frames_per_chunk)
    if num_chunks is None or num_chunks > num_chunks_audio:
      num_chunks = num_chunks_audio

    latent_motion_frames = (self.motion_frames + 3) // self.vae_scale_factor_temporal

    num_channels_latents = self.vae.config.z_dim
    if isinstance(image, Image.Image):
      image = self.video_processor.preprocess(
        resize_min_center_crop(image, height, width),
        height=height,
        width=width,
      ).to(device, dtype=torch.float32)
    else:
      image = self.video_processor.preprocess(image, height=height,
                                              width=width).to(device, dtype=torch.float32)

    video_chunks = []
    for r in range(num_chunks):
      if r > 0:
        self._reset_cache_context()
      # The first chunk of a call starts from the reference image, so it keeps all 73 motion
      # frames; a call that continues from a previous clip drops the re-encoded 19 latents. This
      # used to be written onto `self`, which leaked into the next call.
      drop_motion_frames = r == 0 and not init_first_frame
      latents_outputs = self.prepare_latents(
        image if r == 0 else None,
        batch_size * num_videos_per_prompt,
        latent_motion_frames,
        num_channels_latents,
        height,
        width,
        num_frames_per_chunk,
        torch.float32,
        device,
        generator,
        latents if r == 0 else None,
        init_first_frame,
      )

      if r == 0:
        latents, condition, videos_last_pixels, motion_latents = latents_outputs
      else:
        latents = latents_outputs

      left_idx = r * num_frames_per_chunk
      right_idx = r * num_frames_per_chunk + num_frames_per_chunk
      # Pose conditioning is not implemented: the reference branch of the official model feeds
      # zeros here, which the pose embedder turns into its bias term.
      pose_latents = torch.zeros(
        [
          batch_size * num_videos_per_prompt,
          self.transformer.config.pose_dim,
          num_frames_per_chunk // self.vae_scale_factor_temporal,
          height // self.vae_scale_factor_spatial,
          width // self.vae_scale_factor_spatial,
        ],
        dtype=transformer_dtype,
        device=device,
      )
      audio_embeds_input = audio_embeds[..., left_idx:right_idx]
      motion_latents_input = motion_latents.to(transformer_dtype).clone()

      self.scheduler.set_timesteps(num_inference_steps, device=device)
      timesteps = self.scheduler.timesteps

      num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
      self._num_timesteps = len(timesteps)

      with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
          if self.interrupt:
            continue

          self._current_timestep = t

          latent_model_input = latents.to(transformer_dtype)
          condition = condition.to(transformer_dtype)
          timestep = t.expand(latents.shape[0])

          noise_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            motion_latents=motion_latents_input,
            image_latents=condition,
            pose_latents=pose_latents,
            audio_embeds=audio_embeds_input,
            motion_frames=[self.motion_frames, latent_motion_frames],
            drop_motion_frames=drop_motion_frames,
            attention_kwargs=attention_kwargs,
            return_dict=False,
          )[0]

          if self.do_classifier_free_guidance:
            noise_uncond = self.transformer(
              hidden_states=latent_model_input,
              timestep=timestep,
              encoder_hidden_states=negative_prompt_embeds,
              motion_latents=motion_latents_input,
              image_latents=condition,
              pose_latents=pose_latents,
              audio_embeds=0.0 * audio_embeds_input,
              motion_frames=[self.motion_frames, latent_motion_frames],
              drop_motion_frames=drop_motion_frames,
              attention_kwargs=attention_kwargs,
              return_dict=False,
            )[0]
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

          latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

          if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and
                                         (i + 1) % self.scheduler.order == 0):
            progress_bar.update()

      if not drop_motion_frames:
        decode_latents = torch.cat([motion_latents, latents], dim=2)
      else:
        decode_latents = torch.cat([condition, latents], dim=2)

      decode_latents = decode_latents.to(self.vae.dtype)

      latents_mean = (torch.tensor(self.vae.config.latents_mean).view(
        1, self.vae.config.z_dim, 1, 1, 1).to(decode_latents.device, decode_latents.dtype))
      latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
        1, self.vae.config.z_dim, 1, 1, 1).to(decode_latents.device, decode_latents.dtype)
      decode_latents = decode_latents / latents_std + latents_mean
      video = self.vae.decode(decode_latents, return_dict=False)[0]
      video = video[:, :, -(num_frames_per_chunk):]

      if drop_motion_frames:
        video = video[:, :, 3:]

      num_overlap_frames = min(self.motion_frames, video.shape[2])
      videos_last_pixels = torch.cat(
        [videos_last_pixels[:, :, num_overlap_frames:], video[:, :, -num_overlap_frames:]], dim=2)

      motion_latents = retrieve_latents(self.vae.encode(videos_last_pixels), sample_mode="argmax")
      motion_latents = (motion_latents - latents_mean) * latents_std

      video_chunks.append(video)

    video_chunks = torch.cat(video_chunks, dim=2)

    self._current_timestep = None

    if output_type != "latent":
      video = self.video_processor.postprocess_video(video_chunks, output_type=output_type)
    else:
      video = video_chunks

    self.maybe_free_model_hooks()

    if not return_dict:
      return (video, )

    return WanPipelineOutput(frames=video)
