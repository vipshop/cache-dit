"""Vendored MOVA pipeline (text/image to video + audio).

The released repository is a normal diffusers-style tree, but its ``model_index.json`` points the
components at ``mova.diffusion.models.*``, which is a training package cache-dit must not depend on
(it needs ``yunchang``, ``descript-audiotools`` and ``mmengine``). This module therefore owns loading
end to end, in the shape recommended by ``references/custom_models.md`` §5.

Intentional differences from ``mova/diffusion/pipelines/pipeline_mova.py``:

* the denoising loop is unchanged, but the four released components are handed to
  :class:`~.stack.MOVAStack`, which owns the fused block list; the pipeline only exposes
  ``transformer`` so cache-dit sees a single denoising stack;
* ``cfg_mode`` / ``cfg_merge`` are parameters instead of hard-coded locals, and caching runs are
  rejected when they would break the one-conditional-plus-one-unconditional forward count;
* ``generator`` is honoured (upstream accepted a ``seed`` argument and never used it), which is what
  makes two configurations comparable at the same seed;
* ``vae`` / ``text_encoder`` / ``scheduler`` overrides and the usual ``from_pretrained`` kwargs are
  accepted, so cache-dit's CLI can inject pre-built components.

Not supported: ``save_pretrained`` round-trip (moving the blocks changes the state-dict layout; see
:mod:`~.stack`).
"""

import html
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch
from diffusers import AutoencoderKLWan, DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from PIL import Image
from transformers.models import T5TokenizerFast, UMT5EncoderModel

from .bridge import DualTowerConditionalBridge
from .dac_vae import DAC_KEY_PREFIXES_TO_DROP, MovaAudioDAC
from .scheduler import FlowMatchPairScheduler
from .stack import MOVAStack
from .wan_video_dit import WanAudioModel, WanModel

__all__ = ["MOVAPipeline", "MOVAOutput"]

_PROMPT_MAX_LENGTH = 512


def _prompt_clean(text: str) -> str:
  """Normalize a prompt the way the released pipeline does.

  `ftfy` is imported here, not at module scope: it only ships in the `[dev]` extra, and the MOVAPipeline
  import chain is reached by `--cache` on any installation, so a top-level import would break a plain
  `pip install cache_dit` for every user.

  :param text: Raw prompt.
  :returns: Cleaned prompt.
  """

  import ftfy

  text = ftfy.fix_text(text)
  text = html.unescape(html.unescape(text))
  return re.sub(r"\s+", " ", text).strip()


def _retrieve_latents(encoder_output, generator=None, sample_mode: str = "sample") -> torch.Tensor:
  """Read latents out of a VAE encode output.

  :param encoder_output: Output of ``video_vae.encode``.
  :param generator: Optional generator for the sampling path.
  :param sample_mode: ``"sample"`` or ``"argmax"``.
  :returns: The latents.
  """

  if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
    return encoder_output.latent_dist.sample(generator)
  if hasattr(encoder_output, "latent_dist"):
    return encoder_output.latent_dist.mode()
  if hasattr(encoder_output, "latents"):
    return encoder_output.latents
  raise AttributeError("Could not access latents of the provided encoder output.")


def _resolve_dtypes(torch_dtype: Optional[Union[torch.dtype, dict]]) -> dict:
  """Resolve the per-component dtypes.

  The audio VAE always runs in float32: its decode path is a 1-D convolutional stack and the released
  pipeline decodes it under ``autocast(float32)``.

  :param torch_dtype: A dtype, a per-component mapping, or None.
  :returns: Mapping with a ``default`` key.
  """

  defaults = {
    "default": torch.bfloat16,
    "video_vae": torch.bfloat16,
    "audio_vae": torch.float32,
    "text_encoder": torch.bfloat16,
  }
  if torch_dtype is None:
    return defaults
  if isinstance(torch_dtype, dict):
    resolved = dict(defaults)
    resolved.update(torch_dtype)
    return resolved
  # The audio decoder keeps its float32 default: it is a 1-D convolutional stack that the released
  # pipeline always decodes under autocast(float32), and dtype-folding it loses the tail.
  return {
    "default": torch_dtype,
    "video_vae": torch_dtype,
    "audio_vae": defaults["audio_vae"],
    "text_encoder": torch_dtype,
  }


@dataclass
class MOVAOutput:
  """Minimal output object understood by cache-dit's writers.

  :param frames: Batch of frame lists, one list per request.
  :param audio: Waveform, ``[B, 1, samples]``.
  :param sampling_rate: Sampling rate of ``audio``.
  """

  frames: List[List[Image.Image]]
  audio: torch.Tensor
  sampling_rate: int


class MOVAPipeline(DiffusionPipeline):
  """MOVA text/image-to-video-and-audio pipeline.

  :param transformer: Fused denoising stack.
  :param video_vae: Wan video VAE (diffusers ``AutoencoderKLWan``).
  :param audio_vae: DAC audio decoder.
  :param text_encoder: UMT5 encoder.
  :param tokenizer: T5 tokenizer.
  :param scheduler: Paired flow-matching scheduler.
  :param audio_vae_type: Audio codec family; only ``"dac"`` is supported by the 360p checkpoint.
  :param boundary_ratio: Timestep ratio at which the low-noise video expert takes over.
  """

  model_cpu_offload_seq = "text_encoder->transformer->video_vae->audio_vae"

  def __init__(
    self,
    transformer: MOVAStack,
    video_vae: AutoencoderKLWan,
    audio_vae: MovaAudioDAC,
    text_encoder: UMT5EncoderModel,
    tokenizer: T5TokenizerFast,
    scheduler: FlowMatchPairScheduler,
    audio_vae_type: str = "dac",
    boundary_ratio: float = 0.9,
  ):
    super().__init__()
    self.register_modules(
      transformer=transformer,
      video_vae=video_vae,
      audio_vae=audio_vae,
      text_encoder=text_encoder,
      tokenizer=tokenizer,
      scheduler=scheduler,
    )
    self.register_to_config(audio_vae_type=audio_vae_type, boundary_ratio=boundary_ratio)
    self.audio_vae_type = audio_vae_type
    self.boundary_ratio = boundary_ratio

    self.vae_scale_factor_spatial = self.video_vae.config.scale_factor_spatial
    self.vae_scale_factor_temporal = self.video_vae.config.scale_factor_temporal
    self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
    self.audio_vae_scale_factor = int(self.audio_vae.hop_length)
    self.audio_sample_rate = int(self.audio_vae.sample_rate)

  @property
  def vae(self) -> AutoencoderKLWan:
    """The **video** VAE, under the diffusers component name.

    Everything outside the pipeline (`--parallel-vae`, `--compile-vae`, `--vae-tiling`, and
    ``from_pretrained(vae=...)`` above) addresses the VAE as ``pipe.vae``. Without this alias those
    flags log "no VAE found" and silently do nothing on MOVA.

    The audio decoder is deliberately not exposed: :class:`~.dac_vae.MovaAudioDAC` has no
    registration in `cache_dit.distributed.autoencoders.planners`, so it can only decode on a single
    rank (or be replicated) — aliasing it here would advertise parallelism that does not exist.

    Note that `--parallel-vae` still fails on MOVA, but it now fails *inside* the shared
    `AutoencoderKLWanDataParallelismPlanner`: its patched `tiled_decode` relays p2p messages in an
    order that leaves the second rank reading a stale buffer (`recv_shape` returns garbage integers).
    That is a Wan-VAE planner bug, not a MOVA wiring gap, so it is reported rather than worked around.

    :returns: The video VAE.
    """

    return self.video_vae

  @classmethod
  def from_pretrained(cls,
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
                      audio_vae=None,
                      local_files_only=False,
                      **kwargs) -> "MOVAPipeline":
    """Load MOVA from a checkpoint root holding the released component subdirectories.

    :param pretrained_model_name_or_path: Checkpoint root.
    :param torch_dtype: Dtype, or a per-component mapping such as
      ``{"default": torch.bfloat16, "audio_vae": torch.float32}``; ``dtype`` is an alias.
    :param quantization_config: Ignored; quantize through cache-dit's own flags.
    :param device_map: Rejected; call ``.to(device)`` (or use cache-dit's offloading).
    :param trust_remote_code: Ignored; all components are local.
    :param scheduler: Pre-built scheduler, replaces the checkpoint's.
    :param transformer: Pre-built fused stack, replaces the checkpoint's DiTs.
    :param vae: Pre-built video VAE.
    :param text_encoder: Pre-built text encoder.
    :param tokenizer: Pre-built tokenizer.
    :param audio_vae: Pre-built audio decoder.
    :param local_files_only: Whether to forbid network access while resolving files.
    :param kwargs: Additional diffusers arguments, ignored.
    :returns: The assembled pipeline.
    """

    if args:
      raise ValueError("MOVAPipeline.from_pretrained takes keyword arguments only.")
    dtypes = _resolve_dtypes(torch_dtype if torch_dtype is not None else dtype)
    if quantization_config is not None:
      import logging

      logging.getLogger(__name__).warning(
        "Ignoring quantization_config: quantize MOVA through cache-dit's `--quantize-type` flags so "
        "the fused blocks are discovered.")
    if device_map is not None:
      raise NotImplementedError(
        "MOVA does not support `--device-map-balance`: the CLI skips `pipe.to(device)` for that flag, "
        "which would leave the pipeline on CPU. Drop the flag or use cache-dit offloading.")
    if trust_remote_code:
      import logging

      logging.getLogger(__name__).debug("trust_remote_code is not needed: MOVA code is built in.")
    if kwargs:
      import logging

      logging.getLogger(__name__).warning(
        f"Ignoring unsupported from_pretrained kwargs: {sorted(kwargs)}.")

    root = str(pretrained_model_name_or_path)
    local_only = dict(local_files_only=local_files_only)
    import os

    if transformer is None:
      video_dit = WanModel.from_pretrained(os.path.join(root, "video_dit"),
                                           torch_dtype=dtypes["default"],
                                           **local_only)
      video_dit_2_path = os.path.join(root, "video_dit_2")
      video_dit_2 = (WanModel.from_pretrained(
        video_dit_2_path, torch_dtype=dtypes["default"], **local_only)
                     if os.path.isdir(video_dit_2_path) else None)
      audio_dit = WanAudioModel.from_pretrained(os.path.join(root, "audio_dit"),
                                                torch_dtype=dtypes["default"],
                                                **local_only)
      bridge = DualTowerConditionalBridge.from_pretrained(os.path.join(root, "dual_tower_bridge"),
                                                          torch_dtype=dtypes["default"],
                                                          **local_only)
      transformer = MOVAStack(
        video_dit=video_dit,
        video_dit_2=video_dit_2,
        audio_dit=audio_dit,
        dual_tower_bridge=bridge,
      )

    if vae is None:
      vae = AutoencoderKLWan.from_pretrained(os.path.join(root, "video_vae"),
                                             torch_dtype=dtypes["video_vae"],
                                             **local_only)
    if audio_vae is None:
      audio_vae = _load_audio_vae(os.path.join(root, "audio_vae"), dtypes["audio_vae"], local_only)
    if text_encoder is None:
      text_encoder = UMT5EncoderModel.from_pretrained(os.path.join(root, "text_encoder"),
                                                      torch_dtype=dtypes["text_encoder"],
                                                      **local_only)
    if tokenizer is None:
      tokenizer = T5TokenizerFast.from_pretrained(os.path.join(root, "tokenizer"), **local_only)
    if scheduler is None:
      scheduler = FlowMatchPairScheduler.from_pretrained(os.path.join(root, "scheduler"),
                                                         **local_only)

    return cls(
      transformer=transformer,
      video_vae=vae,
      audio_vae=audio_vae,
      text_encoder=text_encoder,
      tokenizer=tokenizer,
      scheduler=scheduler,
    )

  def normalize_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
    """Normalize latents with the VAE's statistics (diffusers Wan convention).

    :param latents: Raw VAE latents.
    :returns: Normalized latents.
    """

    mean = torch.tensor(self.video_vae.config.latents_mean,
                        device=latents.device,
                        dtype=latents.dtype).view(1, self.video_vae.config.z_dim, 1, 1, 1)
    inv_std = (1.0 / torch.tensor(
      self.video_vae.config.latents_std, device=latents.device, dtype=latents.dtype)).view(
        1, self.video_vae.config.z_dim, 1, 1, 1)
    return (latents - mean) * inv_std

  def denormalize_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
    """Invert :meth:`normalize_video_latents`.

    :param latents: Normalized latents.
    :returns: Raw VAE latents.
    """

    mean = torch.tensor(self.video_vae.config.latents_mean,
                        device=latents.device,
                        dtype=latents.dtype).view(1, self.video_vae.config.z_dim, 1, 1, 1)
    std = torch.tensor(self.video_vae.config.latents_std,
                       device=latents.device,
                       dtype=latents.dtype).view(1, self.video_vae.config.z_dim, 1, 1, 1)
    return latents * std + mean

  def _check_cache_compatible(self, cfg_scale: float, cfg_merge: bool) -> None:
    """Reject the modes that break cache-dit's two-forward-per-step bookkeeping.

    A cached run pairs one conditional and one unconditional forward (`enable_separate_cfg=True`);
    batching the branches into a single forward (`cfg_merge`) or disabling guidance
    (`guidance_scale == 1`) changes that count and would silently halve the step statistics.

    :param cfg_scale: Guidance scale of this call.
    :param cfg_merge: Whether the branches are batched into one forward.
    :raises ValueError: When caching is enabled and the forward count differs from two.
    """

    if not getattr(self.transformer, "_is_cached", False):
      return
    if cfg_merge:
      raise ValueError(
        "cache-dit caching assumes one conditional and one unconditional forward per "
        "step; `cfg_merge=True` batches them into a single forward.")
    if cfg_scale == 1.0:
      raise ValueError(
        "cache-dit caching assumes one conditional and one unconditional forward per "
        "step; use a guidance scale above 1 (or disable caching).")

  def check_inputs(self, height: int, width: int, num_frames: int) -> None:
    """Validate the geometry against the VAE strides.

    :param height: Request height.
    :param width: Request width.
    :param num_frames: Request frame count.
    """

    factor = self.vae_scale_factor_spatial * 2
    if height % factor != 0 or width % factor != 0:
      raise ValueError(f"`height` and `width` must be divisible by {factor}, got {height}x{width}.")
    if num_frames % self.vae_scale_factor_temporal != 1:
      raise ValueError(f"`num_frames - 1` must be divisible by {self.vae_scale_factor_temporal}, "
                       f"got {num_frames - 1}.")

  def prepare_latents(self,
                      image,
                      batch_size: int,
                      num_channels_latents: int,
                      height: int,
                      width: int,
                      num_frames: int,
                      dtype,
                      device,
                      generator,
                      latents=None):
    """Build the video noise and the conditioning block (first-frame latent + mask).

    :param image: Preprocessed first frame, ``[B, 3, H, W]``.
    :param batch_size: Request batch size.
    :param num_channels_latents: VAE latent channels.
    :param height: Request height.
    :param width: Request width.
    :param num_frames: Request frame count.
    :param dtype: Latent dtype.
    :param device: Compute device.
    :param generator: Optional generator for the initial noise.
    :param latents: Optional pre-built noise.
    :returns: ``(latents, condition)``.
    """

    num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
    latent_height = height // self.vae_scale_factor_spatial
    latent_width = width // self.vae_scale_factor_spatial
    shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

    if latents is None:
      latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
      latents = latents.to(device=device, dtype=dtype)

    image = image.unsqueeze(2)
    video_condition = torch.cat(
      [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)],
      dim=2)
    video_condition = video_condition.to(device=device, dtype=self.video_vae.dtype)
    latent_condition = _retrieve_latents(self.video_vae.encode(video_condition),
                                         sample_mode="argmax")
    latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1).to(dtype)
    latent_condition = self.normalize_video_latents(latent_condition)

    mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)
    mask_lat_size[:, :, list(range(1, num_frames))] = 0
    first_frame_mask = mask_lat_size[:, :, 0:1]
    first_frame_mask = torch.repeat_interleave(first_frame_mask,
                                               dim=2,
                                               repeats=self.vae_scale_factor_temporal)
    mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
    mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal,
                                       latent_height, latent_width)
    mask_lat_size = mask_lat_size.transpose(1, 2).to(latent_condition.device)

    return latents, torch.concat([mask_lat_size, latent_condition], dim=1)

  def prepare_audio_latents(self,
                            batch_size: int,
                            num_channels: int,
                            num_samples: int,
                            dtype,
                            device,
                            generator,
                            latents=None):
    """Build the audio noise.

    :param batch_size: Request batch size.
    :param num_channels: Audio latent channels.
    :param num_samples: Requested waveform length in samples.
    :param dtype: Latent dtype.
    :param device: Compute device.
    :param generator: Optional generator for the initial noise.
    :param latents: Optional pre-built noise.
    :returns: ``[B, num_channels, T_a]`` noise.
    """

    latent_t = (num_samples - 1) // self.audio_vae_scale_factor + 1
    shape = (batch_size, num_channels, latent_t)
    if latents is None:
      return randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    return latents.to(device=device, dtype=dtype)

  def _get_t5_prompt_embeds(self,
                            prompt: Union[str, List[str]],
                            num_videos_per_prompt: int = 1,
                            device: Optional[torch.device] = None,
                            dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Encode a prompt with UMT5, right-padded to 512 tokens.

    :param prompt: Prompt or list of prompts.
    :param num_videos_per_prompt: Number of samples per prompt.
    :param device: Device of the encoder call.
    :param dtype: Dtype of the returned embedding.
    :returns: ``[B * num_videos_per_prompt, 512, d_model]``.
    """

    device = device or self._execution_device
    dtype = dtype or self.text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [_prompt_clean(item) for item in prompt]
    batch_size = len(prompt)

    text_inputs = self.tokenizer(
      prompt,
      padding="max_length",
      max_length=_PROMPT_MAX_LENGTH,
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
      torch.cat([u, u.new_zeros(_PROMPT_MAX_LENGTH - u.size(0), u.size(1))]) for u in prompt_embeds
    ],
                                dim=0)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    return prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

  @torch.no_grad()
  def __call__(self,
               prompt: Union[str, List[str]],
               image: Any,
               negative_prompt: Union[str, List[str]] = "",
               height: int = 352,
               width: int = 640,
               num_frames: int = 193,
               video_fps: float = 24.0,
               num_inference_steps: int = 50,
               cfg_scale: float = 5.0,
               guidance_scale: Optional[float] = None,
               cfg_mode: str = "text",
               cfg_merge: bool = False,
               generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
               latents: Optional[torch.Tensor] = None,
               audio_latents: Optional[torch.Tensor] = None,
               output_type: str = "pil") -> MOVAOutput:
    """Generate a video with a synchronized soundtrack.

    :param prompt: Text prompt.
    :param image: First frame (PIL image or a preprocessed tensor).
    :param negative_prompt: Negative prompt.
    :param height: Request height; it must divide by 16, which is why the default is 352 and not the
      360 that the checkpoint name suggests (640x360 would be rejected by :meth:`check_inputs`).
    :param width: Request width. The defaults are the released script's 720x1280 downscaled to a
      360p-class geometry (352x640); the released script itself defaults to 720x1280.
    :param num_frames: Request frame count (``num_frames - 1`` must divide by 4).
    :param video_fps: Request frame rate; also sets the audio duration.
    :param num_inference_steps: Denoising steps.
    :param cfg_scale: Guidance scale; must be above 1 when caching is enabled.
    :param guidance_scale: Alias of ``cfg_scale`` (the name the CLI and diffusers users pass).
    :param cfg_mode: ``"text"`` is the only mode the 360p checkpoint supports.
    :param cfg_merge: Whether the conditional and unconditional branches are batched into one forward.
    :param generator: Optional generator; pass one for reproducible runs.
    :param latents: Optional pre-built video noise (for bit-exact A/B runs).
    :param audio_latents: Optional pre-built audio noise.
    :param output_type: ``"pil"`` for frames, ``"pt"`` for the raw decode output.
    :returns: :class:`MOVAOutput` with the frame batch, the waveform and its rate.
    """

    self.check_inputs(height, width, num_frames)
    if guidance_scale is not None:
      cfg_scale = guidance_scale
    self._check_cache_compatible(cfg_scale, cfg_merge)
    if cfg_mode != "text":
      raise NotImplementedError(f"Unsupported `cfg_mode`: {cfg_mode}.")

    device = self._execution_device
    if isinstance(generator, list):
      generator = generator[0]

    audio_num_samples = int(self.audio_sample_rate * num_frames / video_fps)

    self.scheduler.set_timesteps(num_inference_steps, device=device)
    paired_timesteps = self.scheduler.get_pairs()

    num_channels_latents = self.video_vae.config.z_dim
    if isinstance(image, torch.Tensor):
      image_tensor = image
    else:
      image_tensor = self.video_processor.preprocess(image, height=height, width=width)
    image_tensor = image_tensor.to(device, dtype=torch.float32)
    latents, condition = self.prepare_latents(
      image_tensor,
      1,
      num_channels_latents,
      height,
      width,
      num_frames,
      torch.float32,
      device,
      generator=generator,
      latents=latents,
    )
    audio_latents = self.prepare_audio_latents(
      1,
      self.audio_vae.latent_dim,
      audio_num_samples,
      torch.float32,
      device,
      generator=generator,
      latents=audio_latents,
    )

    prompt_embeds = self._get_t5_prompt_embeds(prompt)
    negative_prompt_embeds = self._get_t5_prompt_embeds(negative_prompt)

    total_steps = paired_timesteps.shape[0]
    boundary_timestep = self.boundary_ratio * self.scheduler.config.num_train_timesteps
    switched = False

    for idx_step in range(total_steps):
      timestep, audio_timestep = paired_timesteps[idx_step]
      use_video_dit_2 = switched or timestep.item() < boundary_timestep
      switched = use_video_dit_2

      latent_model_input = torch.cat([latents, condition], dim=1)
      timestep = timestep.unsqueeze(0).to(device=device, dtype=torch.float32)
      audio_timestep = audio_timestep.unsqueeze(0).to(device=device, dtype=torch.float32)

      visual_posi, audio_posi = self.transformer(
        latent_model_input,
        audio_latents,
        prompt_embeds,
        timestep,
        audio_timestep=audio_timestep,
        video_fps=video_fps,
        use_video_dit_2=use_video_dit_2,
      )
      if cfg_scale == 1.0:
        visual_noise_pred = visual_posi.float()
        audio_noise_pred = audio_posi.float()
      else:
        visual_nega, audio_nega = self.transformer(
          latent_model_input,
          audio_latents,
          negative_prompt_embeds,
          timestep,
          audio_timestep=audio_timestep,
          video_fps=video_fps,
          use_video_dit_2=use_video_dit_2,
        )
        visual_noise_pred = visual_nega.float() + cfg_scale * (visual_posi.float() -
                                                               visual_nega.float())
        audio_noise_pred = audio_nega.float() + cfg_scale * (audio_posi.float() -
                                                             audio_nega.float())

      next_timestep = paired_timesteps[idx_step + 1, 0] if idx_step + 1 < total_steps else None
      next_audio_timestep = paired_timesteps[idx_step + 1,
                                             1] if idx_step + 1 < total_steps else None
      latents = self.scheduler.step_from_to(visual_noise_pred, timestep, next_timestep, latents)
      audio_latents = self.scheduler.step_from_to(audio_noise_pred, audio_timestep,
                                                  next_audio_timestep, audio_latents)

    video_latents = self.denormalize_video_latents(latents)
    with torch.autocast(device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
      video = self.video_vae.decode(video_latents).sample
    with torch.autocast(device_type=device.type, dtype=torch.float32,
                        enabled=device.type == "cuda"):
      audio = self.audio_vae.decode(audio_latents.float())

    if output_type == "pt":
      return MOVAOutput(frames=video, audio=audio.float(), sampling_rate=self.audio_sample_rate)
    frames = self.video_processor.postprocess_video(video, output_type="pil")
    return MOVAOutput(frames=frames, audio=audio.float(), sampling_rate=self.audio_sample_rate)


def _load_audio_vae(path: str, torch_dtype: torch.dtype, local_only: dict) -> MovaAudioDAC:
  """Load the audio VAE, dropping the encode-path weights the decode-only vendor does not build.

  :param path: ``audio_vae/`` directory.
  :param torch_dtype: Dtype of the decoder.
  :param local_only: ``local_files_only`` passthrough.
  :returns: The loaded decoder.
  :raises ValueError: When the checkpoint carries keys outside the declared drop set.
  """

  import os

  from safetensors.torch import load_file

  del local_only
  config = MovaAudioDAC.load_config(path)
  model = MovaAudioDAC.from_config(config)
  state = load_file(os.path.join(path, "diffusion_pytorch_model.safetensors"))

  kept = {}
  dropped = set()
  for key, value in state.items():
    if key.startswith(DAC_KEY_PREFIXES_TO_DROP):
      dropped.add(key.split(".")[0])
      continue
    kept[key] = value
  if dropped != {"encoder", "quant_conv"}:
    raise ValueError(f"Unexpected audio-VAE key layout under {path}: dropped={sorted(dropped)}.")

  missing, unexpected = model.load_state_dict(kept, strict=False)
  if missing or unexpected:
    raise ValueError(f"Audio-VAE decode path mismatch: missing={list(missing)[:5]}, "
                     f"unexpected={list(unexpected)[:5]}.")
  return model.to(dtype=torch_dtype)


MOVA = MOVAPipeline
