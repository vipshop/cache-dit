"""Diffusers-style Ming-Image 0.1 Design and Design-Layer pipeline."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.pipelines.z_image.pipeline_z_image import (
  ZImagePipeline,
  calculate_shift,
  get_default_z_image_sigmas,
  retrieve_timesteps,
)
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

from ....logger import init_logger
from .conditioning import MingImageConditioning
from .transformer import MingImageTransformer2DModel

if TYPE_CHECKING:
  from .mllm import MingImageMLLM

logger = init_logger(__name__)


class MingImagePipeline(ZImagePipeline):
  """Run Ming-Image Design and Design-Layer without a serving runtime."""

  model_cpu_offload_seq = "text_encoder->conditioning->transformer->vae"
  _optional_components = ["text_encoder", "tokenizer", "conditioning"]
  _callback_tensor_inputs = ["latents", "prompt_embeds"]

  def __init__(
    self,
    transformer: MingImageTransformer2DModel,
    scheduler: FlowMatchEulerDiscreteScheduler,
    vae: AutoencoderKLQwenImage,
    conditioning: MingImageConditioning | None = None,
    text_encoder: MingImageMLLM | None = None,
    tokenizer=None,
  ) -> None:
    """Create a shared Design or Design-Layer pipeline.

    :param transformer: Ming DiT whose config selects the output variant.
    :param scheduler: Flow-matching Euler scheduler.
    :param vae: Four-channel Qwen-Image VAE.
    :param conditioning: Query connector and direct projection, optional for preprojected inputs.
    :param text_encoder: Self-contained Bailing-MoE MLLM, optional for precomputed inputs.
    :param tokenizer: Optional tokenizer retained for pipeline component compatibility.
    """
    DiffusionPipeline.__init__(self)
    self.register_modules(
      scheduler=scheduler,
      vae=vae,
      text_encoder=text_encoder,
      tokenizer=tokenizer,
      transformer=transformer,
      conditioning=conditioning,
    )
    self.is_layer_decomposition = bool(transformer.config.multi_frame_output)
    self.default_num_inference_steps = 12
    self.default_guidance_scale = 2.0 if self.is_layer_decomposition else 1.0
    self.vae_scale_factor = 2 ** len(vae.config.temperal_downsample)
    self.image_processor = VaeImageProcessor(
      vae_scale_factor=self.vae_scale_factor * 2,
      do_convert_rgb=False,
      resample="bilinear",
    )
    self.stage_durations: dict[str, float] = {}

  @property
  def device(self) -> torch.device:
    """Return the denoising device, independent of custom conditioning modules."""
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
    tokenizer=None,
    conditioning=None,
    controlnet=None,
    load_mllm: bool = True,
    local_files_only: bool = False,
    **kwargs,
  ) -> "MingImagePipeline":
    """Assemble Ming components directly from the checkpoint directories.

    :param pretrained_model_name_or_path: Checkpoint root; no `model_index.json` is required.
    :param torch_dtype: DiT/condition dtype, or a mapping with `default` and `vae` entries.
    :param dtype: Alias for `torch_dtype`.
    :param quantization_config: Ignored; use cache-dit's DiT quantization options.
    :param device_map: Unsupported; use `.to()` or cache-dit offload instead.
    :param trust_remote_code: Ignored because all model code is built in.
    :param load_mllm: Load the 17B MLLM. Disable only for precomputed-condition workflows.
    :param local_files_only: Forbid network access for every component.
    :returns: The assembled Ming-Image pipeline.
    """
    if args:
      raise ValueError("MingImagePipeline.from_pretrained takes keyword arguments only.")
    if controlnet is not None:
      raise ValueError("Ming-Image does not support ControlNet.")
    if device_map is not None:
      raise NotImplementedError(
        "Ming-Image does not support device_map; use `.to(device)` or cache-dit offloading.")
    if quantization_config is not None:
      logger.warning("Ignoring quantization_config; use cache-dit's DiT quantization options.")
    if trust_remote_code:
      logger.debug("trust_remote_code is unnecessary for the built-in Ming-Image implementation.")
    if kwargs:
      logger.warning(f"Ignoring unsupported from_pretrained kwargs: {sorted(kwargs)}.")

    requested_dtype = torch_dtype if torch_dtype is not None else dtype
    if isinstance(requested_dtype, dict):
      inference_dtype = requested_dtype.get("default", torch.bfloat16)
      vae_dtype = requested_dtype.get("vae", inference_dtype)
    else:
      inference_dtype = requested_dtype or torch.bfloat16
      vae_dtype = inference_dtype

    root = Path(pretrained_model_name_or_path)
    common = {"local_files_only": local_files_only}
    if transformer is None:
      transformer = MingImageTransformer2DModel.from_pretrained(root / "transformer",
                                                                torch_dtype=inference_dtype,
                                                                **common)
    if scheduler is None:
      scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(root / "scheduler", **common)
    if vae is None:
      vae = AutoencoderKLQwenImage.from_pretrained(root / "vae", torch_dtype=vae_dtype, **common)
    if int(vae.config.input_channels) != 4:
      raise ValueError(
        f"Ming-Image requires an RGBA VAE, got input_channels={vae.config.input_channels}.")
    if conditioning is None:
      conditioning = MingImageConditioning.from_pretrained(root,
                                                           torch_dtype=inference_dtype,
                                                           local_files_only=local_files_only)
    if text_encoder is None and load_mllm:
      from .mllm import MingImageMLLM

      text_encoder = MingImageMLLM.from_pretrained(root / "mllm",
                                                   torch_dtype=inference_dtype,
                                                   local_files_only=local_files_only)

    for module in (transformer, vae, conditioning, text_encoder):
      if module is not None:
        module.eval()
    return cls(
      transformer=transformer,
      scheduler=scheduler,
      vae=vae,
      conditioning=conditioning,
      text_encoder=text_encoder,
      tokenizer=tokenizer,
    )

  @property
  def do_classifier_free_guidance(self) -> bool:
    return self._guidance_scale > 1.0

  def _reference_output_size(self, reference, height: int, width: int) -> tuple[int, int]:
    """Preserve reference aspect ratio within the requested output area."""
    if reference is None:
      return height, width
    if isinstance(reference, (list, tuple)):
      if len(reference) != 1:
        raise ValueError("Ming-Image accepts exactly one reference image.")
      reference = reference[0]
    if isinstance(reference, Image.Image):
      reference_width, reference_height = reference.size
    elif hasattr(reference, "shape"):
      reference_height, reference_width = reference.shape[-2:]
    else:
      return height, width
    if reference_height <= 0 or reference_width <= 0:
      raise ValueError("Ming-Image reference dimensions must be positive.")
    area = height * width
    aspect = reference_width / reference_height
    output_width = max(16, round(math.sqrt(area * aspect) / 16) * 16)
    output_height = max(16, round(math.sqrt(area / aspect) / 16) * 16)
    return output_height, output_width

  def _encode_reference(self, reference, height: int, width: int) -> torch.Tensor | None:
    if reference is None:
      return None
    if isinstance(reference, (list, tuple)):
      if len(reference) != 1:
        raise ValueError("Ming-Image accepts exactly one reference image.")
      reference = reference[0]
    if isinstance(reference, Image.Image):
      reference = reference.convert("RGBA")
    if not isinstance(reference, torch.Tensor):
      reference = self.image_processor.preprocess(reference, height=height, width=width)
    if reference.ndim != 4 or reference.shape[0] != 1 or reference.shape[1] != 4:
      raise ValueError("Ming-Image reference must be one RGBA image or a [1, 4, H, W] tensor.")
    reference = reference.to(device=self._execution_device, dtype=self.vae.dtype).unsqueeze(2)
    latent = self.vae.encode(reference).latent_dist.mode()
    return (latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor

  def _decode_latent_frames(self, latents: torch.Tensor) -> torch.Tensor:
    batch, channels, frames, height, width = latents.shape
    flat = latents.permute(2, 0, 1, 3, 4).reshape(frames * batch, channels, height, width)
    flat = flat.to(self.vae.dtype).unsqueeze(2)
    flat = (flat / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    images = self.vae.decode(flat, return_dict=False)[0]
    if images.ndim == 5:
      if images.shape[2] != 1:
        raise ValueError("Ming-Image VAE returned multiple frames for a single latent frame.")
      images = images.squeeze(2)
    return images

  @torch.no_grad()
  def __call__(
    self,
    prompt: str | None = None,
    *,
    image=None,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 12,
    sigmas: list[float] | None = None,
    guidance_scale: float | None = None,
    negative_prompt: str | None = None,
    num_layers: int = 1,
    num_images_per_prompt: int = 1,
    generator=None,
    latents: torch.Tensor | None = None,
    prompt_embeds: list[torch.Tensor] | None = None,
    direct_condition: torch.Tensor | None = None,
    query_hidden_states: torch.Tensor | None = None,
    direct_hidden_states: torch.Tensor | None = None,
    output_type: str = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: dict[str, Any] | None = None,
    callback_on_step_end=None,
    callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    **kwargs,
  ) -> ImagePipelineOutput | tuple:
    """Generate one RGBA design or a composite followed by its RGBA layers.

    :param prompt: Text prompt; required unless raw or projected conditions are supplied.
    :param image: Optional single reference image; required by Design-Layer.
    :param num_layers: Number of output layers for Design-Layer; Design requires 1.
    :param prompt_embeds: Pre-projected caption features, for tests and offline conditioning.
    :param direct_condition: Pre-projected direct-VLM condition matching `prompt_embeds`.
    :param query_hidden_states: Raw MLLM query features, bypassing MLLM inference.
    :param direct_hidden_states: Raw MLLM direct features, bypassing MLLM inference.
    :param output_type: `pil`, `np`, `pt`, or `latent`.
    :returns: RGBA images; Design-Layer returns the composite followed by all layers.
    """
    if kwargs:
      raise TypeError(f"Unexpected Ming-Image call arguments: {sorted(kwargs)}")
    if negative_prompt:
      raise ValueError("Ming-Image uses a zero negative condition; negative_prompt must be empty.")
    if num_images_per_prompt != 1:
      raise ValueError("Ming-Image supports num_images_per_prompt=1 only.")
    if not isinstance(num_layers, int) or num_layers < 1:
      raise ValueError("num_layers must be a positive integer.")
    if self.is_layer_decomposition:
      if image is None:
        raise ValueError("Ming-Image Design-Layer requires a reference image.")
      frames = num_layers + 1
    else:
      if num_layers != 1:
        raise ValueError("Ming-Image Design supports exactly one output frame.")
      frames = 1
    if height <= 0 or width <= 0 or height % 16 or width % 16:
      raise ValueError("height and width must be positive multiples of 16.")
    if output_type not in {"pil", "np", "pt", "latent"}:
      raise ValueError("output_type must be pil, np, pt, or latent.")
    height, width = self._reference_output_size(image, height, width)

    self._guidance_scale = self.default_guidance_scale if guidance_scale is None else float(
      guidance_scale)
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False
    device = self._execution_device

    start = time.perf_counter()
    if (query_hidden_states is None) != (direct_hidden_states is None):
      raise ValueError("query_hidden_states and direct_hidden_states must be supplied together.")
    if prompt_embeds is not None and query_hidden_states is not None:
      raise ValueError("Choose either preprojected conditions or raw MLLM features, not both.")
    if prompt_embeds is None:
      if query_hidden_states is None:
        if prompt is None or self.text_encoder is None:
          raise ValueError(
            "Provide prompt with a loaded MLLM, or both raw/preprojected conditions.")
        query_hidden_states, direct_hidden_states = self.text_encoder.encode(prompt, image=image)
      if self.conditioning is None:
        raise ValueError("Ming conditioning is required for raw MLLM features.")
      query_hidden_states = query_hidden_states.to(device=device)
      direct_hidden_states = direct_hidden_states.to(device=device)
      prompt_embeds, direct_condition = self.conditioning(query_hidden_states, direct_hidden_states)
    elif direct_condition is None:
      raise ValueError("direct_condition is required together with prompt_embeds.")
    if not isinstance(prompt_embeds, list) or len(prompt_embeds) != 1:
      raise ValueError("Ming-Image currently supports one prompt and one condition set.")
    if direct_condition.ndim == 2:
      direct_condition = direct_condition.unsqueeze(0)
    if direct_condition.ndim != 3 or direct_condition.shape[0] != 1:
      raise ValueError("direct_condition must have shape [1, S, C].")
    prompt_embeds = [item.to(device=device) for item in prompt_embeds]
    direct_condition = direct_condition.to(device=device, dtype=self.transformer.dtype)
    reference_latent = self._encode_reference(image, height, width)
    self.stage_durations["conditioning"] = time.perf_counter() - start

    latent_shape = (
      1,
      self.transformer.in_channels,
      frames,
      height // self.vae_scale_factor,
      width // self.vae_scale_factor,
    )
    if latents is None:
      latents = randn_tensor(latent_shape, generator=generator, device=device, dtype=torch.float32)
    else:
      if latents.ndim == 4:
        latents = latents.unsqueeze(2)
      if tuple(latents.shape) != latent_shape:
        raise ValueError(f"latents must have shape {latent_shape}, got {tuple(latents.shape)}.")
      latents = latents.to(device=device, dtype=torch.float32)

    tokens_per_frame = (height // 16) * (width // 16)
    mu = calculate_shift(
      tokens_per_frame,
      self.scheduler.config.get("base_image_seq_len", 256),
      self.scheduler.config.get("max_image_seq_len", 4096),
      self.scheduler.config.get("base_shift", 0.5),
      self.scheduler.config.get("max_shift", 1.15),
    )
    if sigmas is None:
      sigmas = get_default_z_image_sigmas(num_inference_steps)
    self.scheduler.sigma_min = 0.0
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler,
                                                        num_inference_steps,
                                                        device,
                                                        sigmas=sigmas,
                                                        mu=mu)
    self.scheduler.set_begin_index(0)
    self._num_timesteps = len(timesteps)

    # Official inference, vllm-omni and SGLang all use zero caption/direct-VLM unconditions.
    negative_embeds = [torch.zeros_like(item) for item in prompt_embeds]
    start = time.perf_counter()
    with self.progress_bar(total=num_inference_steps) as progress_bar:
      for step_index, timestep_value in enumerate(timesteps):
        if self.interrupt:
          continue
        timestep = (1000 - timestep_value.expand(1)) / 1000
        apply_cfg = self.do_classifier_free_guidance
        if apply_cfg:
          model_latents = latents.to(self.transformer.dtype).repeat(2, 1, 1, 1, 1)
          model_embeds = prompt_embeds + negative_embeds
          model_timestep = timestep.repeat(2)
          model_direct = torch.cat([direct_condition, torch.zeros_like(direct_condition)], dim=0)
          model_reference = (reference_latent.repeat(2, 1, 1, 1, 1)
                             if reference_latent is not None else None)
        else:
          model_latents = latents.to(self.transformer.dtype)
          model_embeds = prompt_embeds
          model_timestep = timestep
          model_direct = direct_condition
          model_reference = reference_latent

        model_output = self.transformer(
          list(model_latents.unbind(dim=0)),
          model_timestep,
          model_embeds,
          ref_latents=model_reference,
          direct_condition=model_direct,
          joint_attention_kwargs=joint_attention_kwargs,
          return_dict=False,
        )[0]
        if apply_cfg:
          positive = model_output[0].float()
          negative = model_output[1].float()
          noise_pred = positive + self._guidance_scale * (positive - negative)
        else:
          noise_pred = model_output[0].float()
        noise_pred = -noise_pred.unsqueeze(0)
        latents = self.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
        if latents.dtype != torch.float32:
          raise RuntimeError("Ming-Image latent updates must remain in fp32.")

        if callback_on_step_end is not None:
          callback_kwargs = {name: locals()[name] for name in callback_on_step_end_tensor_inputs}
          callback_outputs = callback_on_step_end(self, step_index, timestep_value, callback_kwargs)
          latents = callback_outputs.pop("latents", latents)
          prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
        progress_bar.update()
    self.stage_durations["denoising"] = time.perf_counter() - start

    start = time.perf_counter()
    if output_type == "latent":
      images = latents
    else:
      images = self._decode_latent_frames(latents)
      images = self.image_processor.postprocess(images, output_type=output_type)
    self.stage_durations["vae"] = time.perf_counter() - start
    self.maybe_free_model_hooks()
    if not return_dict:
      return (images, )
    return ImagePipelineOutput(images=images)


__all__ = ["MingImagePipeline"]
