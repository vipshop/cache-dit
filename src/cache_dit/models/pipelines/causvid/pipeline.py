"""CausVid-Wan inference pipeline (chunk-wise autoregressive 3-step T2V).

The student generates a clip chunk by chunk: each chunk runs the fixed DMD step table
(`1000 / 757 / 522`), then one extra clean-latent pass at `timestep=0` writes the clean K/V that every
later chunk attends to. Neither pass uses CFG, so a step is exactly one transformer forward.

Cache interaction, which is the reason this pipeline owns the loop:

* the clean-latent pass must run **all** blocks (`force_full_compute=True`), otherwise the skipped
  layers keep the noisy K/V written by the previous denoise step and every later chunk reads stale
  keys;
* the first denoise step of a chunk must recompute as well, because the chunk starts from fresh
  noise. The pipeline therefore arms `force_refresh_step_hint` on cache-dit's live context at every
  chunk boundary; the clean-latent pass is invisible to the cache counter, so the per-chunk pattern
  is exactly "refresh, denoise, denoise, denoise".

Text encoder, VAE and tokenizer come from `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` so that TE-P / VAE-P
keep dispatching on the diffusers/transformers module paths. Only the transformer is replaced by the
CausVid student.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from transformers import AutoTokenizer, UMT5EncoderModel

from ....logger import init_logger
from .scheduler import CausVidFlowMatchScheduler

logger = init_logger(__name__)

# The released student: latent 21 frames = 7 chunks of 3, DMD table 1000/757/522, no CFG.
DEFAULT_CHUNK_FRAMES = 3
DEFAULT_DENOISING_STEPS = (1000, 757, 522)
DEFAULT_NUM_FRAMES = 81
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832
DEFAULT_FPS = 16
DEFAULT_MAX_SEQUENCE_LENGTH = 512


@dataclass
class CausVidPipelineOutput:
  """Video output of :class:`CausVidPipeline`.

  :param frames: Decoded clip, either a `(T, H, W, C)` uint8 array or a list of PIL images.
  """

  frames: Union[np.ndarray, List[Any]]


class CausVidPipeline(DiffusionPipeline):
  """Autoregressive CausVid-Wan text-to-video pipeline.

  :param tokenizer: UMT5 tokenizer of the Wan2.1 base checkpoint.
  :param text_encoder: `UMT5EncoderModel` of the Wan2.1 base checkpoint.
  :param transformer: CausVid student, see :class:`CausVidTransformer3DModel`.
  :param vae: `AutoencoderKLWan` of the Wan2.1 base checkpoint.
  :param scheduler: Sigma grid; defaults to the released `shift=8.0` table.
  """

  model_cpu_offload_seq = "text_encoder->transformer->vae"
  _optional_components: List[str] = []

  def __init__(
    self,
    tokenizer: AutoTokenizer,
    text_encoder: UMT5EncoderModel,
    transformer: torch.nn.Module,
    vae: AutoencoderKLWan,
    scheduler: Optional[CausVidFlowMatchScheduler] = None,
  ) -> None:
    super().__init__()
    self.register_modules(
      tokenizer=tokenizer,
      text_encoder=text_encoder,
      transformer=transformer,
      vae=vae,
    )
    self.scheduler = scheduler if scheduler is not None else CausVidFlowMatchScheduler()
    # Wan VAE decodes 8x spatially (`temperal_downsample` has one entry per downsample stage); the DiT
    # then patches the latent 2x, so the *token* grid is `pixels / 16` while the latent grid is
    # `pixels / 8`.
    self.vae_scale_factor_spatial = 2 ** len(
      getattr(getattr(vae, "config", None), "temperal_downsample", [False, True, True, True]))
    self.fps = DEFAULT_FPS
    # Set once per request, when the per-chunk warmup cap is first applied.
    self._warmup_capped = False

  @classmethod
  def from_pretrained(
    cls,
    pretrained_model_name_or_path: str,
    *,
    transformer: Optional[torch.nn.Module] = None,
    vae: Optional[AutoencoderKLWan] = None,
    text_encoder: Optional[UMT5EncoderModel] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    scheduler: Optional[CausVidFlowMatchScheduler] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    quantization_config: Any = None,
    device_map: Any = None,
    trust_remote_code: bool = False,
    **kwargs,
  ) -> "CausVidPipeline":
    """Load the Wan2.1 base components and attach the pre-built CausVid student.

    `pretrained_model_name_or_path` points at a `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` tree
    (`text_encoder/`, `tokenizer/`, `vae/`). The transformer never comes from that tree: it is the
    distilled student, converted from `tianweiy/CausVid/autoregressive_checkpoint/model.pt`.

    :param pretrained_model_name_or_path: Local path of the Wan2.1 diffusers checkpoint.
    :param transformer: Pre-built student; required.
    :param vae: Pre-built VAE; loaded from the checkpoint when omitted.
    :param text_encoder: Pre-built text encoder; loaded from the checkpoint when omitted.
    :param tokenizer: Pre-built tokenizer; loaded from the checkpoint when omitted.
    :param scheduler: Sigma grid; the released table is used when omitted.
    :param torch_dtype: Dtype of the text encoder and the VAE.
    :param quantization_config: Rejected: cache-dit quantizes the transformer itself.
    :param device_map: Rejected, the CLI skips `to(device)` whenever it is set.
    :param trust_remote_code: Ignored, the code is local.
    :returns: The pipeline instance.
    """
    if device_map is not None:
      raise NotImplementedError(
        "CausVidPipeline builds its components explicitly, so `device_map` is not supported. "
        "Use `--device-map-balance` off, or `--layerwise-offload` for memory pressure.")
    if quantization_config is not None:
      logger.warning("Ignoring `quantization_config`: cache-dit quantizes the transformer itself, "
                     "diffusers' quantizer cannot find the causal blocks.")
    if trust_remote_code:
      logger.debug("`trust_remote_code` is ignored: the CausVid code ships inside cache-dit.")
    if transformer is None:
      raise ValueError(
        "CausVidPipeline.from_pretrained needs the student transformer. Build it with "
        "`CausVidTransformer3DModel(...)` + `load_causvid_state_dict(...)`, or use the "
        "`causvid_wan` CLI example which wires the checkpoint for you.")

    base = os.fspath(pretrained_model_name_or_path)
    if vae is None:
      vae = AutoencoderKLWan.from_pretrained(os.path.join(base, "vae"), torch_dtype=torch.float32)
    if text_encoder is None:
      text_encoder = UMT5EncoderModel.from_pretrained(os.path.join(base, "text_encoder"),
                                                      torch_dtype=torch_dtype)
    if tokenizer is None:
      tokenizer = AutoTokenizer.from_pretrained(os.path.join(base, "tokenizer"))

    return cls(
      tokenizer=tokenizer,
      text_encoder=text_encoder,
      transformer=transformer,
      vae=vae,
      scheduler=scheduler,
    )

  def encode_prompt(
    self,
    prompt: Union[str, List[str]],
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
  ) -> torch.Tensor:
    """Encode a prompt with UMT5 and zero the padding rows, as the Wan reference does.

    :param prompt: Prompt or list of prompts.
    :param device: Device of the text encoder.
    :param dtype: Dtype of the returned embeddings.
    :param max_sequence_length: Token budget of the Wan text conditioner.
    :returns: Text embeddings shaped `(B, max_sequence_length, text_dim)`.
    """
    prompt = [prompt] if isinstance(prompt, str) else list(prompt)
    encodings = self.tokenizer(
      prompt,
      padding="max_length",
      max_length=max_sequence_length,
      truncation=True,
      add_special_tokens=True,
      return_attention_mask=True,
      return_tensors="pt",
    )
    text_input_ids = encodings.input_ids.to(device)
    mask = encodings.attention_mask.to(device)
    prompt_embeds = self.text_encoder(text_input_ids, mask).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    for embed, seq_len in zip(prompt_embeds, seq_lens):
      embed[int(seq_len):] = 0.0
    return prompt_embeds

  @staticmethod
  def _latent_frames(num_frames: int, temporal_compression_ratio: int = 4) -> int:
    return (num_frames - 1) // temporal_compression_ratio + 1

  @staticmethod
  def _resolve_generator(
    generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    device: torch.device,
  ) -> Optional[torch.Generator]:
    """Return one generator bound to `device`, porting the state of a CPU generator when needed.

    :param generator: User generator or per-sample list; `None` keeps the global RNG.
    :param device: Device the latents live on.
    :returns: A generator on `device`, or `None`.
    """
    if isinstance(generator, (list, tuple)):
      generator = generator[0] if generator else None
    if generator is None:
      return None
    if generator.device.type == device.type:
      return generator
    ported = torch.Generator(device=device)
    ported.set_state(generator.get_state())
    return ported

  def _refresh_step_cache(self) -> None:
    # Arming the hint on the live context makes the *next* transformer forward recompute every block,
    # which is what a chunk boundary needs (fresh noise, and a reset residual baseline). cache-dit
    # deep-copies the config on refresh, so the hint has to be re-armed for every chunk.
    #
    # Both fields are written into the config the manager copies from (one shared object per context
    # group), because the copy a refreshed context gets would otherwise drop the warmup cap again.
    # The warmup cap is logged once per request; it never changes what a chunk computes, since the
    # chunk start is a forced full compute either way.
    manager = getattr(self, "_context_manager", None)
    if manager is None:
      return
    context = manager.get_context()
    if context is None:
      return
    cache_config = context.cache_config
    cache_config.force_refresh_step_hint = context.get_current_transformer_step() + 1
    cache_config.force_refresh_step_policy = "once"
    # Every chunk gets its own step counter, so the warmup window (`{0, .., ws-1}` on that counter)
    # has to fit inside one chunk — otherwise every step of every chunk lands in the warmup set and
    # caching can never engage.
    if cache_config.max_warmup_steps > 1:
      cache_config.max_warmup_steps = 1
      if not self._warmup_capped:
        logger.info(
          "Capping the cache warmup window at 1 step: CausVid-Wan runs one denoising loop per chunk "
          "and every chunk start is a forced full compute anyway.")
        self._warmup_capped = True

  def _clear_step_cache_hint(self) -> None:
    """Drop the refresh hint so the next request does not trigger a stale refresh."""
    manager = getattr(self, "_context_manager", None)
    if manager is None:
      return
    context = manager.get_context()
    if context is None:
      return
    context.cache_config.force_refresh_step_hint = None

  def _postprocess(self, video: torch.Tensor, output_type: str):
    # `AutoencoderKLWan.decode` returns `(B, C, T, H, W)`; the writer takes one frame container per
    # sample (a `(T, H, W, C)` array or a list of PIL images), like the other video pipelines.
    video = video.float().clamp(-1, 1) * 0.5 + 0.5
    samples = []
    for item in video:
      frames = item.permute(1, 2, 3, 0).cpu()
      if output_type == "latent":
        samples.append(frames)
        continue
      frames_u8 = (frames * 255).round().to(torch.uint8).numpy()
      if output_type == "np":
        samples.append(frames_u8)
      else:
        from PIL import Image

        samples.append([Image.fromarray(frame) for frame in frames_u8])
    return samples

  @torch.no_grad()
  def __call__(
    self,
    prompt: Union[str, List[str]],
    negative_prompt: Optional[str] = None,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_frames: int = DEFAULT_NUM_FRAMES,
    num_inference_steps: Optional[int] = None,
    guidance_scale: float = 1.0,
    denoising_steps: Optional[List[int]] = None,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    output_type: str = "np",
    return_dict: bool = True,
    **kwargs,
  ) -> Union[CausVidPipelineOutput, Dict[str, Any]]:
    """Generate a clip autoregressively.

    :param prompt: Prompt or list of prompts; batch size 1 is expected.
    :param negative_prompt: Ignored; the distilled student is guidance-free.
    :param height: Pixel height, divisible by 16.
    :param width: Pixel width, divisible by 16.
    :param num_frames: Pixel frames; `(num_frames - 1) / 4 + 1` must be divisible by `chunk_frames`.
    :param num_inference_steps: Ignored; the DMD table is part of the checkpoint.
    :param guidance_scale: Ignored; CausVid distills guidance away.
    :param denoising_steps: Override of the DMD timestep table (`1000 / 757 / 522`).
    :param chunk_frames: Latent frames generated per autoregressive step.
    :param max_sequence_length: UMT5 token budget.
    :param generator: Torch generator for the initial noise and the re-noising.
    :param output_type: `np`, `pil` or `latent`.
    :param return_dict: Whether to wrap the frames in :class:`CausVidPipelineOutput`.
    :returns: The generated clip.
    """
    if negative_prompt is not None:
      logger.debug("`negative_prompt` is ignored: the CausVid student runs without CFG.")
    if guidance_scale not in (None, 1.0):
      logger.warning(f"`guidance_scale={guidance_scale}` is ignored: the CausVid student is "
                     f"guidance-distilled and runs a single conditional forward.")
    if num_inference_steps is not None and num_inference_steps != len(DEFAULT_DENOISING_STEPS):
      logger.warning(f"`num_inference_steps={num_inference_steps}` is ignored: the student's DMD "
                     f"table is fixed at {list(DEFAULT_DENOISING_STEPS)}.")
    if height % 16 != 0 or width % 16 != 0:
      raise ValueError(f"`height`/`width` must be divisible by 16, got {height}x{width}.")

    device = self._execution_device
    transformer = self.transformer
    dtype = next(transformer.parameters()).dtype
    self._warmup_capped = False

    prompt_embeds = self.encode_prompt(prompt,
                                       device=device,
                                       dtype=dtype,
                                       max_sequence_length=max_sequence_length)

    latent_frames = self._latent_frames(num_frames)
    if latent_frames % chunk_frames != 0:
      raise ValueError(
        f"`num_frames={num_frames}` gives {latent_frames} latent frames, which is not divisible by "
        f"`chunk_frames={chunk_frames}`. Use 4 * k * chunk_frames + 1 frames, e.g. 81 for k=7.")
    batch_size = prompt_embeds.shape[0]
    latent_shape = (
      batch_size,
      transformer.config.in_channels,
      latent_frames,
      height // self.vae_scale_factor_spatial,
      width // self.vae_scale_factor_spatial,
    )

    steps = list(DEFAULT_DENOISING_STEPS if denoising_steps is None else denoising_steps)
    if steps[-1] != DEFAULT_DENOISING_STEPS[-1]:
      raise ValueError(
        f"`denoising_steps` must end with {DEFAULT_DENOISING_STEPS[-1]}, got {steps}.")

    chunk_latent_shape = (
      batch_size,
      latent_shape[1],
      chunk_frames,
      latent_shape[3],
      latent_shape[4],
    )
    kv_cache = transformer.init_kv_cache(
      batch_size=batch_size,
      latent_shape=chunk_latent_shape,
      num_frames=latent_frames,
      device=device,
      dtype=dtype,
    )

    frame_seq_len = transformer.frame_seq_len(chunk_latent_shape)
    chunk_tokens = chunk_frames * frame_seq_len
    # One generator drives the initial noise and every re-noising draw, so a fixed seed reproduces
    # the clip bit-exactly — including across ranks, which is what makes PSNR=inf a CP gate.
    noise_gen = self._resolve_generator(generator, device)
    generator_kwargs = {} if noise_gen is None else {"generator": noise_gen}
    noise = torch.randn(latent_shape, device=device, dtype=dtype, **generator_kwargs)

    latents = noise.clone()
    output = torch.zeros_like(latents)
    num_chunks = latent_frames // chunk_frames

    for chunk_index in range(num_chunks):
      start_frame = chunk_index * chunk_frames
      current_start = start_frame * frame_seq_len
      current_end = current_start + chunk_tokens
      chunk = latents[:, :, start_frame:start_frame + chunk_frames].clone()

      # First denoise step of a chunk: full recompute, and a fresh residual baseline.
      self._refresh_step_cache()

      x0 = chunk
      for step_index, timestep in enumerate(steps):
        timestep_batch = torch.full((batch_size, ), timestep, device=device, dtype=torch.float32)
        model_output = transformer(
          chunk,
          timestep_batch,
          prompt_embeds,
          kv_cache=kv_cache,
          current_start=current_start,
          current_end=current_end,
        )
        sigma = self.scheduler.sigma(timestep_batch, device=device).view(-1, 1, 1, 1, 1).to(dtype)
        x0 = chunk - sigma * model_output
        if step_index < len(steps) - 1:
          next_timestep = torch.full((batch_size, ),
                                     steps[step_index + 1],
                                     device=device,
                                     dtype=torch.float32)
          renoise = torch.randn(chunk.shape, device=device, dtype=dtype, **generator_kwargs)
          chunk = self.scheduler.add_noise(x0, renoise, next_timestep)

      output[:, :, start_frame:start_frame + chunk_frames] = x0

      # Clean-latent commit: the only writer of the K/V later chunks attend to.
      commit_timestep = torch.zeros((batch_size, ), device=device, dtype=torch.float32)
      transformer(
        x0,
        commit_timestep,
        prompt_embeds,
        kv_cache=kv_cache,
        current_start=current_start,
        current_end=current_end,
        force_full_compute=True,
      )

    latents = output.to(self.vae.dtype)
    self._clear_step_cache_hint()
    # The KV cache is 6 GiB at the released geometry and is dead once the clip is denoised.
    transformer.release_kv_cache()
    latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1,
                                                                   1).to(
                                                                     latents.device, latents.dtype)
    latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
      1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    video = self.vae.decode(latents, return_dict=False)[0]
    frames = self._postprocess(video, output_type)
    self.maybe_free_model_hooks()

    if not return_dict:
      return (frames, )
    return CausVidPipelineOutput(frames=frames)
