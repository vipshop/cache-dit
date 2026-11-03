"""LongLive 2.0 inference pipeline (chunk-wise autoregressive 4-step T2V).

The generator produces a clip chunk by chunk (8 latent frames per chunk by default): each chunk runs
the 4-step DMD schedule, then one extra clean-latent pass at `timestep=0` writes the clean K/V that
every later chunk attends to. Guidance is distilled away, so a step is exactly one transformer
forward.

Cache interaction, which is the reason this pipeline owns the loop:

* the clean-latent pass must run **all** blocks (`force_full_compute=True`), otherwise the skipped
  layers keep the noisy K/V written by the previous denoise step and every later chunk reads stale
  keys;
* the first denoise step of a chunk must recompute as well, because the chunk starts from fresh
  noise. The pipeline therefore arms `force_refresh_step_hint` on cache-dit's live context at every
  chunk boundary; the clean-latent pass is invisible to the cache counter, so the per-chunk pattern
  is exactly "refresh, denoise, denoise, denoise".

Text encoder, tokenizer, VAE and scheduler come from `Wan-AI/Wan2.2-TI2V-5B-Diffusers` so that TE-P
/ VAE-P keep dispatching on the diffusers / transformers module paths. Only the transformer is
replaced by the LongLive 2.0 generator.

Not covered on purpose: multi-shot prompts (`shot_prompts` / scene cuts / `multi_shot_sink`) and the
I2V first-frame clamp of the reference implementation. This pipeline renders one continuous clip from
one prompt.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import UniPCMultistepScheduler
from transformers import AutoTokenizer, UMT5EncoderModel

from ....logger import init_logger

logger = init_logger(__name__)

DEFAULT_NUM_FRAMES = 61
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832
DEFAULT_NUM_INFERENCE_STEPS = 4
DEFAULT_CHUNK_LATENT_FRAMES = 8
DEFAULT_MAX_SEQUENCE_LENGTH = 512
DEFAULT_FPS = 24

# The scheduler configuration shipped with the checkpoint: UniPC over a flow-matching sigma grid with
# `flow_shift=5.0`, four steps, second-order solver and a terminal sigma of zero.
RELEASED_SCHEDULER_CONFIG = {
  "_class_name": "UniPCMultistepScheduler",
  "beta_end": 0.02,
  "beta_schedule": "linear",
  "beta_start": 0.0001,
  "disable_corrector": [],
  "dynamic_thresholding_ratio": 0.995,
  "final_sigmas_type": "zero",
  "flow_shift": 5.0,
  "lower_order_final": True,
  "num_train_timesteps": 1000,
  "predict_x0": True,
  "prediction_type": "flow_prediction",
  "rescale_betas_zero_snr": False,
  "sample_max_value": 1.0,
  "solver_order": 2,
  "solver_p": None,
  "solver_type": "bh2",
  "steps_offset": 0,
  "thresholding": False,
  "time_shift_type": "exponential",
  "timestep_spacing": "linspace",
  "trained_betas": None,
  "use_beta_sigmas": False,
  "use_dynamic_shifting": False,
  "use_exponential_sigmas": False,
  "use_flow_sigmas": True,
  "use_karras_sigmas": False,
}


@dataclass
class LongLive2PipelineOutput:
  """Video output of :class:`LongLive2Pipeline`.

  :param frames: Decoded clip, either a `(T, H, W, C)` uint8 array or a list of PIL images, one
    entry per sample.
  """

  frames: Union[np.ndarray, List[Any]]


class LongLive2Pipeline(DiffusionPipeline):
  """Autoregressive LongLive 2.0 text-to-video pipeline.

  :param tokenizer: UMT5 tokenizer of the Wan2.2-TI2V-5B components.
  :param text_encoder: `UMT5EncoderModel` of the Wan2.2-TI2V-5B components.
  :param transformer: LongLive 2.0 generator, see :class:`LongLive2Transformer3DModel`.
  :param vae: `AutoencoderKLWan` of the Wan2.2-TI2V-5B components.
  :param scheduler: Flow-matching UniPC grid; defaults to the checkpoint's own configuration.
  """

  model_cpu_offload_seq = "text_encoder->transformer->vae"
  _optional_components: List[str] = []

  def __init__(
    self,
    tokenizer: AutoTokenizer,
    text_encoder: UMT5EncoderModel,
    transformer: torch.nn.Module,
    vae: AutoencoderKLWan,
    scheduler: Optional[UniPCMultistepScheduler] = None,
  ) -> None:
    super().__init__()
    if scheduler is None:
      scheduler = UniPCMultistepScheduler.from_config(dict(RELEASED_SCHEDULER_CONFIG))
    self.register_modules(
      tokenizer=tokenizer,
      text_encoder=text_encoder,
      transformer=transformer,
      vae=vae,
      scheduler=scheduler,
    )
    self.vae_scale_factor_temporal = int(
      getattr(getattr(vae, "config", None), "scale_factor_temporal", 4) or 4)
    # Wan VAE decodes 8x spatially; the DiT then patches the latent 2x, so the *token* grid is
    # `pixels / 16` while the latent grid is `pixels / 8`.
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
    scheduler: Optional[UniPCMultistepScheduler] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    quantization_config: Any = None,
    device_map: Any = None,
    trust_remote_code: bool = False,
    **kwargs,
  ) -> "LongLive2Pipeline":
    """Load the Wan2.2 components and attach the LongLive 2.0 generator.

    `pretrained_model_name_or_path` is a directory holding `transformer/`, `text_encoder/`,
    `tokenizer/`, `vae/` and `scheduler/`; the transformer subdirectory must already be converted
    from the released native state dict (see `convert.py`).

    :param pretrained_model_name_or_path: Local path of the assembled checkpoint directory.
    :param transformer: Pre-built generator; loaded from the checkpoint when omitted.
    :param vae: Pre-built VAE; loaded from the checkpoint when omitted.
    :param text_encoder: Pre-built text encoder; loaded from the checkpoint when omitted.
    :param tokenizer: Pre-built tokenizer; loaded from the checkpoint when omitted.
    :param scheduler: Flow-matching UniPC scheduler; the checkpoint's own config is used when
      omitted.
    :param torch_dtype: Dtype of the text encoder and the VAE.
    :param quantization_config: Rejected: cache-dit quantizes the transformer itself.
    :param device_map: Rejected, the CLI skips `to(device)` whenever it is set.
    :param trust_remote_code: Ignored, the code is local.
    :returns: The pipeline instance.
    """
    if device_map is not None:
      raise NotImplementedError(
        "LongLive2Pipeline builds its components explicitly, so `device_map` is not supported. "
        "Use `--device-map-balance` off, or `--layerwise-offload` for memory pressure.")
    if quantization_config is not None:
      logger.warning("Ignoring `quantization_config`: cache-dit quantizes the transformer itself, "
                     "diffusers' quantizer cannot find the causal blocks.")
    if trust_remote_code:
      logger.debug("`trust_remote_code` is ignored: the LongLive 2.0 code ships inside cache-dit.")

    base = os.fspath(pretrained_model_name_or_path)
    if transformer is None:
      from .transformer import LongLive2Transformer3DModel

      transformer = LongLive2Transformer3DModel.from_pretrained(os.path.join(base, "transformer"),
                                                                torch_dtype=torch_dtype)
    if vae is None:
      vae = AutoencoderKLWan.from_pretrained(os.path.join(base, "vae"), torch_dtype=torch.float32)
    if text_encoder is None:
      text_encoder = UMT5EncoderModel.from_pretrained(os.path.join(base, "text_encoder"),
                                                      torch_dtype=torch_dtype)
    if tokenizer is None:
      tokenizer = AutoTokenizer.from_pretrained(os.path.join(base, "tokenizer"))
    if scheduler is None:
      scheduler = UniPCMultistepScheduler.from_pretrained(os.path.join(base, "scheduler"))

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

  def _latent_frames(self, num_frames: int) -> int:
    ratio = self.vae_scale_factor_temporal
    return (num_frames - 1) // ratio + 1

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
    # has to fit inside one chunk - otherwise every step of every chunk lands in the warmup set and
    # caching can never engage.
    if cache_config.max_warmup_steps > 1:
      cache_config.max_warmup_steps = 1
      if not self._warmup_capped:
        logger.info(
          "Capping the cache warmup window at 1 step: LongLive 2.0 runs one denoising loop per "
          "chunk and every chunk start is a forced full compute anyway.")
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

  def _render_latents(
    self,
    *,
    prompt_embeds: torch.Tensor,
    latents: torch.Tensor,
    chunk_latent_frames: int,
    frame_seq_len: int,
    chunk_tokens: int,
    kv_cache: list,
    num_inference_steps: int,
    context_noise: float,
    device: torch.device,
  ) -> torch.Tensor:
    """Run the chunk-wise autoregressive loop and return the final latents.

    Factored out of :meth:`__call__` so the loop can be exercised without a VAE: the CPU tests build
    a tiny transformer and compare this against a hand-written reference loop.

    :param prompt_embeds: Text embeddings `(B, L, text_dim)`.
    :param latents: Initial noise `(B, C, F_total, H, W)`.
    :param chunk_latent_frames: Latent frames generated per autoregressive step.
    :param frame_seq_len: Token count of one latent frame.
    :param chunk_tokens: Token count of one chunk.
    :param kv_cache: Container returned by `transformer.init_kv_cache`.
    :param num_inference_steps: Denoising steps per chunk.
    :param context_noise: Timestep of the clean-latent pass.
    :param device: Device of the latents.
    :returns: The denoised latents, still in the generator's normalized space.
    """
    transformer = self.transformer
    batch_size = latents.shape[0]
    num_chunks = latents.shape[2] // chunk_latent_frames
    output = torch.zeros_like(latents)
    # Per-frame timesteps: the transformer expands them to one value per token, and under Ulysses CP
    # the hook on `transformer.forward` slices the frame axis (a 1-D timestep could not be split).
    commit_timestep = torch.full((batch_size, chunk_latent_frames),
                                 float(context_noise),
                                 device=device,
                                 dtype=torch.float32)

    for chunk_index in range(num_chunks):
      start_frame = chunk_index * chunk_latent_frames
      current_start = start_frame * frame_seq_len
      current_end = current_start + chunk_tokens
      chunk = latents[:, :, start_frame:start_frame + chunk_latent_frames].clone()

      # The schedule of one chunk is independent: `set_timesteps` also resets UniPC's solver state,
      # which is what the reference implementation does per chunk.
      self.scheduler.set_timesteps(num_inference_steps, device=device)
      timesteps = self.scheduler.timesteps

      # First denoise step of a chunk: full recompute, and a fresh residual baseline.
      self._refresh_step_cache()
      for timestep in timesteps:
        timestep_batch = timestep.expand(batch_size, chunk_latent_frames).to(device=device,
                                                                             dtype=torch.float32)
        model_output = transformer(
          chunk,
          timestep_batch,
          prompt_embeds,
          kv_cache=kv_cache,
          current_start=current_start,
          current_end=current_end,
        )
        chunk = self.scheduler.step(model_output, timestep, chunk, return_dict=False)[0]

      output[:, :, start_frame:start_frame + chunk_latent_frames] = chunk

      # Clean-latent commit: the only writer of the K/V later chunks attend to.
      transformer(
        chunk,
        commit_timestep,
        prompt_embeds,
        kv_cache=kv_cache,
        current_start=current_start,
        current_end=current_end,
        force_full_compute=True,
      )

    self._clear_step_cache_hint()
    return output

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
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = 1.0,
    chunk_latent_frames: int = DEFAULT_CHUNK_LATENT_FRAMES,
    local_attn_frames: int = 32,
    sink_frames: int = 8,
    context_noise: float = 0.0,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    output_type: str = "np",
    return_dict: bool = True,
    **kwargs,
  ) -> Union[LongLive2PipelineOutput, Dict[str, Any]]:
    """Generate a clip autoregressively.

    :param prompt: Prompt or list of prompts; batch size 1 is expected.
    :param negative_prompt: Rejected: the DMD student is guidance-distilled.
    :param height: Pixel height, divisible by 16.
    :param width: Pixel width, divisible by 16.
    :param num_frames: Pixel frames; `(num_frames - 1) / 4 + 1` must be divisible by
      `chunk_latent_frames`.
    :param num_inference_steps: Denoising steps per chunk; the released model uses 4.
    :param guidance_scale: Must stay 1.0: the student runs a single conditional forward per step,
      and a second one would desynchronise cache-dit's step bookkeeping.
    :param chunk_latent_frames: Latent frames generated per autoregressive step.
    :param local_attn_frames: Recent-window attention length in latent frames.
    :param sink_frames: Leading latent frames pinned as attention sinks.
    :param context_noise: Timestep of the clean-latent pass that commits the chunk K/V.
    :param max_sequence_length: UMT5 token budget.
    :param generator: Torch generator for the initial noise.
    :param output_type: `np`, `pil` or `latent`.
    :param return_dict: Whether to wrap the frames in :class:`LongLive2PipelineOutput`.
    :returns: The generated clip.
    """
    if negative_prompt is not None:
      logger.debug("`negative_prompt` is ignored: the LongLive 2.0 student runs without CFG.")
    if guidance_scale not in (None, 1.0):
      raise ValueError(
        f"LongLive 2.0 is guidance-distilled and runs one conditional forward per step; "
        f"`guidance_scale={guidance_scale}` would add a second one and corrupt the cache step "
        f"counter. Leave it at 1.0.")
    if height % 16 != 0 or width % 16 != 0:
      raise ValueError(f"`height`/`width` must be divisible by 16, got {height}x{width}.")

    device = self._execution_device
    transformer = self.transformer
    # Not `next(transformer.parameters()).dtype`: the fp32 keep list makes the first parameter
    # float32 while the latent grid runs in the checkpoint dtype.
    dtype = transformer.input_dtype()
    self._warmup_capped = False

    prompt_embeds = self.encode_prompt(prompt,
                                       device=device,
                                       dtype=dtype,
                                       max_sequence_length=max_sequence_length)

    latent_frames = self._latent_frames(num_frames)
    if latent_frames % chunk_latent_frames != 0:
      raise ValueError(
        f"`num_frames={num_frames}` gives {latent_frames} latent frames, which is not divisible by "
        f"`chunk_latent_frames={chunk_latent_frames}`. Use 4 * k * chunk_latent_frames + 1 frames, "
        f"e.g. 61 for k=2 or 129 for k=4.")
    batch_size = prompt_embeds.shape[0]
    latent_shape = (
      batch_size,
      transformer.config.in_channels,
      latent_frames,
      height // self.vae_scale_factor_spatial,
      width // self.vae_scale_factor_spatial,
    )
    chunk_latent_shape = (
      batch_size,
      latent_shape[1],
      chunk_latent_frames,
      latent_shape[3],
      latent_shape[4],
    )
    kv_cache = transformer.init_kv_cache(
      batch_size=batch_size,
      latent_shape=chunk_latent_shape,
      num_frames=latent_frames,
      device=device,
      dtype=dtype,
      local_attn_frames=local_attn_frames,
      sink_frames=sink_frames,
    )

    frame_seq_len = transformer.frame_seq_len(chunk_latent_shape)
    chunk_tokens = chunk_latent_frames * frame_seq_len
    noise_gen = self._resolve_generator(generator, device)
    generator_kwargs = {} if noise_gen is None else {"generator": noise_gen}
    latents = torch.randn(latent_shape, device=device, dtype=dtype, **generator_kwargs)
    latents = self._render_latents(
      prompt_embeds=prompt_embeds,
      latents=latents,
      chunk_latent_frames=chunk_latent_frames,
      frame_seq_len=frame_seq_len,
      chunk_tokens=chunk_tokens,
      kv_cache=kv_cache,
      num_inference_steps=num_inference_steps,
      context_noise=context_noise,
      device=device,
    )

    transformer.release_kv_cache()
    latents = latents.to(self.vae.dtype)
    # The Wan VAE works in a normalized latent space; denormalize before decoding.
    shape = (1, self.vae.config.z_dim, 1, 1, 1)
    latents_mean = torch.tensor(self.vae.config.latents_mean).view(shape).to(latents)
    latents_std = torch.tensor(self.vae.config.latents_std).view(shape).to(latents)
    latents = latents * latents_std + latents_mean
    video = self.vae.decode(latents, return_dict=False)[0]
    frames = self._postprocess(video, output_type)
    self.maybe_free_model_hooks()

    if not return_dict:
      return (frames, )
    return LongLive2PipelineOutput(frames=frames)
