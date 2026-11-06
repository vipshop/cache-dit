"""HY-Nomas text-to-video pipeline: segment-autoregressive generation with a persistent KV cache.

The release is not a diffusers checkpoint: the repo ships weights under ``DiT/`` and ``VAE/`` with a
3-byte ``config.json`` and no ``model_index.json``, so the components are assembled from those
sub-directories here (the MOSS-Image layout) instead of through
``DiffusionPipeline.from_pretrained`` discovery.

Two properties of the loop are load-bearing and easy to break:

* **The denoising steps must call ``transformer(...)``** — the cached runtime patches
  ``forward``, so a step that goes through `predict()` would never be cached, and every gate
  would still pass (bit-identical output, zero speedup).
* **The K/V write must call ``predict(collect_cache=True)``** and commit only after every layer
  produced an entry; a skipped block would otherwise publish partial history for every later
  segment.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput

from ....logger import init_logger
from .components import NomasPixelUMMDecoder
from .sampling import (
  DEFAULT_CACHE_NOISE_SIGMA,
  DEFAULT_FLOW_SHIFT,
  DEFAULT_SEGMENT_LATENT_FRAMES,
  DEFAULT_TEXT_LEN,
  NomasSegment,
  action_metadata,
  build_text_inputs,
  cache_capacity,
  canonical_camera,
  canonical_caption,
  check_cache_budget,
  nomas_sigmas,
  plan_segments,
  postprocess_text,
  segment_camera_embedding,
  to_unit_range,
)
from .transformer import NomasTransformer3DModel

logger = init_logger(__name__)

_TEXT_ENCODER_ENV = "HY_NOMAS_TEXT_ENCODER_DIR"
_DEFAULT_TEXT_ENCODER = "Qwen/Qwen2.5-VL-7B-Instruct"
# Preference order when both decoder variants are on disk; fp8 dequantises to the same bf16 weights
# at ~45% of the fp32 footprint.
_DECODER_VARIANTS = ("VAE/PixelUMM_FT_FP8", "VAE/PixelUMM_FT")


@dataclass
class NomasPipelineOutput(BaseOutput):
  """Pipeline output.

  :param frames: ``[1, 3, F, H, W]`` video in ``[0, 1]``.
  """

  frames: torch.Tensor


def _single_file(pattern: str) -> str:
  matches = sorted(glob.glob(pattern))
  if len(matches) != 1:
    raise FileNotFoundError(f"expected exactly one checkpoint matching {pattern}, found {matches}")
  return matches[0]


class NomasPipeline(DiffusionPipeline):
  """Segment-autoregressive text-to-video pipeline of HY-Nomas."""

  model_cpu_offload_seq = "text_encoder->transformer->vae"

  def __init__(
    self,
    transformer: NomasTransformer3DModel,
    vae: NomasPixelUMMDecoder,
    text_encoder: torch.nn.Module,
    tokenizer,
    scheduler=None,
  ):
    super().__init__()
    self.register_modules(transformer=transformer,
                          vae=vae,
                          text_encoder=text_encoder,
                          tokenizer=tokenizer,
                          scheduler=scheduler)
    self.text_len = DEFAULT_TEXT_LEN
    self.segment_latent_frames = DEFAULT_SEGMENT_LATENT_FRAMES
    self.flow_shift = DEFAULT_FLOW_SHIFT
    self.cache_noise_sigma = DEFAULT_CACHE_NOISE_SIGMA
    self.max_cache_gib = 32.0
    self._warmup_capped = False

  @classmethod
  def from_pretrained(
    cls,
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    quantization_config=None,
    device_map=None,
    trust_remote_code: bool = False,
    **kwargs,
  ) -> "NomasPipeline":
    """Assemble the pipeline from the release layout.

    :param model_path: Release root (holds ``DiT/`` and ``VAE/``).
    :param torch_dtype: Component dtype.
    :param quantization_config: Dropped with a warning; quantize through cache-dit instead.
    :param device_map: Rejected: the CLI would skip ``pipe.to(device)`` and run on CPU.
    :param trust_remote_code: Ignored; the code is local.
    :param kwargs: ``transformer`` / ``vae`` / ``text_encoder`` / ``tokenizer`` overrides,
      ``transformer_variant`` (default ``distilled``), ``text_encoder_path``.
    :returns: The assembled pipeline.
    """
    if device_map is not None:
      raise NotImplementedError("HY-Nomas does not support `device_map`; the pipeline must be "
                                "moved to its device explicitly so the KV cache follows it.")
    if quantization_config is not None:
      logger.warning("Ignoring `quantization_config` on load: the release ships plain weights, and "
                     "cache-dit quantizes the transformer itself.")
    if trust_remote_code:
      logger.debug("Ignoring `trust_remote_code`: the vendored code is local.")

    variant = kwargs.pop("transformer_variant", "distilled")
    text_encoder_path = kwargs.pop("text_encoder_path", None)
    transformer = kwargs.pop("transformer", None)
    vae = kwargs.pop("vae", None)
    text_encoder = kwargs.pop("text_encoder", None)
    tokenizer = kwargs.pop("tokenizer", None)
    if kwargs:
      logger.debug("Ignoring unsupported from_pretrained arguments: %s", sorted(kwargs))

    if transformer is None:
      weight = _single_file(
        os.path.join(model_path, "DiT", variant, "diffusion_pytorch_model*.safetensors"))
      transformer = NomasTransformer3DModel(attention_backend="flex").to(torch_dtype)
      from safetensors.torch import load_file

      state = load_file(weight)
      transformer.load_state_dict({
        key: value.to(torch_dtype)
        for key, value in state.items()
      },
                                  strict=True)
      transformer.post_load_weights()
      transformer.eval().requires_grad_(False)
    if vae is None:
      vae = NomasPixelUMMDecoder.from_checkpoint(_decoder_weight(model_path), dtype=torch_dtype)
    if text_encoder is None or tokenizer is None:
      from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

      source = text_encoder_path or os.environ.get(_TEXT_ENCODER_ENV) or _DEFAULT_TEXT_ENCODER
      tokenizer = tokenizer or AutoTokenizer.from_pretrained(source)
      text_encoder = (text_encoder or Qwen2_5_VLForConditionalGeneration.from_pretrained(
        source, torch_dtype=torch_dtype).eval().requires_grad_(False))
    return cls(transformer=transformer, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer)

  def _refresh_segment_cache(self) -> None:
    """Force the next transformer call to recompute every block.

    A cached step would reuse the previous segment's residual, but the segment boundary changes the
    history, the noise and the RoPE offset.  The hint is armed again for every segment because
    cache-dit deep-copies the config when it refreshes.
    """
    # cache-dit attaches the manager to the patched transformer, not to the pipeline.
    transformer = getattr(self, "transformer", None)
    manager = transformer.__dict__.get("_context_manager") if transformer is not None else None
    manager = manager or self.__dict__.get("_context_manager")
    if manager is None:
      return
    context = manager.get_context()
    if context is None or context.cache_config is None:
      return
    config = context.cache_config
    config.force_refresh_step_hint = context.get_current_transformer_step() + 1
    config.force_refresh_step_policy = "once"
    if config.max_warmup_steps > 1:
      config.max_warmup_steps = 1
      if not self._warmup_capped:
        logger.info("HY-Nomas: capping the cache warmup window at 1 step; every segment start is a "
                    "forced full compute anyway.")
        self._warmup_capped = True

  @torch.inference_mode()
  def _encode_text(self, captions: Sequence[str], device, dtype):
    inputs = build_text_inputs(self.tokenizer, list(captions), max_length=self.text_len)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    outputs = self.text_encoder(input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=True)
    states, mask = postprocess_text(outputs.hidden_states, inputs)
    return states.to(dtype), mask.to(device)

  def _plan(self, prompt, num_frames, segments, continuation_prompts,
            prompt_format) -> List[NomasSegment]:
    planned = plan_segments(num_frames, prompt, segments, continuation_prompts,
                            self.segment_latent_frames)
    return [
      NomasSegment(segment.start, segment.end, canonical_caption(segment.caption, prompt_format))
      for segment in planned
    ]

  def _noise(self, shape, generator, device, dtype):
    """Draw latents with the caller's generator; its device decides where the noise is sampled."""
    generator = generator or torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn(shape, generator=generator, dtype=dtype, device=generator.device)
    return noise.to(device)

  @torch.inference_mode()
  def __call__(
    self,
    prompt: Union[str, Sequence[str]],
    height: int = 352,
    width: int = 640,
    num_frames: int = 40,
    num_inference_steps: int = 8,
    guidance_scale: float = 1.0,
    prompt_format: str = "json",
    segments: Optional[Sequence[dict]] = None,
    continuation_prompts: Optional[Sequence[str]] = None,
    negative_prompt: Optional[str] = None,
    norm_extrinsics=None,
    generator: Optional[torch.Generator] = None,
    fps: int = 24,
    output_type: str = "np",
  ) -> NomasPipelineOutput:
    """Generate one clip event by event.

    :param prompt: Caption of the first event (a JSON event in ``json`` mode).
    :param height: Pixel height; must be a multiple of 32.
    :param width: Pixel width; must be a multiple of 32.
    :param num_frames: Requested pixel frame count.
    :param num_inference_steps: Euler steps per segment (the distilled recipe is 8).
    :param guidance_scale: Must stay ``1.0``: the distilled checkpoint has no CFG branch.
    :param prompt_format: ``json`` (official event schema) or ``plain`` (smoke tests only).
    :param segments: Optional explicit ``[{"frames": int, "caption": str}, ...]``.
    :param continuation_prompts: Optional captions of the events after the first.
    :param negative_prompt: Unused; present for CLI symmetry.
    :param norm_extrinsics: Optional canonical normalized camera timeline ``(L-1)*4+1``.
    :param generator: Generator; one fixed seed reproduces the clip on its device.
    :param fps: Frame rate, used to place action windows.
    :param output_type: ``np`` (default, what the CLI video writers consume) or ``pt``.
    :returns: The decoded clip.
    """
    if guidance_scale != 1.0:
      raise ValueError("HY-Nomas is distilled without classifier-free guidance: guidance_scale "
                       "must be 1.0, and a second forward per step would break cache bookkeeping.")
    if negative_prompt:
      logger.debug("Ignoring negative_prompt: the distilled checkpoint has no uncond branch.")
    if prompt_format == "plain":
      logger.warning("HY-Nomas plain captions are an experimental shortcut; pretrained validation "
                     "requires prompt_format=json.")
    if height % 32 or width % 32:
      raise ValueError("HY-Nomas height and width must be multiples of 32")
    if torch.is_tensor(prompt):
      raise TypeError("prompt must be text")
    prompt = prompt if isinstance(prompt, str) else prompt[0]

    device = self._execution_device
    dtype = self.transformer.dtype
    plan = self._plan(prompt, num_frames, segments, continuation_prompts, prompt_format)
    latent_frames = (num_frames - 1) // 4 + 1
    latent_hw = (height // 32, width // 32)
    conditions = []
    for segment in plan:
      states, mask = self._encode_text([segment.caption], device, dtype)
      text_offsets = video_windows = None
      action_start = -1
      if prompt_format == "json":
        action_start, text_offsets, video_windows = action_metadata(segment.caption,
                                                                    self.tokenizer,
                                                                    mask,
                                                                    segment.end - segment.start,
                                                                    fps,
                                                                    text_len=self.text_len)
      conditions.append({
        "text_states": states,
        "text_mask": mask,
        "text_offsets": text_offsets,
        "video_windows": video_windows,
        "action_start": action_start,
      })

    capacity = sum(
      cache_capacity((segment.end - segment.start, *latent_hw), condition["text_mask"].shape[1])
      for segment, condition in zip(plan[:-1], conditions[:-1]))
    free_bytes = None
    if device.type == "cuda":
      free_bytes = torch.cuda.mem_get_info(device)[0] + torch.cuda.memory_reserved(device) \
        - torch.cuda.memory_allocated(device)
    check_cache_budget(max(capacity, 1),
                       self.transformer.config.num_layers,
                       self.transformer.config.hidden_size,
                       self.transformer.parallel.head_shards,
                       self.max_cache_gib,
                       free_bytes=free_bytes)
    cache = self.transformer.allocate_cache(max(capacity, 1))

    camera = canonical_camera(norm_extrinsics, latent_frames, device)
    camera_embedding = self.transformer.embed_camera(camera[None]) if camera is not None else None
    sigmas = nomas_sigmas(num_inference_steps, self.flow_shift)
    latents: List[torch.Tensor] = []
    for index, (segment, condition) in enumerate(zip(plan, conditions)):
      latents.append(
        self._run_segment(segment,
                          condition,
                          latent_hw,
                          sigmas,
                          cache,
                          camera_embedding,
                          generator,
                          device,
                          dtype,
                          cache_segment=index < len(plan) - 1))
    video_latents = torch.cat(latents, dim=2)
    pixels = self.vae.decode(video_latents).sample
    pixels = to_unit_range(pixels)[:, :, :num_frames]
    # The CLI video writers iterate frames as HWC arrays on the host, so `np` returns
    # `[F, H, W, 3]` instead of the raw `[1, 3, F, H, W]` batch layout.
    if output_type == "np":
      pixels = pixels[0].permute(1, 2, 3, 0).float().cpu().numpy()
    else:
      pixels = pixels.float().cpu()
    if self.text_encoder is not None and getattr(self, "model_cpu_offload_seq", None):
      self.maybe_free_model_hooks()
    return NomasPipelineOutput(frames=pixels)

  def _run_segment(self, segment: NomasSegment, condition, latent_hw, sigmas, cache,
                   camera_embedding, generator, device, dtype, cache_segment: bool):
    """Denoise one segment and, unless it is the last one, publish its K/V."""
    frames, height, width = segment.end - segment.start, *latent_hw
    shape = (1, self.transformer.config.in_channels, frames, height, width)
    latent = self._noise(shape, generator, device, dtype)
    camera = segment_camera_embedding(camera_embedding, segment.start, segment.end)
    context = self.transformer.prepare_segment_context(latent,
                                                       condition["text_mask"],
                                                       cache,
                                                       text_offsets=condition["text_offsets"],
                                                       video_windows=condition["video_windows"],
                                                       hide_cached_action_tails=True)
    # A cached step would reuse the previous segment's residual baseline.
    self._refresh_segment_cache()
    for index in range(sigmas.shape[0] - 1):
      timestep = (sigmas[index:index + 1] * 1000).to(device)
      prediction = self.transformer(
        latent,
        condition["text_states"],
        timestep,
        text_mask=condition["text_mask"],
        cache=cache,
        camera_embedding=camera,
        text_offsets=condition["text_offsets"],
        video_windows=condition["video_windows"],
        hide_cached_action_tails=True,
        context=context,
      )
      latent = (latent.float() + prediction.float() * (sigmas[index + 1] - sigmas[index])).to(dtype)
    if not cache_segment:
      return latent
    # The K/V write is a separate forward on a lightly renoised latent, and it must run every block:
    # `predict` is not patched by the cached runtime.
    sigma = self.cache_noise_sigma
    cache_latent = latent
    if sigma:
      noise = self._noise(shape, generator, device, dtype)
      cache_latent = ((1 - sigma) * latent.float() + sigma * noise.float()).to(dtype)
    _, entries = self.transformer.predict(
      cache_latent,
      torch.tensor([sigma * 1000.0], device=device),
      condition["text_states"],
      condition["text_mask"],
      cache=cache,
      camera_embedding=camera,
      text_offsets=condition["text_offsets"],
      video_windows=condition["video_windows"],
      hide_cached_action_tails=True,
      collect_cache=True,
    )
    cache.commit(entries, condition["text_mask"], frames * height * width, frames,
                 condition["action_start"] if condition["action_start"] >= 0 else -1)
    return latent


def _decoder_weight(model_path: str) -> str:
  """Locate the pixel decoder weight, preferring the fp8 variant.

  :param model_path: Release root.
  :returns: Path of a decodable checkpoint.
  :raises FileNotFoundError: When neither variant is present.
  """
  for variant in _DECODER_VARIANTS:
    pattern = os.path.join(model_path, variant, "diffusion_pytorch_model*.safetensors")
    matches = sorted(glob.glob(pattern))
    if matches:
      if len(matches) != 1:
        raise FileNotFoundError(f"expected one decoder checkpoint in {variant}, found {matches}")
      return matches[0]
  raise FileNotFoundError(f"no PixelUMM decoder under {model_path} "
                          f"(looked for {', '.join(_DECODER_VARIANTS)})")
