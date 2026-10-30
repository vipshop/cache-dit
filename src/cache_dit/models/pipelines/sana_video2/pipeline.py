"""SANA-Video 2.0 text-to-video / text-image-to-video pipeline.

Vendored from NVlabs/Sana (Apache-2.0, Copyright 2026 NVIDIA CORPORATION & AFFILIATES) — the
released checkpoint is a merged ``state_dict`` plus a ``config.yaml`` and the official entry point
is ``inference_video_scripts/inference_sana_video.py`` in that repository, so diffusers cannot load
it. The vendored copy keeps the reference numerics:

* prompt side: ``instructions + prompt + motion suffix``, right padding, ``model_max_length=300``
  after dropping the instruction tokens, and the negative prompt encoded *without* the instruction;
* latents: ``(B, 128, (F - 1) // 8 + 1, H // 32, W // 32)`` in fp32, ``H``/``W`` divisible by 32
  and ``(F - 1) % 8 == 0``;
* VAE: the LTX-2.3 contract — ``z = (z - mean) * scaling_factor / std`` going in and
  ``z = z * std / scaling_factor + mean`` coming out;
* CFG: one batched forward per step (``cat([x, x])``), so ``has_separate_cfg`` is False;
* T2V uses the second-order flow DPM-Solver++, TI2V the flow Euler solver with a clean first frame.

Note:
  Keep the constraints below intact, cache-dit's dispatch depends on them:

  * :class:`SanaVideo2Pipeline` must inherit `DiffusionPipeline` and the transformer must be a
    `ModelMixin` + `ConfigMixin`; `vae` must stay a diffusers `AutoencoderKLLTX2Video` and
    `text_encoder` a `transformers` module, otherwise `--parallel-vae` / `--parallel-text`
    silently do nothing.
  * Caching assumes exactly one ``transformer.forward()`` per denoising step. CFG is folded into
    the batch axis, so ``guidance_scale`` must stay above 1 while caching; the guard below
    enforces it instead of silently halving the effective step count.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import yaml
from diffusers import AutoencoderKLLTX2Video, DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from transformers import AutoTokenizer, Gemma2Model

from ....logger import init_logger

logger = init_logger(__name__)

_DEFAULT_NEGATIVE_PROMPT = (
  "A chaotic sequence with misshapen, deformed limbs in every frame, distorted and inconsistent "
  "motion, blurry details, and jittery, flickering visuals.")
_MOTION_SUFFIX_TEMPLATE = " motion score: {}."
_TEXT_ENCODER_DIR_ENV = "SANA_VIDEO2_TEXT_ENCODER_DIR"
_VAE_DIR_ENV = "SANA_VIDEO2_VAE_DIR"


@dataclass
class SanaVideo2PipelineOutput:
  """Video output of :class:`SanaVideo2Pipeline`.

  Deliberately a plain dataclass, not a `diffusers.utils.BaseOutput`: that class subclasses
  `OrderedDict`, and cache-dit's CLI would then treat the result as a modular-pipeline dict and
  fail to find the `frames` field.

  :param frames: One frame sequence per generated sample, `np.ndarray` for ``output_type="np"``
    and `PIL.Image` for ``output_type="pil"``.
  :param fps: Native frame rate of the checkpoint.
  """

  frames: List[Any]
  fps: int = 24


def _resolve_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
  """Pick the transformer dtype.

  :param dtype: Requested dtype, or `None` for the checkpoint default.
  :returns: ``torch.bfloat16`` unless an explicit dtype was given.
  """
  return torch.bfloat16 if dtype is None else dtype


def _resolve_component_dir(root: Path, name: str, env_var: str) -> Path:
  """Resolve a component directory next to the checkpoint or through the environment.

  :param root: Checkpoint root directory.
  :param name: Sub-directory name used by the reference layout (``vae`` / ``text_encoder``).
  :param env_var: Environment variable that may point at the component instead.
  :returns: The resolved directory.
  """
  candidate = root / name
  if candidate.is_dir():
    return candidate
  override = os.environ.get(env_var)
  if override:
    return Path(override)
  raise FileNotFoundError(
    f"SANA-Video 2.0 needs the `{name}` component next to the checkpoint ({candidate}) or in "
    f"`{env_var}`.")


def _prepare_image(image: Image.Image, height: int, width: int) -> torch.Tensor:
  """Resize, center-crop and normalize a conditioning frame.

  :param image: Conditioning image.
  :param height: Target height.
  :param width: Target width.
  :returns: Pixels `[3, H, W]` in ``[-1, 1]``.
  """
  image = image.convert("RGB")
  image_width, image_height = image.size
  scale = max(height / image_height, width / image_width)
  resized = image.resize((round(image_width * scale), round(image_height * scale)),
                         Image.Resampling.BICUBIC)
  top = round((resized.height - height) / 2.0)
  left = round((resized.width - width) / 2.0)
  pixels = np.array(resized)[top:top + height, left:left + width].copy()
  return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255).sub_(0.5).div_(0.5)


class SanaVideo2Pipeline(DiffusionPipeline):
  """Text-to-video and text-image-to-video pipeline for SANA-Video 2.0 5B."""

  model_cpu_offload_seq = "text_encoder->transformer->vae"
  _optional_components = ["text_encoder", "tokenizer", "vae"]

  def __init__(self,
               transformer,
               text_encoder=None,
               tokenizer=None,
               vae=None,
               scheduler=None,
               instruction: str = "",
               max_sequence_length: int = 300,
               fps: int = 24):
    super().__init__()
    self.register_modules(transformer=transformer,
                          text_encoder=text_encoder,
                          tokenizer=tokenizer,
                          vae=vae)
    self.instruction = instruction
    self.max_sequence_length = max_sequence_length
    self.fps = fps
    if tokenizer is not None:
      tokenizer.padding_side = "right"

  @classmethod
  def from_pretrained(cls,
                      pretrained_model_name_or_path,
                      *args,
                      torch_dtype=None,
                      dtype=None,
                      quantization_config=None,
                      device_map=None,
                      trust_remote_code=False,
                      transformer=None,
                      text_encoder=None,
                      tokenizer=None,
                      vae=None,
                      scheduler=None,
                      vae_path=None,
                      text_encoder_path=None,
                      tokenizer_path=None,
                      local_files_only=False,
                      **kwargs) -> "SanaVideo2Pipeline":
    """Load the checkpoint root, its LTX-2.3 VAE and the Gemma-2 text encoder.

    :param pretrained_model_name_or_path: Root holding ``config.yaml`` and ``checkpoints/*.pth``.
    :param torch_dtype: Transformer / text-encoder dtype. ``dtype`` is accepted as an alias.
    :param quantization_config: Ignored; quantize through cache-dit's own flags.
    :param device_map: Unsupported; call ``.to(device)`` or use cache-dit offloading.
    :param trust_remote_code: Ignored, the code is local.
    :param transformer: Pre-built transformer, replaces the checkpoint's.
    :param text_encoder: Pre-built text encoder.
    :param tokenizer: Pre-built tokenizer.
    :param vae: Pre-built VAE.
    :param scheduler: Unused; the pipeline owns its flow solvers.
    :param vae_path: LTX-2.3 VAE directory, overriding ``<root>/vae``.
    :param text_encoder_path: Gemma-2 directory, overriding ``<root>/text_encoder``.
    :param tokenizer_path: Tokenizer directory, defaults to the text encoder directory.
    :param local_files_only: Kept for API compatibility.
    :returns: The assembled pipeline.
    """
    if args:
      raise ValueError("SanaVideo2Pipeline.from_pretrained takes keyword arguments only.")
    if device_map is not None:
      raise NotImplementedError(
        "SANA-Video 2.0 does not support `--device-map-balance`: the CLI skips `pipe.to(device)` "
        "for that flag, which would leave the whole pipeline on CPU.")
    if quantization_config is not None:
      logger.warning("Ignoring quantization_config: quantize SANA-Video 2.0 through cache-dit's "
                     "`--quantize-type` flags so the DiT blocks are discovered.")
    if trust_remote_code:
      logger.debug("trust_remote_code is not needed: the SANA-Video 2.0 code is built into "
                   "cache-dit.")
    if scheduler is not None:
      logger.warning("Ignoring the provided scheduler: the pipeline owns its flow solvers.")
    if kwargs:
      logger.warning(f"Ignoring unsupported from_pretrained kwargs: {sorted(kwargs)}.")

    root = Path(str(pretrained_model_name_or_path))
    if not root.is_dir():
      # Repo id instead of a directory: pull the two files the release actually ships. The LTX-2.3
      # VAE and the Gemma-2 encoder live in their own repos, so they still have to be point at
      # through `vae_path` / `text_encoder_path` or the two environment variables.
      from huggingface_hub import snapshot_download
      root = Path(
        snapshot_download(
          str(pretrained_model_name_or_path),
          allow_patterns=["config.yaml", "checkpoints/*"],
          local_files_only=local_files_only,
        ))
      logger.info(f"Resolved {pretrained_model_name_or_path} to {root}.")
    inference_dtype = _resolve_dtype(torch_dtype if torch_dtype is not None else dtype)
    config_path = root / "config.yaml"
    source: Dict[str, Any] = {}
    instruction = ""
    if config_path.is_file():
      with config_path.open() as handle:
        source = yaml.safe_load(handle)
      instruction = "\n".join(source.get("text_encoder", {}).get("chi_prompt", []))
    if transformer is None:
      transformer = _load_transformer(root, source, inference_dtype)
    if text_encoder is None:
      encoder_dir = Path(
        text_encoder_path) if text_encoder_path is not None else _resolve_component_dir(
          root, "text_encoder", _TEXT_ENCODER_DIR_ENV)
      text_encoder = Gemma2Model.from_pretrained(encoder_dir, dtype=inference_dtype).eval()
    if tokenizer is None:
      if tokenizer_path is not None:
        tokenizer_dir = Path(tokenizer_path)
      elif text_encoder_path is not None:
        tokenizer_dir = Path(text_encoder_path)
      elif (root / "tokenizer").is_dir():
        tokenizer_dir = root / "tokenizer"
      else:
        tokenizer_dir = _resolve_component_dir(root, "text_encoder", _TEXT_ENCODER_DIR_ENV)
      tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    if vae is None:
      vae_dir = Path(vae_path) if vae_path is not None else _resolve_component_dir(
        root, "vae", _VAE_DIR_ENV)
      vae = AutoencoderKLLTX2Video.from_pretrained(vae_dir, torch_dtype=torch.bfloat16).eval()
    for module in (transformer, text_encoder, vae):
      if module is not None:
        module.eval()
    max_sequence_length = int(source.get("text_encoder", {}).get("model_max_length", 300))
    fps = int(source.get("data", {}).get("target_fps", 24))
    return cls(
      transformer=transformer,
      text_encoder=text_encoder,
      tokenizer=tokenizer,
      vae=vae,
      instruction=instruction,
      max_sequence_length=max_sequence_length,
      fps=fps,
    )

  @property
  def _scaling_factor(self) -> float:
    return float(self.vae.config.scaling_factor)

  def _latent_device(self, generator: Optional[torch.Generator] = None) -> torch.device:
    """Pick the device the latents live on while denoising.

    `--layerwise-offload` parks the transformer weights on the host and streams them per block, so
    `transformer.device` reports CPU while the blocks still need accelerator tensors. Fall back to
    the generator's device first (it is created on the accelerator by cache-dit) and to the current
    CUDA device otherwise.

    :param generator: Optional torch generator.
    :returns: Device for latents, prompt embeddings and the decoded video.
    """
    device = getattr(self.transformer, "device", torch.device("cpu"))
    if device.type != "cpu":
      return device
    if generator is not None and generator.device.type != "cpu":
      return generator.device
    if torch.cuda.is_available():
      return torch.device("cuda", torch.cuda.current_device())
    return device

  def _latent_stats(self, dtype: torch.dtype, device: torch.device):
    mean = self.vae.latents_mean.to(device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    std = self.vae.latents_std.to(device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    return mean, std

  @torch.no_grad()
  def encode_prompt(self,
                    prompt: Union[str, List[str]],
                    negative_prompt: Optional[Union[str, List[str]]],
                    do_classifier_free_guidance: bool,
                    motion_score: int = 10):
    """Encode the prompts exactly like the reference entry point.

    :param prompt: Text prompt(s).
    :param negative_prompt: Negative prompt(s), encoded without the instruction prefix.
    :param do_classifier_free_guidance: Whether a negative branch is needed.
    :param motion_score: Motion hint appended to the prompt (>0 appends the numeric score).
    :returns: ``(prompt_embeds, prompt_mask, negative_embeds, negative_mask)``; the negative pair is
      `None` when CFG is disabled.
    """
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    suffix = _MOTION_SUFFIX_TEMPLATE.format(int(motion_score)) if motion_score > 0 else ""
    length = self.max_sequence_length
    if self.instruction:
      texts = [self.instruction + item.strip() + suffix for item in prompts]
      encoded_length = len(self.tokenizer.encode(self.instruction)) + length - 2
    else:
      texts = [item.strip() + suffix for item in prompts]
      encoded_length = length
    device = self.text_encoder.device
    encoded = self.tokenizer(
      texts,
      max_length=encoded_length,
      padding="max_length",
      truncation=True,
      return_tensors="pt",
    ).to(device)
    hidden = self.text_encoder(encoded.input_ids, encoded.attention_mask)[0]
    select_index = [0] + list(range(-length + 1, 0))
    prompt_embeds = hidden[:, select_index]
    prompt_mask = encoded.attention_mask[:, select_index]
    if not do_classifier_free_guidance:
      return prompt_embeds, prompt_mask, None, None
    negative = negative_prompt
    if negative is None:
      negative = _DEFAULT_NEGATIVE_PROMPT
    negatives = [negative] if isinstance(negative, str) else list(negative)
    negatives = negatives * len(prompts)
    negative_encoded = self.tokenizer(
      negatives,
      max_length=length,
      padding="max_length",
      truncation=True,
      return_tensors="pt",
    ).to(device)
    negative_hidden = self.text_encoder(negative_encoded.input_ids,
                                        negative_encoded.attention_mask)[0]
    return (prompt_embeds, prompt_mask, negative_hidden[:, select_index],
            negative_encoded.attention_mask[:, select_index])

  def _predict_factory(self, embeds: torch.Tensor, mask: torch.Tensor, guidance_scale: float,
                       is_ti2v: bool):

    def predict(x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
      inputs = torch.cat([x, x]) if guidance_scale > 1.0 else x
      if time.ndim == 0:
        timestep = (time * 1000).expand(inputs.shape[0])
      else:
        timestep = torch.cat([time, time]) if guidance_scale > 1.0 else time
      output = self.transformer(
        hidden_states=inputs,
        timestep=timestep,
        encoder_hidden_states=embeds,
        encoder_attention_mask=mask,
      )
      prediction = output.sample if hasattr(output, "sample") else output[0]
      if not is_ti2v:
        sigma = time.reshape((1, ) * x.ndim).to(inputs)
        prediction = (1 - sigma) * prediction + inputs
      if guidance_scale > 1.0:
        uncond, cond = prediction.chunk(2)
        prediction = uncond + guidance_scale * (cond - uncond)
      return prediction

    return predict

  @torch.no_grad()
  def __call__(self,
               prompt: Union[str, List[str]] = None,
               negative_prompt: Optional[Union[str, List[str]]] = None,
               height: int = 736,
               width: int = 1280,
               num_frames: int = 97,
               num_inference_steps: int = 30,
               guidance_scale: float = 8.0,
               flow_shift: Optional[float] = None,
               motion_score: int = 10,
               image: Optional[Union[Image.Image, Sequence[Image.Image]]] = None,
               latents: Optional[torch.Tensor] = None,
               generator: Optional[torch.Generator] = None,
               output_type: str = "np",
               **kwargs: Any) -> SanaVideo2PipelineOutput:
    """Generate a video from a text prompt, optionally conditioned on a first frame.

    :param prompt: Text prompt(s).
    :param negative_prompt: Negative prompt(s); defaults to the reference prompt.
    :param height: Output height, divisible by 32.
    :param width: Output width, divisible by 32.
    :param num_frames: Frame count, must satisfy ``(num_frames - 1) % 8 == 0``.
    :param num_inference_steps: Number of flow solver steps.
    :param guidance_scale: CFG scale; must stay above 1 while cache-dit caching is enabled.
    :param flow_shift: Flow shift of the sigma schedule; the checkpoint default is 12.0.
    :param motion_score: Motion hint appended to the prompt.
    :param image: Optional first-frame conditioning image(s); enables the TI2V path.
    :param latents: Optional initial noise of the expected shape.
    :param generator: Torch generator used for the initial noise.
    :param output_type: ``"pil"`` (list of frames) or ``"np"``.
    :returns: The generated frames.
    """
    if prompt is None:
      raise ValueError("SANA-Video 2.0 needs a prompt.")
    if height % 32 or width % 32:
      raise ValueError("SANA-Video 2.0 height and width must be divisible by 32.")
    if (num_frames - 1) % 8:
      adjusted = (num_frames - 1) // 8 * 8 + 1
      logger.warning(f"num_frames={num_frames} is not 8n+1; using {adjusted}.")
      num_frames = adjusted
    if getattr(self.transformer, "_is_cached", False) and guidance_scale <= 1.0:
      raise ValueError(
        "cache-dit caching assumes one batched conditional/unconditional forward per step; "
        "run with guidance_scale > 1 or disable caching.")
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    batch_size = len(prompts)
    device = self._latent_device(generator)
    do_cfg = guidance_scale > 1.0
    prompt_embeds, prompt_mask, negative_embeds, negative_mask = self.encode_prompt(
      prompts, negative_prompt, do_cfg, motion_score=motion_score)
    if do_cfg:
      embeds = torch.cat([negative_embeds, prompt_embeds])
      mask = torch.cat([negative_mask, prompt_mask])
    else:
      embeds, mask = prompt_embeds, prompt_mask
    embeds = embeds.to(device)
    if mask is not None:
      mask = mask.to(device)
    latent_frames = (num_frames - 1) // 8 + 1
    shape = (batch_size, self.transformer.config.in_channels, latent_frames, height // 32,
             width // 32)
    if latents is None:
      latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
    else:
      if tuple(latents.shape) != shape:
        raise ValueError(f"Expected latents with shape {shape}, got {tuple(latents.shape)}.")
      latents = latents.to(device=device, dtype=torch.float32)
    is_ti2v = image is not None
    if is_ti2v:
      images = [image] if isinstance(image, Image.Image) else list(image)
      if len(images) == 1:
        images = images * batch_size
      if len(images) != batch_size:
        raise ValueError("TI2V requires one conditioning image per prompt.")
      pixels = torch.stack([_prepare_image(item, height, width) for item in images])[:, :, None]
      pixels = pixels.to(device=self.vae.device, dtype=self.vae.dtype)
      image_latents = self.vae.encode(pixels).latent_dist.mode()
      mean, std = self._latent_stats(image_latents.dtype, image_latents.device)
      latents[:, :, :1] = ((image_latents - mean) * self._scaling_factor / std).float()
    shift = 12.0 if flow_shift is None else float(flow_shift)
    steps = max(1, int(num_inference_steps))
    predict = self._predict_factory(embeds, mask, guidance_scale, is_ti2v)
    from .sampling import sample_flow_dpm, sample_ltx_euler
    if is_ti2v:
      latents = sample_ltx_euler(predict, latents, steps, shift)
    else:
      latents = sample_flow_dpm(predict, latents, max(2, steps), shift)
    mean, std = self._latent_stats(latents.dtype, latents.device)
    decoded = latents * std / self._scaling_factor + mean
    samples = self.vae.decode(decoded.to(device=self.vae.device, dtype=self.vae.dtype),
                              temb=None,
                              return_dict=False)[0]
    # `[0, 1]` floats, the diffusers video convention: the CLI's `export_to_video` scales numpy
    # frames by 255 itself, so handing it uint8 values would overflow into rainbow chroma noise.
    videos = (samples.float() * 0.5 + 0.5).clamp_(0, 1).permute(0, 2, 3, 4, 1).cpu().numpy()
    if output_type == "np":
      frames = list(videos)
    else:
      frames = [[Image.fromarray((frame * 255.0).round().astype(np.uint8)) for frame in video]
                for video in videos]
    return SanaVideo2PipelineOutput(frames=frames, fps=self.fps)


def _load_transformer(root: Path, source: Dict[str, Any], dtype: torch.dtype):
  """Load the merged ``state_dict`` into a meta-initialized transformer, strictly.

  :param root: Checkpoint root holding ``checkpoints/*.pth``.
  :param source: Parsed ``config.yaml``; its ``model`` / ``text_encoder`` sections carry the arch.
  :param dtype: Runtime dtype of the transformer.
  :returns: The transformer in eval mode.
  """
  from .transformer import SanaVideo2Transformer3DModel

  checkpoints = sorted((root / "checkpoints").glob("*.pth"))
  if not checkpoints:
    raise FileNotFoundError(f"No `checkpoints/*.pth` under {root}.")
  if len(checkpoints) > 1:
    logger.warning(
      f"Found {len(checkpoints)} checkpoints under {root / 'checkpoints'}; loading "
      f"{checkpoints[0].name}. The 4-step distilled variant needs its own solver recipe, so move "
      "it out of this directory (or pass a root that only holds one checkpoint) if that is not "
      "what you meant.")
  logger.info(f"Loading SANA-Video 2.0 checkpoint {checkpoints[0]}.")
  state = torch.load(checkpoints[0], map_location="cpu", weights_only=True, mmap=True)
  if "state_dict" in state:
    state = state["state_dict"]
  fields = set(SanaVideo2Transformer3DModel.__init__.__code__.co_varnames)
  values: Dict[str, Any] = {}
  for section in ("model", "text_encoder"):
    for key, value in source.get(section, {}).items():
      if key in fields:
        values[key] = value
  if "pos_embed" in state:
    values["input_size"] = int(state["pos_embed"].shape[1] ** 0.5)
  logger.info(f"Building SanaVideo2Transformer3DModel with {values}.")
  with torch.device("meta"):
    transformer = SanaVideo2Transformer3DModel(**values)
  transformer.load_state_dict(state, strict=True, assign=True)
  transformer.to(dtype=dtype)
  transformer.post_load_weights()
  return transformer.eval().requires_grad_(False)


__all__ = ["SanaVideo2Pipeline", "SanaVideo2PipelineOutput"]
