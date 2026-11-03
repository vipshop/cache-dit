"""Vendored MAGI-2 Preview pipeline (text/image to video with audio).

The released ``sand-ai/MAGI-2-preview`` checkpoint is not a diffusers repository: it has no
``model_index.json``, ships its own transformer / text encoder / decoders, and its weights are a plain
``safetensors`` shard tree. This pipeline therefore owns loading end to end.

Adapted from ``vllm-omni @ 7f217a1`` ``vllm_omni/diffusion/models/magi2/pipeline_magi2.py``
(Apache-2.0, Copyright (c) 2026 SandAI). Intentional differences:

* no ``od_config`` / vLLM worker plumbing: the pipeline takes a checkpoint root and the usual
  ``from_pretrained`` kwargs, and cache-dit's CLI drives it;
* the released 1080p refiner stage is out of scope (this is the preview stage);
* auxiliary components (text encoder, VAE, TurboVAE, audio decoder) are optional so cache / CP / TP
  work can run on the transformer alone and report latent-level agreement;
* cache-dit style (2-space indent, Sphinx docstrings).
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from diffusers import DiffusionPipeline

from .configuration import MAGI2_GENERATION_CONFIG, Magi2PreviewConfig
from .data_proxy import Magi2DataProxy
from .sampler import CFGConfig, Magi2PreviewSampler, Magi2SamplerInput, build_magi2_preview_schedulers
from .transformer import Magi2PreviewTransformer
from .wan_vae import denormalize_wan22_latent

__all__ = ["Magi2PreviewPipeline", "load_transformer_weights"]

logger = logging.getLogger(__name__)

_REQUIRED_CHECKPOINT_PATHS = ("preview/model.safetensors.index.json", )


def _load_audio_decoder(root: Path, dtype):
  """Load the released Stable Audio decoder, or warn when the checkpoint does not ship it.

  :param root: Checkpoint root holding ``stable-audio-open-1.0/``.
  :param dtype: Decoder dtype.
  :returns: The decoder, or ``None`` when the checkpoint has no audio component.
  """

  from .audio_decoder import Magi2AudioDecoder

  audio_dir = Path(root) / "stable-audio-open-1.0"
  if not (audio_dir / "model.safetensors").is_file():
    logger.warning(
      "MAGI-2 audio decoder is missing from the checkpoint; the run returns video only.")
    return None
  return Magi2AudioDecoder(audio_dir, dtype=torch.float32)


def _load_text_encoder(root: Path, dtype):
  """Load the released Qwen3.5 conditioner with `transformers`, not with a vendored model.

  :param root: Checkpoint root holding ``text_encoder/``.
  :param dtype: Text-model dtype.
  :returns: The released feature extractor.
  :raises FileNotFoundError: When the conditioner directory is absent.
  """

  from .conditioning import Magi2Qwen35TextEncoder

  text_encoder_dir = Path(root) / "text_encoder"
  if not (text_encoder_dir / "config.json").is_file():
    raise FileNotFoundError(f"MAGI-2 conditioner does not exist: {text_encoder_dir}")
  return Magi2Qwen35TextEncoder(str(text_encoder_dir), dtype=dtype, device="cpu")


DEFAULT_NEGATIVE_PROMPT = (
  "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
  "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly "
  "drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy "
  "background, three legs, many people in the background, walking backwards")


def load_transformer_weights(transformer: Magi2PreviewTransformer,
                             checkpoint_root,
                             *,
                             prefix: str = "transformer."):
  """Load the released preview shards into ``transformer``.

  The published index keys are already in the module tree's own namespace
  (``block.layers.0.attention...``); some re-uploads keep a ``transformer.`` prefix, so it is stripped
  when present instead of being required.

  :param transformer: Transformer whose parameter tree matches the checkpoint keys.
  :param checkpoint_root: Directory holding ``preview/`` (the repository root).
  :param prefix: Legacy key prefix stripped when the checkpoint carries one.
  :returns: The set of loaded parameter names.
  :raises ValueError: When the index or a shard is missing, or the checkpoint is incomplete.
  """

  from safetensors import safe_open

  preview = Path(checkpoint_root) / "preview"
  index_path = preview / "model.safetensors.index.json"
  if not index_path.is_file():
    raise ValueError(f"incomplete MAGI-2 checkpoint: missing {index_path}")
  weight_map = json.loads(index_path.read_text())["weight_map"]

  loaded = set()
  for shard_name in sorted(set(weight_map.values())):
    shard_path = preview / shard_name
    if not shard_path.is_file():
      raise ValueError(f"incomplete MAGI-2 checkpoint: missing shard {shard_path}")
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
      pairs = [(name[len(prefix):] if name.startswith(prefix) else name, name)
               for name in handle.keys()]
      loaded |= transformer.load_weights(
        (stripped, handle.get_tensor(name)) for stripped, name in pairs)
  transformer.finalize_loading(loaded)
  return loaded


class Magi2PreviewPipeline(DiffusionPipeline):
  """MAGI-2 Preview text/image-to-video-and-audio pipeline.

  :param transformer: Preview denoising transformer.
  :param text_encoder: Optional Qwen3.5 feature extractor; without it the pipeline expects pre-
    computed text features.
  :param video_decoder: Optional video VAE / TurboVAE decoder.
  :param image_vae: Optional image VAE used to encode an input still.
  :param audio_decoder: Optional audio VAE decoder.
  """

  # The released conditioner is CPU-resident: Qwen3.5 runs a hand-written causal conv on a raw weight,
  # so streaming it through the offload hooks would leave that weight on the host.
  layerwise_offload_skip_components = ("text_encoder", )

  def __init__(
    self,
    transformer: Magi2PreviewTransformer = None,
    text_encoder: nn.Module = None,
    video_decoder: nn.Module = None,
    image_vae: nn.Module = None,
    audio_decoder: nn.Module = None,
  ) -> None:
    super().__init__()
    self.register_modules(
      transformer=transformer,
      text_encoder=text_encoder,
      video_decoder=video_decoder,
      image_vae=image_vae,
      audio_decoder=audio_decoder,
    )
    self.data_proxy = Magi2DataProxy()
    self.checkpoint_root = None

  @classmethod
  def from_pretrained(
    cls,
    pretrained_model_name_or_path,
    *,
    torch_dtype=None,
    dtype=None,
    quantization_config=None,
    device_map=None,
    trust_remote_code=False,
    transformer=None,
    text_encoder=None,
    video_decoder=None,
    image_vae=None,
    audio_decoder=None,
    load_auxiliary: bool = True,
    **kwargs,
  ) -> "Magi2PreviewPipeline":
    """Build the pipeline from a local checkpoint root.

    :param pretrained_model_name_or_path: Checkpoint directory holding ``preview/``.
    :param torch_dtype: Transformer dtype (``dtype`` is accepted as an alias).
    :param dtype: Alias of ``torch_dtype``.
    :param quantization_config: Rejected with a warning: cache-dit quantizes the DiT itself.
    :param device_map: Rejected; the CLI skips ``pipe.to(device)`` for balanced device maps, which
      would leave the pipeline on the CPU.
    :param trust_remote_code: Ignored; the code is local.
    :param transformer: Pre-built transformer to use instead of loading one.
    :param text_encoder: Pre-built text encoder.
    :param video_decoder: Pre-built video decoder.
    :param image_vae: Pre-built image VAE.
    :param audio_decoder: Pre-built audio decoder.
    :param load_auxiliary: Load the text encoder and decoders from the checkpoint when they are not
      injected.
    :param kwargs: Additional components are rejected.
    :returns: The pipeline.
    :raises NotImplementedError: When ``device_map`` is requested.
    :raises ValueError: When the checkpoint is incomplete.
    """

    if device_map is not None:
      raise NotImplementedError(
        "MAGI-2 does not support `device_map`: the preview transformer is 228 GiB and the CLI skips "
        "`pipe.to(device)` whenever a balanced device map is requested, which would leave the whole "
        "pipeline on the CPU. Use `--layerwise-offload` (cache-dit) or `--parallel tp` instead.")
    if quantization_config is not None:
      import warnings

      warnings.warn("MAGI-2 ignores `quantization_config`; quantize through cache-dit's own flags.",
                    stacklevel=2)
    del trust_remote_code
    if kwargs:
      names = ", ".join(sorted(kwargs))
      raise TypeError(f"unexpected MAGI-2 pipeline initialization argument(s): {names}")

    resolved_dtype = torch_dtype if torch_dtype is not None else (
      dtype if dtype is not None else torch.bfloat16)
    if isinstance(resolved_dtype, str):
      resolved_dtype = getattr(torch, resolved_dtype)

    root = Path(pretrained_model_name_or_path).expanduser()
    if not root.is_dir():
      raise ValueError(
        f"MAGI-2 needs a local checkpoint directory, got {pretrained_model_name_or_path!r}")
    missing = [
      relative for relative in _REQUIRED_CHECKPOINT_PATHS if not (root / relative).is_file()
    ]
    if missing:
      raise ValueError(f"incomplete MAGI-2 checkpoint at {root}: missing " + ", ".join(missing))

    if transformer is None:
      config = Magi2PreviewConfig(params_dtype=resolved_dtype)
      config.validate()
      transformer = Magi2PreviewTransformer(config)
      load_transformer_weights(transformer, root)
    transformer.eval().requires_grad_(False)

    components = dict(
      text_encoder=text_encoder,
      video_decoder=video_decoder,
      image_vae=image_vae,
      audio_decoder=audio_decoder,
    )
    if load_auxiliary:
      if components["text_encoder"] is None:
        components["text_encoder"] = _load_text_encoder(root, resolved_dtype)
      if components["audio_decoder"] is None:
        components["audio_decoder"] = _load_audio_decoder(root, resolved_dtype)

    pipe = cls(transformer=transformer, **components)
    pipe.checkpoint_root = str(root)
    pipe.scheduler = None
    return pipe

  def encode_prompt(self, prompt: str) -> torch.Tensor:
    """Encode one prompt with the released text encoder.

    :param prompt: Raw prompt.
    :returns: ``[1, L, D]`` text features.
    """

    if self.text_encoder is None:
      raise ValueError(
        "MAGI-2 pipeline has no text encoder; provide `text_encoder` or pass features.")
    return self.text_encoder.encode(prompt)

  @torch.inference_mode()
  def __call__(
    self,
    prompt: str = None,
    *,
    negative_prompt: str = None,
    num_inference_steps: int = 4,
    height: int = 256,
    width: int = 448,
    seed: int = 42,
    latent: torch.Tensor = None,
    audio_latent: torch.Tensor = None,
    txt_feat: torch.Tensor = None,
    null_txt_feat: torch.Tensor = None,
    latent_frames: int = None,
    audio_frames: int = None,
    decode: bool = True,
    generator: torch.Generator = None,
    image=None,
  ):
    """Generate one 10-second audio-video clip (or its latents).

    The released model only supports 10 s clips, but the latent geometry is a request parameter, which
    is what makes small development runs possible: ``latent_frames`` / ``audio_frames`` override the
    official 32 / 250 frames.

    :param prompt: Text prompt (structured JSON is normalized like the release).
    :param negative_prompt: Negative prompt; the released default is used when omitted.
    :param num_inference_steps: Denoising steps.
    :param height: Output height (multiple of the VAE spatial stride).
    :param width: Output width (multiple of the VAE spatial stride).
    :param seed: Torch seed for the initial noise.
    :param latent: Optional pre-built video noise ``[1, C, T, H, W]``.
    :param audio_latent: Optional pre-built audio noise ``[1, L, C]``.
    :param txt_feat: Optional pre-computed conditional text features.
    :param null_txt_feat: Optional pre-computed unconditional text features.
    :param latent_frames: Video latent frame count; the official timeline when omitted.
    :param audio_frames: Audio latent frame count; the official timeline when omitted.
    :param decode: Decode to pixels/audio when the decoders are present.
    :param generator: Optional torch generator for the initial noise.
    :param image: Optional conditioning still for image-to-video; not supported yet.
    :returns: Dict with ``latent`` / ``audio_latent`` and, when decoded, ``frames`` / ``audio``.
    :raises NotImplementedError: When ``image`` is given: the released image path needs the Wan2.2
        VAE encoder, which is not vendored yet.
    """

    if image is not None:
      raise NotImplementedError(
        "image-to-video needs the released Wan2.2 VAE encoder (`Magi2PreviewPipeline` currently "
        "decodes with `AutoencoderKLWan` but does not encode a conditioning still). Drop `image` to "
        "run text-to-video-and-audio.")

    generation = MAGI2_GENERATION_CONFIG
    # The pre-adapter is the always-resident compute device: under layerwise offload every layer
    # weight streams in and out, while the adapters stay on the device, so quoting the first parameter
    # of the whole transformer would report the host.
    device = self.transformer.pre_adapter.video_embedder.weight.device
    if txt_feat is None:
      txt_feat = self.encode_prompt(prompt)
    if null_txt_feat is None:
      null_txt_feat = self.encode_prompt(negative_prompt or DEFAULT_NEGATIVE_PROMPT)
    # `transformer.dtype` reports the first parameter, and the released pre/post adapters are FP32 by
    # design, so the checkpoint dtype comes from the architecture config instead.
    compute_dtype = self.transformer.config.params_dtype
    txt_feat = txt_feat.to(device=device, dtype=compute_dtype)
    null_txt_feat = null_txt_feat.to(device=device, dtype=compute_dtype)

    if latent_frames is None:
      public_frames = int(round(generation.duration_seconds * generation.fps))
      video_frames = public_frames * 2  # the DiT/decoder timeline runs at 25 fps
      latent_frames = (video_frames - 1) // generation.video_vae_stride[0] + 1
    if audio_frames is None:
      audio_frames = int(round(generation.duration_seconds * generation.audio_latent_fps))

    if latent is None:
      if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
      # Draw order is parity-critical: video noise precedes audio noise.
      latent = torch.randn(
        1,
        generation.video_latent_channels,
        latent_frames,
        height // generation.video_vae_stride[1],
        width // generation.video_vae_stride[2],
        generator=generator,
        dtype=torch.float32,
        # The CLI hands a generator that lives on the pipeline device, so the draw has to happen there.
        device=generator.device,
      ).to(device)
    if audio_latent is None:
      if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
      audio_latent = torch.randn(
        1,
        audio_frames,
        generation.audio_latent_channels,
        generator=generator,
        dtype=torch.float32,
        device=generator.device,
      ).to(device)

    # A caller-supplied noise tensor may live on the host (e.g. a CPU generator); the model runs on the
    # compute device, and the CP gather is a device collective.
    latent = latent.to(device)
    audio_latent = audio_latent.to(device)

    video_scheduler, audio_scheduler = build_magi2_preview_schedulers(
      num_inference_steps,
      device=device,
      shift=generation.shift,
    )
    sampler = Magi2PreviewSampler(self.transformer,
                                  self.data_proxy,
                                  device=device,
                                  dtype=compute_dtype)
    sampler_input = Magi2SamplerInput(
      video_t_list=video_scheduler.timesteps,
      audio_t_list=audio_scheduler.timesteps,
      latent=latent,
      audio_latent=audio_latent,
      txt_feat=txt_feat,
      null_txt_feat=null_txt_feat,
      ref_audio_feat=None,
      ref_video_feat=None,
      video_scheduler=video_scheduler,
      audio_scheduler=audio_scheduler,
      cfg_config=CFGConfig(
        video_txt_guidance_scale=generation.video_guidance_scale,
        audio_txt_guidance_scale=generation.audio_guidance_scale,
      ),
    )
    latent, audio_latent = sampler.sample(sampler_input)

    output = {"latent": latent, "audio_latent": audio_latent}
    if decode:
      decoder = self.resolve_video_decoder()
      video = decoder.decode(denormalize_wan22_latent(latent.float().to(decoder.device)))
      video = video.sample if hasattr(video, "sample") else video
      # A 4-D `[T, H, W, 3]` array: the CLI unwraps one *batch* level from a list, which would keep a
      # single frame, while `export_to_video` iterates a 4-D array frame by frame.
      output["frames"] = preview_output_to_numpy(video)
      if self.audio_decoder is not None:
        audio = self.audio_decoder.decode(audio_latent.transpose(1, 2))
        # `(batch, channels, samples)` as a tensor: the CLI unwraps one batch level and expects tensors.
        output["audio"] = audio.float().cpu()
        output["sampling_rate"] = generation.audio_sample_rate
    return output

  def resolve_video_decoder(self):
    """Return the video decoder, loading the released Wan2.2 VAE on first use.

    :returns: Decoder module with a `decode` method.
    :raises NotImplementedError: When no decoder was injected and the checkpoint has no image VAE.
    """

    if self.video_decoder is not None:
      return self.video_decoder
    if getattr(self, "_resolved_video_decoder", None) is not None:
      return self._resolved_video_decoder
    root = Path(self.checkpoint_root) if self.checkpoint_root else None
    vae_path = root / "vae" / "Wan2.2_VAE.pth" if root else None
    if vae_path is None or not vae_path.is_file():
      raise NotImplementedError(
        "MAGI-2 decoding needs the released Wan2.2 VAE (`vae/Wan2.2_VAE.pth`); run with "
        "`decode=False` to get latents, or inject `video_decoder`.")
    from .wan_vae import load_wan22_vae

    decoder = load_wan22_vae(vae_path, dtype=torch.float32)
    decoder = decoder.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    decoder.requires_grad_(False)
    self._resolved_video_decoder = decoder
    return decoder

  def set_progress_bar_config(self, **kwargs) -> None:
    """Accept the diffusers progress-bar API (the sampled loop reports through cache-dit)."""

    self._progress_bar_config = kwargs


def default_negative_prompt() -> str:
  """Return the released negative prompt.

  :returns: The prompt string.
  """

  return os.environ.get("MAGI2_NEGATIVE_PROMPT", DEFAULT_NEGATIVE_PROMPT)


def preview_output_to_numpy(video: torch.Tensor) -> np.ndarray:
  """Convert a ``[1, 3, T, H, W]`` decoder output to ``uint8`` frames.

  :param video: Decoder output in ``[-1, 1]``.
  :returns: ``[T, H, W, 3]`` uint8 array.
  """

  return np.stack([np.asarray(frame) for frame in preview_output_to_pil(video)])


def preview_output_to_pil(video: torch.Tensor) -> list:
  """Convert a ``[1, 3, T, H, W]`` decoder output to PIL frames.

  :param video: Decoder output in ``[-1, 1]``.
  :returns: List of ``PIL.Image`` frames, one per decoded frame.
  """

  from PIL import Image

  # Out-of-place throughout: `Tensor.float()` is a no-op for float32, so in-place scaling here would
  # rewrite the decoder output the caller still holds.
  frames = (video[0].float() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 3, 0)
  array = (frames * 255).byte().cpu().numpy()
  return [Image.fromarray(frame) for frame in array]
