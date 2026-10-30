import os
import time
import types
import torch
import argparse
import dataclasses
from PIL import Image
from enum import Enum
import numpy as np
from typing import Dict, Any, Union, Optional, List, Callable, ClassVar
from diffusers.utils import export_to_video, load_image
from diffusers.schedulers import SchedulerMixin
from diffusers import DiffusionPipeline, ModelMixin
from transformers import GenerationMixin
from diffusers.loaders.lora_base import LoraBaseMixin
from diffusers.quantizers import PipelineQuantizationConfig

from ..summary import summary
from ..utils import is_modular_pipeline
from ..logger import init_logger

from .utils import (
  strify,
  get_rank_device,
  maybe_destroy_distributed,
  maybe_init_distributed,
  maybe_apply_optimization,
  maybe_finalize_deferred_svdq_pipe_move,
  pipe_quant_bnb_4bit_config,
  create_profiler_from_args,
  MemoryTracker,
)

logger = init_logger(__name__)


class ExampleType(Enum):
  T2V = "T2V - Text to Video"
  T2VA = "T2VA - Text to Video with Audio"
  I2V = "I2V - Image to Video"
  T2I = "T2I - Text to Image"
  IE2I = "IE2I - Image Editing to Image"
  FLF2V = "FLF2V - First Last Frames to Video"
  VACE = "VACE - Video All-in-one Creation and Editing"
  # Same first/last-frame conditioning as FLF2V, but the pipeline also generates a soundtrack, so
  # the output parse has to go through the audio-video path instead of `output.frames`.
  FL2VA = "FL2VA - First Last Frames to Video with Audio"
  # Reference-conditioned: image / video / audio references plus a prompt.
  REF2VA = "REF2VA - Reference to Video with Audio"
  # Audio-only: the pipeline returns a waveform and no frames, so the output parse and the writer
  # take the waveform path instead of the image / video one.
  T2A = "T2A - Text to Audio"
  # Character animation: a reference character image plus a driving video produce a video.
  ANIMATE = "ANIMATE - Reference Image + Driving Video to Video"


# Pipelines of these tasks return a video stream plus an optional soundtrack. `call_pipe` (which
# asks a modular pipeline for those state keys) and `run` (which parses them) must agree on this
# set: a mismatch leaves `output` as the whole `PipelineState`, and the parse step then fails only
# after the full inference has already run.
_AUDIO_VIDEO_TASKS = frozenset({ExampleType.T2VA, ExampleType.FL2VA, ExampleType.REF2VA})
# Audio-only tasks: no frames, the whole output is a waveform.
_AUDIO_TASKS = frozenset({ExampleType.T2A})

# Video-only tasks. A modular pipeline returns the whole `PipelineState` unless it is asked for the
# output key, so `call_pipe` requests `videos` for these and `run` parses a dict or an object alike.
_VIDEO_TASKS = frozenset({
  ExampleType.T2V,
  ExampleType.I2V,
  ExampleType.FLF2V,
  ExampleType.VACE,
  ExampleType.ANIMATE,
})

# `--reference-*-path` flags and the diffusers class that loads them. Order is semantic: a pipeline
# consumes references in the order it receives them.
_REFERENCE_PATH_ARGS = (
  ("reference_image_path", "MiniMaxH3ImageReference"),
  ("reference_video_path", "MiniMaxH3VideoReference"),
  ("reference_audio_path", "MiniMaxH3AudioReference"),
)


def _references_from_args(args: argparse.Namespace) -> Optional[list]:
  """Build reference objects from the `--reference-*-path` flags.

  :param args: CLI arguments.
  :returns: The reference objects in image -> video -> audio order, or None when no flag is set.
  :raises ImportError: When the installed diffusers has no reference classes for this model.
  """
  selected = [(attr, cls_name) for attr, cls_name in _REFERENCE_PATH_ARGS
              if getattr(args, attr, None)]
  if not selected:
    return None
  import diffusers

  # The reference data classes come from the pipeline package when the installed diffusers has
  # them; an older release only exposes some of them at the top level.
  try:
    from diffusers.modular_pipelines import minimax_h3 as _h3_references
  except ImportError:  # pragma: no cover - older diffusers
    _h3_references = None

  references = []
  for attr, cls_name in selected:
    reference_cls = getattr(_h3_references, cls_name, None)
    if reference_cls is None:
      reference_cls = getattr(diffusers, cls_name, None)
    if reference_cls is None:
      raise ImportError(f"`{cls_name}` is not exposed by the installed diffusers, so "
                        f"--{attr.replace('_', '-')} cannot be used. Pass `references` "
                        "programmatically instead.")
    paths = getattr(args, attr)
    if isinstance(paths, str):
      # A programmatic caller may pass a single path; iterating it would walk the characters.
      paths = [paths]
    for path in paths:
      references.append(reference_cls.from_file(path))
  if references and all(getattr(reference, "kind", None) == "audio" for reference in references):
    # Fail here rather than after the 60 GiB conditioner has been loaded: the setup step rejects a
    # soundtrack that has no visual reference to go with it.
    raise ValueError("An audio reference has to be paired with at least one image or video "
                     "reference.")
  return references


@dataclasses.dataclass
class ExampleInputData:
  # This class provides default input data for examples.
  # The default values may be overridden by command line
  # args or other means.
  # General inputs for both image and video generation
  prompt: Optional[str] = None
  negative_prompt: Optional[str] = None
  height: Optional[int] = None
  width: Optional[int] = None
  guidance_scale: Optional[float] = None
  guidance_scale_2: Optional[float] = None  # for dual guidance scale
  true_cfg_scale: Optional[float] = None
  num_inference_steps: Optional[int] = None
  num_images_per_prompt: Optional[int] = None
  num_frames: Optional[int] = None
  # Specific inputs for image editing
  image: Optional[Union[List[Image.Image], Image.Image]] = None
  mask_image: Optional[Union[List[Image.Image], Image.Image]] = None
  # Specific inputs for first/last-frame conditioning, e.g. MiniMax-H3 `fl2va`
  last_image: Optional[Image.Image] = None
  # Skip the client-side resize of `image` / `last_image`. A pipeline that derives its canvas from
  # the source aspect ratio and crops or stretches the conditioning frame itself (MiniMax-H3's
  # resize step) would otherwise resample twice, which changes the composition it asked for.
  skip_image_resize: bool = False
  # Preserve alpha for RGBA-conditioned pipelines. Builder-only, like `skip_image_resize`.
  keep_alpha: bool = False
  # Forward `--num-inference-steps` to the pipeline. Pipelines that schedule their own passes with
  # an explicit sigma list (LTX-2.5 `LTX2DFRPipeline`) take no such argument, so they clear it.
  # Builder-only, like `skip_image_resize`: `data()` pops it before forwarding.
  use_num_inference_steps: bool = True
  # Specific inputs for video generation, e.g, Wan VACE
  video: Optional[List[Image.Image]] = None
  mask: Optional[List[Image.Image]] = None
  # Specific inputs for controlnet, e.g, Qwen-Image-ControlNet-Inpainting
  control_image: Optional[Union[List[Image.Image], Image.Image]] = None
  control_mask: Optional[Union[List[Image.Image], Image.Image]] = None
  controlnet_conditioning_scale: Optional[float] = None
  # Specific inputs for Qwen Image Layered
  layers: Optional[int] = None
  resolution: Optional[int] = None
  cfg_normalize: Optional[bool] = None
  use_en_prompt: Optional[bool] = None
  # Other inputs
  seed: int = 0
  # `None` means "the current rank's accelerator, CPU when there is none". Sampling on the
  # accelerator is not a neutral choice: torch 2.12 rewrote the CPU path of `randn` for
  # half/bf16 (pytorch/pytorch#175988), so a CPU generator samples a different latent -- and
  # therefore draws a different image -- on 2.11 and on 2.12+ with the same seed.
  gen_device: Optional[str] = None
  # Some extra args, e.g, editing model specific inputs
  extra_input_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

  # Reference media for reference-conditioned pipelines (MiniMax-H3 `ref2va`). Declared as a
  # `ClassVar` — i.e. **not** a field — on purpose: `data()` runs `dataclasses.asdict`, which deep
  # copies every value and turns a reference dataclass into a plain dict that the pipeline can no
  # longer read. Set it through `set_references`; `data()` forwards the objects unchanged.
  references: ClassVar[Any] = None
  # Set by `Example.prepare_input_data` from the task type. A workflow that does not take references
  # must not decode them either — the flags are global, and decoding a video or importing the
  # reference loaders would fail a run that never asked for them.
  allow_references: ClassVar[bool] = True

  def set_references(self, references: Any) -> "ExampleInputData":
    """Attach reference media for a reference-conditioned pipeline.

    :param references: List of pipeline-specific reference objects, e.g. `MiniMaxH3ImageReference`.
    :returns: This instance, so the call can wrap the constructor.
    """
    self.references = list(references)
    return self

  def data(self, args: argparse.Namespace) -> Dict[str, Any]:
    data = dataclasses.asdict(self)
    # `asdict` keeps falsy values, so a field that only steers this builder has to leave before the
    # `is not None` filter below: forwarding `skip_image_resize=False` to the pipeline would be a
    # `TypeError` for models that never declare it.
    skip_image_resize = data.pop("skip_image_resize", False)
    keep_alpha = data.pop("keep_alpha", False)
    use_num_inference_steps = data.pop("use_num_inference_steps", True)
    # Reference objects stay out of `asdict` on purpose (see the field comment), so they are
    # forwarded separately, after the arg overrides.
    allow_references = getattr(self, "allow_references", True)
    references = getattr(self, "references", None) if allow_references else None
    # Decoding reference media is expensive (a video is decoded frame by frame), and `data()` is
    # called twice per run — once for the inference and once for the summary. Cache by the flags so
    # the media is decoded once.
    references_from_args = None
    if allow_references:
      cache_key = tuple(tuple(getattr(args, attr, None) or ()) for attr, _ in _REFERENCE_PATH_ARGS)
      cached = getattr(self, "_references_from_args_cache", None)
      if cached is not None and cached[0] == cache_key:
        references_from_args = cached[1]
      else:
        references_from_args = _references_from_args(args)
        self._references_from_args_cache = (cache_key, references_from_args)
    # Flatten extra_args and merge into main dict
    extra_args = data.pop("extra_input_kwargs")  # {key: value, ...}
    extra_args = extra_args if extra_args is not None else {}
    # Remove None values from extra_args
    extra_data = {k: v for k, v in extra_args.items() if v is not None}
    input_data = {k: v for k, v in data.items() if v is not None}
    input_data.update(extra_data)
    # Override with CLI-provided extra_input_kwargs (takes precedence over programmatic extras)
    if getattr(args, "extra_input_kwargs", None):
      input_data.update(args.extra_input_kwargs)
    # Override with args if provided
    if args.prompt is not None:
      input_data["prompt"] = args.prompt
    if args.negative_prompt is not None:
      input_data["negative_prompt"] = args.negative_prompt
    if args.skip_negative_prompt:
      input_data.pop("negative_prompt", None)
    if args.height is not None:
      input_data["height"] = args.height
    if args.width is not None:
      input_data["width"] = args.width
    if args.num_inference_steps is not None and use_num_inference_steps:
      input_data["num_inference_steps"] = args.num_inference_steps
    if args.num_frames is not None:
      input_data["num_frames"] = args.num_frames
    # `--extra-input-kwargs` is merged above, so drop the key after every source was considered.
    if not use_num_inference_steps:
      input_data.pop("num_inference_steps", None)
    # Determine resize target: explicit --input-height/--input-width first,
    # then fall back to output height/width.
    input_height = getattr(args, "input_height", None)
    input_width = getattr(args, "input_width", None)
    resize_h = input_height if input_height is not None else input_data.get("height")
    resize_w = input_width if input_width is not None else input_data.get("width")

    def _prepare(image: Image.Image) -> Image.Image:
      # `skip_image_resize` leaves the geometry to the pipeline: MiniMax-H3 derives its canvas from
      # the source aspect ratio and stretches the first frame / cover-crops the last one itself, so
      # resizing here would resample twice and change the composition it asked for.
      if skip_image_resize or resize_h is None or resize_w is None:
        return image
      return image.resize((resize_w, resize_h))

    if skip_image_resize and (getattr(args, "mask_image_path", None) is not None
                              or input_data.get("mask_image") is not None):
      # The mask has to line up with the image pixel for pixel, so leaving one of them to the
      # pipeline and resizing the other here would silently mismatch them.
      raise ValueError("`skip_image_resize` also leaves the mask geometry to the pipeline, which "
                       "this CLI cannot express: drop the mask, or resize both here.")

    if args.image_path is not None:
      if keep_alpha:
        image = load_image(args.image_path, convert_method=lambda value: value.convert("RGBA"))
      else:
        image = load_image(args.image_path).convert("RGB")
      if isinstance(input_data.get("image"), list):
        if len(input_data["image"]) > 1:
          logger.warning("Overriding multiple input images with a single image "
                         "from args.image_path.")
        input_data["image"] = [_prepare(image)]
      else:
        input_data["image"] = _prepare(image)
    last_image_path = getattr(args, "last_image_path", None)
    if last_image_path is not None:
      # First/last-frame conditioning takes the trailing frame as its own argument (`last_image`),
      # so it must not be folded into `image`.
      input_data["last_image"] = _prepare(load_image(last_image_path).convert("RGB"))
    if args.mask_image_path is not None:
      if "mask_image" in input_data:
        if isinstance(input_data["mask_image"], list):
          if len(input_data["mask_image"]) > 1:
            logger.warning("Overriding multiple input mask images with a single mask "
                           "image from args.mask_image_path.")
      if isinstance(input_data["mask_image"], list):
        input_data["mask_image"] = [load_image(args.mask_image_path).convert("RGB")]
        if resize_h is not None and resize_w is not None:
          input_data["mask_image"] = [
            img.resize((resize_w, resize_h)) for img in input_data["mask_image"]
          ]
      else:
        input_data["mask_image"] = load_image(args.mask_image_path).convert("RGB")
        if resize_h is not None and resize_w is not None:
          input_data["mask_image"] = input_data["mask_image"].resize((resize_w, resize_h))
    # Set generator with seed from input data or args
    if args.generator_device is not None:
      self.gen_device = args.generator_device
    if args.seed is not None:
      self.seed = args.seed
    input_data["generator"] = torch.Generator(self.resolved_gen_device()).manual_seed(self.seed)
    # Remove redundant keys from input data
    input_data.pop("seed", None)
    input_data.pop("gen_device", None)
    if references_from_args:
      input_data["references"] = references_from_args
    elif references:
      input_data["references"] = references
    return input_data

  def resolved_gen_device(self) -> str:
    """Resolve the device used to build the `generator`.

    :returns: The explicit `gen_device` when one is set, otherwise the current rank's
      accelerator, falling back to CPU when it is not a CUDA device.
    """
    if self.gen_device is not None:
      return self.gen_device
    _, device = get_rank_device()
    # Only CUDA generators are verified to sample the same latents across torch versions.
    # Other backends keep the CPU behaviour they had before this default changed, instead
    # of trading one reproducibility problem for an unverified one.
    if device.type != "cuda":
      return "cpu"
    return str(device)

  def new_generator(
    self,
    args: argparse.Namespace = None,
    *,
    seed_override: Optional[int] = None,
  ) -> torch.Generator:
    # NOTE: We should always create a new generator before each inference to
    # ensure reproducibility when using the same seed. The generator lives on the
    # accelerator by default, which keeps the sampled latents stable across torch
    # versions (see the `gen_device` comment above).
    if args is not None and args.generator_device is not None:
      self.gen_device = args.generator_device
    gen_device = self.resolved_gen_device()
    if seed_override is not None:
      return torch.Generator(gen_device).manual_seed(seed_override)
    if args is not None and args.seed is not None:
      return torch.Generator(gen_device).manual_seed(args.seed)
    elif self.seed is not None:
      return torch.Generator(gen_device).manual_seed(self.seed)
    else:
      return torch.Generator(gen_device).manual_seed(0)

  def summary(self, args: argparse.Namespace) -> str:
    summary_str = "🤖 Example Input Summary:\n"
    data = self.data(args)
    for k, v in data.items():
      if k in ["prompt", "negative_prompt"]:
        summary_str += f"- {k}: {v}\n"
      elif k in ["height", "width", "num_inference_steps", "num_frames"]:
        summary_str += f"- {k}: {v}\n"
      elif k == "references":
        count = len(v) if isinstance(v, (list, tuple)) else 1
        summary_str += f"- {k}: {count} item(s)\n"
      elif k in ["image", "mask_image", "control_image", "control_mask", "last_image"]:
        if isinstance(v, Image.Image):
          W, H = v.size
          summary_str += f"- {k}: Single Image ({H}x{W})\n"
        elif isinstance(v, list):
          if len(v) > 0:
            summary_str += f"- {k}: List Images ({len(v)} images)\n"
            for i in range(min(len(v), 3)):  # show up to 3 images
              if isinstance(v[i], Image.Image):
                W, H = v[i].size
                summary_str += f"    - Image {i}: ({H}x{W})\n"
              else:
                summary_str += f"    - Image {i}: Not a valid PIL Image\n"
          elif len(v) == 1:
            if isinstance(v[0], Image.Image):
              W, H = v[0].size
              summary_str += f"- {k}: Single Image ({H}x{W})\n"
            else:
              summary_str += f"- {k}: Not a valid PIL Image\n"
          else:
            summary_str += f"- {k}: Empty List\n"
      elif k in ["video", "mask"]:
        if isinstance(v, list):
          if len(v) > 0:
            summary_str += f"- {k}: List of Frames ({len(v)} frames)\n"
            for i in range(min(len(v), 1)):  # show up to 1 frames
              if isinstance(v[i], Image.Image):
                W, H = v[i].size
                summary_str += f"    - Frame {i}: ({H}x{W})\n"
              else:
                summary_str += f"    - Frame {i}: Not a valid PIL Image\n"
          else:
            summary_str += f"- {k}: Empty List\n"
        else:
          summary_str += f"- {k}: Not a valid list of frames\n"
      elif k == "generator":
        # Show seed and device info
        if isinstance(v, torch.Generator):
          gen_device = v.device if hasattr(v, "device") else "cpu"
          gen_seed = v.initial_seed() if hasattr(v, "initial_seed") else "N/A"
          summary_str += f"- {k}: device {gen_device}, seed {gen_seed}\n"
        else:
          summary_str += f"- {k}: Not a valid torch.Generator\n"
      else:
        summary_str += f"- {k}: {v}\n"
    summary_str = summary_str.rstrip("\n")
    logger.info(summary_str)
    return summary_str


@dataclasses.dataclass
class ExampleOutputData:
  # Tag
  model_tag: Optional[str] = None
  strify_tag: Optional[str] = None
  # Generated image or video
  image: Optional[Image.Image | List[Image.Image]] = (
    None  # Single PIL Images or list of PIL Images
  )
  video: Optional[List[Image.Image]] = None  # List of PIL Images or video frames
  # Generated soundtrack, when the model produces one (e.g. joint video + audio models)
  audio: Optional[torch.Tensor] = None  # of shape (channels, num_samples)
  audio_sampling_rate: Optional[int] = None
  # Native frame rate of the model, used instead of `args.saved_fps` when set
  video_fps: Optional[float] = None
  # Performance metrics
  load_time: Optional[float] = None
  warmup_time: Optional[float] = None
  inference_time: Optional[float] = None
  memory_usage: Optional[float] = None
  # Other outputs
  extra_output_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

  def save(self, args: argparse.Namespace) -> None:
    # TODO: Handle other extra outputs as needed
    save_path = args.save_path
    if save_path is None:
      save_path = self._default_save_path()
      if save_path is None:
        logger.warning("No valid save path found for output data.")
        return

    if self.image is not None:
      if isinstance(self.image, Image.Image):
        self.image.save(save_path)
        logger.info(f"Image saved to {save_path}")
      elif isinstance(self.image, list):
        save_pre = ".".join(save_path.split(".")[:-1])
        save_ext = save_path.split(".")[-1]
        for i, img in enumerate(self.image):
          img_save_path = f"{save_pre}_{i}.{save_ext}"
          img.save(img_save_path)
          logger.info(f"Image {i} saved to {img_save_path}")

    if self.video is None and self.audio is not None:
      self._save_audio(save_path)
    elif self.video is not None:
      fps = self.video_fps if self.video_fps is not None else args.saved_fps
      # `encode_video` needs both the waveform and its rate; a model that reports only one of them
      # falls back to the video-only writer instead of failing after a long inference.
      encoder = None
      if self.audio is not None:
        if self.audio_sampling_rate is None:
          logger.warning("Audio was generated but no sampling rate was reported: writing the video "
                         "stream only.")
        else:
          try:
            from diffusers.utils.export_utils import encode_video

            encoder = encode_video
          except ImportError:
            logger.warning("This diffusers version has no `encode_video`: writing the video stream "
                           "only, the soundtrack is dropped.")
      if encoder is not None:
        try:
          encoder(
            video=self.video,
            fps=fps,
            output_path=save_path,
            audio=self.audio,
            audio_sample_rate=self.audio_sampling_rate,
          )
        except ImportError:
          # `encode_video` needs PyAV, which is not a cache-dit dependency and is only imported
          # when the muxing path actually runs.
          logger.warning("PyAV is not available to mux the soundtrack: writing the video stream "
                         "only.")
          export_to_video(self.video, save_path, fps=fps)
      else:
        export_to_video(self.video, save_path, fps=fps)
      logger.info(f"Video saved to {save_path}")

  def _save_audio(self, save_path: str) -> bool:
    """Write the generated waveform to `save_path` as a WAV file.

    `self.audio` follows the pipeline convention `(channels, samples)`, while `soundfile` expects
    `(frames, channels)`, so the waveform is transposed before it is written.

    :param save_path: Destination path, normally ending in `.wav`.
    :returns: Whether the file was written.
    """
    if self.audio_sampling_rate is None:
      logger.warning("Audio was generated but no sampling rate was reported: skipping the write.")
      return False
    try:
      import soundfile as sf
    except ImportError:
      # `soundfile` is not a cache-dit dependency; only audio models need it.
      logger.warning("`soundfile` is required to write the generated audio: skipping the write.")
      return False
    audio = self.audio
    if hasattr(audio, "detach"):
      audio = audio.detach().float().cpu().numpy()
    audio = np.ascontiguousarray(np.asarray(audio).T)
    sf.write(save_path, audio, int(self.audio_sampling_rate))
    logger.info(f"Audio saved to {save_path}")
    return True

  def _default_save_path(self) -> Optional[str]:
    if self.image is not None:
      try:
        W, H = self.image.size
        HxW_str = f"{H}x{W}"
      except Exception:
        HxW_str = None
      if HxW_str is not None:
        if HxW_str not in self.strify_tag:
          return f"{self.model_tag}.{HxW_str}.{self.strify_tag}.png"
        else:
          return f"{self.model_tag}.{self.strify_tag}.png"
      else:
        return f"{self.model_tag}.{self.strify_tag}.png"
    elif self.video is not None:
      try:
        if isinstance(self.video, (list, np.ndarray)) and len(self.video) > 0:
          if isinstance(self.video[0], Image.Image):
            W, H = self.video[0].size
          elif isinstance(self.video[0], np.ndarray):
            frame = self.video[0]  # type: np.ndarray
            H, W = frame.shape[:2]
          else:
            raise ValueError("Invalid video frame type.")
          if isinstance(self.video, list):
            num_frames = len(self.video)
          elif isinstance(self.video, np.ndarray):
            num_frames = self.video.shape[0]
          else:
            raise ValueError("Invalid video type.")
          HxW_str = f"{H}x{W}x{num_frames}"
        else:
          HxW_str = None
      except Exception:
        HxW_str = None
      if HxW_str is not None:
        if HxW_str not in self.strify_tag:
          return f"{self.model_tag}.{HxW_str}.{self.strify_tag}.mp4"
        else:
          return f"{self.model_tag}.{self.strify_tag}.mp4"
      else:
        return f"{self.model_tag}.{self.strify_tag}.mp4"
    elif self.audio is not None:
      return f"{self.model_tag}.{self.strify_tag}.wav"
    else:
      return None

  def summary(self, args: argparse.Namespace) -> str:
    from cache_dit.platforms import current_platform

    logger.info("🤖 Example Output Summary:")
    summary_str = f"- Model: {args.example}\n- Optimization: {self.strify_tag}\n"
    device_name = current_platform.get_device_name()
    world_size = (1
                  if not torch.distributed.is_initialized() else torch.distributed.get_world_size())
    summary_str += f"- Device: {device_name} x {world_size}\n"
    if self.load_time is not None:
      summary_str += f"- Load Time: {self.load_time:.2f}s\n"
    if self.warmup_time is not None:
      summary_str += f"- Warmup Time: {self.warmup_time:.2f}s\n"
    if self.inference_time is not None:
      summary_str += f"- Inference Time: {self.inference_time:.2f}s\n"
    if self.memory_usage is not None:
      summary_str += f"- Memory Usage: {self.memory_usage:.2f}GiB\n"
    summary_str = summary_str.rstrip("\n")
    logger.info(summary_str)
    return summary_str


@dataclasses.dataclass
class ExampleInitConfig:
  # This class provides default initialization config for examples.
  # The default values may be overridden by command line args or other means.
  task_type: ExampleType
  model_name_or_path: str
  pipeline_class: Optional[type[DiffusionPipeline]] = DiffusionPipeline
  torch_dtype: Optional[torch.dtype] = torch.bfloat16
  bnb_4bit_components: Optional[List[str]] = dataclasses.field(default_factory=list)
  scheduler: Optional[Union[SchedulerMixin, Callable]] = None  # lora case
  transformer: Optional[Union[ModelMixin, Callable]] = None  # lora or nunchaku case
  vae: Optional[Union[ModelMixin, Callable]] = None
  text_encoder: Optional[Union[GenerationMixin, Callable]] = None
  controlnet: Optional[Union[ModelMixin, Callable]] = None
  # Optional components to leave unloaded even when the checkpoint ships them, e.g. an LTX-2.5
  # `prompt_enhancer`: it rewrites the prompt (changing the input distribution) and dominates the
  # measured latency. Passing them as `None` is also required to skip a component whose folder is
  # absent, since `from_pretrained` otherwise falls back to the checkpoint root and fails.
  skip_components: Optional[List[str]] = None
  lora_weights_path: Optional[str] = None
  lora_weights_name: Optional[str] = None
  # For parallelism compatibility, tensor parallelism requires fused LoRA
  force_fuse_lora: bool = True
  pre_init_hook: Optional[Callable[[Any], None]] = None  # For future use
  post_init_hook: Optional[Callable[[DiffusionPipeline], None]] = None
  # For DBCache, Parallelism optimization.
  extra_optimize_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
  # For third-party models that require trust_remote_code
  trust_remote_code: bool = False
  # Preserve every image returned by multi-output pipelines instead of selecting the first.
  preserve_image_list: bool = False
  # Modular pipelines only: factory of the pipeline blocks. Needed for checkpoints whose repo has no
  # `modular_model_index.json` (e.g. Qwen-Image 2.1), so `from_pretrained` cannot resolve the blocks.
  modular_blocks_factory: Optional[Callable[[], Any]] = None
  # Modular pipelines only: workflow whose execution blocks, and therefore whose components, are
  # loaded. Repositories that hold several checkpoint partitions (MiniMax-H3 keeps `transformer/`
  # and `transformer_ref/`) would otherwise load every partition.
  modular_workflow: Optional[str] = None
  # Modular pipelines only: per-component loading dtype, e.g. `{"transformer": torch.bfloat16,
  # "default": torch.float32}` for a checkpoint whose VAEs ship float32.
  modular_component_dtypes: Optional[Dict[str, Any]] = None
  # Modular pipelines only: extra `load_components` overrides, e.g.
  # `{"pretrained_model_name_or_path": {"text_encoder": "/local/gemma-3-12b-it"}}` to load a
  # component referenced by `modular_model_index.json` from a local path (the JoyAI-Echo checkpoint
  # points its text encoder at the gated `google/gemma-3-12b-it`). Values must be per-component
  # maps: `load_components` applies a plain value to every component.
  modular_load_overrides: Optional[Dict[str, Any]] = None

  @property
  def is_modular_pipeline(self) -> bool:
    """Whether the target pipeline is a diffusers `ModularPipeline`.

    :returns: Whether `pipeline_class` is a `ModularPipeline` subclass.
    """
    try:
      from diffusers import ModularPipeline
    except ImportError:  # diffusers without modular pipelines
      return False
    return (isinstance(self.pipeline_class, type)
            and issubclass(self.pipeline_class, ModularPipeline))

  def __post_init__(self):
    if not self.bnb_4bit_components:
      self.bnb_4bit_components = ["text_encoder"]
    if self.extra_optimize_kwargs:
      # remove None values
      self.extra_optimize_kwargs = {
        k: v
        for k, v in self.extra_optimize_kwargs.items() if v is not None
      }

  def get_pipe(self, args: argparse.Namespace, **kwargs) -> DiffusionPipeline:
    if self.pipeline_class is None:
      raise ValueError("pipeline_class must be provided to get the pipeline instance.")
    model_path = self.model_name_or_path if args.model_path is None else args.model_path
    if self.is_modular_pipeline:
      # diffusers modular pipelines load the pipeline class from the model config first, then
      # load the components separately (`load_components`), instead of one-shot `from_pretrained`.
      if self.modular_blocks_factory is not None:
        # Checkpoints without `modular_model_index.json` cannot resolve their blocks in
        # `from_pretrained`, so build the pipeline from explicit blocks instead.
        pipe = self.pipeline_class(
          blocks=self.modular_blocks_factory(),
          pretrained_model_name_or_path=model_path,
        )
      else:
        pipe = self.pipeline_class.from_pretrained(
          model_path,
          trust_remote_code=self.trust_remote_code,
        )
      # `workflow` belongs to `load_components`: passing it to `from_pretrained` as well would
      # prune the blocks twice, and the pruned tree no longer declares its `_workflow_map`.
      load_kwargs: Dict[str, Any] = {
        "dtype":
        self.modular_component_dtypes if self.modular_component_dtypes else self.torch_dtype,
      }
      if self.modular_workflow is not None:
        load_kwargs["workflow"] = self.modular_workflow
      if self.modular_load_overrides:
        load_kwargs.update(self.modular_load_overrides)
      pipe.load_components(**load_kwargs)
      if self.post_init_hook is not None:
        self.post_init_hook(pipe, **kwargs)
      return pipe

    pipeline_quantization_config = self._pipeline_quantization_config(args)
    pipe = self.pipeline_class.from_pretrained(
      model_path,
      torch_dtype=self.torch_dtype,
      quantization_config=pipeline_quantization_config,
      device_map="balanced" if args.device_map_balance else None,
      trust_remote_code=self.trust_remote_code,
      **self._custom_components_kwargs(),
    )  # type: LoraBaseMixin
    if self.post_init_hook is not None:
      self.post_init_hook(pipe, **kwargs)

    # Load lora and fuse if needed
    if self.has_lora:
      assert issubclass(
        type(pipe),
        LoraBaseMixin), "Pipeline class must inherit from LoraBaseMixin to load LoRA weights."
      assert hasattr(pipe, "load_lora_weights"
                     ), "Pipeline instance must have load_lora_weights method to load LoRA weights."
      if self.lora_weights_name is None:
        # TODO: Support adapter name in the future
        pipe.load_lora_weights(self.lora_weights_path)
      else:
        pipe.load_lora_weights(self.lora_weights_path, weight_name=self.lora_weights_name)
      if not args.disable_fuse_lora and (pipeline_quantization_config is None or "transformer"
                                         not in pipeline_quantization_config.components_to_quantize
                                         or self.force_fuse_lora):
        pipe.fuse_lora()
        pipe.unload_lora_weights()
        logger.info("Fused and unloaded LoRA weights into the transformer.")
      else:
        logger.warning("Keep LoRA weights in memory since transformer is quantized.")

    return pipe

  def summary(self, args: argparse.Namespace, **kwargs) -> str:
    logger.info("🤖 Example Init Config Summary:")
    extra_model_path = kwargs.get("extra_model_path", "")
    model_name_or_path = self.model_name_or_path if args.model_path is None else args.model_path
    summary_str = "- Model: "
    if (os.path.basename(extra_model_path).lower() != os.path.basename(model_name_or_path).lower()):
      summary_str += f"\n    - {model_name_or_path}\n"
      summary_str += f"    - {extra_model_path}\n"
    else:
      summary_str += f"{model_name_or_path}\n"
    summary_str += f"- Task Type: {self.task_type.value}\n"
    summary_str += f"- Torch Dtype: {self.torch_dtype}\n"
    summary_str += f"- Warmup Seed: {getattr(args, 'warmup_seed', None)}\n"
    summary_str += f"- Warmup Prompt: {getattr(args, 'warmup_prompt', None)}\n"
    if self.lora_weights_path is not None and self.lora_weights_name is not None:
      summary_str += (
        f"- LoRA Weights: {os.path.join(self.lora_weights_path, self.lora_weights_name)}\n")
    elif self.lora_weights_path is not None:
      summary_str += f"- LoRA Path: {self.lora_weights_path}\n"
    else:
      summary_str += "- LoRA Weights: None\n"
    summary_str = summary_str.rstrip("\n")
    logger.info(summary_str)
    return summary_str

  def _custom_components_kwargs(self) -> Dict[str, Any]:
    custom_components_kwargs = {}

    custom_components_kwargs["scheduler"] = (
      self.scheduler if not _is_function_or_method(self.scheduler, ) else
      self.scheduler()  # get scheduler instance
    )
    custom_components_kwargs["transformer"] = (
      self.transformer if not _is_function_or_method(self.transformer, ) else
      self.transformer()  # get transformer instance
    )
    custom_components_kwargs["vae"] = (
      self.vae if not _is_function_or_method(self.vae, ) else self.vae()  # get vae instance
    )
    custom_components_kwargs["text_encoder"] = (
      self.text_encoder if not _is_function_or_method(self.text_encoder, ) else
      self.text_encoder()  # get text_encoder instance
    )
    custom_components_kwargs["controlnet"] = (
      self.controlnet if not _is_function_or_method(self.controlnet, ) else
      self.controlnet()  # get controlnet instance
    )
    # Remove None components
    custom_components_kwargs = {k: v for k, v in custom_components_kwargs.items() if v is not None}
    # Skipped optional components are re-added as `None`: `from_pretrained` then leaves them
    # unloaded instead of looking them up on disk.
    for name in self.skip_components or []:
      custom_components_kwargs[name] = None
    return custom_components_kwargs

  @property
  def has_lora(self) -> bool:
    return (self.lora_weights_path is not None and os.path.exists(self.lora_weights_path)
            and self.lora_weights_name is not None
            and os.path.exists(os.path.join(self.lora_weights_path, self.lora_weights_name)))

  def _pipeline_quantization_config(
      self, args: argparse.Namespace) -> Optional[PipelineQuantizationConfig]:
    if self.bnb_4bit_components is None or len(self.bnb_4bit_components) == 0:
      return None
    return pipe_quant_bnb_4bit_config(
      args=args,
      components_to_quantize=self.bnb_4bit_components,
    )


def _is_function_or_method(component: Any) -> bool:
  func_types = (
    types.FunctionType,
    types.BuiltinFunctionType,
    types.LambdaType,
  )
  excluded_module_classes = (
    SchedulerMixin,
    ModelMixin,
    GenerationMixin,
    torch.nn.Module,
  )

  is_basic_func = isinstance(component, func_types)
  is_excluded_instance = isinstance(component, excluded_module_classes)
  is_method = isinstance(
    component,
    (
      types.MethodType,
      types.ClassMethodDescriptorType,
    ),
  )
  return is_basic_func and not is_excluded_instance and not is_method


def _resolve_audio_sampling_rate(pipe: Any) -> Optional[int]:
  """Read the rate the generated waveform runs at from the pipeline's own components.

  Some pipelines return a waveform without reporting a rate (LTX-2 is one of them), which would
  make the writer drop the soundtrack. The vocoder knows the rate it produced: LTX-2 decodes audio
  latents with `audio_vae` at 16 kHz and upsamples through `vocoder` to
  `config.output_sampling_rate` (24/48 kHz), so `pipe.audio_sampling_rate` — a *mel-domain* rate —
  is deliberately not consulted.

  `vae` is consulted last: Stable Audio 3's `AutoencoderSAME` carries the rate on its own config
  (`sampling_rate`, 44.1 kHz) and has neither a vocoder nor an `audio_vae`. Image / video VAEs do
  not expose any of the three keys, so they still resolve to `None` and the writer keeps its
  existing behaviour.

  :param pipe: Pipeline instance, may be `None`.
  :returns: The output sampling rate in Hz, or `None` when no component reports one.
  """
  for name in ("vocoder", "audio_vae", "vae"):
    component = getattr(pipe, name, None)
    config = getattr(component, "config", None)
    for key in ("output_sampling_rate", "sample_rate", "sampling_rate"):
      rate = getattr(config, key, None)
      if rate is not None:
        return int(rate)
  return None


def parse_audio_video_output(
  output: Any,
  task_type: ExampleType,
  native_fps: Optional[float] = None,
  pipe: Any = None,
) -> Dict[str, Any]:
  """Parse the output of a task that returns a video stream plus an optional soundtrack.

  A modular pipeline returns the state values it was asked for, i.e. a dict, while a regular
  pipeline returns an object with attributes; both shapes are accepted so the same task type works
  on either. The video stream is read from `videos` or, for pipelines that follow the older
  `DiffusionPipeline` convention (LTX-2), from `frames`.

  :param output: Pipeline output.
  :param task_type: Task being parsed, used in the error message.
  :param native_fps: Frame rate the model generates at, kept for the writer.
  :param pipe: Pipeline instance, used to recover a missing sampling rate from its components.
  :returns: `video` / `audio` / `audio_sampling_rate` / `video_fps` keys.
  :raises ValueError: When the output carries neither `videos` nor `frames`.
  """
  if isinstance(output, dict):
    videos = output.get("videos")
    if videos is None:
      videos = output.get("frames")
    audio = output.get("audio")
    sampling_rate = output.get("sampling_rate")
  else:
    videos = getattr(output, "videos", None)
    if videos is None:
      videos = getattr(output, "frames", None)
    audio = getattr(output, "audio", None)
    sampling_rate = getattr(output, "sampling_rate", None)
  if videos is None:
    raise ValueError(f"{task_type.name} expects the pipeline to return `videos` (or `frames`), got "
                     f"{type(output).__name__}.")
  if sampling_rate is None and audio is not None:
    sampling_rate = _resolve_audio_sampling_rate(pipe)
  return {
    "video": videos[0] if isinstance(videos, (list, tuple)) else videos,
    # The pipeline returns `(batch, channels, num_samples)`; store one request's waveform.
    "audio": audio[0] if audio is not None and audio.dim() == 3 else audio,
    "audio_sampling_rate": sampling_rate if audio is not None else None,
    # Keep the native rate: the soundtrack is cut to match it.
    "video_fps": native_fps,
  }


def parse_audio_output(
  output: Any,
  task_type: ExampleType,
  pipe: Any = None,
) -> Dict[str, Any]:
  """Parse the output of an audio-only task.

  Audio pipelines return a waveform under `audios` with shape `(batch, channels, num_samples)` and
  usually do not report the rate on the output object, so it is read from the pipeline.

  :param output: Pipeline output.
  :param task_type: Task being parsed, used in the error message.
  :param pipe: Pipeline instance, used to read the sampling rate.
  :returns: `audio` / `audio_sampling_rate` keys.
  :raises ValueError: When the output carries no `audios`.
  """
  if isinstance(output, dict):
    audios = output.get("audios")
    sampling_rate = output.get("sampling_rate")
  else:
    audios = getattr(output, "audios", None)
    sampling_rate = getattr(output, "sampling_rate", None)
  if audios is None:
    raise ValueError(f"{task_type.name} expects the pipeline to return `audios`, got "
                     f"{type(output).__name__}.")
  if sampling_rate is None:
    sampling_rate = getattr(pipe, "sample_rate", None) or _resolve_audio_sampling_rate(pipe)
  # `(batch, channels, num_samples)` -> `(channels, num_samples)` for the writer.
  audio = audios[0]
  return {
    "audio": audio,
    "audio_sampling_rate": int(sampling_rate) if sampling_rate is not None else None,
  }


class Example:

  def __init__(
    self,
    args: argparse.Namespace,
    init_config: Optional[ExampleInitConfig] = None,
    input_data: Optional[ExampleInputData] = None,
  ):
    self.args = args
    self.init_config: Optional[ExampleInitConfig] = init_config
    self.input_data: Optional[ExampleInputData] = input_data
    self.output_data: Optional[ExampleOutputData] = None
    self.rank, self.device = maybe_init_distributed(self.args)

  def check_valid(self) -> bool:
    if self.args is None:
      raise ValueError("args must be provided.")
    if self.input_data is None:
      raise ValueError("input_data must be provided.")
    if self.init_config is None:
      raise ValueError("init_config must be provided.")
    return True

  def prepare_input_data(self):
    # `--reference-*-path` is a global flag, and building references decodes media, so the decision
    # is taken here — before `data()` reads the flags — instead of rejecting the result afterwards:
    # a workflow that does not take references ignores them, and one that does not even have the
    # reference loaders installed never tries to import them.
    allow_references = self.init_config.task_type is ExampleType.REF2VA
    self.input_data.allow_references = allow_references
    if not allow_references and any(
        getattr(self.args, attr, None) for attr, _ in _REFERENCE_PATH_ARGS):
      logger.warning("Ignoring reference media: %s does not take references.",
                     self.init_config.task_type.name)
    input_kwargs = self.input_data.data(self.args)
    default_num_inference_steps = input_kwargs.get("num_inference_steps", None)
    extra_optimize_kwargs = self.init_config.extra_optimize_kwargs
    extra_optimize_kwargs["default_num_inference_steps"] = default_num_inference_steps
    return input_kwargs, extra_optimize_kwargs

  def run(self) -> None:
    self.check_valid()
    # Inputs are built and validated before the model load: a bad combination should fail in a
    # second, not after 100+ GiB of weights are on the device.
    input_kwargs, extra_optimize_kwargs = self.prepare_input_data()
    start_time = time.time()
    pipe = self.init_config.get_pipe(self.args)
    load_time = time.time() - start_time

    maybe_apply_optimization(self.args, pipe, **extra_optimize_kwargs)

    pipe.set_progress_bar_config(disable=self.rank != 0)

    # track memory if needed
    memory_tracker = MemoryTracker() if self.args.track_memory else None
    if memory_tracker:
      memory_tracker.__enter__()

    # warm up
    start_time = time.time()
    for _ in range(self.args.warmup):
      warmup_kwargs = self.new_generator(dict(input_kwargs), self.args, warmup=True)
      if self.args.warmup_num_inference_steps is not None:
        warmup_kwargs["num_inference_steps"] = self.args.warmup_num_inference_steps
      _ = self.call_pipe(pipe, warmup_kwargs)
      maybe_finalize_deferred_svdq_pipe_move(pipe)
    if self.args.warmup > 0:
      warmup_time = (time.time() - start_time) / self.args.warmup
    else:
      warmup_time = None

    start_time = time.time()
    # actual inference
    model_tag = self.args.example if self.args.example is not None else "None"
    if self.args.profile:
      requested_profile_name = getattr(self.args, "profile_name", None)
      profile_name = requested_profile_name or f"{model_tag}_profile"
      profiler = create_profiler_from_args(self.args, profile_name=profile_name)
      with profiler:
        for _ in range(self.args.repeat):
          step_kwargs = self.new_generator(dict(input_kwargs), self.args)
          output = self.call_pipe(pipe, step_kwargs)
          maybe_finalize_deferred_svdq_pipe_move(pipe)
      if self.rank == 0:
        logger.info(f"Profiler traces saved to: {profiler.output_dir}/{profiler.trace_path.name}")
    else:
      for _ in range(self.args.repeat):
        step_kwargs = self.new_generator(dict(input_kwargs), self.args)
        output = self.call_pipe(pipe, step_kwargs)
        maybe_finalize_deferred_svdq_pipe_move(pipe)
    if self.args.repeat > 0:
      inference_time = (time.time() - start_time) / self.args.repeat
    else:
      inference_time = None

    if self.args.cache_summary:
      if self.rank == 0:
        summary(pipe)

    if memory_tracker:
      memory_tracker.__exit__(None, None, None)
      peak_gb = memory_tracker.report()
    else:
      peak_gb = None

    # Prepare output data
    output_data = ExampleOutputData(
      model_tag=model_tag,
      strify_tag=f"{strify(self.args, pipe)}",
      load_time=load_time,
      warmup_time=warmup_time,
      inference_time=inference_time,
      memory_usage=peak_gb,
    )

    if self.init_config.task_type in [ExampleType.T2I, ExampleType.IE2I]:
      images = output.images if hasattr(output, "images") else output
      if isinstance(images, list) and not self.init_config.preserve_image_list:
        images = images[0]
      output_data.image = images
    elif self.init_config.task_type in _AUDIO_VIDEO_TASKS:
      parsed = parse_audio_video_output(output,
                                        self.init_config.task_type,
                                        native_fps=getattr(pipe, "fps", None),
                                        pipe=pipe)
      output_data.video = parsed["video"]
      output_data.audio = parsed["audio"]
      output_data.audio_sampling_rate = parsed["audio_sampling_rate"]
      output_data.video_fps = parsed["video_fps"]
    elif self.init_config.task_type in _AUDIO_TASKS:
      parsed = parse_audio_output(output, self.init_config.task_type, pipe=pipe)
      output_data.audio = parsed["audio"]
      output_data.audio_sampling_rate = parsed["audio_sampling_rate"]
    elif self.init_config.task_type in _VIDEO_TASKS:
      # A modular pipeline asked for `output="videos"` returns the value itself, while a regular
      # pipeline hides it behind `output.frames`. Unwrap exactly one batch level, keyed on the rank
      # of the container rather than on its type, so a `list[frames]` result stays a frame list.
      if isinstance(output, dict):
        videos = output.get("videos", output)
      elif getattr(output, "videos", None) is not None:
        videos = output.videos
      else:
        videos = getattr(output, "frames", output)
      if torch.is_tensor(videos) or isinstance(videos, np.ndarray):
        if videos.ndim == 5:
          videos = videos[0]
      elif isinstance(videos, (list, tuple)):
        videos = videos[0]
      output_data.video = videos

    self.output_data = output_data

    if self.rank == 0:
      self.init_config.summary(
        self.args,
        # path for extra model, e.g., lora weights, svdq int4 weights, etc.
        extra_model_path=ExampleRegister.get_default(self.args.example, ),
      )
      self.input_data.summary(self.args)
      self.output_data.summary(self.args)
      self.output_data.save(self.args)

    maybe_destroy_distributed()

  def call_pipe(self, pipe, kwargs: Dict[str, Any]):
    """Run one inference, asking a modular pipeline for the output it should return.

    :param pipe: Pipeline instance.
    :param kwargs: Inference inputs.
    :returns: The pipeline output.
    """
    if not is_modular_pipeline(pipe):
      return pipe(**kwargs)
    # diffusers modular pipelines return the whole `PipelineState` unless `output` is given
    kwargs = dict(kwargs)
    if self.init_config.task_type in [ExampleType.T2I, ExampleType.IE2I]:
      kwargs["output"] = "images"
    elif self.init_config.task_type in _AUDIO_VIDEO_TASKS:
      # Joint video + audio models return both streams plus the rate the soundtrack runs at.
      kwargs["output"] = ["videos", "audio", "sampling_rate"]
    elif self.init_config.task_type in _VIDEO_TASKS:
      kwargs["output"] = "videos"
    return pipe(**kwargs)

  def new_generator(
    self,
    input_kwargs: Dict[str, Any],
    args: argparse.Namespace,
    *,
    warmup: bool = False,
  ) -> Dict[str, Any]:
    # NOTE: We should always create a new generator before each inference to
    # ensure reproducibility when using the same seed.
    warmup_seed = getattr(args, "warmup_seed", None) if warmup else None
    input_kwargs["generator"] = self.input_data.new_generator(
      args=args,
      seed_override=warmup_seed,
    )
    if warmup:
      warmup_prompt = getattr(args, "warmup_prompt", None)
      if warmup_prompt is not None:
        input_kwargs["prompt"] = warmup_prompt
    return input_kwargs


class ExampleRegister:
  _example_registry: Dict[str, Callable[..., Example]] = {}
  _example_registry_defaults: Dict[str, str] = {}

  @classmethod
  def register(cls, name: str, default: str = ""):

    def decorator(example_func: Callable[..., Example]):
      if name in cls._example_registry:
        raise ValueError(f"Example '{name}' is already registered.")
      cls._example_registry[name] = example_func
      cls._example_registry_defaults[name] = default
      return example_func

    return decorator

  @classmethod
  def get_example(cls, args: argparse.Namespace, name: str, **kwargs) -> Example:
    if name not in cls._example_registry:
      raise ValueError(f"Example '{name}' is not registered.")
    example_func = cls._example_registry[name]
    return example_func(args, **kwargs)

  @classmethod
  def list_examples(cls) -> List[str]:
    return list(cls._example_registry.keys())

  @classmethod
  def get_default(cls, name: str) -> str:
    return cls._example_registry_defaults.get(name, "")
