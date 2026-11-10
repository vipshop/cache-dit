"""Boogu-Image pipeline: instruction encoding, denoising and VAE decoding.

Adapted from Boogu-Image (`boogu/pipelines/boogu/pipeline_boogu.py`),
Apache-2.0. Only the inference path cache-dit runs is vendored: the local /
remote instruction rewriter, boosted orthogonal guidance, the turbo and LoRA
pipeline variants and the prompt-tuning pipeline are gone, and requesting them
raises `NotImplementedError` instead of silently degrading.

`__call__` keeps the upstream signature so callers do not break, and
`from_pretrained` assembles the components from the checkpoint subdirectories
(`transformer/`, `vae/`, `scheduler/`, `mllm/`, `processor/`) instead of going
through `model_index.json`.
"""

import inspect
import os
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from ....logger import init_logger
from .image_processor import BooguImageProcessor
from .rope import BooguImageRotaryPosEmbed
from .scheduler import FlowMatchEulerDiscreteScheduler
from .transformer_boogu import BooguImageTransformer2DModel

logger = init_logger(__name__)


def _resolve_dtypes(
  spec: Optional[Union[torch.dtype, str, Dict[str, torch.dtype]]]
) -> Dict[str, Optional[torch.dtype]]:
  """Expand a dtype spec into one entry per component.

  :param spec: A single dtype, `"auto"`/`None`, or a `{component: dtype}` map.
  :returns: Mapping with `transformer`, `vae` and `mllm` keys.
  """

  names = ("transformer", "vae", "mllm")
  if isinstance(spec, dict):
    default = spec.get("default")
    return {name: spec.get(name, default) for name in names}
  if spec == "auto":
    spec = None
  return {name: spec for name in names}


@contextmanager
def debug_timer(label: str):
  """Log the wall time of a pipeline stage at debug level.

  :param label: Human-readable stage name.
  """

  start = time.perf_counter()
  try:
    yield
  finally:
    logger.debug("[Boogu-Image] %s took %.2fs", label, time.perf_counter() - start)


@dataclass
class FMPipelineOutput(BaseOutput):
  """Output class for BooguImagePipeline.

  Args:
      images (Union[List[PIL.Image.Image], np.ndarray]):
          List of denoised PIL images of length `batch_size` or numpy array of shape
          `(batch_size, height, width, num_channels)`. Contains the generated images.
  """

  images: Union[List[PIL.Image.Image], np.ndarray]


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
  scheduler,
  num_inference_steps: Optional[int] = None,
  device: Optional[Union[str, torch.device]] = None,
  timesteps: Optional[List[int]] = None,
  **kwargs,
):
  """Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after
  the call. Handles custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

  Args:
      scheduler (`SchedulerMixin`):
          The scheduler to get timesteps from.
      num_inference_steps (`int`):
          The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
          must be `None`.
      device (`str` or `torch.device`, *optional*):
          The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
      timesteps (`List[int]`, *optional*):
          Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
          `num_inference_steps` and `sigmas` must be `None`.
      sigmas (`List[float]`, *optional*):
          Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
          `num_inference_steps` and `timesteps` must be `None`.

  Returns:
      `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
      second element is the number of inference steps.
  """
  if timesteps is not None:
    accepts_timesteps = "timesteps" in set(
      inspect.signature(scheduler.set_timesteps).parameters.keys())
    if not accepts_timesteps:
      raise ValueError(
        f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
        f" timestep schedules. Please check whether you are using the correct scheduler.")
    scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
    timesteps = scheduler.timesteps
    num_inference_steps = len(timesteps)
  else:
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    timesteps = scheduler.timesteps

  return timesteps, num_inference_steps


class BooguImagePipeline(DiffusionPipeline):
  """Boogu-Image text-to-image and instruction-based image-editing pipeline.

  The MLLM encodes the instruction (and optional reference images) into hidden states, the
  single/double-stream transformer predicts latent updates, and the VAE maps between image and
  latent space under a flow-matching scheduler with training-aligned time shifting.

  :param transformer: Boogu transformer denoiser for latent prediction.
  :param vae: Autoencoder used for latent/image encoding and decoding.
  :param scheduler: Diffusion scheduler controlling the denoising trajectory.
  :param mllm: Multimodal language model used as the instruction encoder.
  :param processor: Processor paired with the MLLM for text/image inputs.
  """

  model_cpu_offload_seq = "mllm->transformer->vae"

  def __init__(
    self,
    transformer: BooguImageTransformer2DModel,
    vae: AutoencoderKL,
    scheduler: FlowMatchEulerDiscreteScheduler,
    mllm: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
  ) -> None:
    """Initialize the Boogu-Image pipeline.

    Args:
        transformer: Boogu transformer denoiser for latent prediction.
        vae: Autoencoder used for latent/image encoding and decoding.
        scheduler: Diffusion scheduler that controls denoising steps.
        mllm: Multimodal language model used to encode instructions.
        processor: Processor paired with the MLLM for text/image inputs.
    """
    super().__init__()

    # The released checkpoint registers `Qwen3VLForConditionalGeneration`, but
    # instruction features come from hidden states, so only the inner `model`
    # is needed.
    if hasattr(mllm, "lm_head"):
      mllm = mllm.model

    self.register_modules(
      transformer=transformer,
      vae=vae,
      scheduler=scheduler,
      mllm=mllm,
      processor=processor,
    )
    self.vae_scale_factor = (2 ** (len(self.vae.config.block_out_channels) -
                                   1) if hasattr(self, "vae") and self.vae is not None else 8)
    self.image_processor = BooguImageProcessor(vae_scale_factor=self.vae_scale_factor * 2,
                                               do_resize=True)
    self.default_sample_size = 128

    self.MASK_VISION_TOKENS_FEATURE: bool = False
    self.VISION_TOKEN_IDs: List[int] = []

    # System prompts matching dataset logic (specific to this pipeline)

    self.SYSTEM_PROMPT_4_TI2I_UNIFIED = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
    self.SYSTEM_PROMPT_4_T2I_UNIFIED = "You are a helpful assistant that generates high-quality images based on user instructions. The instructions are as follows."

    self.SYSTEM_PROMPT_4_T2I = self.SYSTEM_PROMPT_4_T2I_UNIFIED
    self.SYSTEM_PROMPT_DROP = self.SYSTEM_PROMPT_4_TI2I_UNIFIED  # This is for empty negative instruction for image guidance in double guidance.
    self.SYSTEM_PROMPT_4_TI2I = self.SYSTEM_PROMPT_4_TI2I_UNIFIED
    self.SYSTEM_PROMPT_4_I2I = self.SYSTEM_PROMPT_4_TI2I_UNIFIED

    self.user_set_pipe_device = None

    self.enable_model_cpu_offload_flag = False
    self.enable_sequential_cpu_offload_flag = False
    self.enable_group_offload_flag = False

  @classmethod
  def from_pretrained(
    cls,
    pretrained_model_name_or_path: Union[str, os.PathLike],
    torch_dtype: Optional[Union[torch.dtype, Dict[str, torch.dtype]]] = None,
    quantization_config: Any = None,
    device_map: Any = None,
    trust_remote_code: Optional[bool] = None,
    transformer: Optional[BooguImageTransformer2DModel] = None,
    vae: Optional[AutoencoderKL] = None,
    scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
    mllm: Optional[Qwen3VLForConditionalGeneration] = None,
    processor: Optional[Qwen3VLProcessor] = None,
    **kwargs,
  ) -> "BooguImagePipeline":
    """Assemble the pipeline from a Boogu-Image checkpoint directory.

    The components are read from their subdirectories (`transformer/`, `vae/`,
    `scheduler/`, `mllm/`, `processor/`) rather than through `model_index.json`,
    because the released index points at remote-code module names.

    :param pretrained_model_name_or_path: Local checkpoint root.
    :param torch_dtype: Single dtype, `"auto"`, or a per-component mapping.
    :param quantization_config: Rejected with a warning; quantize through
      cache-dit's own flags instead, since the loader cannot find the blocks.
    :param device_map: Rejected; cache-dit skips `pipe.to(device)` whenever
      `--device-map-balance` is set, so accepting it would leave the pipeline on
      the CPU.
    :param trust_remote_code: Ignored; the code is local.
    :param transformer: Optional pre-built transformer.
    :param vae: Optional pre-built VAE.
    :param scheduler: Optional pre-built scheduler.
    :param mllm: Optional pre-built instruction encoder.
    :param processor: Optional pre-built processor.
    :returns: The assembled pipeline.
    """

    if device_map is not None:
      raise NotImplementedError(
        "BooguImagePipeline.from_pretrained does not implement `device_map`; "
        "load the pipeline and call `pipe.to(device)` instead.")
    if quantization_config is not None:
      logger.warning("Ignoring `quantization_config`: cache-dit quantizes the transformer "
                     "itself, and the checkpoint loader cannot see its blocks.")
    if trust_remote_code:
      logger.debug("`trust_remote_code` is ignored; the model code is local.")
    if kwargs:
      logger.debug("Ignoring unsupported from_pretrained kwargs: %s", sorted(kwargs))

    root = os.path.abspath(os.fspath(pretrained_model_name_or_path))
    dtypes = _resolve_dtypes(torch_dtype)

    if mllm is None:
      mllm = Qwen3VLForConditionalGeneration.from_pretrained(os.path.join(root, "mllm"),
                                                             dtype=dtypes["mllm"])
    if processor is None:
      processor = Qwen3VLProcessor.from_pretrained(os.path.join(root, "processor"))
    if vae is None:
      vae = AutoencoderKL.from_pretrained(os.path.join(root, "vae"), torch_dtype=dtypes["vae"])
    if scheduler is None:
      scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(root, "scheduler"))
    if transformer is None:
      transformer = BooguImageTransformer2DModel.from_pretrained(os.path.join(root, "transformer"),
                                                                 torch_dtype=dtypes["transformer"])

    transformer = transformer.eval()
    vae = vae.eval()
    mllm = mllm.eval()

    return cls(
      transformer=transformer,
      vae=vae,
      scheduler=scheduler,
      mllm=mllm,
      processor=processor,
    )

  def set_scheduler(self, scheduler):
    """Scheduler's setter."""
    assert scheduler is not None, "`scheduler` must not be None."

    # Re-register the scheduler so both the instance attribute and pipeline config stay in sync.
    self.register_modules(scheduler=scheduler)

  def set_transformer(self, transformer, device=None):
    """Transformer's setter."""
    assert transformer is not None, "`transformer` must not be None."

    # Re-register the transformer so both the instance attribute and pipeline config stay in sync.
    self.register_modules(transformer=transformer)
    logger.debug("[Setter Info]: `self.transformer` has been registered.")

    if (self.enable_model_cpu_offload_flag or self.enable_sequential_cpu_offload_flag
        or self.enable_group_offload_flag or getattr(self, "_all_hooks", None)):
      warnings.warn(
        "[Setter Warning]: `set_transformer(...)` is being called after this pipeline may have enabled "
        "device/offload hooks. Re-registering `transformer` at this point can leave stale Accelerate/"
        "Diffusers hook state. Prefer setting the transformer before enabling CPU/group offload or "
        "running inference.",
        UserWarning,
      )

    if device is not None:
      self.transformer.to(device)
      logger.debug("[Setter Info]: `self.transformer` has been moved to the requested device. "
                   f"device={device!r}.")

  def _get_max_image_pixels(
    self,
    num_images: int,
    max_input_image_pixels: Union[int, list, tuple] = 1024 * 1024,
  ):
    """Resolve the per-image pixel budget from an int or a per-count list.

    :param num_images: Reference images in this sample.
    :param max_input_image_pixels: Scalar budget or one budget per image count.
    :returns: The pixel budget to use.
    """

    if (num_images <= 0) or (not max_input_image_pixels):
      return 1024 * 1024

    if isinstance(max_input_image_pixels, (list, tuple)):
      assert len(max_input_image_pixels) >= num_images, (
        f"`len(max_input_image_pixels)` should be >= number of input images per "
        f"sample, i.e., {num_images}")
      return max_input_image_pixels[num_images - 1]

    return max_input_image_pixels

  def prepare_latents(
    self,
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: Union[torch.device, str],
    generator: Optional[torch.Generator],
    latents: Optional[torch.FloatTensor] = None,
  ) -> torch.FloatTensor:
    """Prepare the initial latents for the diffusion process.

    Args:
        batch_size: The number of images to generate.
        num_channels_latents: The number of channels in the latent space.
        height: The height of the generated image.
        width: The width of the generated image.
        dtype: The data type of the latents.
        device: The device to place the latents on.
        generator: The random number generator to use.
        latents: Optional pre-computed latents to use instead of random initialization.

    Returns:
        torch.FloatTensor: The prepared latents tensor.
    """
    height = int(height) // self.vae_scale_factor
    width = int(width) // self.vae_scale_factor

    shape = (batch_size, num_channels_latents, height, width)

    if latents is None:
      latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
      latents = latents.to(device)
    return latents

  def encode_vae(self, img: torch.FloatTensor) -> torch.FloatTensor:
    """Encode an image into the VAE latent space.

    The condition latent is sampled from the global RNG, exactly like upstream.
    Seed `torch.manual_seed` (cache-dit's CLI does) for a reproducible run.

    :param img: The input image tensor to encode.
    :returns: The encoded latent representation.
    """

    z0 = self.vae.encode(img.to(dtype=self.vae.dtype)).latent_dist.sample()
    if self.vae.config.shift_factor is not None:
      z0 = z0 - self.vae.config.shift_factor
    if self.vae.config.scaling_factor is not None:
      z0 = z0 * self.vae.config.scaling_factor
    z0 = z0.to(dtype=self.vae.dtype)
    return z0

  def preprocess_vlm_input_pil_images(
    self,
    input_pil_images: List[PIL.Image.Image],
    height: Optional[int] = None,
    width: Optional[int] = None,
    max_pixels: Optional[int] = None,
    max_side_length: Optional[int] = None,
    resize_mode: str = "default",
    crops_coords: List[Tuple[int, int, int, int]] = None,
  ) -> List[PIL.Image.Image]:
    """Resize input PIL images for VLM encoding, matching dataset behavior exactly as in
    BOOGUTrainTorchIterableTI2IDataset.preprocess_vlm_input_pil_images.

    max_pixels is an int or None; per-image selection is handled by caller before passing here.
    """

    if input_pil_images is None or len(input_pil_images) <= 0:
      return input_pil_images

    assert isinstance(input_pil_images, list), ("`input_pil_images` should be a list.")
    assert all(
      isinstance(x, PIL.Image.Image)
      for x in input_pil_images), ("`input_pil_images` should be a list of PIL.Image.Image.")

    processed_input_pil_images = []
    for image in input_pil_images:
      if crops_coords is not None:
        image = [i.crop(crops_coords) for i in image]
      height, width = self.image_processor.get_new_height_width(image, height, width, max_pixels,
                                                                max_side_length)
      processed_input_pil_images.append(
        self.image_processor.resize(image, height, width, resize_mode=resize_mode))
    return processed_input_pil_images

  def prepare_image(
    self,
    images: Union[List[List[PIL.Image.Image]], List[PIL.Image.Image]],
    batch_size: int,
    num_images_per_instruction: int,
    max_input_image_pixels: Union[int, list, tuple],
    max_side_length: int,
    device: torch.device,
    dtype: torch.dtype,
    generator=None,
  ) -> List[Optional[torch.FloatTensor]]:
    """Prepare input images for processing by encoding them into the VAE latent space.

    Args:
        images: Single image or list of images to process.
        batch_size: The number of images to generate per prompt.
        num_images_per_instruction: The number of images to generate for each prompt.
        device: The device to place the encoded latents on.
        dtype: The data type of the encoded latents.

    Returns:
        List[Optional[torch.FloatTensor]]: List of encoded latent representations for each image.
    """

    success, max_images_per_sample, wrapped_input_images = (
      self._check_and_wrap_input_images(images))

    if wrapped_input_images is not None:
      assert len(wrapped_input_images) == batch_size, (
        "`wrapped_input_images` should be List[List[PIL.Image.Image]] and the `len(wrapped_input_images)` should be equal to `batch_size`."
      )
    else:
      wrapped_input_images = [None] * batch_size

    latents = []

    for i, img in enumerate(wrapped_input_images):
      if img is not None and len(img) > 0:
        ref_latents = []
        for j, img_j in enumerate(img):
          max_pixels = self._get_max_image_pixels(
            num_images=len(img),
            max_input_image_pixels=max_input_image_pixels,
          )
          img_j = self.image_processor.preprocess(img_j,
                                                  max_pixels=max_pixels,
                                                  max_side_length=max_side_length)
          ref_latents.append(self.encode_vae(img_j.to(device=device)).squeeze(0))
      else:
        ref_latents = None

      for _ in range(num_images_per_instruction):
        latents.append(ref_latents)

    return latents

  def _check_and_wrap_input_images(
    self,
    input_images: Any,
    treat_empty_list_as_none: bool = False,
  ) -> Tuple[bool, int, Optional[Union[List[List[PIL.Image.Image]], List[List[str]]]]]:
    """
        Normalize input_images into a two-level batch structure with per-sample lists:
            - List[List[PIL.Image.Image]]  or
            - List[List[str]]              (each str is an image path)
            - Allowed per-sample "empty" markers: [] or None

        ***This function may not be actually used for singe generation tasks (i.e., [text,[image,...]] -> image),
            but it might be useful for batch generation.***

        Rules:
            - If input_images is None or []:
                return (True, 0, None)
            - If already in batch form such as [[image], [image,image], [], None] or [[str], [], [str,str], None],
              return as is (optionally convert [] -> None if treat_empty_list_as_none=True).
            - If List[PIL.Image.Image] / List[str] / List[None|PIL|str], wrap each non-None element as a single-image sample:
              e.g. [img1, img2, None] -> [[img1], [img2], None]
            - If single PIL.Image.Image / single str, wrap as [[item]]
            - Otherwise attempt to iterate and collect valid items (PIL first, else paths) into a single batch sample.

        Returns:
            (success, max_images_per_sample, wrapped_input_images)
            - success: whether input_images is successfully wrapped
            - max_images_per_sample: max number of images in any sample of the batch
            - wrapped_input_images: List[List[PIL.Image.Image]] or List[List[str]] or None
        """

    # Case 0: input is None or empty
    if input_images is None:
      return True, 0, None
    try:
      # Safely check for emptiness without assuming it is a sequence
      if hasattr(input_images, "__len__") and len(input_images) == 0:
        return True, 0, None
    except TypeError:
      # If __len__ raises, ignore here; further logic will handle it
      pass

    def is_pil_image(x: Any) -> bool:
      return isinstance(x, Image.Image)

    def is_path(x: Any) -> bool:
      return isinstance(x, str)

    def is_list_of_pil_images(x: Any) -> bool:
      return isinstance(x, list) and all(is_pil_image(i) for i in x)

    def is_list_of_paths(x: Any) -> bool:
      return isinstance(x, list) and all(is_path(i) for i in x)

    def is_list_of_list_of_pil_images(x: Any) -> bool:
      return (isinstance(x, list) and len(x) > 0 and all(is_list_of_pil_images(i) for i in x))

    def is_list_of_list_of_paths(x: Any) -> bool:
      return (isinstance(x, list) and len(x) > 0 and all(is_list_of_paths(i) for i in x))

    def is_batch_two_level_with_none(x: Any) -> bool:
      """
            Accept batch-shaped inputs where each sample is:
              - None (represents no image)
              - []   (empty sample, can be converted to None if treat_empty_list_as_none=True)
              - List[PIL.Image.Image] or List[str]
            """
      if not isinstance(x, list) or len(x) == 0:
        return False
      for sample in x:
        if sample is None:
          continue
        if isinstance(sample, list):
          if len(sample) == 0:
            continue
          # Allow mixed PIL/str but all elements must be either PIL or str
          all_pil = all(is_pil_image(i) for i in sample)
          all_str = all(is_path(i) for i in sample)
          if not (all_pil or all_str):
            return False
        else:
          # Non-list, non-None found => not batch two-level
          return False
      return True

    # Case 1: already in normalized batch form (with None/[] allowed)
    if is_batch_two_level_with_none(input_images):
      wrapped = list(input_images)  # shallow copy
      # Optionally convert empty lists to None per sample
      if treat_empty_list_as_none:
        for idx, sample in enumerate(wrapped):
          if isinstance(sample, list) and len(sample) == 0:
            wrapped[idx] = None
      max_len = 0
      for sample in wrapped:
        if isinstance(sample, list):
          max_len = max(max_len, len(sample))
      return True, max_len, wrapped

    # Case 2: List[PIL.Image.Image] -> single batch
    if is_list_of_pil_images(input_images):
      wrapped = [input_images]
      max_len = len(input_images)
      return True, max_len, wrapped

    # Case 2b: List[str] (paths) -> single batch
    if is_list_of_paths(input_images):
      wrapped = [input_images]
      max_len = len(input_images)
      return True, max_len, wrapped

    # Case 2c: Flat batch where elements can be PIL/str/None
    if isinstance(input_images, list) and all(
      (is_pil_image(x) or is_path(x) or x is None or (isinstance(x, list))) for x in input_images):
      wrapped: List[Optional[List[Any]]] = []
      max_len = 0
      for item in input_images:
        if item is None:
          wrapped.append(None)
        elif is_pil_image(item) or is_path(item):
          wrapped.append([item])
          max_len = max(max_len, 1)
        elif isinstance(item, list):
          # Clean sublist: keep only PIL or str
          pil_sub = [i for i in item if is_pil_image(i)]
          str_sub = [i for i in item if is_path(i)]
          if len(pil_sub) > 0 and len(str_sub) == 0:
            wrapped.append(pil_sub)
            max_len = max(max_len, len(pil_sub))
          elif len(str_sub) > 0 and len(pil_sub) == 0:
            wrapped.append(str_sub)
            max_len = max(max_len, len(str_sub))
          else:
            # Empty or mixed invalid -> treat as empty
            wrapped.append(None if treat_empty_list_as_none else [])
        else:
          # Unknown element -> treat as empty
          wrapped.append(None if treat_empty_list_as_none else [])
      # If all are None and we prefer None, keep as batch-level structure per spec
      return True, max_len, wrapped

    # Case 3: single PIL.Image.Image -> [[image]]
    if is_pil_image(input_images):
      wrapped = [[input_images]]
      return True, 1, wrapped

    # Case 3b: single path str -> [[path]]
    if is_path(input_images):
      wrapped = [[input_images]]
      return True, 1, wrapped

    # Case 4: other types -> try to interpret as iterable and collect images/paths as a single sample
    try:
      as_list = list(input_images)
    except TypeError:
      # Cannot iterate; normalization fails
      return False, 0, None

    pil_items = [x for x in as_list if is_pil_image(x)]
    path_items = [x for x in as_list if is_path(x)]

    if pil_items:
      # Treat all collected PIL images as one sample in a single batch
      wrapped = [pil_items]
      max_len = len(pil_items)
      return True, max_len, wrapped

    if path_items:
      # Treat all collected paths as one sample in a single batch
      wrapped = [path_items]
      max_len = len(path_items)
      return True, max_len, wrapped

    # No valid entries found
    return False, 0, None

  def _get_instruction_feature_embeds(
    self,
    instruction: Union[str, List[str]],
    input_pil_images: Optional[List[List[PIL.Image.Image]]],
    device: Optional[torch.device] = None,
    max_sequence_length: int = 256,
    truncate_instruction_sequence: bool = False,
    use_prompt_tuning_embedding: bool = False,
    max_vlm_input_pil_pixels: Optional[Union[int, List[int]]] = None,
    max_vlm_input_pil_side_length: Optional[int] = None,
    system_prompt_follows_task_type: bool = False,
    task_type: str = "ti2i",
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get interleaved instruction embeddings from VLM (self.mllm), aligned with training:

    - Build VLM inputs via processor.apply_chat_template (images + text)
    - Optionally prepend trainable prompt embeddings
    - Optionally remove vision-token features by truncation
    - Return last layer or last-N layers and the corresponding attention mask

    Args:
        instruction: The instruction or list of instructions to encode.
        input_pil_images: A list of PIL images to be included in the prompt (TI2I/I2I).
        device: The device to place the embeddings on. If None, uses the pipeline's device.
        max_sequence_length: Maximum sequence length for tokenization.
        use_prompt_tuning_embedding: Whether to prepend trainable prompt embeddings.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The instruction embeddings tensor (or list of last-N layers)
            - The attention mask tensor

    Raises:
        Warning: If the input text is truncated due to sequence length limitations.
    """
    if use_prompt_tuning_embedding:
      raise NotImplementedError("Prompt tuning is not vendored in cache-dit; the released Edit "
                                "checkpoint ships a disabled prompt-tuning configuration.")

    device = device or self._execution_device
    instruction = [instruction] if isinstance(instruction, str) else instruction
    batch_size = len(instruction)
    has_offload_strategy = (bool(getattr(self, "enable_model_cpu_offload_flag", False))
                            or bool(getattr(self, "enable_sequential_cpu_offload_flag", False))
                            or bool(getattr(self, "enable_group_offload_flag", False)))

    def _module_execution_device(module, fallback_device):
      """Return the best execution device for a possibly offloaded module."""
      hook = getattr(module, "_hf_hook", None)
      hook_device = getattr(hook, "execution_device", None)
      if hook_device is not None:
        return torch.device(hook_device)

      for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        if tensor.device.type != "meta":
          return tensor.device

      return torch.device(fallback_device)

    # Build prompts with images+text.
    # input_pil_images: Optional[List[List[PIL.Image.Image]]], outer length == batch_size,
    # inner list contains K_i images for sample i.
    prompts: List[list] = []
    processed_samples: List[Optional[List[PIL.Image.Image]]] = []

    if input_pil_images is None or len(input_pil_images) == 0:
      # No images for any sample -> pass None per sample
      processed_samples = [None for _ in range(batch_size)
                           ]  # type: List[Optional[List[PIL.Image.Image]]]
    else:
      # Validate shape: outer length must match batch_size
      assert (isinstance(input_pil_images, list) and len(input_pil_images) == batch_size), (
        "When provided, `input_pil_images` must be a List[List[PIL.Image.Image]] with len == batch size."
      )
      for imgs in input_pil_images:
        if imgs and len(imgs) > 0:
          # Determine per-sample max_pixels as in dataset logic:
          # - If max_vlm_input_pil_pixels is a list/tuple, require len >= K_i and take index K_i-1
          # - If it's an int, use it for all images in this sample
          # - If None, do not constrain by pixels
          max_pixels_i: Optional[int] = None
          if isinstance(max_vlm_input_pil_pixels, (list, tuple)):
            assert len(max_vlm_input_pil_pixels) >= len(imgs), (
              "`max_vlm_input_pil_pixels` length must be >= number of images in each sample")
            max_pixels_i = int(max_vlm_input_pil_pixels[len(imgs) - 1])
          elif isinstance(max_vlm_input_pil_pixels, int):
            max_pixels_i = max_vlm_input_pil_pixels
          else:
            max_pixels_i = None
          proc = self.preprocess_vlm_input_pil_images(
            imgs,  # List[PIL.Image.Image] for this sample
            max_pixels=max_pixels_i,
            max_side_length=max_vlm_input_pil_side_length,
          )
          processed_samples.append(proc)
        else:
          # Empty inner list -> treat as no images for this sample
          processed_samples.append(None)

    # Build the batched prompts; for each sample i, pass instruction[i] and its image list (or None)
    for i in range(batch_size):
      sample_imgs: Optional[List[PIL.Image.Image]] = None
      if processed_samples and i < len(processed_samples):
        sample_imgs = processed_samples[i]
      # _apply_chat_template expects (instruction: str, input_pil_images: Optional[List[PIL.Image.Image]])
      prompts.append(
        self._apply_chat_template(
          instruction[i],
          sample_imgs,
          system_prompt_follows_task_type=system_prompt_follows_task_type,
          task_type=task_type,
        ))

    # Processor produces dict with 'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'
    with debug_timer("VLM processor (tokenize+image)"):
      # Note: padding/truncation/return_tensors/etc. are processor-level
      # kwargs and must go inside `processor_kwargs`. Passing them as
      # top-level **kwargs triggers a transformers warning:
      #   "Kwargs passed to `processor.__call__` have to be in
      #    `processor_kwargs` dict, not in `**kwargs`"
      # because those keys are not Jinja2 template variables.
      # Old (warning-triggering)写法:
      # vlm_inputs = self.processor.apply_chat_template(
      #     prompts,
      #     padding="longest",
      #     max_length=max_sequence_length,
      #     truncation=truncate_instruction_sequence,
      #     padding_side="right",
      #     return_tensors="pt",
      #     tokenize=True,
      #     return_dict=True,
      # )
      # return_dict is a top-level arg of apply_chat_template (controls return format).
      # return_tensors is also accepted at top-level and forwarded to processor_kwargs.
      # Only padding/truncation/max_length/etc. are pure processor kwargs.
      vlm_inputs = self.processor.apply_chat_template(
        prompts,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
          "padding": "longest",
          "max_length": max_sequence_length,
          "truncation": truncate_instruction_sequence,
          "padding_side": "right",
        },
      )
    _pv = vlm_inputs.get("pixel_values")
    _iids = vlm_inputs.get("input_ids")
    logger.debug(f"VLM inputs: pixel_values={_pv.shape if _pv is not None else 'None'} "
                 f"input_ids={_iids.shape if _iids is not None else 'None'} "
                 f"grid_thw={vlm_inputs.get('image_grid_thw', 'None')}")
    move_vlm_inputs_to_device = not (use_prompt_tuning_embedding and has_offload_strategy)
    for k in vlm_inputs.keys():
      if isinstance(vlm_inputs[k], torch.Tensor) and move_vlm_inputs_to_device:
        vlm_inputs[k] = vlm_inputs[k].to(device)

    input_ids = vlm_inputs["input_ids"]
    instruction_mask = vlm_inputs["attention_mask"]

    num_instruction_feature_layers = (self.transformer.instruction_feature_configs.get(
      "num_instruction_feature_layers", 1))
    final_instruction_mask = instruction_mask

    try:
      _lm_p = next(self.mllm.language_model.parameters())
      logger.debug(
        f"MLLM forward before: lm_device={_lm_p.device} has_offload={has_offload_strategy}")
    except Exception as _e:
      logger.debug(f"MLLM device check failed: {_e}")

    with torch.no_grad():
      if num_instruction_feature_layers > 1:
        text_encoder_outputs = self.mllm(**vlm_inputs, output_hidden_states=True, return_dict=True)
        all_hidden_states = (text_encoder_outputs.hidden_states
                             )  # Tuple of [B, extended_seq_len, text_hidden_dim]
        instruction_feats = list(all_hidden_states)[
          -num_instruction_feature_layers:]  # Convert to list for model processing
      else:
        try:
          instruction_feats = self.mllm(**vlm_inputs, output_hidden_states=False).last_hidden_state
        except Exception as e:
          text_encoder_outputs = self.mllm(**vlm_inputs,
                                           output_hidden_states=True,
                                           return_dict=True)

          # Get all hidden states from all layers
          all_hidden_states = (text_encoder_outputs.hidden_states
                               )  # Tuple of [B, extended_seq_len, text_hidden_dim]

          # Get last layer's feature for model processing
          instruction_feats = all_hidden_states[-1]

          # #################verbose exception############
          # print("Exception Type:", repr(e))
          # print("Exception:", str(e))
          # traceback.print_exc()
          # ########################################
          warnings.warn(f"{type(e).__name__}: {e}", UserWarning)

    logger.debug("✅ No prompt tuning: use the original instruction features")
    logger.debug(f"✅ Extracted {num_instruction_feature_layers} layers of instruction features")
    logger.debug(
      f"✅ Each layer shape: {instruction_feats[0].shape if isinstance(instruction_feats, list) else instruction_feats.shape}"
    )

    # Optionally remove vision-token features by truncation
    if (self.MASK_VISION_TOKENS_FEATURE and (self.VISION_TOKEN_IDs is not None)
        and len(self.VISION_TOKEN_IDs) > 0):
      mask_device = input_ids.device
      vision_ids = torch.as_tensor(self.VISION_TOKEN_IDs, device=mask_device, dtype=input_ids.dtype)
      vision_mask_core = torch.isin(input_ids, vision_ids)  # [B, L_core]
      keep_core_mask = instruction_mask.to(dtype=torch.bool) & (~vision_mask_core)  # [B, L_core]
      if use_prompt_tuning_embedding:
        prefix_keep = torch.ones(batch_size,
                                 num_prompt_tokens,
                                 dtype=torch.bool,
                                 device=mask_device)
        keep_mask = torch.cat([prefix_keep, keep_core_mask], dim=1)
      else:
        keep_mask = keep_core_mask
      kept_lengths = keep_mask.sum(dim=1)
      max_kept_len = (int(kept_lengths.max().item()) if kept_lengths.numel() > 0 else 0)

      def compress_features(feats: torch.Tensor, keep_m: torch.Tensor,
                            max_len: int) -> torch.Tensor:
        keep_m = keep_m.to(feats.device)
        B, L, D = feats.shape
        out = feats.new_zeros((B, max_len, D))
        for b in range(B):
          idx = torch.nonzero(keep_m[b], as_tuple=False).squeeze(-1)
          if idx.numel() > 0:
            cur = feats[b].index_select(dim=0, index=idx)
            out[b, :idx.numel()] = cur
        return out

      new_mask = final_instruction_mask.new_zeros((batch_size, max_kept_len))
      for b in range(batch_size):
        kept_len_b = int(kept_lengths[b].item())
        if kept_len_b > 0:
          new_mask[b, :kept_len_b] = 1
      if isinstance(instruction_feats, list):
        instruction_feats = [
          compress_features(feat, keep_mask, max_kept_len) for feat in instruction_feats
        ]
      else:
        instruction_feats = compress_features(instruction_feats, keep_mask, max_kept_len)
      final_instruction_mask = new_mask

    if self.mllm is not None:
      dtype = self.mllm.dtype
    elif self.transformer is not None:
      dtype = self.transformer.dtype
    else:
      dtype = None

    if isinstance(instruction_feats, (list, tuple)):
      final_instruction_feats = [feat.to(dtype=dtype, device=device) for feat in instruction_feats]
    else:
      final_instruction_feats = instruction_feats.to(dtype=dtype, device=device)
    # Keep the attention mask on the same execution device as the features
    # before passing both into the diffusion transformer.
    final_instruction_mask = final_instruction_mask.to(device=device)

    return final_instruction_feats, final_instruction_mask

  def _apply_chat_template(
    self,
    instruction: str,
    input_pil_images: Optional[List[PIL.Image.Image]] = None,
    system_prompt_follows_task_type: bool = False,
    task_type: str = "ti2i",
  ):
    """Build chat template content with interleaved text and images.

    If `system_prompt_follows_task_type` is True, the system prompt will be selected based on the task type.
    If `system_prompt_follows_task_type` is False, the system prompt will be selected based on the input images.
    Returns the prompt structure (list of messages with typed contents).
    """
    user_text_content = [{"type": "text", "text": instruction}]

    if system_prompt_follows_task_type:
      if task_type.lower() == "t2i":
        system_prompt = self.SYSTEM_PROMPT_4_T2I
      else:
        system_prompt = self.SYSTEM_PROMPT_4_TI2I
    else:
      # Pick system prompt adaptively based on the input images and instruction.
      if input_pil_images is None or len(input_pil_images) == 0:
        if instruction is None or len(instruction.strip()) == 0:
          system_prompt = self.SYSTEM_PROMPT_DROP
        else:
          system_prompt = self.SYSTEM_PROMPT_4_T2I
      else:
        if instruction is None or len(instruction.strip()) == 0:
          system_prompt = self.SYSTEM_PROMPT_4_I2I
        else:
          system_prompt = self.SYSTEM_PROMPT_4_TI2I

    system_role = {
      "role": "system",
      "content": [{
        "type": "text",
        "text": system_prompt
      }],
    }
    if input_pil_images is None or len(input_pil_images) == 0:
      prompt = [system_role, {"role": "user", "content": user_text_content}]
    else:
      images_content = [{"type": "image", "image": pil_img} for pil_img in input_pil_images]
      prompt = [
        system_role,
        {
          "role": "user",
          "content": images_content + user_text_content
        },
      ]
    return prompt

  def _reshape_embeds_and_mask(self, embeds, mask, num_images_per_instruction):
    """To duplicate text embeddings and attention mask for each generation per instruction, using
    mps friendly method."""
    if isinstance(embeds, (list, tuple)):
      batch_size, seq_len, _ = embeds[0].shape
      reshaped_embeds = []
      for embed in embeds:
        embed = embed.repeat(1, num_images_per_instruction, 1)
        reshaped_embeds.append(embed.view(batch_size * num_images_per_instruction, seq_len, -1))
    else:
      batch_size, seq_len, _ = embeds.shape
      embeds = embeds.repeat(1, num_images_per_instruction, 1)
      reshaped_embeds = embeds.view(batch_size * num_images_per_instruction, seq_len, -1)

    # `embeds` are expanded as `[b0, b0, b1, b1, ...]`, so the mask must use the
    # same interleaved order (a plain `repeat` would order `[b0, b1, b0, b1]`).
    mask = mask.repeat_interleave(num_images_per_instruction, dim=0)
    reshaped_mask = mask.view(batch_size * num_images_per_instruction, -1)

    return batch_size, seq_len, reshaped_embeds, reshaped_mask

  def encode_instruction(
    self,
    instruction: Union[str, List[str]],
    do_classifier_free_guidance: bool = True,
    negative_instruction: Optional[Union[str, List[str]]] = None,
    input_images: Optional[Union[List[List[PIL.Image.Image]], List[PIL.Image.Image]]] = None,
    use_input_images_4_neg_instruct: bool = False,
    use_input_images_4_empty_instruct: bool = False,
    max_vlm_input_pil_pixels: Optional[Union[int, List[int]]] = 384 * 384,
    max_vlm_input_pil_side_length: Optional[int] = 384 * 2,
    num_images_per_instruction: int = 1,
    device: Optional[torch.device] = None,
    instruction_embeds: Optional[torch.Tensor] = None,
    negative_instruction_embeds: Optional[torch.Tensor] = None,
    instruction_attention_mask: Optional[torch.Tensor] = None,
    negative_instruction_attention_mask: Optional[torch.Tensor] = None,
    # For double guidance
    empty_instruction: Optional[Union[str, List[str]]] = " ",
    empty_instruction_embeds: Optional[torch.Tensor] = None,
    empty_instruction_attention_mask: Optional[torch.Tensor] = None,
    use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide: bool = False,
    use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide: bool = False,
    max_sequence_length: int = 256,
    truncate_instruction_sequence: bool = False,
    use_rewrite_text_instruction: bool = False,
    system_prompt_follows_task_type: bool = False,
    task_type: str = "ti2i",
    input_image_paths: Optional[Union[List[List[str]], List[str]]] = None,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode the instruction into the Qwen3-VL hidden states the DiT consumes.

    Pre-computed embeddings are honoured when supplied. The negative / empty variants are only
    encoded when the matching guidance mode is enabled, and the instruction rewriter is not vendored
    (requesting it raises).

    :returns:`(instruction_embeds, instruction_mask, negative_embeds, negative_mask)`.
    """
    device = device or self._execution_device

    instruction = [instruction] if isinstance(instruction, str) else instruction
    # Chat template with images is handled inside _get_instruction_feature_embeds
    batch_size = len(instruction)

    if use_rewrite_text_instruction:
      raise NotImplementedError(
        "The instruction rewriter is not vendored in cache-dit; rewrite the "
        "instruction beforehand and pass the result as `instruction`.")

    if instruction_embeds is None:
      with debug_timer("VLM _get_instruction_feature_embeds (positive)"):
        instruction_embeds, instruction_attention_mask = (self._get_instruction_feature_embeds(
          instruction=instruction,
          input_pil_images=input_images,
          device=device,
          max_sequence_length=max_sequence_length,
          truncate_instruction_sequence=truncate_instruction_sequence,
          use_prompt_tuning_embedding=False,
          max_vlm_input_pil_pixels=max_vlm_input_pil_pixels,
          max_vlm_input_pil_side_length=max_vlm_input_pil_side_length,
          system_prompt_follows_task_type=system_prompt_follows_task_type,
          task_type=task_type,
        ))

    batch_size, seq_len, _ = instruction_embeds.shape
    # # duplicate text embeddings and attention mask for each generation per instruction, using mps friendly method

    batch_size, seq_len, instruction_embeds, instruction_attention_mask = (
      self._reshape_embeds_and_mask(
        instruction_embeds,
        instruction_attention_mask,
        num_images_per_instruction,
      ))

    # Get negative embeddings for classifier free guidance
    if do_classifier_free_guidance and negative_instruction_embeds is None:
      negative_instruction = (negative_instruction if negative_instruction is not None else "")

      # Normalize str to list
      negative_instruction = (batch_size * [negative_instruction] if isinstance(
        negative_instruction, str) else negative_instruction)

      if instruction is not None and type(instruction) is not type(negative_instruction):
        raise TypeError(
          f"`negative_instruction` should be the same type to `instruction`, but got {type(negative_instruction)} !="
          f" {type(instruction)}.")
      # elif isinstance(negative_instruction, str): # not needed since negative_instruction is already a list

      elif batch_size != len(negative_instruction):
        raise ValueError(
          f"`negative_instruction`: {negative_instruction} has batch size {len(negative_instruction)}, but `instruction`:"
          f" {instruction} has batch size {batch_size}. Please make sure that passed `negative_instruction` matches"
          " the batch size of `instruction`.")
      negative_instruction_embeds, negative_instruction_attention_mask = (
        self._get_instruction_feature_embeds(
          instruction=negative_instruction,
          input_pil_images=input_images if use_input_images_4_neg_instruct else None,
          device=device,
          max_sequence_length=max_sequence_length,
          truncate_instruction_sequence=truncate_instruction_sequence,
          use_prompt_tuning_embedding=False,
          max_vlm_input_pil_pixels=max_vlm_input_pil_pixels
          if use_input_images_4_neg_instruct else None,
          max_vlm_input_pil_side_length=max_vlm_input_pil_side_length
          if use_input_images_4_neg_instruct else None,
          system_prompt_follows_task_type=system_prompt_follows_task_type,
          task_type=task_type,
        ))

      # batch_size, seq_len, _ = negative_instruction_embeds.shape
      # # duplicate text embeddings and attention mask for each generation per instruction, using mps friendly method
      #     batch_size * num_images_per_instruction, -1
      # )

      (
        batch_size,
        seq_len,
        negative_instruction_embeds,
        negative_instruction_attention_mask,
      ) = self._reshape_embeds_and_mask(
        negative_instruction_embeds,
        negative_instruction_attention_mask,
        num_images_per_instruction,
      )

    if (use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide
        or use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide):
      if do_classifier_free_guidance and (empty_instruction_embeds is None):
        empty_instruction = (empty_instruction if empty_instruction is not None else [" "] *
                             batch_size)

        empty_instruction = (batch_size * [empty_instruction]
                             if isinstance(empty_instruction, str) else empty_instruction)

        if instruction is not None and type(instruction) is not type(empty_instruction):
          raise TypeError(
            f"`empty_instruction` should be the same type as `instruction`, but got {type(empty_instruction)} !="
            f" {type(instruction)}.")

        elif batch_size != len(empty_instruction):
          raise ValueError(
            f"`empty_instruction`: {empty_instruction} has batch size {len(empty_instruction)}, but `instruction`:"
            f" {instruction} has batch size {batch_size}. Please make sure that passed `empty_instruction` matches"
            " the batch size of `instruction`.")

        empty_instruction_embeds, empty_instruction_attention_mask = (
          self._get_instruction_feature_embeds(
            instruction=empty_instruction,
            input_pil_images=input_images if use_input_images_4_empty_instruct else None,
            device=device,
            max_sequence_length=max_sequence_length,
            truncate_instruction_sequence=truncate_instruction_sequence,
            use_prompt_tuning_embedding=self.prompt_embedding is not None,
            max_vlm_input_pil_pixels=max_vlm_input_pil_pixels
            if use_input_images_4_empty_instruct else None,
            max_vlm_input_pil_side_length=max_vlm_input_pil_side_length
            if use_input_images_4_empty_instruct else None,
            system_prompt_follows_task_type=system_prompt_follows_task_type,
            task_type=task_type,
          ))
        (
          batch_size,
          seq_len,
          empty_instruction_embeds,
          empty_instruction_attention_mask,
        ) = self._reshape_embeds_and_mask(
          empty_instruction_embeds,
          empty_instruction_attention_mask,
          num_images_per_instruction,
        )

    return (
      instruction_embeds,
      instruction_attention_mask,
      negative_instruction_embeds,
      negative_instruction_attention_mask,
      empty_instruction_embeds,
      empty_instruction_attention_mask,
    )

  @property
  def num_timesteps(self):
    return self._num_timesteps

  @property
  def text_guidance_scale(self):
    return self._text_guidance_scale

  @property
  def image_guidance_scale(self):
    return self._image_guidance_scale

  @property
  def empty_instruction_guidance_scale(self):
    return self._empty_instruction_guidance_scale

  @property
  def cfg_range(self):
    return self._cfg_range

  @torch.no_grad()
  def __call__(
    self,
    instruction: Optional[Union[str, List[str]]] = None,
    negative_instruction: Optional[Union[str, List[str]]] = None,
    instruction_embeds: Optional[torch.FloatTensor] = None,
    negative_instruction_embeds: Optional[torch.FloatTensor] = None,
    instruction_attention_mask: Optional[torch.LongTensor] = None,
    negative_instruction_attention_mask: Optional[torch.LongTensor] = None,
    # For double guidance
    empty_instruction: Optional[Union[str, List[str]]] = " ",
    empty_instruction_embeds: Optional[torch.Tensor] = None,
    empty_instruction_attention_mask: Optional[torch.Tensor] = None,
    use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide: bool = False,
    use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide: bool = False,
    max_sequence_length: int = 1280,
    truncate_instruction_sequence: bool = False,
    callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
    input_images: Optional[Union[List[List[PIL.Image.Image]], List[PIL.Image.Image]]] = None,
    use_input_images_4_neg_instruct: bool = False,
    use_input_images_4_empty_instruct: bool = False,
    max_vlm_input_pil_pixels: Optional[Union[int, List[int]]] = 384 * 384,
    max_vlm_input_pil_side_length: Optional[int] = 384 * 2,
    num_images_per_instruction: int = 1,
    height: Optional[int] = None,
    width: Optional[int] = None,
    max_input_image_pixels: Union[int, list, tuple] = 2048 * 2048,
    max_input_image_side_length: int = 2048 * 2,
    align_res: bool = True,
    num_inference_steps: int = 50,
    text_guidance_scale: float = 4.0,
    image_guidance_scale: float = 1.0,
    empty_instruction_guidance_scale: float = 0.0,
    cfg_range: Tuple[float, float] = (0.0, 1.0),
    use_rewrite_text_instruction: bool = False,
    rewriter_max_new_tokens: int = 512,
    resize_rewriter_ref_images: bool = True,
    rewriter_ref_images_max_pixels: Optional[Union[int, List[int]]] = 768 * 768,
    rewriter_ref_images_max_side_length: Optional[int] = 1664,
    rewriter_system_prompt_type: str = "default",
    custom_rewriter_system_prompts_list: List[str] = None,
    merge_original_and_rewritten_instructions: bool = True,
    do_sample_for_local_rewriter: bool = True,
    save_rewritten_instruction: bool = False,
    save_rewritten_instruction_path: Optional[str] = None,
    input_image_paths: Optional[Union[List[List[str]], List[str]]] = None,
    use_dashscope_remote_rewriting: bool = False,
    dashscope_remote_rewriting_model: str = "qwen-vl-max-latest",
    dashscope_base_http_api_url: str = "https://dashscope.aliyuncs.com/api/v1",
    dashscope_api_key: str = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxx",
    system_prompt_follows_task_type: bool = False,
    # Momentum config (boosted orthogonal guidance, not vendored).
    use_boosted_orthogonal_guidance: bool = False,
    text_momentum_rolling_sum_momentum_weight: float = 0.1,
    text_momentum_rolling_sum_current_weight: float = 0.9,
    image_momentum_rolling_sum_momentum_weight: float = 0.1,
    image_momentum_rolling_sum_current_weight: float = 0.9,
    empty_momentum_rolling_sum_momentum_weight: float = 0.1,
    empty_momentum_rolling_sum_current_weight: float = 0.9,
    bog_mu: float = 0.1,
    bog_range=[0.0, 1.0],
    bog_interval: int = 3,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    timesteps: List[int] = None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    verbose: bool = False,
    step_func=None,
    device: Optional[str] = None,
    rewriter_device: Optional[str] = None,
    unload_rewriter_level: Optional[str] = None,
    enable_inner_devices_manager: bool = False,
  ):

    if enable_inner_devices_manager is not None:
      self.enable_inner_devices_manager = enable_inner_devices_manager

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    self._text_guidance_scale = text_guidance_scale
    self._image_guidance_scale = image_guidance_scale
    self._empty_instruction_guidance_scale = empty_instruction_guidance_scale

    self._cfg_range = cfg_range
    self._attention_kwargs = attention_kwargs

    # 1. Define call parameters
    if instruction is not None and isinstance(instruction, str):
      batch_size = 1
      instruction = [instruction]
    elif instruction is not None and isinstance(instruction, (list, tuple)):
      batch_size = len(instruction)
    else:
      batch_size = instruction_embeds.shape[0]

    if use_rewrite_text_instruction or use_dashscope_remote_rewriting:
      raise NotImplementedError(
        "The instruction rewriter is not vendored in cache-dit; rewrite the "
        "instruction beforehand and pass the result as `instruction`.")
    if use_boosted_orthogonal_guidance:
      raise NotImplementedError("Boosted orthogonal guidance is not vendored in cache-dit.")
    if self.enable_model_cpu_offload_flag or self.enable_sequential_cpu_offload_flag:
      raise NotImplementedError(
        "Use `--cpu-offload` / `--layerwise-offload` from cache-dit instead of "
        "the pipeline's own offload flags.")

    # The pipeline no longer owns device placement: the caller moves the
    # pipeline (cache-dit's CLI does `pipe.to(device)`), so the execution device
    # defaults to wherever the components already live.
    execution_device = self._execution_device if device is None else torch.device(device)
    self.user_set_pipe_device = execution_device

    max_images_per_sample = 0
    if input_images:
      success, max_images_per_sample, input_images = (
        self._check_and_wrap_input_images(input_images))

    if input_image_paths:
      success, max_image_paths_per_sample, input_image_paths = (
        self._check_and_wrap_input_images(input_image_paths))
      assert (max_image_paths_per_sample == max_images_per_sample
              ), """The size of `input_image_paths` must be equal to that of `input_images`.
                    `input_image_paths` contains the paths to `input_images`, so they correspond to each other.
                    """

    # task_type = self._get_task_type_by_ref_latents(ref_latents)
    task_type = self._get_task_type_by_input_images(input_images)

    # cache-dit's separate-CFG bookkeeping assumes exactly one conditional and
    # one unconditional forward per step. Boogu adds a forward per active
    # guidance mode, so a cached run must keep exactly one of them enabled.
    if getattr(self.transformer, "_is_cached", False):
      if tuple(cfg_range) != (0.0, 1.0):
        raise ValueError("cache-dit caching assumes guidance is active on every step; "
                         f"`cfg_range={tuple(cfg_range)}` turns it off outside the window and "
                         "changes the per-step forward count. Leave `cfg_range` unset when "
                         "caching.")
      if (empty_instruction_guidance_scale != 0.0
          and use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide
          != use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide):
        raise ValueError("cache-dit caching does not support the empty-instruction "
                         "double-guidance variant; set "
                         "`empty_instruction_guidance_scale=0` or disable caching.")
      text_on = text_guidance_scale > 1.0
      image_on = task_type == "ti2i" and image_guidance_scale > 1.0
      n_forwards = 1 + int(text_on) + int(image_on)
      if n_forwards != 2:
        raise ValueError("cache-dit caching runs exactly two transformer forwards per step, "
                         f"but this call runs {n_forwards} "
                         f"(text_guidance_scale={text_guidance_scale}, "
                         f"image_guidance_scale={image_guidance_scale}). Enable exactly one of "
                         "text / image guidance, or disable caching.")

    # 2. Encode input instruction (VLM forward)
    with debug_timer("VLM encode (instruction)"):
      (
        instruction_embeds,
        instruction_attention_mask,
        negative_instruction_embeds,
        negative_instruction_attention_mask,
        empty_instruction_embeds,
        empty_instruction_attention_mask,
      ) = self.encode_instruction(
        instruction,
        self.text_guidance_scale > 1.0,
        negative_instruction=negative_instruction,
        input_images=input_images,
        use_input_images_4_neg_instruct=use_input_images_4_neg_instruct,
        use_input_images_4_empty_instruct=use_input_images_4_empty_instruct,
        max_vlm_input_pil_pixels=max_vlm_input_pil_pixels,
        max_vlm_input_pil_side_length=max_vlm_input_pil_side_length,
        num_images_per_instruction=num_images_per_instruction,
        device=self.user_set_pipe_device,
        instruction_embeds=instruction_embeds,
        negative_instruction_embeds=negative_instruction_embeds,
        instruction_attention_mask=instruction_attention_mask,
        negative_instruction_attention_mask=negative_instruction_attention_mask,
        # For double guidance
        empty_instruction=empty_instruction,
        empty_instruction_embeds=empty_instruction_embeds,
        empty_instruction_attention_mask=empty_instruction_attention_mask,
        use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide=
        use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide,
        use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide=
        use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide,
        max_sequence_length=max_sequence_length,
        truncate_instruction_sequence=truncate_instruction_sequence,
        use_rewrite_text_instruction=use_rewrite_text_instruction,
        input_image_paths=input_image_paths,
        system_prompt_follows_task_type=system_prompt_follows_task_type,
        task_type=task_type,
      )

    # Put ref_latents here before encoding instruction.
    dtype = self.vae.dtype

    # 3. Prepare control image
    with debug_timer("VAE encode (prepare_image)"):
      ref_latents = self.prepare_image(
        images=input_images,
        batch_size=batch_size,
        num_images_per_instruction=num_images_per_instruction,
        max_input_image_pixels=max_input_image_pixels,
        max_side_length=max_input_image_side_length,
        device=self.user_set_pipe_device,
        dtype=dtype,
        generator=generator,
      )

    input_images, width, height, ori_width, ori_height = (self._resolve_output_and_original_size(
      input_images=input_images,
      ref_latents=ref_latents,
      align_res=align_res,
      width=width,
      height=height,
      max_input_image_pixels=max_input_image_pixels,
      max_images_per_sample=max_images_per_sample,
      img_scale_num=self.vae_scale_factor * 2,
    ))

    if len(input_images) == 0:
      self._image_guidance_scale = 1

    # 4. Prepare latents.
    latent_channels = self.transformer.config.in_channels
    latents = self.prepare_latents(
      batch_size * num_images_per_instruction,
      latent_channels,
      height,
      width,
      instruction_embeds.dtype,
      self.user_set_pipe_device,
      generator,
      latents,
    )

    freqs_cis = BooguImageRotaryPosEmbed.get_freqs_cis(
      self.transformer.config.axes_dim_rope,
      self.transformer.config.axes_lens,
      theta=10000,
    )

    with debug_timer("DiT + VAE decode (processing)"):
      image = self.processing(
        latents=latents,
        ref_latents=ref_latents,
        instruction_embeds=instruction_embeds,
        freqs_cis=freqs_cis,
        negative_instruction_embeds=negative_instruction_embeds,
        instruction_attention_mask=instruction_attention_mask,
        negative_instruction_attention_mask=negative_instruction_attention_mask,
        num_inference_steps=num_inference_steps,
        timesteps=timesteps,
        device=self.user_set_pipe_device,
        dtype=dtype,
        verbose=verbose,
        generator=generator,
        step_func=step_func,
        # For double guidance
        empty_instruction_embeds=empty_instruction_embeds,
        empty_instruction_attention_mask=empty_instruction_attention_mask,
        use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide=
        use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide,
        use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide=
        use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide,
      )

    image = F.interpolate(image, size=(ori_height, ori_width), mode="bilinear")

    image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
      return image
    else:
      return FMPipelineOutput(images=image)

  def _resolve_output_and_original_size(
    self,
    input_images,
    ref_latents: List[Union[List[torch.FloatTensor], None]],
    align_res: bool,
    width: int,
    height: int,
    max_input_image_pixels: Union[int, list, tuple],
    max_images_per_sample: int,
    img_scale_num: int = 16,
  ) -> Tuple[List, int, int, int, int]:
    if input_images is None:
      input_images = []

    if len(input_images) == 1 and align_res:
      width, height = (
        ref_latents[0][0].shape[-1] * self.vae_scale_factor,
        ref_latents[0][0].shape[-2] * self.vae_scale_factor,
      )
      ori_width, ori_height = width, height
    else:
      ori_width, ori_height = width, height

      cur_pixels = height * width

      if isinstance(max_input_image_pixels, (list, tuple)):
        if ((input_images is not None) and (len(input_images) > 0) and max_images_per_sample > 0):
          assert len(max_input_image_pixels) >= max_images_per_sample, (
            f"When `max_input_image_pixels` is a list or tuple, the length of it (here is {len(max_input_image_pixels)}) should be >= max number of input images in all the samples (here is {max_images_per_sample})."
          )
          max_pixels = max_input_image_pixels[max_images_per_sample - 1]
        else:
          max_pixels = max_input_image_pixels[0]
      else:
        max_pixels = max_input_image_pixels

      ratio = (max_pixels / cur_pixels) ** 0.5
      ratio = min(ratio, 1.0)

      height, width = (
        int(height * ratio) // img_scale_num * img_scale_num,
        int(width * ratio) // img_scale_num * img_scale_num,
      )

    return input_images, width, height, ori_width, ori_height

  def _get_task_type_by_ref_latents(self, ref_latents: List[Union[List[torch.FloatTensor], None]]):
    if not ref_latents:
      return "t2i"

    if isinstance(ref_latents, (list, tuple)):
      for x in ref_latents:
        if x:
          return "ti2i"
    return "t2i"

  def _get_task_type_by_input_images(self, input_images: Union[List[List[PIL.Image.Image]],
                                                               List[PIL.Image.Image]]):
    if not input_images:
      return "t2i"

    if isinstance(input_images, (list, tuple)):
      for x in input_images:
        if x:
          return "ti2i"
    return "t2i"

  def processing(
    self,
    latents,
    ref_latents,
    instruction_embeds,
    freqs_cis,
    negative_instruction_embeds,
    instruction_attention_mask,
    negative_instruction_attention_mask,
    num_inference_steps,
    timesteps,
    device,
    dtype,
    verbose,
    generator=None,
    step_func=None,
    empty_instruction_embeds=None,
    empty_instruction_attention_mask=None,
    use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide=False,
    use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide=False,
  ):
    """Run the denoising loop and decode the final latents.

    One conditional forward per step is always run; the guidance modes add the
    matching unconditional forwards, so a cache-dit cached run must keep
    `text_guidance_scale > 1` (two forwards per step) or disable caching.

    :param latents: Initial noise latents `[B, C, H, W]`.
    :param ref_latents: Reference-image latents, or `None` for text-to-image.
    :param instruction_embeds: Encoded instruction hidden states.
    :param freqs_cis: Per-axis RoPE tables.
    :param negative_instruction_embeds: Encoded negative instruction.
    :param instruction_attention_mask: Instruction padding mask.
    :param negative_instruction_attention_mask: Negative-instruction mask.
    :param num_inference_steps: Number of denoising steps.
    :param timesteps: Optional explicit timestep schedule.
    :param device: Execution device.
    :param dtype: Latent dtype.
    :param verbose: Whether the scheduler should log its progress.
    :param generator: Optional RNG used by the scheduler.
    :param step_func: Optional per-step callback `(step, total)`.
    :param empty_instruction_embeds: Empty-instruction embeds for double guidance.
    :param empty_instruction_attention_mask: Mask paired with the above.
    :param use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide:
      Use the empty instruction for the image-guidance branch.
    :param use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide:
      Use the empty instruction for the text-guidance branch.
    :returns: The decoded image tensor.
    """

    task_type = self._get_task_type_by_ref_latents(ref_latents)
    logger.debug("[Boogu-Image] task type: %s", task_type)

    timesteps, num_inference_steps = retrieve_timesteps(
      self.scheduler,
      num_inference_steps,
      device,
      timesteps,
      num_tokens=latents.shape[-2] * latents.shape[-1],
    )

    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    empty_image_branch = (use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide)
    empty_text_branch = (use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide)

    with self.progress_bar(total=num_inference_steps) as progress_bar:
      for i, t in enumerate(timesteps):
        model_pred = self.predict(
          t=t,
          latents=latents,
          instruction_embeds=instruction_embeds,
          freqs_cis=freqs_cis,
          instruction_attention_mask=instruction_attention_mask,
          ref_image_hidden_states=ref_latents,
        )

        # `cfg_range` disables guidance outside the configured step window.
        in_cfg = self.cfg_range[0] <= i / len(timesteps) <= self.cfg_range[1]
        text_scale = self.text_guidance_scale if in_cfg else 1.0
        image_scale = self.image_guidance_scale if in_cfg else 1.0
        empty_scale = self.empty_instruction_guidance_scale if in_cfg else 0.0

        if task_type == "ti2i" and text_scale > 1.0 and image_scale > 1.0:
          # Double guidance: first drop the text condition, then both.
          pred_drop_text = self.predict(
            t=t,
            latents=latents,
            instruction_embeds=negative_instruction_embeds,
            freqs_cis=freqs_cis,
            instruction_attention_mask=negative_instruction_attention_mask,
            ref_image_hidden_states=ref_latents,
          )
          pred_drop_all = self.predict(
            t=t,
            latents=latents,
            instruction_embeds=negative_instruction_embeds,
            freqs_cis=freqs_cis,
            instruction_attention_mask=negative_instruction_attention_mask,
            ref_image_hidden_states=None,
          )
          delta_text = model_pred - pred_drop_text
          delta_image = pred_drop_text - pred_drop_all

          if empty_scale != 0.0 and (empty_image_branch != empty_text_branch):
            pred_drop_text_empty = self.predict(
              t=t,
              latents=latents,
              instruction_embeds=empty_instruction_embeds,
              freqs_cis=freqs_cis,
              instruction_attention_mask=empty_instruction_attention_mask,
              ref_image_hidden_states=ref_latents,
            )
            pred_pos = (pred_drop_text_empty if empty_image_branch else pred_drop_text)
            pred_neg = (pred_drop_text_empty if empty_text_branch else pred_drop_text)
            model_pred = (model_pred + (text_scale - 1) * delta_text +
                          (image_scale - 1) * delta_image + empty_scale * (pred_pos - pred_neg))
          else:
            model_pred = (model_pred + (text_scale - 1) * delta_text +
                          (image_scale - 1) * delta_image)

        elif task_type == "ti2i" and text_scale > 1.0:
          pred_drop_text = self.predict(
            t=t,
            latents=latents,
            instruction_embeds=negative_instruction_embeds,
            freqs_cis=freqs_cis,
            instruction_attention_mask=negative_instruction_attention_mask,
            ref_image_hidden_states=ref_latents,
          )
          model_pred = model_pred + (text_scale - 1) * (model_pred - pred_drop_text)

        elif task_type == "ti2i" and image_scale > 1.0:
          pred_drop_image = self.predict(
            t=t,
            latents=latents,
            instruction_embeds=instruction_embeds,
            freqs_cis=freqs_cis,
            instruction_attention_mask=instruction_attention_mask,
            ref_image_hidden_states=None,
          )
          model_pred = model_pred + (image_scale - 1) * (model_pred - pred_drop_image)

        elif text_scale > 1.0:
          pred_drop_all = self.predict(
            t=t,
            latents=latents,
            instruction_embeds=negative_instruction_embeds,
            freqs_cis=freqs_cis,
            instruction_attention_mask=negative_instruction_attention_mask,
            ref_image_hidden_states=None,
          )
          model_pred = model_pred + (text_scale - 1) * (model_pred - pred_drop_all)

        latents = self.scheduler.step(model_pred,
                                      t,
                                      latents,
                                      generator=generator,
                                      return_dict=False)[0]
        latents = latents.to(dtype=dtype)

        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and
                                       (i + 1) % self.scheduler.order == 0):
          progress_bar.update()

        if step_func is not None:
          step_func(i, self._num_timesteps)

    latents = latents.to(dtype=dtype)
    if self.vae.config.scaling_factor is not None:
      latents = latents / self.vae.config.scaling_factor
    if self.vae.config.shift_factor is not None:
      latents = latents + self.vae.config.shift_factor
    return self.vae.decode(latents, return_dict=False)[0]

  def predict(
    self,
    t,
    latents,
    instruction_embeds,
    freqs_cis,
    instruction_attention_mask,
    ref_image_hidden_states,
  ):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(latents.shape[0]).to(latents.dtype)

    batch_size, num_channels_latents, height, width = latents.shape

    optional_kwargs = {}
    if "ref_image_hidden_states" in set(
        inspect.signature(self.transformer.forward).parameters.keys()):
      optional_kwargs["ref_image_hidden_states"] = ref_image_hidden_states

    model_pred = self.transformer(
      latents,
      timestep,
      instruction_embeds,
      freqs_cis,
      instruction_attention_mask,
      **optional_kwargs,
    )
    return model_pred
