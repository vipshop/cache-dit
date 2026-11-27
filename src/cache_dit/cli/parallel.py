"""`--parallel-params` resolution, extra parallel modules and distributed lifecycle."""

import os
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline

from ..caching import BlockAdapter, load_parallelism_config
from ..logger import init_logger
from ..platforms import current_platform
from .params import resolve_parallel_params

logger = init_logger(__name__)


def _resolve_cli_parallelism(args) -> None:
  """Resolve `--parallel-params` into the plain attributes the pipeline code reads.

  `parallel_type` names the active axes as the former `--parallel` values did (`ulysses`, `tp`,
  `ring`, `usp`, `ulysses_tp`, `ring_tp`, `usp_tp`), so the readers of `args.parallel_type` and
  the output naming keep working. The degrees are validated against the world size where the
  parallelism groups are built.

  :param args: Parsed CLI namespace, updated in place.
  """

  args.parallel_type, args.parallel_axes, fields = resolve_parallel_params(args.parallel_params)
  if args.parallel_params is not None and args.parallel_type is None and any(
      axis in fields for axis in ("ulysses", "ring", "tp")):
    logger.warning("--parallel-params only has degrees of 1, so parallelism is not enabled; "
                   "raise a degree, e.g. 'ulysses=2'.")
  args.parallel_vae = bool(fields.get("vae", False))
  args.parallel_text_encoder = bool(fields.get("text", False))
  args.parallel_controlnet = bool(fields.get("controlnet", False))
  args.ulysses_anything = bool(fields.get("ulysses_anything", False))
  args.ulysses_async = bool(fields.get("ulysses_async", False))
  args.ring_rotate_method = fields.get("ring_rotate_method", "p2p")
  args.ring_no_convert_to_fp32 = not fields.get("ring_convert_to_fp32", True)
  args.parallel_degrees = {axis: fields.get(axis, 1) for axis in ("ulysses", "ring", "tp")}
  if args.parallel_type is not None:
    # torchrun exports WORLD_SIZE; fail here rather than after the model has been loaded. Without it
    # (plain `python`), the world size is checked where the groups are built.
    world_size = os.environ.get("WORLD_SIZE")
    if world_size is not None:
      product = args.parallel_degrees["ulysses"] * args.parallel_degrees[
        "ring"] * args.parallel_degrees["tp"]
      if product != int(world_size):
        raise ValueError(
          f"--parallel-params degrees (ulysses={args.parallel_degrees['ulysses']},"
          f"ring={args.parallel_degrees['ring']},tp={args.parallel_degrees['tp']}) multiply to "
          f"{product}, which must equal the world size {world_size}.")


def get_text_encoder_from_pipe(
  pipe: DiffusionPipeline, ) -> Tuple[Optional[torch.nn.Module], Optional[str]]:
  pipe_cls_name = pipe.__class__.__name__
  if (hasattr(pipe, "text_encoder_2") and not pipe_cls_name.startswith("Hunyuan")
      and not pipe_cls_name.startswith("Kandinsky")):
    # Specific for FluxPipeline, FLUX.1-dev
    return getattr(pipe, "text_encoder_2"), "text_encoder_2"
  elif hasattr(pipe, "text_encoder_3"):  # HiDream pipeline
    return getattr(pipe, "text_encoder_3"), "text_encoder_3"
  elif hasattr(
      pipe,
      "vision_language_encoder") and pipe_cls_name.startswith("GlmImage"):  # GLM Image pipeline
    return getattr(pipe, "vision_language_encoder"), "vision_language_encoder"
  elif hasattr(pipe, "text_encoder"):  # General case
    return getattr(pipe, "text_encoder"), "text_encoder"
  else:
    return None, None


def prepare_extra_parallel_modules(
  args,
  pipe_or_adapter: DiffusionPipeline | BlockAdapter,
  custom_extra_modules: Optional[List[torch.nn.Module]] = None,
) -> list:
  if custom_extra_modules is not None:
    return custom_extra_modules

  if isinstance(pipe_or_adapter, BlockAdapter):
    pipe = pipe_or_adapter.pipe
    assert pipe is not None, "Please set extra_parallel_modules manually if pipe is None."
  else:
    pipe = pipe_or_adapter

  extra_parallel_modules = []

  if args.parallel_text_encoder:
    text_encoder, _ = get_text_encoder_from_pipe(pipe)
    if text_encoder is not None:
      extra_parallel_modules.append(text_encoder)
    else:
      logger.warning("parallel-text-encoder is set but no text encoder found in the pipeline.")

  if args.parallel_vae:
    assert not args.vae_tiling, "VAE tiling is not compatible with VAE parallelism."
    assert not args.vae_slicing, "VAE slicing is not compatible with VAE parallelism."
    if hasattr(pipe, "vae"):
      extra_parallel_modules.append(getattr(pipe, "vae"))
    else:
      logger.warning("parallel-vae is set but no VAE found in the pipeline.")

  if args.parallel_controlnet:
    if hasattr(pipe, "controlnet"):
      extra_parallel_modules.append(getattr(pipe, "controlnet"))
    else:
      logger.warning("parallel-controlnet is set but no ControlNet found in the pipeline.")

  return extra_parallel_modules


def get_rank_device():
  available = current_platform.is_accelerator_available()
  device_type = current_platform.device_type
  if dist.is_initialized():
    rank = dist.get_rank()
    device = torch.device(device_type, rank % current_platform.device_count())
    return rank, device
  return 0, torch.device(device_type if available else "cpu")


def maybe_init_distributed(args=None):
  from ..platforms.platform import CpuPlatform
  from ..logger import suppress_stdout

  platform_full_backend = current_platform.full_dist_backend
  cpu_full_backend = CpuPlatform.full_dist_backend
  # Always use hybrid comm backend since there no side effect.
  backend = f"{cpu_full_backend},{platform_full_backend}"
  if args is not None:
    if args.parallel_type is not None:
      with suppress_stdout():
        dist.init_process_group(backend=backend, )
      rank, device = get_rank_device()
      current_platform.set_device(device)
      return rank, device
    elif args.config_path is not None:
      # check if distributed is needed from config file
      has_parallelism_config = load_parallelism_config(
        args.config_path,
        check_only=True,
      )
      if has_parallelism_config:
        if not dist.is_initialized():
          with suppress_stdout():
            dist.init_process_group(backend=backend, )
        rank, device = get_rank_device()
        current_platform.set_device(device)
        return rank, device
      else:
        # no distributed needed
        rank, device = get_rank_device()
        return rank, device
    else:
      # no distributed needed
      rank, device = get_rank_device()
      return rank, device
  else:
    # always init distributed for other examples
    if not dist.is_initialized():
      with suppress_stdout():
        dist.init_process_group(backend=backend, )
    rank, device = get_rank_device()
    current_platform.set_device(device)
    return rank, device


def maybe_destroy_distributed():
  if dist.is_initialized():
    dist.destroy_process_group()
