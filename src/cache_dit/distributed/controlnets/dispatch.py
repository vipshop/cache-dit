from typing import Optional

import torch

from diffusers.models.modeling_utils import ModelMixin

from ...attention import _maybe_register_custom_attn_backends
from ...distributed.core import (
  _ContextParallelConfig,
  _enable_context_parallelism,
)
from ...logger import init_logger
from ..backend import ParallelismBackend
from ..config import ParallelismConfig
from .planners import _activate_controlnet_cp_planners
from .register import ControlNetContextParallelismPlannerRegister

logger = init_logger(__name__)


def _ensure_controlnet_cp_planners_activated() -> None:
  _maybe_register_custom_attn_backends()
  if ControlNetContextParallelismPlannerRegister.supported_planners()[0] == 0:
    _activate_controlnet_cp_planners()


def _parallelize_controlnet_cp(
  controlnet: torch.nn.Module,
  parallelism_config: Optional[ParallelismConfig],
) -> torch.nn.Module:
  assert isinstance(controlnet,
                    ModelMixin), ("controlnet must be an instance of diffusers' ModelMixin, "
                                  f"but got {type(controlnet)}")
  if parallelism_config is None:
    return controlnet

  assert isinstance(
    parallelism_config,
    ParallelismConfig), ("parallelism_config must be an instance of ParallelismConfig"
                         f" but got {type(parallelism_config)}")

  _ensure_controlnet_cp_planners_activated()

  if parallelism_config.cp_enabled():
    assert (not parallelism_config.hybrid_enabled()
            ), "Hybrid parallelism is not supported for ControlNet now."
    cp_config = _ContextParallelConfig(
      ulysses_degree=parallelism_config.ulysses_size,
      ring_degree=parallelism_config.ring_size,
      convert_to_fp32=parallelism_config.ring_convert_to_fp32,
      rotate_method=parallelism_config.ring_rotate_method,
      ulysses_anything=parallelism_config.ulysses_anything,
      ulysses_float8=parallelism_config.ulysses_float8,
      ulysses_async=parallelism_config.ulysses_async,
    )

    cp_plan = parallelism_config.cp_plan
    if cp_plan is not None:
      logger.info(f"Using custom context parallelism plan: {cp_plan}")
    else:
      cp_plan = ControlNetContextParallelismPlannerRegister.get_planner(controlnet)().apply(
        controlnet=controlnet,
        parallelism_config=parallelism_config,
      )

    _enable_context_parallelism(controlnet, config=cp_config, cp_plan=cp_plan)

  return controlnet


def parallelize_controlnet(
  controlnet: torch.nn.Module | ModelMixin,
  parallelism_config: Optional[ParallelismConfig],
) -> torch.nn.Module:
  """Parallelize one ControlNet with the configured CP strategy.

  :param controlnet: ControlNet module to parallelize.
  :param parallelism_config: Parallelism configuration shared with the transformer.
  :returns: The parallelized ControlNet.
  """
  assert isinstance(
    controlnet, (torch.nn.Module,
                 ModelMixin)), ("controlnet must be an instance of torch.nn.Module or ModelMixin, "
                                f"but got {type(controlnet)}")

  if parallelism_config is None:
    return controlnet

  if parallelism_config.backend != ParallelismBackend.NATIVE_DIFFUSER:
    logger.warning(f"Parallelism backend {parallelism_config.backend} is not supported "
                   "for ControlNet now, skip context parallelism for ControlNet.")
    return controlnet

  if parallelism_config.cp_enabled():
    controlnet = _parallelize_controlnet_cp(
      controlnet=controlnet,
      parallelism_config=parallelism_config,
    )
    controlnet._is_parallelized = True  # type: ignore[attr-defined]
    # Use `parallelism` not `parallel` to avoid name conflict with diffusers.
    controlnet._parallelism_config = parallelism_config  # type: ignore[attr-defined]
    logger.info(f"Parallelize ControlNet: {controlnet.__class__.__name__}, "
                f"id:{id(controlnet)}, {parallelism_config.strify(True)}")
  else:
    logger.warning("Please set ulysses_size or ring_size in parallelism_config to enable "
                   "context parallelism for ControlNet. Skipping parallelism for ControlNet.")
  return controlnet
