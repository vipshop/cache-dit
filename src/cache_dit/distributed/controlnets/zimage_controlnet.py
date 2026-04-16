from typing import Optional

import torch
from diffusers.models.modeling_utils import ModelMixin

from ...distributed.core import (
  _ContextParallelInput,
  _ContextParallelModelPlan,
  _ContextParallelOutput,
)
from ...logger import init_logger
from ..async_ulysses import AsyncUlyssesRegistry
from .register import (
  ControlNetContextParallelismPlanner,
  ControlNetContextParallelismPlannerRegister,
  ParallelismConfig,
)

logger = init_logger(__name__)


@ControlNetContextParallelismPlannerRegister.register("ZImageControlNetModel")
class ZImageControlNetContextParallelismPlanner(ControlNetContextParallelismPlanner):

  def _apply(
    self,
    controlnet: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:

    if parallelism_config.ulysses_async and controlnet is not None:
      AsyncUlyssesRegistry.apply(controlnet)
      logger.info("Enabled experimental Async QKV Projection with Ulysses style "
                  "Context Parallelism for ZImageControlNetModel.")

    # The cp plan for ZImage ControlNet is very complicated, I [HATE] it.
    n_control_layers = len(controlnet.control_layers)  # 15
    n_control_noise_refiner_layers = len(controlnet.control_noise_refiner)  # 2
    _cp_plan = {
      "control_noise_refiner.0": {
        "c": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      "control_noise_refiner.*": {
        "x": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "freqs_cis": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      f"control_noise_refiner.{n_control_noise_refiner_layers - 1}":
      _ContextParallelOutput(gather_dim=2, expected_dims=4),
      "control_layers.0": {
        "c": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      "control_layers.*": {
        "x": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "freqs_cis": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      f"control_layers.{n_control_layers - 1}":
      _ContextParallelOutput(gather_dim=2, expected_dims=4),
    }
    return _cp_plan
