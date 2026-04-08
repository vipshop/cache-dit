import torch
import logging
from abc import abstractmethod
from typing import Dict
from torch.distributed import init_device_mesh
from ...config import ParallelismConfig
from ....logger import init_logger

logger = init_logger(__name__)


class AutoEncoderDataParallelismPlanner:

  def apply(
    self,
    auto_encoder: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> torch.nn.Module:
    # TODO: May add some common checks before or after applying the parallelism plan,
    # e.g., check if the auto_encoder is already parallelized, check if the parallelism_config
    # is valid, etc.
    auto_encoder = self._apply(
      auto_encoder=auto_encoder,
      parallelism_config=parallelism_config,
      **kwargs,
    )
    return auto_encoder

  @abstractmethod
  def _apply(
    self,
    auto_encoder: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> torch.nn.Module:
    raise NotImplementedError("apply method must be implemented by subclasses")

  def mesh(self, parallelism_config: ParallelismConfig, **kwargs):
    auto_encoder_world_size = parallelism_config.auto_encoder_world_size
    device_type = torch.accelerator.current_accelerator().type
    dp_mesh = init_device_mesh(
      device_type=device_type,
      mesh_shape=[auto_encoder_world_size],
    )
    return dp_mesh


class AutoEncoderDataParallelismPlannerRegister:
  _auto_encoder_dp_planner_registry: Dict[str, AutoEncoderDataParallelismPlanner] = {}

  @classmethod
  def register(cls, name: str):

    def decorator(planner_cls: type[AutoEncoderDataParallelismPlanner]):
      assert (name not in cls._auto_encoder_dp_planner_registry
              ), f"AutoEncoderDataParallelismPlanner with name {name} is already registered."
      if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Registering AutoEncoderDataParallelismPlanner: {name}")
      cls._auto_encoder_dp_planner_registry[name] = planner_cls
      return planner_cls

    return decorator

  @classmethod
  def get_planner(cls,
                  auto_encoder: str | torch.nn.Module) -> type[AutoEncoderDataParallelismPlanner]:
    if isinstance(auto_encoder, torch.nn.Module):
      name = auto_encoder.__class__.__name__
    else:
      name = auto_encoder
    planner_cls = None
    if name in cls._auto_encoder_dp_planner_registry:
      planner_cls = cls._auto_encoder_dp_planner_registry[name]
    if planner_cls is None:
      raise ValueError(f"No planner registered under name: {name}")
    return planner_cls

  @classmethod
  def supported_planners(cls, ) -> tuple[int, list[str]]:
    val_planners = cls._auto_encoder_dp_planner_registry.keys()
    return len(val_planners), [p for p in val_planners]
