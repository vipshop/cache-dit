import torch
import logging
from abc import abstractmethod
from typing import Optional
from diffusers.models.modeling_utils import ModelMixin
from ...distributed.core import _ContextParallelModelPlan

from ..config import ParallelismConfig
from ...logger import init_logger

logger = init_logger(__name__)

__all__ = [
  "ControlNetContextParallelismPlanner",
  "ControlNetContextParallelismPlannerRegister",
]


class ControlNetContextParallelismPlanner:
  # Prefer native diffusers implementation if available
  _cp_planner_preferred_native_diffusers: bool = True

  def apply(
    self,
    controlnet: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    """Apply the context parallelism plan to the given controlnet.

    :param controlnet: The controlnet model to which the CP plan will be applied.
    :param parallelism_config: The parallelism configuration containing mesh and other settings.
    :param kwargs: Additional arguments that may be needed for specific planners.
    :returns: An internal `_ContextParallelModelPlan` that describes how to apply context
      parallelism to the
        model.
    """
    assert (controlnet
            is not None), "ControlNet model must be provided to apply context parallelism."
    assert (parallelism_config
            is not None), "ParallelismConfig must be provided to apply context parallelism."
    cp_plan = self._apply(
      controlnet=controlnet,
      parallelism_config=parallelism_config,
      **kwargs,
    )
    controlnet._cp_plan = cp_plan
    cls_name = controlnet.__class__.__name__
    if logger.isEnabledFor(logging.DEBUG):
      logger.debug(f"Generated CP plan: {cp_plan}")
    if parallelism_config and getattr(parallelism_config, "ulysses_async", False):
      logger.info(f"Async Ulysses Attention is enabled for {cls_name}.")
    return cp_plan

  @abstractmethod
  def _apply(
    self,
    # NOTE: Keep this kwarg for future extensions
    controlnet: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    # NOTE: This method should only return the CP plan dictionary.
    raise NotImplementedError("apply method must be implemented by subclasses")


class ControlNetContextParallelismPlannerRegister:
  _cp_planner_registry: dict[str, ControlNetContextParallelismPlanner] = {}

  @classmethod
  def register(cls, name: str):

    def decorator(planner_cls: type[ControlNetContextParallelismPlanner]):
      assert (name not in cls._cp_planner_registry
              ), f"ControlNetContextParallelismPlanner with name {name} is already registered."
      if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Registering ControlNetContextParallelismPlanner: {name}")
      cls._cp_planner_registry[name] = planner_cls
      return planner_cls

    return decorator

  @classmethod
  def get_planner(
      cls,
      controlnet: str | torch.nn.Module | ModelMixin) -> type[ControlNetContextParallelismPlanner]:
    if isinstance(controlnet, (torch.nn.Module, ModelMixin)):
      name = controlnet.__class__.__name__
    else:
      name = controlnet
    planner_cls = None
    for planner_name in cls._cp_planner_registry:
      if name.startswith(planner_name):
        planner_cls = cls._cp_planner_registry.get(planner_name)
        break
    if planner_cls is None:
      raise ValueError(f"No planner registered under name: {name}")
    return planner_cls

  @classmethod
  def supported_planners(cls, ) -> tuple[int, list[str]]:
    val_planners = cls._cp_planner_registry.keys()
    return len(val_planners), [p for p in val_planners]
