import torch
import logging
from abc import abstractmethod
from typing import Optional
from diffusers.models.modeling_utils import ModelMixin

try:
    from diffusers.models._modeling_parallel import (
        ContextParallelModelPlan,
    )
except ImportError:
    raise ImportError(
        "Context parallelism requires the 'diffusers>=0.36.dev0'."
        "Please install latest version of diffusers from source: \n"
        "pip3 install git+https://github.com/huggingface/diffusers.git"
    )

from cache_dit.logger import init_logger

logger = init_logger(__name__)


__all__ = [
    "ControlNetContextParallelismPlanner",
    "ControlNetContextParallelismPlannerRegister",
]


class ControlNetContextParallelismPlanner:
    # Prefer native diffusers implementation if available
    _cp_planner_preferred_native_diffusers: bool = True

    @abstractmethod
    def apply(
        self,
        # NOTE: Keep this kwarg for future extensions
        controlnet: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:
        # NOTE: This method should only return the CP plan dictionary.
        raise NotImplementedError("apply method must be implemented by subclasses")


class ControlNetContextParallelismPlannerRegister:
    _cp_planner_registry: dict[str, ControlNetContextParallelismPlanner] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(planner_cls: type[ControlNetContextParallelismPlanner]):
            assert (
                name not in cls._cp_planner_registry
            ), f"ControlNetContextParallelismPlanner with name {name} is already registered."
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Registering ControlNetContextParallelismPlanner: {name}")
            cls._cp_planner_registry[name] = planner_cls
            return planner_cls

        return decorator

    @classmethod
    def get_planner(
        cls, controlnet: str | torch.nn.Module | ModelMixin
    ) -> type[ControlNetContextParallelismPlanner]:
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
    def supported_planners(
        cls,
    ) -> tuple[int, list[str]]:
        val_planners = cls._cp_planner_registry.keys()
        return len(val_planners), [p for p in val_planners]
