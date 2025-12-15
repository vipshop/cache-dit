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
    "ContextParallelismPlanner",
    "ContextParallelismPlannerRegister",
]


class ContextParallelismPlanner:
    # Prefer native diffusers implementation if available
    _cp_planner_preferred_native_diffusers: bool = True

    @abstractmethod
    def apply(
        self,
        # NOTE: Keep this kwarg for future extensions
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:
        # NOTE: This method should only return the CP plan dictionary.
        raise NotImplementedError("apply method must be implemented by subclasses")


class ContextParallelismPlannerRegister:
    _cp_planner_registry: dict[str, ContextParallelismPlanner] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(planner_cls: type[ContextParallelismPlanner]):
            assert (
                name not in cls._cp_planner_registry
            ), f"ContextParallelismPlanner with name {name} is already registered."
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Registering ContextParallelismPlanner: {name}")
            cls._cp_planner_registry[name] = planner_cls
            return planner_cls

        return decorator

    @classmethod
    def get_planner(
        cls, transformer: str | torch.nn.Module | ModelMixin
    ) -> type[ContextParallelismPlanner]:
        if isinstance(transformer, (torch.nn.Module, ModelMixin)):
            name = transformer.__class__.__name__
        else:
            name = transformer
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
