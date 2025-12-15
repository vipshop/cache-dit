import torch
import logging
from abc import abstractmethod
from typing import Dict
from cache_dit.parallelism.parallel_config import ParallelismConfig
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class VAEDataParallelismPlanner:

    @abstractmethod
    def apply(
        self,
        vae: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        raise NotImplementedError("apply method must be implemented by subclasses")


class VAEDataParallelismPlannerRegister:
    _vae_dp_planner_registry: Dict[str, VAEDataParallelismPlanner] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(planner_cls: type[VAEDataParallelismPlanner]):
            assert (
                name not in cls._vae_dp_planner_registry
            ), f"VAEDataParallelismPlanner with name {name} is already registered."
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Registering VAEDataParallelismPlanner: {name}")
            cls._vae_dp_planner_registry[name] = planner_cls
            return planner_cls

        return decorator

    @classmethod
    def get_planner(cls, vae: str | torch.nn.Module) -> type[VAEDataParallelismPlanner]:
        if isinstance(vae, torch.nn.Module):
            name = vae.__class__.__name__
        else:
            name = vae
        planner_cls = None
        for planner_name in cls._vae_dp_planner_registry:
            if name.startswith(planner_name):
                planner_cls = cls._vae_dp_planner_registry.get(planner_name)
                break
        if planner_cls is None:
            raise ValueError(f"No planner registered under name: {name}")
        return planner_cls

    @classmethod
    def supported_planners(
        cls,
    ) -> tuple[int, list[str]]:
        val_planners = cls._vae_dp_planner_registry.keys()
        return len(val_planners), [p for p in val_planners]
