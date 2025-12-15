import torch
import logging
from abc import abstractmethod
from typing import Dict
from cache_dit.parallelism.parallel_config import ParallelismConfig
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class TextEncoderTensorParallelismPlanner:

    @abstractmethod
    def apply(
        self,
        text_encoder: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        raise NotImplementedError("apply method must be implemented by subclasses")


class TextEncoderTensorParallelismPlannerRegister:
    _tp_planner_registry: Dict[str, TextEncoderTensorParallelismPlanner] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(planner_cls: type[TextEncoderTensorParallelismPlanner]):
            assert (
                name not in cls._tp_planner_registry
            ), f"TensorParallelismPlanner with name {name} is already registered."
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Registering TensorParallelismPlanner: {name}")
            cls._tp_planner_registry[name] = planner_cls
            return planner_cls

        return decorator

    @classmethod
    def get_planner(
        cls, text_encoder: str | torch.nn.Module
    ) -> type[TextEncoderTensorParallelismPlanner]:
        if isinstance(text_encoder, torch.nn.Module):
            name = text_encoder.__class__.__name__
        else:
            name = text_encoder
        planner_cls = None
        for planner_name in cls._tp_planner_registry:
            if name.startswith(planner_name):
                planner_cls = cls._tp_planner_registry.get(planner_name)
                break
        if planner_cls is None:
            raise ValueError(f"No planner registered under name: {name}")
        return planner_cls

    @classmethod
    def supported_planners(
        cls,
    ) -> tuple[int, list[str]]:
        val_planners = cls._tp_planner_registry.keys()
        return len(val_planners), [p for p in val_planners]
