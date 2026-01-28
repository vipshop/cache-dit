import torch
import logging
from abc import abstractmethod
from typing import Dict
from torch.distributed import init_device_mesh
from cache_dit.parallelism.config import ParallelismConfig
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

    def mesh(self, parallelism_config: ParallelismConfig, **kwargs):
        text_encoder_world_size = parallelism_config.text_encoder_world_size
        device_type = torch.accelerator.current_accelerator().type
        tp_mesh = init_device_mesh(
            device_type=device_type,
            mesh_shape=[text_encoder_world_size],
        )
        return tp_mesh


class TextEncoderTensorParallelismPlannerRegister:
    _text_encoder_tp_planner_registry: Dict[str, TextEncoderTensorParallelismPlanner] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(planner_cls: type[TextEncoderTensorParallelismPlanner]):
            assert (
                name not in cls._text_encoder_tp_planner_registry
            ), f"TextEncoderTensorParallelismPlanner with name {name} is already registered."
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Registering TextEncoderTensorParallelismPlanner: {name}")
            cls._text_encoder_tp_planner_registry[name] = planner_cls
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
        for planner_name in cls._text_encoder_tp_planner_registry:
            if name.startswith(planner_name):
                planner_cls = cls._text_encoder_tp_planner_registry.get(planner_name)
                break
        if planner_cls is None:
            raise ValueError(f"No planner registered under name: {name}")
        return planner_cls

    @classmethod
    def supported_planners(
        cls,
    ) -> tuple[int, list[str]]:
        val_planners = cls._text_encoder_tp_planner_registry.keys()
        return len(val_planners), [p for p in val_planners]
