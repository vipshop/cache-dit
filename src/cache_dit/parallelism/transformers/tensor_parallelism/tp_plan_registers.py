import torch
import logging
import copy
from abc import abstractmethod
from typing import Dict, List, Optional
from torch.distributed import init_device_mesh
from ...config import ParallelismConfig
from ....logger import init_logger
from ....envs import ENV

logger = init_logger(__name__)


class TensorParallelismPlanner:

    @abstractmethod
    def apply(
        self,
        transformer: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        raise NotImplementedError("apply method must be implemented by subclasses")

    def mesh(self, parallelism_config: ParallelismConfig, **kwargs):
        assert (
            parallelism_config.tp_enabled()
        ), "tp_size must be set and greater than 1 for tensor parallelism"
        # currently, in hybrid mode, we use the _tp_mesh from ParallelismConfig.
        if parallelism_config.hybrid_enabled():
            if parallelism_config._tp_mesh is not None:
                return parallelism_config._tp_mesh

        device_type = torch.accelerator.current_accelerator().type

        tp_mesh = init_device_mesh(
            device_type=device_type,
            mesh_shape=[parallelism_config.tp_size],
        )
        return tp_mesh

    def exclude_for_quantize(
        self,
        transformer: torch.nn.Module,
        exclude_layers: Optional[List[str]] = None,
        **kwargs,
    ):
        # Temporarily exclude some layers for quantization, e.g, layers applied RowwiseParallel.
        # Avoid torch._scaled_mm error: "RuntimeError: Expected b.stride(0) == 1 to be true,
        # but got false", RowwiseParallel (TP) seems will cause the layout of the linear weights
        # changedly after '_dispatch_get_local_results_slow_path'. Why? Need further investigation.
        if ENV.CACHE_DIT_DISABLE_EXCLUDE_FOR_QUANTIZE_AFTER_TP:
            return
        if exclude_layers is not None and isinstance(exclude_layers, list):
            transformer._exclude_for_quantize = copy.deepcopy(exclude_layers)


class TensorParallelismPlannerRegister:
    _tp_planner_registry: Dict[str, TensorParallelismPlanner] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(planner_cls: type[TensorParallelismPlanner]):
            assert (
                name not in cls._tp_planner_registry
            ), f"TensorParallelismPlanner with name {name} is already registered."
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Registering TensorParallelismPlanner: {name}")
            cls._tp_planner_registry[name] = planner_cls
            return planner_cls

        return decorator

    @classmethod
    def get_planner(cls, transformer: str | torch.nn.Module) -> type[TensorParallelismPlanner]:
        if isinstance(transformer, torch.nn.Module):
            name = transformer.__class__.__name__
        else:
            name = transformer
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
