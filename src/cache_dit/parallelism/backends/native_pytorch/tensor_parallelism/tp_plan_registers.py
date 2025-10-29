import torch
from abc import abstractmethod
from cache_dit.parallelism.parallel_config import ParallelismConfig
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class TensorParallelismPlaner:
    @abstractmethod
    def apply(
        self,
        transformer: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        raise NotImplementedError(
            "apply method must be implemented by subclasses"
        )


class TensorParallelismPlanerRegister:
    _tp_planer_registry = {}

    @classmethod
    def register(
        cls,
        name: str,
        planer_cls: TensorParallelismPlaner,
    ):
        assert (
            name not in cls._tp_planer_registry
        ), f"TensorParallelismPlaner with name {name} is already registered."
        cls._tp_planer_registry[name] = planer_cls

    @classmethod
    def get_planer(
        cls, transformer: str | torch.nn.Module
    ) -> TensorParallelismPlaner:
        if isinstance(transformer, torch.nn.Module):
            name = transformer.__class__.__name__
        else:
            name = transformer
        planer_cls = None
        for planer_name in cls._tp_planer_registry:
            if name.startswith(planer_name):
                planer_cls = cls._tp_planer_registry.get(planer_name)
        if planer_cls is None:
            raise ValueError(f"No planer registered under name: {name}")
        return planer_cls
