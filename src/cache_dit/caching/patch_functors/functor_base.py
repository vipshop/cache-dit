import torch
from abc import abstractmethod

from cache_dit.logger import init_logger

logger = init_logger(__name__)


class PatchFunctor:

    @abstractmethod
    def apply(
        self,
        transformer: torch.nn.Module,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
        raise NotImplementedError("apply method is not implemented.")

    @staticmethod
    def assert_from_diffusers(transformer: torch.nn.Module) -> bool:
        assert transformer.__module__.startswith(
            "diffusers"
        ), "Internal PatchFunctor in cache-dit only support diffusers transformers now."
