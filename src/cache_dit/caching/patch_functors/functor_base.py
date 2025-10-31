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
