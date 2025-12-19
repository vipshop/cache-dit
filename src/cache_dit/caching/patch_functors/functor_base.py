import torch
from abc import abstractmethod

from cache_dit.logger import init_logger
from cache_dit.envs import ENV

logger = init_logger(__name__)


class PatchFunctor:

    def apply(
        self,
        transformer: torch.nn.Module,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
        if not ENV.CACHE_DIT_PATCH_FUNCTOR_DISABLE_DIFFUSERS_CHECK:
            if not self.is_from_diffusers(transformer):
                return transformer
        return self._apply(transformer, *args, **kwargs)

    @abstractmethod
    def _apply(
        self,
        transformer: torch.nn.Module,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
        raise NotImplementedError("_apply method is not implemented.")

    @classmethod
    def is_from_diffusers(cls, transformer: torch.nn.Module) -> bool:
        if ENV.CACHE_DIT_PATCH_FUNCTOR_DISABLE_DIFFUSERS_CHECK:
            return True
        if transformer.__module__.startswith("diffusers"):
            return True
        logger.warning("Found transformer not from diffusers. Skipping patch functor.")
        return False
