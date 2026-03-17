import torch
from abc import abstractmethod

from ...logger import init_logger
from ...envs import ENV
from ...utils import check_parallelized
from ...utils import check_patched

logger = init_logger(__name__)


class PatchFunctor:

    def apply(
        self,
        transformer: torch.nn.Module,
        *args,
        **kwargs,
    ) -> torch.nn.Module:
        if check_patched(transformer):
            logger.warning(
                "Transformer is already patched. Skipping patch functor. "
                f"Transformer: {transformer.__class__.__name__}, "
                f"Patch Functor: {self.__class__.__name__}"
            )
            return transformer

        if not ENV.CACHE_DIT_PATCH_FUNCTOR_DISABLE_DIFFUSERS_CHECK:
            if not self.is_from_diffusers(transformer):
                return transformer

        if check_parallelized(transformer):
            logger.warning(
                "Patch functor is not applied for parallelized transformer. "
                f"Skipping {self.__class__.__name__} for {transformer.__class__.__name__}."
            )
            return transformer
        transformer = self._apply(transformer, *args, **kwargs)
        self.logging_patched(transformer)
        return transformer

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

    def logging_patched(self, transformer: torch.nn.Module):
        is_patched = getattr(transformer, "_is_patched", False)
        logger.info(
            f"Applied patch functor {self.__class__.__name__} for "
            f"{transformer.__class__.__name__}, patched: {is_patched}"
        )
