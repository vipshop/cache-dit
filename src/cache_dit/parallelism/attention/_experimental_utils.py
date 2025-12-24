import torch
import functools
import diffusers
from typing import List, Union

try:
    from diffusers import ContextParallelConfig  # noqa: F401
    from diffusers.hooks.context_parallel import (
        _find_submodule_by_name as _find_submodule_by_name_for_context_parallel,
    )

    def _is_diffusers_parallelism_available() -> bool:
        return True

except ImportError:
    ContextParallelConfig = None
    _find_submodule_by_name_for_context_parallel = None

    def _is_diffusers_parallelism_available() -> bool:
        return False


from cache_dit.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "_is_diffusers_parallelism_available",
    "_maybe_patch_find_submodule",
]

# NOTE: Add this utility function to diffusers to support ModuleDict, such as 'all_final_layer', like ZImage
# Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/hooks/context_parallel.py#L283
# This function is only used when diffusers native context parallelism is enabled and can compatible with the
# original one.
if (
    _is_diffusers_parallelism_available()
    and _find_submodule_by_name_for_context_parallel is not None
):

    @functools.wraps(_find_submodule_by_name_for_context_parallel)
    def _patch_find_submodule_by_name(
        model: torch.nn.Module, name: str
    ) -> Union[torch.nn.Module, List[torch.nn.Module]]:
        if name == "":
            return model
        first_atom, remaining_name = name.split(".", 1) if "." in name else (name, "")
        if first_atom == "*":
            if not isinstance(model, torch.nn.ModuleList):
                raise ValueError("Wildcard '*' can only be used with ModuleList")
            submodules = []
            for submodule in model:
                subsubmodules = _patch_find_submodule_by_name(submodule, remaining_name)
                if not isinstance(subsubmodules, list):
                    if isinstance(subsubmodules, torch.nn.ModuleDict):
                        subsubmodules = list(subsubmodules.values())
                    else:
                        subsubmodules = [subsubmodules]
                submodules.extend(subsubmodules)
            return submodules
        else:
            if hasattr(model, first_atom):
                submodule = getattr(model, first_atom)
                if isinstance(submodule, torch.nn.ModuleDict):  # e.g, 'all_final_layer' in ZImage
                    if remaining_name == "":
                        submodule = list(submodule.values())
                        # Make sure all values are Modules, not support other complex cases.
                        for v in submodule:
                            if not isinstance(v, torch.nn.Module):
                                raise ValueError(
                                    f"Value '{v}' in ModuleDict '{first_atom}' is not a Module"
                                )
                        return submodule
                    else:
                        raise ValueError(
                            f"Cannot access submodule '{remaining_name}' of ModuleDict '{first_atom}' directly. "
                            f"Please specify the key of the ModuleDict first."
                        )
                return _patch_find_submodule_by_name(submodule, remaining_name)
            else:
                raise ValueError(
                    f"'{first_atom}' is not a submodule of '{model.__class__.__name__}'"
                )

    def _maybe_patch_find_submodule():
        if (
            diffusers.hooks.context_parallel._find_submodule_by_name
            != _patch_find_submodule_by_name
        ):
            diffusers.hooks.context_parallel._find_submodule_by_name = _patch_find_submodule_by_name
            logger.debug("Patched _find_submodule_by_name to support ModuleDict.")

else:

    def _maybe_patch_find_submodule():
        pass
