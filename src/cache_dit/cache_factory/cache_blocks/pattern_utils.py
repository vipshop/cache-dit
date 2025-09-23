import torch

from typing import Any
from cache_dit.cache_factory import CachedContext
from cache_dit.cache_factory import CachedContextManager


def patch_cached_stats(
    module: torch.nn.Module | Any,
    cache_context: CachedContext | str = None,
    cache_manager: CachedContextManager = None,
):
    # Patch the cached stats to the module, the cached stats
    # will be reset for each calling of pipe.__call__(**kwargs).
    if module is None or cache_manager is None:
        return

    if cache_context is not None:
        cache_manager.set_context(cache_context)

    # TODO: Patch more cached stats to the module
    module._cached_steps = cache_manager.get_cached_steps()
    module._residual_diffs = cache_manager.get_residual_diffs()
    module._cfg_cached_steps = cache_manager.get_cfg_cached_steps()
    module._cfg_residual_diffs = cache_manager.get_cfg_residual_diffs()


def remove_cached_stats(
    module: torch.nn.Module | Any,
):
    if module is None:
        return

    if hasattr(module, "_cached_steps"):
        del module._cached_steps
    if hasattr(module, "_residual_diffs"):
        del module._residual_diffs
    if hasattr(module, "_cfg_cached_steps"):
        del module._cfg_cached_steps
    if hasattr(module, "_cfg_residual_diffs"):
        del module._cfg_residual_diffs
