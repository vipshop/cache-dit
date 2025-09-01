import torch

from typing import Any
from cache_dit.cache_factory import CachedContext


@torch.compiler.disable
def patch_cached_stats(
    module: torch.nn.Module | Any, cache_context: str = None
):
    # Patch the cached stats to the module, the cached stats
    # will be reset for each calling of pipe.__call__(**kwargs).
    if module is None:
        return

    if cache_context is not None:
        CachedContext.set_cache_context(cache_context)

    # TODO: Patch more cached stats to the module
    module._cached_steps = CachedContext.get_cached_steps()
    module._residual_diffs = CachedContext.get_residual_diffs()
    module._cfg_cached_steps = CachedContext.get_cfg_cached_steps()
    module._cfg_residual_diffs = CachedContext.get_cfg_residual_diffs()
