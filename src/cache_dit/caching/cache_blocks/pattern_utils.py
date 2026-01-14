import torch

from typing import Any
from ..cache_contexts import CachedContext
from ..cache_contexts import CachedContextManager
from ..cache_contexts import PrunedContextManager


def apply_stats(
    module: torch.nn.Module | Any,
    cache_context: CachedContext | str = None,
    context_manager: CachedContextManager | PrunedContextManager = None,
):
    # Patch the cached stats to the module, the cached stats
    # will be reset for each calling of pipe.__call__(**kwargs).
    if module is None or context_manager is None:
        return

    if cache_context is not None:
        context_manager.set_context(cache_context)

    # Cache stats for Dual Block Cache
    module._cached_steps = context_manager.get_cached_steps()
    module._residual_diffs = context_manager.get_residual_diffs()
    module._cfg_cached_steps = context_manager.get_cfg_cached_steps()
    module._cfg_residual_diffs = context_manager.get_cfg_residual_diffs()
    # Pruned stats for Dynamic Block Prune
    if not isinstance(context_manager, PrunedContextManager):
        return
    module._pruned_steps = context_manager.get_pruned_steps()
    module._cfg_pruned_steps = context_manager.get_cfg_pruned_steps()
    module._pruned_blocks = context_manager.get_pruned_blocks()
    module._cfg_pruned_blocks = context_manager.get_cfg_pruned_blocks()
    module._actual_blocks = context_manager.get_actual_blocks()
    module._cfg_actual_blocks = context_manager.get_cfg_actual_blocks()
    # Caculate pruned ratio
    if len(module._pruned_blocks) > 0 and sum(module._actual_blocks) > 0:
        module._pruned_ratio = sum(module._pruned_blocks) / sum(module._actual_blocks)
    else:
        module._pruned_ratio = None
    if len(module._cfg_pruned_blocks) > 0 and sum(module._cfg_actual_blocks) > 0:
        module._cfg_pruned_ratio = sum(module._cfg_pruned_blocks) / sum(module._cfg_actual_blocks)
    else:
        module._cfg_pruned_ratio = None


def remove_stats(
    module: torch.nn.Module | Any,
):
    if module is None:
        return

    # Dual Block Cache
    if hasattr(module, "_cached_steps"):
        del module._cached_steps
    if hasattr(module, "_residual_diffs"):
        del module._residual_diffs
    if hasattr(module, "_cfg_cached_steps"):
        del module._cfg_cached_steps
    if hasattr(module, "_cfg_residual_diffs"):
        del module._cfg_residual_diffs

    # Dynamic Block Prune
    if hasattr(module, "_pruned_steps"):
        del module._pruned_steps
    if hasattr(module, "_cfg_pruned_steps"):
        del module._cfg_pruned_steps
    if hasattr(module, "_pruned_blocks"):
        del module._pruned_blocks
    if hasattr(module, "_cfg_pruned_blocks"):
        del module._cfg_pruned_blocks
    if hasattr(module, "_actual_blocks"):
        del module._actual_blocks
    if hasattr(module, "_cfg_actual_blocks"):
        del module._cfg_actual_blocks
    if hasattr(module, "_pruned_ratio"):
        del module._pruned_ratio
    if hasattr(module, "_cfg_pruned_ratio"):
        del module._cfg_pruned_ratio
