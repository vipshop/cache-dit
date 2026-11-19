import torch

from typing import Any
from ..cache_contexts import CachedContext
from ..cache_contexts import CachedContextManager


def apply_stats(
  module: torch.nn.Module | Any,
  cache_context: CachedContext | str = None,
  context_manager: CachedContextManager = None,
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
  module._accumulated_cached_steps = context_manager.get_accumulated_cached_steps()
  module._cfg_accumulated_cached_steps = context_manager.get_cfg_accumulated_cached_steps()
  module._accumulated_executed_steps = context_manager.get_accumulated_executed_steps()
  module._accumulated_transformer_executed_steps = (
    context_manager.get_accumulated_transformer_executed_steps())


def remove_stats(module: torch.nn.Module | Any, ):
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
  if hasattr(module, "_accumulated_cached_steps"):
    del module._accumulated_cached_steps
  if hasattr(module, "_cfg_accumulated_cached_steps"):
    del module._cfg_accumulated_cached_steps
  if hasattr(module, "_accumulated_executed_steps"):
    del module._accumulated_executed_steps
  if hasattr(module, "_accumulated_transformer_executed_steps"):
    del module._accumulated_transformer_executed_steps
