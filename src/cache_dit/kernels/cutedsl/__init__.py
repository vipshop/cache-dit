import importlib
from typing import Callable

import torch

from ...logger import init_logger

logger = init_logger(__name__)


def _import_cutedsl_ops_error(
  *args,
  **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
  raise ImportError("CuTe DSL backend requires the `nvidia-cutlass-dsl` runtime to be installed.")


def _safe_import(
  module_name: str,
  func_name: str,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
  try:
    package = __package__ if __package__ is not None else ""
    module = importlib.import_module(module_name, package=package)
    return getattr(module, func_name)
  except (ImportError, AttributeError) as exc:
    logger.debug(f"Failed to import {func_name} from {module_name}: {exc}")
    return _import_cutedsl_ops_error


fused_merge_attn_states = _safe_import(
  "._ops_registery",
  "fused_merge_attn_states",
)

__all__ = ["fused_merge_attn_states"]
