"""Register built-in ControlNet context-parallel planners.

The ControlNet planner registry follows the same split/gather plan model as the transformer
registry, but targets auxiliary ControlNet branches. Each planner resolves a
`_ContextParallelModelPlan` that describes which tensors should be sharded before a module runs
and which tensors should be gathered again before returning to the caller.
"""

import importlib
from ...logger import init_logger
from .register import ControlNetContextParallelismPlanner

logger = init_logger(__name__)


class ImportErrorContextParallelismPlanner(ControlNetContextParallelismPlanner):

  def _apply(
    self,
    controlnet,
    parallelism_config,
    **kwargs,
  ):
    raise ImportError(
      "This ControlNetContextParallelismPlanner requires latest diffusers to be installed. "
      "Please install diffusers from source.")


def _safe_import(module_name: str, class_name: str) -> type[ControlNetContextParallelismPlanner]:
  try:
    # e.g., module_name = ".zimage_controlnet", class_name = "ZImageControlNetContextParallelismPlanner"
    package = __package__ if __package__ is not None else ""
    module = importlib.import_module(module_name, package=package)
    target_class = getattr(module, class_name)
    return target_class
  except (ImportError, AttributeError) as e:
    logger.debug(f"Failed to import {class_name} from {module_name}: {e}")
    return ImportErrorContextParallelismPlanner


def _activate_controlnet_cp_planners():
  """Function to register all built-in context parallelism planners."""
  ZImageControlNetContextParallelismPlanner = _safe_import(  # noqa: F841
    ".zimage_controlnet", "ZImageControlNetContextParallelismPlanner")


__all__ = ["_activate_controlnet_cp_planners"]
