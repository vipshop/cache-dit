import importlib

from ...logger import init_logger

logger = init_logger(__name__)

__all__ = []


def _safe_import(module_name: str, planner_name: str):
  try:
    module = importlib.import_module(f".{module_name}", package=__package__)
  except Exception as exc:
    logger.debug(f"Failed to activate async Ulysses planner {planner_name}: {exc}")
    globals()[planner_name] = None
    return None

  planner = getattr(module, planner_name, None)
  if planner is None:
    logger.debug(f"Async Ulysses planner {planner_name} is unavailable in module {module_name}")
    globals()[planner_name] = None
    return None

  globals()[planner_name] = planner
  __all__.append(planner_name)
  return planner


FluxAsyncUlyssesPlanner = _safe_import("flux", "FluxAsyncUlyssesPlanner")
Flux2AsyncUlyssesPlanner = _safe_import("flux2", "Flux2AsyncUlyssesPlanner")
QwenImageAsyncUlyssesPlanner = _safe_import("qwen_image", "QwenImageAsyncUlyssesPlanner")
ZImageAsyncUlyssesPlanner = _safe_import("zimage", "ZImageAsyncUlyssesPlanner")
OvisImageAsyncUlyssesPlanner = _safe_import("ovis_image", "OvisImageAsyncUlyssesPlanner")
LongCatImageAsyncUlyssesPlanner = _safe_import("longcat_image", "LongCatImageAsyncUlyssesPlanner")
