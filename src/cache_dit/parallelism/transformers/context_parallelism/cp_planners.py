"""Register built-in transformer context-parallel planners.

This module mirrors the plan concepts used by diffusers' `_modeling_parallel` support. Each
planner ultimately returns a `ContextParallelModelPlan`, which maps module names to split/gather
rules applied around that module's forward pass.

In practice a plan usually contains three kinds of entries:

- root-input rules such as `""`, which shard model inputs before the first forward call;
- intermediate rules such as `"rope"` or block names, which describe how layer outputs should be
  split or preserved across ranks;
- final gather rules such as `"proj_out"`, which restore full-sequence tensors before returning to
  user code.
"""
import importlib
from ....logger import init_logger
from .cp_plan_registers import ContextParallelismPlanner

logger = init_logger(__name__)


class ImportErrorContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer,
    parallelism_config,
    **kwargs,
  ):
    raise ImportError("This ContextParallelismPlanner requires latest diffusers to be installed. "
                      "Please install diffusers from source.")


def _safe_import(module_name: str, class_name: str) -> type[ContextParallelismPlanner]:
  try:
    # e.g., module_name = ".cp_plan_dit", class_name = "DiTContextParallelismPlanner"
    package = __package__ if __package__ is not None else ""
    module = importlib.import_module(module_name, package=package)
    target_class = getattr(module, class_name)
    return target_class
  except (ImportError, AttributeError) as e:
    logger.debug(f"Failed to import {class_name} from {module_name}: {e}")
    return ImportErrorContextParallelismPlanner


def _activate_cp_planners():
  """Function to register all built-in context parallelism planners."""
  FluxContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_flux", "FluxContextParallelismPlanner")
  QwenImageContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_qwen_image", "QwenImageContextParallelismPlanner")
  WanContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_wan", "WanContextParallelismPlanner")
  WanVACEContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_wan", "WanVACEContextParallelismPlanner")
  LTXVideoContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_ltxvideo", "LTXVideoContextParallelismPlanner")
  LTX2ContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_ltx2", "LTX2ContextParallelismPlanner")
  HunyuanImageContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_hunyuan", "HunyuanImageContextParallelismPlanner")
  HunyuanVideoContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_hunyuan", "HunyuanVideoContextParallelismPlanner")
  CogVideoXContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_cogvideox", "CogVideoXContextParallelismPlanner")
  CogView3PlusContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_cogview", "CogView3PlusContextParallelismPlanner")
  CogView4ContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_cogview", "CogView4ContextParallelismPlanner")
  CosisIDContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_cosisid", "CosisIDContextParallelismPlanner")
  ChromaContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_chroma", "ChromaContextParallelismPlanner")
  PixArtContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_pixart", "PixArtContextParallelismPlanner")
  DiTContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_dit", "DiTContextParallelismPlanner")
  Kandinsky5ContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_kandinsky", "Kandinsky5ContextParallelismPlanner")
  SkyReelsV2ContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_skyreels", "SkyReelsV2ContextParallelismPlanner")
  Flux2ContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_flux2", "Flux2ContextParallelismPlanner")
  ZImageContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_zimage", "ZImageContextParallelismPlanner")
  ChronoEditContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_chrono_edit", "ChronoEditContextParallelismPlanner")
  OvisImageContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_ovis_image", "OvisImageContextParallelismPlanner")
  LongCatImageContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_longcat_image", "LongCatImageContextParallelismPlanner")
  HeliosContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cp_plan_helios", "HeliosContextParallelismPlanner")

  try:
    import nunchaku  # noqa: F401

    _nunchaku_available = True
  except ImportError:
    _nunchaku_available = False

  if _nunchaku_available:
    NunchakuFluxContextParallelismPlanner = _safe_import(  # noqa: F841
      ".cp_plan_nunchaku", "NunchakuFluxContextParallelismPlanner")
    NunchakuQwenImageContextParallelismPlanner = _safe_import(  # noqa: F841
      ".cp_plan_nunchaku", "NunchakuQwenImageContextParallelismPlanner")
    NunchakuZImageContextParallelismPlanner = _safe_import(  # noqa: F841
      ".cp_plan_nunchaku", "NunchakuZImageContextParallelismPlanner")


__all__ = ["_activate_cp_planners"]
