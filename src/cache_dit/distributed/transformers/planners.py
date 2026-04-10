"""Register built-in transformer context- and tensor-parallel planners.

The flattened transformer package keeps CP and TP model planners in the same directory, but still
exposes two independent activation paths and fallback planner types. This module centralizes those
activations so runtime entrypoints, summary generation, and tests all resolve planners from one
canonical place.
"""

import importlib
from typing import Dict, List, Tuple

import torch
from torch.distributed.tensor.parallel import ParallelStyle

from ...logger import init_logger
from .register import ContextParallelismPlanner, TensorParallelismPlanner

logger = init_logger(__name__)

__all__ = ["_activate_cp_planners", "_activate_tp_planners"]


class ImportErrorContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer,
    parallelism_config,
    **kwargs,
  ):
    raise ImportError("This ContextParallelismPlanner requires latest diffusers to be installed. "
                      "Please install diffusers from source.")


class ImportErrorTensorParallelismPlanner(TensorParallelismPlanner):

  def _apply(
    self,
    transformer,
    parallelism_config,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    raise ImportError("This TensorParallelismPlanner requires latest diffusers to be installed. "
                      "Please install diffusers from source.")


def _safe_import(module_name: str, class_name: str, fallback_cls):
  try:
    package = __package__ if __package__ is not None else ""
    module = importlib.import_module(module_name, package=package)
    target_class = getattr(module, class_name)
    return target_class
  except (ImportError, AttributeError) as e:
    logger.debug(f"Failed to import {class_name} from {module_name}: {e}")
    return fallback_cls


def _activate_cp_planners():
  """Register all built-in context-parallel transformer planners."""
  FluxContextParallelismPlanner = _safe_import(  # noqa: F841
    ".flux", "FluxContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  QwenImageContextParallelismPlanner = _safe_import(  # noqa: F841
    ".qwen_image", "QwenImageContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  WanContextParallelismPlanner = _safe_import(  # noqa: F841
    ".wan", "WanContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  WanVACEContextParallelismPlanner = _safe_import(  # noqa: F841
    ".wan", "WanVACEContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  LTXVideoContextParallelismPlanner = _safe_import(  # noqa: F841
    ".ltxvideo", "LTXVideoContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  LTX2ContextParallelismPlanner = _safe_import(  # noqa: F841
    ".ltx2", "LTX2ContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  HunyuanImageContextParallelismPlanner = _safe_import(  # noqa: F841
    ".hunyuan", "HunyuanImageContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  HunyuanVideoContextParallelismPlanner = _safe_import(  # noqa: F841
    ".hunyuan", "HunyuanVideoContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  CogVideoXContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cogvideox", "CogVideoXContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  CogView3PlusContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cogview", "CogView3PlusContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  CogView4ContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cogview", "CogView4ContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  CosisIDContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cosisid", "CosisIDContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  ChromaContextParallelismPlanner = _safe_import(  # noqa: F841
    ".chroma", "ChromaContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  PixArtContextParallelismPlanner = _safe_import(  # noqa: F841
    ".pixart", "PixArtContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  DiTContextParallelismPlanner = _safe_import(  # noqa: F841
    ".dit", "DiTContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  Kandinsky5ContextParallelismPlanner = _safe_import(  # noqa: F841
    ".kandinsky", "Kandinsky5ContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  SkyReelsV2ContextParallelismPlanner = _safe_import(  # noqa: F841
    ".skyreels", "SkyReelsV2ContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  Flux2ContextParallelismPlanner = _safe_import(  # noqa: F841
    ".flux2", "Flux2ContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  ZImageContextParallelismPlanner = _safe_import(  # noqa: F841
    ".zimage", "ZImageContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  ChronoEditContextParallelismPlanner = _safe_import(  # noqa: F841
    ".chrono_edit", "ChronoEditContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  OvisImageContextParallelismPlanner = _safe_import(  # noqa: F841
    ".ovis_image", "OvisImageContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  LongCatImageContextParallelismPlanner = _safe_import(  # noqa: F841
    ".longcat_image", "LongCatImageContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  HeliosContextParallelismPlanner = _safe_import(  # noqa: F841
    ".helios", "HeliosContextParallelismPlanner", ImportErrorContextParallelismPlanner)

  try:
    import nunchaku  # noqa: F401

    _nunchaku_available = True
  except ImportError:
    _nunchaku_available = False

  if _nunchaku_available:
    NunchakuFluxContextParallelismPlanner = _safe_import(  # noqa: F841
      ".nunchaku", "NunchakuFluxContextParallelismPlanner", ImportErrorContextParallelismPlanner)
    NunchakuQwenImageContextParallelismPlanner = _safe_import(  # noqa: F841
      ".nunchaku",
      "NunchakuQwenImageContextParallelismPlanner",
      ImportErrorContextParallelismPlanner,
    )
    NunchakuZImageContextParallelismPlanner = _safe_import(  # noqa: F841
      ".nunchaku", "NunchakuZImageContextParallelismPlanner", ImportErrorContextParallelismPlanner)


def _activate_tp_planners():
  """Register all built-in tensor-parallel transformer planners."""
  CogViewTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".cogview", "CogViewTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  FluxTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".flux", "FluxTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  Flux2TensorParallelismPlanner = _safe_import(  # noqa: F841
    ".flux2", "Flux2TensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  HunyuanDiTTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".hunyuan_dit", "HunyuanDiTTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  Kandinsky5TensorParallelismPlanner = _safe_import(  # noqa: F841
    ".kandinsky5", "Kandinsky5TensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  MochiTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".mochi", "MochiTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  LTXVideoTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".ltx_video", "LTXVideoTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  LTX2VideoTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".ltx2_video", "LTX2VideoTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  PixArtTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".pixart", "PixArtTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  QwenImageTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".qwen_image", "QwenImageTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  WanTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".wan", "WanTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  SkyReelsV2TensorParallelismPlanner = _safe_import(  # noqa: F841
    ".skyreels", "SkyReelsV2TensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  ZImageTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".zimage", "ZImageTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  OvisImageTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".ovis_image", "OvisImageTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  LongCatImageTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".longcat_image", "LongCatImageTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  GlmImageTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".glm_image", "GlmImageTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  HeliosTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".helios", "HeliosTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
