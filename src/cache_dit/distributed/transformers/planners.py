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
  Flux3ContextParallelismPlanner = _safe_import(  # noqa: F841
    ".flux3", "Flux3ContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  MiniMaxH3ContextParallelismPlanner = _safe_import(  # noqa: F841
    ".minimax_h3", "MiniMaxH3ContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  FastH3ContextParallelismPlanner = _safe_import(  # noqa: F841
    ".fast_h3", "FastH3ContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  # "QwenImage21" is imported before "QwenImage": planner lookup matches the first registered
  # prefix, and `QwenImage21Transformer2DModel` also starts with "QwenImage".
  QwenImage21ContextParallelismPlanner = _safe_import(  # noqa: F841
    ".qwen_image21", "QwenImage21ContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  QwenImageContextParallelismPlanner = _safe_import(  # noqa: F841
    ".qwen_image", "QwenImageContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  # "WanAnimate2" is imported before "Wan": planner lookup matches the first registered prefix, and
  # `WanAnimate2Transformer3DModel` also starts with "Wan".
  WanAnimate2ContextParallelismPlanner = _safe_import(  # noqa: F841
    ".wan_animate_2", "WanAnimate2ContextParallelismPlanner", ImportErrorContextParallelismPlanner)
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
  LongCatAudioDiTContextParallelismPlanner = _safe_import(  # noqa: F841
    ".longcat_audio_dit", "LongCatAudioDiTContextParallelismPlanner",
    ImportErrorContextParallelismPlanner)
  StableAudio3ContextParallelismPlanner = _safe_import(  # noqa: F841
    ".stable_audio_3", "StableAudio3ContextParallelismPlanner",
    ImportErrorContextParallelismPlanner)
  LongCatImageContextParallelismPlanner = _safe_import(  # noqa: F841
    ".longcat_image", "LongCatImageContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  HeliosContextParallelismPlanner = _safe_import(  # noqa: F841
    ".helios", "HeliosContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  BooguImageContextParallelismPlanner = _safe_import(  # noqa: F841
    ".boogu_image", "BooguImageContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  ErnieImageContextParallelismPlanner = _safe_import(  # noqa: F841
    ".ernie_image", "ErnieImageContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  Krea2ContextParallelismPlanner = _safe_import(  # noqa: F841
    ".krea2", "Krea2ContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  BriaFiboContextParallelismPlanner = _safe_import(  # noqa: F841
    ".bria_fibo", "BriaFiboContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  BriaFibo2ContextParallelismPlanner = _safe_import(  # noqa: F841
    ".bria_fibo2", "BriaFibo2ContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  JoyImageContextParallelismPlanner = _safe_import(  # noqa: F841
    ".joy_image", "JoyImageContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  AnyFlowContextParallelismPlanner = _safe_import(  # noqa: F841
    ".anyflow", "AnyFlowContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  PrxContextParallelismPlanner = _safe_import(  # noqa: F841
    ".prx", "PrxContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  CosmosContextParallelismPlanner = _safe_import(  # noqa: F841
    ".cosmos", "CosmosContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  MotifVideoContextParallelismPlanner = _safe_import(  # noqa: F841
    ".motif_video", "MotifVideoContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  Ideogram4ContextParallelismPlanner = _safe_import(  # noqa: F841
    ".ideogram4", "Ideogram4ContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  MOSSImageContextParallelismPlanner = _safe_import(  # noqa: F841
    ".moss_image", "MOSSImageContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  Kandinsky6ContextParallelismPlanner = _safe_import(  # noqa: F841
    ".kandinsky6", "Kandinsky6ContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  HiDreamO1ImageContextParallelismPlanner = _safe_import(  # noqa: F841
    ".hidream_o1_image", "HiDreamO1ImageContextParallelismPlanner",
    ImportErrorContextParallelismPlanner)
  SanaVideo2ContextParallelismPlanner = _safe_import(  # noqa: F841
    ".sana_video2", "SanaVideo2ContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  SanaWMContextParallelismPlanner = _safe_import(  # noqa: F841
    ".sana_wm", "SanaWMContextParallelismPlanner", ImportErrorContextParallelismPlanner)
  LingBotVideoContextParallelismPlanner = _safe_import(  # noqa: F841
    ".lingbot_video", "LingBotVideoContextParallelismPlanner", ImportErrorContextParallelismPlanner)

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
  Flux3TensorParallelismPlanner = _safe_import(  # noqa: F841
    ".flux3", "Flux3TensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  MiniMaxH3TensorParallelismPlanner = _safe_import(  # noqa: F841
    ".minimax_h3", "MiniMaxH3TensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  FastH3TensorParallelismPlanner = _safe_import(  # noqa: F841
    ".fast_h3", "FastH3TensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
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
  QwenImage21TensorParallelismPlanner = _safe_import(  # noqa: F841
    ".qwen_image21", "QwenImage21TensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  QwenImageTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".qwen_image", "QwenImageTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  # Before "Wan" for the same prefix reason as the CP planner above.
  WanAnimate2TensorParallelismPlanner = _safe_import(  # noqa: F841
    ".wan_animate_2", "WanAnimate2TensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  WanTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".wan", "WanTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  SkyReelsV2TensorParallelismPlanner = _safe_import(  # noqa: F841
    ".skyreels", "SkyReelsV2TensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  ZImageTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".zimage", "ZImageTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  OvisImageTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".ovis_image", "OvisImageTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  LongCatAudioDiTTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".longcat_audio_dit", "LongCatAudioDiTTensorParallelismPlanner",
    ImportErrorTensorParallelismPlanner)
  StableAudio3TensorParallelismPlanner = _safe_import(  # noqa: F841
    ".stable_audio_3", "StableAudio3TensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  LongCatImageTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".longcat_image", "LongCatImageTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  GlmImageTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".glm_image", "GlmImageTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  HeliosTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".helios", "HeliosTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  BooguImageTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".boogu_image", "BooguImageTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  ErnieImageTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".ernie_image", "ErnieImageTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  Krea2TensorParallelismPlanner = _safe_import(  # noqa: F841
    ".krea2", "Krea2TensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  BriaFiboTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".bria_fibo", "BriaFiboTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  BriaFibo2TensorParallelismPlanner = _safe_import(  # noqa: F841
    ".bria_fibo2", "BriaFibo2TensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  JoyImageTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".joy_image", "JoyImageTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  AnyFlowTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".anyflow", "AnyFlowTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  PrxTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".prx", "PrxTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  CosmosTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".cosmos", "CosmosTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  MotifVideoTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".motif_video", "MotifVideoTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  Ideogram4TensorParallelismPlanner = _safe_import(  # noqa: F841
    ".ideogram4", "Ideogram4TensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  MOSSImageTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".moss_image", "MOSSImageTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  NucleusMoETensorParallelismPlanner = _safe_import(  # noqa: F841
    ".nucleus_moe", "NucleusMoETensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  HiDreamO1ImageTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".hidream_o1_image", "HiDreamO1ImageTensorParallelismPlanner",
    ImportErrorTensorParallelismPlanner)
  Kandinsky6TensorParallelismPlanner = _safe_import(  # noqa: F841
    ".kandinsky6", "Kandinsky6TensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  SanaVideo2TensorParallelismPlanner = _safe_import(  # noqa: F841
    ".sana_video2", "SanaVideo2TensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  SanaWMTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".sana_wm", "SanaWMTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
  LingBotVideoTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".lingbot_video", "LingBotVideoTensorParallelismPlanner", ImportErrorTensorParallelismPlanner)
