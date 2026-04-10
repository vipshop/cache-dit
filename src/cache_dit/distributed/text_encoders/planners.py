import importlib
import torch
from typing import Dict, List, Tuple
from torch.distributed.tensor.parallel import ParallelStyle
from .register import TextEncoderTensorParallelismPlanner
from ...logger import init_logger

logger = init_logger(__name__)


class ImportErrorTextEncoderTensorParallelismPlanner(TextEncoderTensorParallelismPlanner):

  def _apply(
    self,
    text_encoder,
    parallelism_config,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    raise ImportError(
      "This TextEncoderTensorParallelismPlanner requires latest diffusers to be installed. "
      "Please install diffusers from source.")


def _safe_import(module_name: str, class_name: str) -> type[TextEncoderTensorParallelismPlanner]:
  try:
    # e.g., module_name = ".t5_encoder", class_name = "T5EncoderTensorParallelismPlanner"
    package = __package__ if __package__ is not None else ""
    module = importlib.import_module(module_name, package=package)
    target_class = getattr(module, class_name)
    return target_class
  except (ImportError, AttributeError) as e:
    logger.debug(f"Failed to import {class_name} from {module_name}: {e}")
    return ImportErrorTextEncoderTensorParallelismPlanner


def _activate_text_encoder_tp_planners():
  """Function to register all built-in tensor parallelism planners."""
  T5EncoderTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".t5_encoder", "T5EncoderTensorParallelismPlanner")
  UMT5EncoderTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".umt5_encoder", "UMT5EncoderTensorParallelismPlanner")
  MistralTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".mistral", "MistralTensorParallelismPlanner")
  Qwen2_5_VLTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".qwen2_5", "Qwen2_5_VLTensorParallelismPlanner")
  Qwen3TensorParallelismPlanner = _safe_import(  # noqa: F841
    ".qwen3", "Qwen3TensorParallelismPlanner")
  LlamaTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".llama", "LlamaTensorParallelismPlanner")
  GemmaTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".gemma", "GemmaTensorParallelismPlanner")
  GlmTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".glm", "GlmTensorParallelismPlanner")
  GlmImageTensorParallelismPlanner = _safe_import(  # noqa: F841
    ".glm_image", "GlmImageTensorParallelismPlanner")


__all__ = ["_activate_text_encoder_tp_planners"]
