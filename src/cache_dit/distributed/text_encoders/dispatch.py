from typing import Optional

try:
  import einops  # noqa: F401
except ImportError as exc:
  _TP_IMPORT_ERROR = exc
else:
  _TP_IMPORT_ERROR = None

import torch

from ...logger import init_logger
from ..config import ParallelismConfig
from .planners import _activate_text_encoder_tp_planners
from .register import TextEncoderTensorParallelismPlannerRegister

logger = init_logger(__name__)


def _ensure_text_encoder_tp_planners_activated() -> None:
  if _TP_IMPORT_ERROR is not None:
    raise ImportError("parallelism functionality requires the 'parallelism' extra dependencies. "
                      "Install with:\npip install cache-dit[parallelism]") from _TP_IMPORT_ERROR
  if TextEncoderTensorParallelismPlannerRegister.supported_planners()[0] == 0:
    _activate_text_encoder_tp_planners()


def maybe_enable_tensor_parallelism(
  text_encoder: torch.nn.Module,
  parallelism_config: Optional[ParallelismConfig],
) -> torch.nn.Module:
  assert isinstance(
    text_encoder, torch.nn.Module
  ), f"text_encoder must be an instance of torch.nn.Module, but got {type(text_encoder)}"

  if parallelism_config is None:
    return text_encoder

  _ensure_text_encoder_tp_planners_activated()

  # We don't check backend here because text encoder may use different
  # parallelism backend with transformer.
  return TextEncoderTensorParallelismPlannerRegister.get_planner(text_encoder)().apply(
    text_encoder=text_encoder,
    parallelism_config=parallelism_config,
  )


def maybe_enable_parallelism_for_text_encoder(
  text_encoder: torch.nn.Module,
  parallelism_config: Optional[ParallelismConfig],
) -> torch.nn.Module:
  assert isinstance(
    text_encoder, torch.nn.Module
  ), f"text_encoder must be an instance of torch.nn.Module, but got {type(text_encoder)}"
  if getattr(text_encoder, "_is_parallelized", False):
    logger.warning("The text encoder is already parallelized. Skipping parallelism enabling.")
    return text_encoder

  if parallelism_config is None:
    return text_encoder

  text_encoder = maybe_enable_tensor_parallelism(
    text_encoder=text_encoder,
    parallelism_config=parallelism_config,
  )

  text_encoder._is_parallelized = True  # type: ignore[attr-defined]
  text_encoder._parallelism_config = parallelism_config  # type: ignore[attr-defined]

  logger.info(f"Parallelize Text Encoder: {text_encoder.__class__.__name__}, "
              f"id:{id(text_encoder)}, {parallelism_config.strify(True, text_encoder=True)}")

  return text_encoder
