from collections.abc import Callable
from functools import cache, partial

import torch

from ...logger import init_logger
from ..config import ParallelismConfig
from .planners import _activate_text_encoder_tp_planners
from .register import TextEncoderTensorParallelismPlannerRegister

logger = init_logger(__name__)
_activate_builtin_planners = cache(_activate_text_encoder_tp_planners)


@cache
def _require_parallelism_extra() -> None:
  try:
    import einops  # noqa: F401
  except ImportError as exc:
    raise ImportError("parallelism functionality requires the 'parallelism' extra dependencies. "
                      "Install with:\npip install cache-dit[parallelism]") from exc


def _prepare_text_encoder_tp(
  text_encoder: torch.nn.Module,
  parallelism_config: ParallelismConfig,
) -> tuple[Callable[..., torch.nn.Module], bool]:
  _require_parallelism_extra()
  get_planner = TextEncoderTensorParallelismPlannerRegister.get_planner
  try:
    planner_cls = get_planner(text_encoder)
  except ValueError:
    _activate_builtin_planners()
    planner_cls = get_planner(text_encoder)
  planner = planner_cls()
  if parallelism_config.local_weight_shards and planner_cls._supports_local_weight_shards:
    mesh = planner._preflight_local_weight_shards(text_encoder, parallelism_config)
    return partial(planner.apply, local_shard_mesh=mesh), True
  return planner.apply, False


def parallelize_text_encoder(
  text_encoder: torch.nn.Module,
  parallelism_config: ParallelismConfig | None,
  *,
  tp_apply: Callable[..., torch.nn.Module] | None = None,
) -> torch.nn.Module:
  """Parallelize one text encoder with the configured TP strategy.

  :param text_encoder: Text encoder module to parallelize.
  :param parallelism_config: Parallelism configuration shared with the transformer.
  :param tp_apply: Planner apply method prepared before installing parallelism.
  :returns: The parallelized text encoder.
  """
  assert isinstance(
    text_encoder, torch.nn.Module
  ), f"text_encoder must be an instance of torch.nn.Module, but got {type(text_encoder)}"
  if getattr(text_encoder, "_is_parallelized", False):
    logger.warning("The text encoder is already parallelized. Skipping parallelism enabling.")
    return text_encoder

  if parallelism_config is None:
    return text_encoder

  if tp_apply is None:
    tp_apply, _ = _prepare_text_encoder_tp(text_encoder, parallelism_config)
  text_encoder = tp_apply(text_encoder=text_encoder, parallelism_config=parallelism_config)

  text_encoder._is_parallelized = True  # type: ignore[attr-defined]
  text_encoder._parallelism_config = parallelism_config  # type: ignore[attr-defined]

  logger.info(f"Parallelize Text Encoder: {text_encoder.__class__.__name__}, "
              f"id:{id(text_encoder)}, {parallelism_config.strify(True, text_encoder=True)}")

  return text_encoder
