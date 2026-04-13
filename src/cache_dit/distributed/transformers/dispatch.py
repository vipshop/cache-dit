from typing import Optional

try:
  import einops  # noqa: F401
except ImportError as exc:
  _TP_IMPORT_ERROR = exc
else:
  _TP_IMPORT_ERROR = None

import torch

from diffusers.models.modeling_utils import ModelMixin

from ...attention import _maybe_register_custom_attn_backends
from ...distributed.core import _ContextParallelConfig, _enable_context_parallelism
from ...logger import init_logger
from ..backend import ParallelismBackend
from ..config import ParallelismConfig
from .planners import _activate_cp_planners, _activate_tp_planners
from .register import ContextParallelismPlannerRegister, TensorParallelismPlannerRegister

logger = init_logger(__name__)


def _ensure_cp_planners_activated() -> None:
  _maybe_register_custom_attn_backends()
  if ContextParallelismPlannerRegister.supported_planners()[0] == 0:
    _activate_cp_planners()


def _ensure_tp_planners_activated() -> None:
  if _TP_IMPORT_ERROR is not None:
    raise ImportError("parallelism functionality requires the 'parallelism' extra dependencies. "
                      "Install with:\npip install cache-dit[parallelism]") from _TP_IMPORT_ERROR
  if TensorParallelismPlannerRegister.supported_planners()[0] == 0:
    _activate_tp_planners()


def _parallelize_transformer_context(
  transformer: torch.nn.Module | ModelMixin,
  parallelism_config: Optional[ParallelismConfig] = None,
) -> torch.nn.Module:
  """Enable context parallelism for one transformer instance.

  :param transformer: Transformer module or diffusers model to patch.
  :param parallelism_config: Parallelism configuration describing CP settings.
  :returns: The patched transformer.
  """
  assert isinstance(
    transformer,
    (torch.nn.Module,
     ModelMixin)), ("transformer must be an instance of torch.nn.Module or ModelMixin, "
                    f"but got {type(transformer)}")
  if parallelism_config is None:
    return transformer

  assert isinstance(
    parallelism_config,
    ParallelismConfig), ("parallelism_config must be an instance of ParallelismConfig"
                         f" but got {type(parallelism_config)}")

  _ensure_cp_planners_activated()

  if parallelism_config.cp_enabled():
    cp_config = _ContextParallelConfig(
      ulysses_degree=parallelism_config.ulysses_size,
      ring_degree=parallelism_config.ring_size,
      convert_to_fp32=parallelism_config.ring_convert_to_fp32,
      rotate_method=parallelism_config.ring_rotate_method,
      ulysses_anything=parallelism_config.ulysses_anything,
      ulysses_float8=parallelism_config.ulysses_float8,
      ulysses_async=parallelism_config.ulysses_async,
    )
    if parallelism_config.hybrid_enabled():
      cp_config.setup(
        rank=parallelism_config._cp_rank,
        world_size=parallelism_config._cp_world_size,
        device=parallelism_config._device,
        mesh=parallelism_config._cp_mesh,
      )

    cp_plan = parallelism_config.cp_plan
    if cp_plan is not None:
      logger.info(f"Using custom context parallelism plan: {cp_plan}")
    else:
      cp_plan = ContextParallelismPlannerRegister.get_planner(transformer)().apply(
        transformer=transformer,
        parallelism_config=parallelism_config,
      )

    _enable_context_parallelism(transformer, config=cp_config, cp_plan=cp_plan)
    _maybe_patch_parallel_config(transformer)

  return transformer


def _parallelize_transformer_tensor(
  transformer: torch.nn.Module | ModelMixin,
  parallelism_config: Optional[ParallelismConfig],
) -> torch.nn.Module:
  """Enable tensor parallelism for one transformer instance.

  :param transformer: Transformer module or diffusers model to patch.
  :param parallelism_config: Parallelism configuration describing TP settings.
  :returns: The patched transformer.
  """
  assert isinstance(
    transformer,
    (torch.nn.Module,
     ModelMixin)), ("transformer must be an instance of torch.nn.Module or ModelMixin, "
                    f"but got {type(transformer)}")
  assert isinstance(transformer,
                    ModelMixin), ("transformer must be an instance of diffusers' ModelMixin, "
                                  f"but got {type(transformer)}")
  if parallelism_config is None:
    return transformer
  if not parallelism_config.tp_enabled():
    return transformer

  _ensure_tp_planners_activated()
  return TensorParallelismPlannerRegister.get_planner(transformer)().apply(
    transformer=transformer,
    parallelism_config=parallelism_config,
  )


def _maybe_patch_parallel_config(
  transformer: torch.nn.Module,
  **kwargs,
) -> torch.nn.Module:
  cls_name = transformer.__class__.__name__
  if not cls_name.startswith("Nunchaku"):
    return transformer

  try:
    from nunchaku.models.transformers.transformer_flux_v2 import (
      NunchakuFluxAttention,
      NunchakuFluxFA2Processor,
      NunchakuFluxTransformer2DModelV2,
    )
    from nunchaku.models.transformers.transformer_qwenimage import (
      NunchakuQwenAttention,
      NunchakuQwenImageNaiveFA2Processor,
      NunchakuQwenImageTransformer2DModel,
    )
    from nunchaku.models.transformers.transformer_zimage import (
      NunchakuZImageAttention,
      NunchakuZImageTransformer2DModel,
      NunchakuZSingleStreamAttnProcessor,
    )
  except ImportError:
    raise ImportError("NunchakuZImageTransformer2DModel, NunchakuFluxTransformer2DModelV2 and "
                      "NunchakuQwenImageTransformer2DModel requires the 'nunchaku' package. "
                      "Please install nunchaku>=1.10 before using the context parallelism for "
                      "nunchaku 4-bits models.")

  assert isinstance(
    transformer,
    (
      NunchakuFluxTransformer2DModelV2,
      NunchakuQwenImageTransformer2DModel,
      NunchakuZImageTransformer2DModel,
    ),
  )
  config = getattr(transformer, "_cp_config", None)
  if config is None:
    config = getattr(transformer, "_parallel_config", None)
  if config is None:
    logger.warning(f"The transformer {cls_name} does not have _cp_config attribute. "
                   "Skipping patching native parallel config.")
    return transformer

  attention_classes = (
    NunchakuFluxAttention,
    NunchakuFluxFA2Processor,
    NunchakuQwenAttention,
    NunchakuQwenImageNaiveFA2Processor,
    NunchakuZImageAttention,
    NunchakuZSingleStreamAttnProcessor,
  )
  for module in transformer.modules():
    if not isinstance(module, attention_classes):
      continue
    processor = getattr(module, "processor", None)
    if processor is None:
      continue
    if not hasattr(processor, "_cp_config"):
      processor._cp_config = None
    if not hasattr(processor, "_parallel_config"):
      processor._parallel_config = None
    if getattr(processor, "_cp_config", None) is not None:
      logger.warning(f"The attention processor {processor.__class__.__name__} already has "
                     "_cp_config attribute set. Skipping patching native parallel config.")
      continue
    processor._cp_config = config
    processor._parallel_config = config

  return transformer


def parallelize_transformer(
  transformer: torch.nn.Module | ModelMixin,
  parallelism_config: Optional[ParallelismConfig],
) -> torch.nn.Module:
  """Dispatch transformer parallelization by configured backend.

  :param transformer: Transformer module or diffusers model to parallelize.
  :param parallelism_config: Parallelism configuration selecting the backend and layout.
  :returns: The transformer after backend-specific parallelization.
  """
  assert isinstance(
    transformer,
    (torch.nn.Module,
     ModelMixin)), ("transformer must be an instance of torch.nn.Module or ModelMixin, "
                    f"but got {type(transformer)}")

  if parallelism_config is None:
    return transformer

  # Currently, we can dispatch the parallelism based on the backend type.
  if parallelism_config.backend == ParallelismBackend.NATIVE_HYBRID:
    return _parallelize_transformer_hybrid(
      transformer=transformer,
      parallelism_config=parallelism_config,
    )
  elif parallelism_config.backend == ParallelismBackend.NATIVE_DIFFUSER:
    return _parallelize_transformer_cp(
      transformer=transformer,
      parallelism_config=parallelism_config,
    )
  elif parallelism_config.backend == ParallelismBackend.NATIVE_PYTORCH:
    return _parallelize_transformer_tp(
      transformer=transformer,
      parallelism_config=parallelism_config,
    )
  else:
    raise ValueError(f"{parallelism_config.backend} backend is not supported yet")


def _parallelize_transformer_hybrid(
  transformer: torch.nn.Module | ModelMixin,
  parallelism_config: ParallelismConfig,
):
  assert parallelism_config.hybrid_enabled(), "hybrid_enabled() must be True for HYBRID backend."
  # 0. First enable context parallelism
  transformer = _parallelize_transformer_cp(
    transformer=transformer,
    parallelism_config=parallelism_config,
  )
  # 1. Then enable tensor parallelism
  transformer = _parallelize_transformer_tp(
    transformer=transformer,
    parallelism_config=parallelism_config,
  )
  transformer._is_parallelized = True  # type: ignore[attr-defined]
  # Use `parallelism` not `parallel` to avoid name conflict with diffusers.
  transformer._parallelism_config = parallelism_config  # type: ignore[attr-defined]
  logger.info(f"Parallelize Transformer: {transformer.__class__.__name__}, "
              f"id:{id(transformer)}, {parallelism_config.strify(True)}")
  return transformer


def _parallelize_transformer_cp(
  transformer: torch.nn.Module | ModelMixin,
  parallelism_config: ParallelismConfig,
) -> torch.nn.Module:
  assert isinstance(
    transformer,
    (torch.nn.Module,
     ModelMixin)), ("transformer must be an instance of torch.nn.Module or ModelMixin, "
                    f"but got {type(transformer)}")

  if parallelism_config is None:
    return transformer

  assert isinstance(
    parallelism_config,
    ParallelismConfig), ("parallelism_config must be an instance of ParallelismConfig"
                         f" but got {type(parallelism_config)}")

  # Ensure the backend is correct, NAITIVE_DIFFUSER or HYBRID
  assert parallelism_config.backend in (
    ParallelismBackend.NATIVE_DIFFUSER,
    ParallelismBackend.NATIVE_HYBRID,
  ), ("parallelism_config.backend must be ParallelismBackend.NATIVE_DIFFUSER "
      f"or ParallelismBackend.NATIVE_HYBRID but got {parallelism_config.backend}")

  if parallelism_config.cp_enabled() or parallelism_config.hybrid_enabled():
    transformer = _parallelize_transformer_context(
      transformer,
      parallelism_config,
    )
    if not parallelism_config.hybrid_enabled():
      transformer._is_parallelized = True  # type: ignore[attr-defined]
      # Use `parallelism` not `parallel` to avoid name conflict with diffusers.
      transformer._parallelism_config = parallelism_config  # type: ignore[attr-defined]
      logger.info(f"Parallelize Transformer: {transformer.__class__.__name__}, "
                  f"id:{id(transformer)}, {parallelism_config.strify(True)}")
  else:
    raise ValueError("NATIVE_DIFFUSER backend only support context parallelism now. "
                     "Please set ulysses_size or ring_size in parallelism_config.")
  return transformer


def _parallelize_transformer_tp(
  transformer: torch.nn.Module | ModelMixin,
  parallelism_config: ParallelismConfig,
) -> torch.nn.Module:
  assert isinstance(
    transformer,
    (torch.nn.Module,
     ModelMixin)), ("transformer must be an instance of torch.nn.Module or ModelMixin, "
                    f"but got {type(transformer)}")

  if parallelism_config is None:
    return transformer

  # Ensure the backend is correct, NATIVE_PYTORCH or HYBRID
  assert parallelism_config.backend in (
    ParallelismBackend.NATIVE_PYTORCH,
    ParallelismBackend.NATIVE_HYBRID,
  ), ("parallelism_config.backend must be ParallelismBackend.NATIVE_PYTORCH "
      f"or ParallelismBackend.NATIVE_HYBRID but got {parallelism_config.backend}")

  assert isinstance(
    parallelism_config,
    ParallelismConfig), ("parallelism_config must be an instance of ParallelismConfig"
                         f" but got {type(parallelism_config)}")

  if parallelism_config.tp_enabled() or parallelism_config.hybrid_enabled():
    transformer = _parallelize_transformer_tensor(
      transformer=transformer,
      parallelism_config=parallelism_config,
    )
    if not parallelism_config.hybrid_enabled():
      transformer._is_parallelized = True  # type: ignore[attr-defined]
      # Use `parallelism` not `parallel` to avoid name conflict with diffusers.
      transformer._parallelism_config = parallelism_config  # type: ignore[attr-defined]
      logger.info(f"Parallelize Transformer: {transformer.__class__.__name__}, "
                  f"id:{id(transformer)}, {parallelism_config.strify(True)}")
  else:
    raise ValueError("NATIVE_PYTORCH only supported tensor parallelism now. "
                     "Please set tp_size > 1 for tensor parallelism.")
  return transformer
