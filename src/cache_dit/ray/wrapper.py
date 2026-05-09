from __future__ import annotations

import types
from typing import Any

import torch
from diffusers import DiffusionPipeline
from diffusers.models.modeling_utils import ModelMixin

from ..distributed import ParallelismConfig
from ..logger import init_logger
from ..quantization import QuantizeConfig
from .engine import RayParallelEngine
from .engine import RayPipelineEngine

logger = init_logger(__name__)


def enable_ray_parallelism(
  transformer: torch.nn.Module | ModelMixin,
  parallelism_config: ParallelismConfig,
  cache_context_kwargs: dict[str, Any] | None = None,
  quantize_config: QuantizeConfig | None = None,
) -> torch.nn.Module | ModelMixin:
  """Patch a transformer so its forward is executed by Ray worker actors.

  :param transformer: User-visible transformer module to patch in-place.
  :param parallelism_config: Ray-enabled parallelism configuration.
  :param cache_context_kwargs: Optional cache context keyword arguments to apply inside Ray workers.
  :param quantize_config: Optional quantization configuration to apply inside Ray workers.
  :returns: The same transformer object with a proxied forward method.
  """

  if getattr(transformer, "_cache_dit_ray_enabled", False):
    logger.warning("Ray parallelism is already enabled for this transformer. Skipping.")
    return transformer

  engine = RayParallelEngine(transformer, parallelism_config, cache_context_kwargs, quantize_config)
  transformer._cache_dit_ray_original_forward = transformer.forward  # type: ignore[attr-defined]
  transformer._cache_dit_ray_original_to = transformer.to  # type: ignore[attr-defined]
  transformer._cache_dit_ray_engine = engine  # type: ignore[attr-defined]

  def ray_forward(self, *args, **kwargs):
    return self._cache_dit_ray_engine.forward(*args, **kwargs)

  def ray_to(self, *args, **kwargs):
    logger.info(f"Skipping .to(...) for Ray-owned {self.__class__.__name__}; "
                "worker actors own the executable transformer copies.")
    return self

  transformer.forward = types.MethodType(ray_forward, transformer)
  transformer.to = types.MethodType(ray_to, transformer)
  transformer._cache_dit_ray_enabled = True  # type: ignore[attr-defined]
  logger.info(f"Enabled Ray parallelism for {transformer.__class__.__name__} with "
              f"world_size={engine.world_size}.")
  return transformer


def disable_ray_parallelism(transformer: torch.nn.Module | ModelMixin) -> None:
  """Restore a transformer patched by :func:`enable_ray_parallelism`.

  :param transformer: Transformer module that may own a Ray engine.
  """

  engine = getattr(transformer, "_cache_dit_ray_engine", None)
  if hasattr(transformer, "_cache_dit_ray_original_to"):
    transformer.to = transformer._cache_dit_ray_original_to
    del transformer._cache_dit_ray_original_to
  if engine is not None:
    engine.shutdown()
    del transformer._cache_dit_ray_engine
  if hasattr(transformer, "_cache_dit_ray_original_forward"):
    transformer.forward = transformer._cache_dit_ray_original_forward
    del transformer._cache_dit_ray_original_forward
  if hasattr(transformer, "_cache_dit_ray_enabled"):
    del transformer._cache_dit_ray_enabled


def enable_ray_pipeline_parallelism(
  pipe: DiffusionPipeline,
  parallelism_config: ParallelismConfig,
  cache_context_kwargs: dict[str, Any] | None = None,
  quantize_config: QuantizeConfig | None = None,
) -> DiffusionPipeline:
  """Patch a pipeline so each full inference call is executed by Ray workers.

  :param pipe: User-visible diffusion pipeline to patch in-place.
  :param parallelism_config: Ray-enabled parallelism configuration.
  :param cache_context_kwargs: Optional cache context keyword arguments to apply inside Ray workers.
  :param quantize_config: Optional quantization configuration to apply inside Ray workers.
  :returns: The same pipeline object with a proxied ``__call__`` method.
  """

  if getattr(pipe, "_cache_dit_ray_pipeline_enabled", False):
    logger.warning("Ray parallelism is already enabled for this pipeline. Skipping.")
    return pipe

  engine = RayPipelineEngine(pipe, parallelism_config, cache_context_kwargs, quantize_config)
  original_class = pipe.__class__

  def ray_pipeline_call(self, *args, **kwargs):
    return self._cache_dit_ray_pipeline_engine.call(*args, **kwargs)

  ray_class = type(
    f"CacheDitRay{original_class.__name__}",
    (original_class, ),
    {"__call__": ray_pipeline_call},
  )
  pipe._cache_dit_ray_pipeline_original_class = original_class  # type: ignore[attr-defined]
  pipe._cache_dit_ray_pipeline_engine = engine  # type: ignore[attr-defined]
  pipe.__class__ = ray_class
  pipe._cache_dit_ray_pipeline_enabled = True  # type: ignore[attr-defined]
  logger.info(f"Enabled Ray parallelism for {original_class.__name__} with "
              f"world_size={engine.world_size}.")
  return pipe


def disable_ray_pipeline_parallelism(pipe: DiffusionPipeline) -> None:
  """Restore a pipeline patched by :func:`enable_ray_pipeline_parallelism`.

  :param pipe: Pipeline that may own a Ray pipeline engine.
  """

  engine = getattr(pipe, "_cache_dit_ray_pipeline_engine", None)
  if engine is not None:
    engine.shutdown()
    del pipe._cache_dit_ray_pipeline_engine
  original_class = getattr(pipe, "_cache_dit_ray_pipeline_original_class", None)
  if original_class is not None:
    pipe.__class__ = original_class
    del pipe._cache_dit_ray_pipeline_original_class
  if hasattr(pipe, "_cache_dit_ray_pipeline_enabled"):
    del pipe._cache_dit_ray_pipeline_enabled
