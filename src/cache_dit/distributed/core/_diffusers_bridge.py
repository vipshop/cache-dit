"""Adapted from diffusers config normalization and attention-backend checks."""

from typing import Optional

import torch

from ...logger import init_logger
from ._modeling_parallel import _ContextParallelConfig

logger = init_logger(__name__)


def _normalize_parallel_config(config) -> _ContextParallelConfig:
  """Normalize diffusers or cache-dit CP configs into `_ContextParallelConfig`.

  :param config: Parallel config-like object from cache-dit or diffusers.
  :returns: Cache-dit owned internal `_ContextParallelConfig`.
  """

  if isinstance(config, _ContextParallelConfig):
    return config

  context_parallel_config = getattr(config, "context_parallel_config", None)
  if context_parallel_config is not None:
    return _normalize_context_parallel_config(context_parallel_config)

  if hasattr(config, "ring_degree") or hasattr(config, "ulysses_degree"):
    return _normalize_context_parallel_config(config)

  raise TypeError(f"Unsupported parallel config type: {type(config)}")


def _normalize_context_parallel_config(config) -> _ContextParallelConfig:
  if isinstance(config, _ContextParallelConfig):
    return config

  normalized = _ContextParallelConfig(
    ring_degree=getattr(config, "ring_degree", None),
    ulysses_degree=getattr(config, "ulysses_degree", None),
    convert_to_fp32=getattr(config, "convert_to_fp32", True),
    rotate_method=getattr(config, "rotate_method", "p2p"),
    mesh=getattr(config, "mesh", None),
    ulysses_anything=getattr(config, "ulysses_anything", False),
    ulysses_float8=getattr(config, "ulysses_float8", False),
    ulysses_async=getattr(config, "ulysses_async", False),
    extra_kwargs=getattr(config, "extra_kwargs", None),
  )

  normalized._rank = getattr(config, "_rank", None)
  normalized._world_size = getattr(config, "_world_size", None)
  normalized._device = getattr(config, "_device", None)
  normalized._mesh = getattr(config, "_mesh", None) or normalized.mesh
  normalized._flattened_mesh = getattr(config, "_flattened_mesh", None)
  normalized._ring_mesh = getattr(config, "_ring_mesh", None)
  normalized._ulysses_mesh = getattr(config, "_ulysses_mesh", None)
  normalized._ring_local_rank = getattr(config, "_ring_local_rank", None)
  normalized._ulysses_local_rank = getattr(config, "_ulysses_local_rank", None)
  return normalized


def get_diffusers_attention_classes(attn_classes_extra: Optional[tuple] = None) -> tuple[type, ...]:
  """Load diffusers attention classes used by CP backend validation.

  :param attn_classes_extra: Optional additional attention-like classes.
  :returns: Tuple of diffusers attention classes.
  """

  from diffusers.models.attention import AttentionModuleMixin
  from diffusers.models.attention_processor import Attention, MochiAttention

  attention_classes = (Attention, MochiAttention, AttentionModuleMixin)
  if attn_classes_extra is not None:
    attention_classes += attn_classes_extra
  return attention_classes


def validate_context_parallel_attention_backend(
  model: torch.nn.Module,
  parallel_config: _ContextParallelConfig,
  *,
  attn_classes_extra: Optional[tuple] = None,
) -> None:
  """Ensure diffusers attention processors use CP-compatible backends.

  :param model: Module tree to inspect.
  :param parallel_config: Cache-dit normalized parallel config.
  :param attn_classes_extra: Optional additional attention classes.
  """

  from ...attention._attention_dispatch import _AttnBackendRegistry

  attention_classes = get_diffusers_attention_classes(attn_classes_extra=attn_classes_extra)
  for module in model.modules():
    if not isinstance(module, attention_classes):
      continue

    processor = getattr(module, "processor", None)
    if processor is None:
      continue
    if not hasattr(processor, "_attention_backend"):
      processor._attention_backend = None

    attention_backend = processor._attention_backend
    if attention_backend is None:
      attention_backend, _ = _AttnBackendRegistry.get_active_backend()

    try:
      attention_backend = _AttnBackendRegistry.normalize_backend(attention_backend)
    except ValueError as exc:
      compatible_backends = sorted(_AttnBackendRegistry.list_context_parallel_backends())
      raise ValueError(
        "Context parallelism requires a cache-dit registered attention backend. "
        f"Got unsupported backend: {attention_backend}. Compatible backends: {compatible_backends}."
      ) from exc
    _AttnBackendRegistry.ensure_backend_registered(attention_backend)

    if not _AttnBackendRegistry.is_context_parallel_available(attention_backend):
      compatible_backends = sorted(_AttnBackendRegistry.list_context_parallel_backends())
      raise ValueError(
        f"Context parallelism is enabled but attention processor '{processor.__class__.__name__}' "
        f"uses backend '{attention_backend.value}' which does not support context parallelism. "
        f"Compatible backends: {compatible_backends}.")
    break
