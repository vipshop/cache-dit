from typing import Any, Optional

import torch
from diffusers.models.modeling_utils import ModelMixin

from ..logger import init_logger
from ._attention_dispatch import _AttnBackendRegistry

logger = init_logger(__name__)


def _iter_attention_backend_targets(pipe_or_adapter: Any):
  """Yield transformer-like targets whose attention backend should be updated.

  This helper intentionally relies on duck typing instead of importing cache-dit's
  `BlockAdapter` type, which keeps `cache_dit.attention` free of caching-layer import
  cycles while still handling pipeline-like objects that expose a `transformer`
  attribute.

  :param pipe_or_adapter: Pipeline-like, adapter-like, or module-like object.
  :returns: An iterator of concrete modules that should receive backend updates.
  """

  transformer = getattr(pipe_or_adapter, "transformer", None)
  if transformer is None:
    yield pipe_or_adapter
    return

  if isinstance(transformer, list):
    for module in transformer:
      yield module
    return

  yield transformer


def _set_attn_backend_impl(
  pipe_or_adapter: Any,
  attention_backend: Optional[str] = None,
  *,
  default_diffusers_backend: Optional[str] = None,
  wrap_runtime_error: bool = True,
) -> None:
  """Shared attention-backend setter used by public and distributed entrypoints.

  :param pipe_or_adapter: Pipeline-like object, adapter-like object, transformer module,
    or any module exposing attention processors with `_attention_backend` state.
  :param attention_backend: Explicit attention backend name to apply.
  :param default_diffusers_backend: Fallback backend used only for diffusers modules when
    `attention_backend` is omitted.
  :param wrap_runtime_error: Whether to wrap backend-application failures in a
    user-facing `RuntimeError`.
  """

  if attention_backend is None and default_diffusers_backend is None:
    return

  resolved_cache_dit_backend = None
  try:
    if attention_backend is not None:
      resolved_cache_dit_backend = _AttnBackendRegistry.normalize_backend(attention_backend).value
  except ValueError:
    resolved_cache_dit_backend = None

  if resolved_cache_dit_backend is not None:
    try:
      from . import _maybe_register_custom_attn_backends

      _maybe_register_custom_attn_backends()
    except Exception as e:
      logger.warning("Failed to register custom attention backends. "
                     f"Proceeding to set attention backend anyway. Error: {e}")

  def _set_backend_locally(module: torch.nn.Module, backend_name: str) -> bool:
    applied = False
    for current in module.modules():
      processor = getattr(current, "processor", None)
      if processor is not None:
        processor._attention_backend = backend_name
        current._attention_backend = backend_name
        applied = True
        continue
      if hasattr(current, "_attention_backend"):
        current._attention_backend = backend_name
        applied = True
    return applied

  def _set_backend(module: Any) -> None:
    if module is None:
      return

    if hasattr(module, "set_attention_backend") and isinstance(module, ModelMixin):
      backend_name = resolved_cache_dit_backend or attention_backend
      if backend_name is None:
        if default_diffusers_backend is None:
          return
        module.set_attention_backend(default_diffusers_backend)
        logger.warning("attention_backend is None, set default attention backend of "
                       f"{module.__class__.__name__} to: <{default_diffusers_backend}>.")
        return

      module.set_attention_backend(backend_name)
      logger.info(f"Set attention backend to <{backend_name}> for module: "
                  f"{module.__class__.__name__}.")
      return

    if isinstance(module, torch.nn.Module):
      if attention_backend is None:
        return
      if resolved_cache_dit_backend is None:
        raise ValueError("Non-diffusers modules only support cache-dit attention backends. "
                         f"Got unsupported backend: {attention_backend}.")
      if not _set_backend_locally(module, resolved_cache_dit_backend):
        logger.warning("--attn was provided but module does not expose local attention backend "
                       f"state: {module.__class__.__name__}.")
        return
      logger.info("Set local cache-dit attention backend to "
                  f"<{resolved_cache_dit_backend}> for module: {module.__class__.__name__}.")
      return

    logger.warning("--attn was provided but module does not support set_attention_backend: "
                   f"{module.__class__.__name__}.")

  def _apply() -> None:
    for module in _iter_attention_backend_targets(pipe_or_adapter):
      _set_backend(module)

  if not wrap_runtime_error:
    _apply()
    return

  try:
    _apply()
  except Exception as e:
    raise RuntimeError(
      f"Failed to set attention backend to <{attention_backend}>. "
      "This usually means the backend is unavailable (e.g., FlashAttention-3 not installed) "
      "or the model/shape/dtype is unsupported. "
      f"Original error: {e}") from e


def set_attn_backend(
  pipe_or_adapter: Any,
  attention_backend: Optional[str] = None,
) -> None:
  """Set the attention backend on diffusers or plain transformer modules.

  :param pipe_or_adapter: Pipeline-like object, adapter-like object, transformer module,
    or any module exposing attention processors with `_attention_backend` state.
  :param attention_backend: Attention backend name to apply. When omitted, the function
    returns without making changes.
  """

  _set_attn_backend_impl(
    pipe_or_adapter,
    attention_backend=attention_backend,
  )
