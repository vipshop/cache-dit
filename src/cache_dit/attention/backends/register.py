import inspect
import os
from enum import Enum
from typing import Any, Callable, Optional

import torch

from ...logger import init_logger
from ...distributed.core import (
  _ContextParallelConfig,
  _normalize_parallel_config,
  RingAttention,
  UlyssesAttention,
  USPAttention,
)

logger = init_logger(__name__)
_TRUE_VALUES = {"1", "true", "yes", "on"}
_CACHE_DIT_ATTN_BACKEND_ENV = "CACHE_DIT_ATTN_BACKEND"
_CACHE_DIT_ATTN_CHECKS_ENV = "CACHE_DIT_ATTN_CHECKS"


def _supports_enable_gqa() -> bool:
  try:
    return "enable_gqa" in inspect.signature(
      torch.nn.functional.scaled_dot_product_attention).parameters
  except (TypeError, ValueError):
    return False


class _AttnBackend(str, Enum):
  """Cache-DiT owned attention backend identifiers."""

  NATIVE = "native"
  FLASH = "flash"
  FLASH_VARLEN = "flash_varlen"
  SAGE = "sage"
  _FLASH_3 = "_flash_3"
  _SDPA_CUDNN = "_sdpa_cudnn"
  _NATIVE_NPU = "_native_npu"
  _NPU_FIA = "_npu_fia"
  _MINDIESD_LASER = "_mindiesd_laser"


def _default_active_backend() -> _AttnBackend:
  try:
    return _AttnBackend(os.getenv(_CACHE_DIT_ATTN_BACKEND_ENV, _AttnBackend.NATIVE.value))
  except ValueError:
    logger.warning(f"Unsupported {_CACHE_DIT_ATTN_BACKEND_ENV} value, falling back to native.")
    return _AttnBackend.NATIVE


def _resolve_cp_config(
  _cp_config: Optional[Any] = None,
  _parallel_config: Optional[Any] = None,
) -> Optional[_ContextParallelConfig]:
  if _cp_config is not None and _parallel_config is not None and _cp_config is not _parallel_config:
    logger.warning("Both _cp_config and legacy _parallel_config were provided; "
                   "preferring _cp_config.")

  config = _cp_config if _cp_config is not None else _parallel_config
  if config is None:
    return None
  if isinstance(config, _ContextParallelConfig):
    return config
  return _normalize_parallel_config(config)


class _AttnBackendRegistry:
  _backends: dict[_AttnBackend, Callable[..., torch.Tensor]] = {}
  _constraints: dict[_AttnBackend, list[Callable[..., None]]] = {}
  _supported_arg_names: dict[_AttnBackend, set[str]] = {}
  _supports_context_parallel: set[str] = set()
  _bridge_to_diffusers: set[str] = set()
  _active_backend: _AttnBackend = _default_active_backend()
  _checks_enabled: bool = os.getenv(_CACHE_DIT_ATTN_CHECKS_ENV, "0").lower() in _TRUE_VALUES

  @classmethod
  def normalize_backend(cls, backend: str | _AttnBackend) -> _AttnBackend:
    if isinstance(backend, _AttnBackend):
      return backend
    return _AttnBackend(backend)

  @classmethod
  def register(
    cls,
    backend: str | _AttnBackend,
    constraints: Optional[list[Callable[..., None]]] = None,
    supports_context_parallel: bool = False,
    bridge_to_diffusers: bool = True,
  ):

    backend_name = cls.normalize_backend(backend)

    def decorator(func: Callable[..., torch.Tensor]):
      cls._backends[backend_name] = func
      cls._constraints[backend_name] = constraints or []
      cls._supported_arg_names[backend_name] = set(inspect.signature(func).parameters.keys())
      if supports_context_parallel:
        cls._supports_context_parallel.add(backend_name.value)
      else:
        cls._supports_context_parallel.discard(backend_name.value)
      if bridge_to_diffusers:
        cls._bridge_to_diffusers.add(backend_name.value)
      else:
        cls._bridge_to_diffusers.discard(backend_name.value)
      return func

    return decorator

  @classmethod
  def get_active_backend(cls) -> tuple[_AttnBackend, Callable[..., torch.Tensor]]:
    cls.ensure_backend_registered(cls._active_backend)
    return cls._active_backend, cls._backends[cls._active_backend]

  @classmethod
  def set_active_backend(cls, backend: str | _AttnBackend) -> None:
    cls._active_backend = cls.normalize_backend(backend)
    cls.ensure_backend_registered(cls._active_backend)

  @classmethod
  def get_backend(cls, backend: str | _AttnBackend) -> Optional[Callable[..., torch.Tensor]]:
    backend_name = cls.normalize_backend(backend)
    cls.ensure_backend_registered(backend_name)
    return cls._backends.get(backend_name)

  @classmethod
  def get_constraints(cls, backend: str | _AttnBackend) -> list[Callable[..., None]]:
    return cls._constraints.get(cls.normalize_backend(backend), [])

  @classmethod
  def get_supported_arg_names(cls, backend: str | _AttnBackend) -> set[str]:
    return cls._supported_arg_names.get(cls.normalize_backend(backend), set())

  @classmethod
  def list_context_parallel_backends(cls) -> list[str]:
    return sorted(cls._supports_context_parallel)

  @classmethod
  def list_bridge_backends(cls) -> list[_AttnBackend]:
    return [backend for backend in cls._backends if backend.value in cls._bridge_to_diffusers]

  @classmethod
  def ensure_backend_registered(cls, backend: str | _AttnBackend) -> bool:
    backend_name = cls.normalize_backend(backend)
    if backend_name in cls._backends:
      return True
    return _maybe_register_diffusers_backend_proxy(backend_name)

  @classmethod
  def is_context_parallel_available(cls, backend: str | _AttnBackend) -> bool:
    backend_name = cls.normalize_backend(backend)
    cls.ensure_backend_registered(backend_name)
    return backend_name.value in cls._supports_context_parallel


def _maybe_register_diffusers_backend_proxy(backend: _AttnBackend) -> bool:
  try:
    from diffusers.models.attention_dispatch import (
      AttentionBackendName as _DiffusersAttentionBackendName,
      _AttentionBackendRegistry as _DiffusersAttentionBackendRegistry,
      dispatch_attention_fn as _diffusers_dispatch_attention_fn,
    )
  except ImportError:
    return False

  try:
    diffusers_backend = _DiffusersAttentionBackendName(backend.value)
  except ValueError:
    return False

  if diffusers_backend not in _DiffusersAttentionBackendRegistry._backends:
    return False

  def _diffusers_attention_proxy(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    attention_kwargs: Optional[dict[str, Any]] = None,
    _cp_config: Optional["_ContextParallelConfig"] = None,
    _parallel_config: Optional["_ContextParallelConfig"] = None,
  ) -> torch.Tensor:
    return _diffusers_dispatch_attention_fn(
      query,
      key,
      value,
      attn_mask=attn_mask,
      dropout_p=dropout_p,
      is_causal=is_causal,
      scale=scale,
      enable_gqa=enable_gqa,
      attention_kwargs=attention_kwargs,
      backend=diffusers_backend,
      parallel_config=_resolve_cp_config(
        _cp_config=_cp_config,
        _parallel_config=_parallel_config,
      ),
    )

  _AttnBackendRegistry._backends[backend] = _diffusers_attention_proxy
  _AttnBackendRegistry._constraints[backend] = list(
    _DiffusersAttentionBackendRegistry._constraints.get(diffusers_backend, []))
  _AttnBackendRegistry._supported_arg_names[backend] = set(
    inspect.signature(_diffusers_attention_proxy).parameters.keys())
  if _DiffusersAttentionBackendRegistry._is_context_parallel_available(diffusers_backend):
    _AttnBackendRegistry._supports_context_parallel.add(backend.value)
  _AttnBackendRegistry._bridge_to_diffusers.discard(backend.value)
  return True


# Helper checks moved from former _attention_dispatch.py
def _check_device(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs) -> None:
  if query.device != key.device or query.device != value.device:
    raise ValueError("Query, key, and value must be on the same device.")
  if query.dtype != key.dtype or query.dtype != value.dtype:
    raise ValueError("Query, key, and value must have the same dtype.")


def _check_device_cuda(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                       **kwargs) -> None:
  _check_device(query, key, value)
  if query.device.type != "cuda":
    raise ValueError("Query, key, and value must be on a CUDA device.")


def _check_qkv_dtype_bf16_or_fp16(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  **kwargs,
) -> None:
  if query.dtype != key.dtype or query.dtype != value.dtype:
    raise ValueError("Query, key, and value must have the same dtype.")
  if query.dtype not in (torch.bfloat16, torch.float16):
    raise ValueError("Query, key, and value must be either bfloat16 or float16.")


def _check_shape(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  **kwargs,
) -> None:
  if query.shape[-1] != key.shape[-1]:
    raise ValueError("Query and key must have the same head dimension.")
  if key.shape[-3] != value.shape[-3]:
    raise ValueError("Key and value must have the same sequence length.")
  if attn_mask is not None and attn_mask.shape[-1] != key.shape[-3]:
    raise ValueError("Attention mask must match the key's sequence length.")


# Context-parallel dispatcher helper moved here so backends can import it
def _context_parallel_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  cp_gqa_strategy: Optional[str] = None,
  *,
  forward_op,
  backward_op,
  _cp_config: Optional["_ContextParallelConfig"] = None,
):
  if attn_mask is not None:
    forward_op_name = forward_op.__name__
    # allowlist of forward op names that support attn_mask
    _ATTENTION_OPS_ALLOW_ATTN_MASK = [
      "_native_attention_forward_op",
      "_sdpa_cudnn_attention_forward_op",
      "_flash_varlen_attention_forward_op",
      "_npu_attention_forward_op",
      "_npu_fused_infer_attention_forward_op",
    ]
    if forward_op_name not in _ATTENTION_OPS_ALLOW_ATTN_MASK:
      raise ValueError("Templated context parallel attention with attn_mask "
                       "is only supported for native attention backend.")
  if is_causal:
    raise ValueError("Causal attention is not yet supported for templated attention.")
  if cp_gqa_strategy not in (None, "replicate_kv_sequence", "group_aligned_flash_varlen"):
    raise ValueError(f"Unsupported CP GQA strategy: {cp_gqa_strategy}.")
  if enable_gqa and cp_gqa_strategy is None:
    raise ValueError("GQA is not yet supported for templated attention.")

  if _cp_config is None:
    raise ValueError("Context parallel config must be provided for templated attention.")

  if _cp_config.ring_degree > 1 and _cp_config.ulysses_degree > 1:
    if cp_gqa_strategy is not None:
      raise ValueError("CP GQA strategy is only supported for pure Ulysses attention.")
    return USPAttention.apply(
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      enable_gqa,
      return_lse,
      forward_op,
      backward_op,
      _cp_config,
    )
  elif _cp_config.ring_degree > 1:
    if cp_gqa_strategy is not None:
      raise ValueError("CP GQA strategy is only supported for pure Ulysses attention.")
    return RingAttention.apply(
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      enable_gqa,
      return_lse,
      forward_op,
      backward_op,
      _cp_config,
    )
  elif _cp_config.ulysses_degree > 1:
    return UlyssesAttention.apply(
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      enable_gqa,
      return_lse,
      cp_gqa_strategy,
      forward_op,
      backward_op,
      _cp_config,
    )
  else:
    raise ValueError("Reaching this branch of code is unexpected. Please report a bug.")


# Dispatcher moved here
def _dispatch_attention_fn(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  attention_kwargs: Optional[dict[str, Any]] = None,
  cp_gqa_strategy: Optional[str] = None,
  *,
  backend: Optional[str | _AttnBackend] = None,
  cp_config: Optional[_ContextParallelConfig] = None,
  parallel_config: Optional[_ContextParallelConfig] = None,
) -> torch.Tensor:
  attention_kwargs = attention_kwargs or {}
  cp_config = _resolve_cp_config(
    _cp_config=cp_config,
    _parallel_config=parallel_config,
  )

  if backend is None:
    backend_name, backend_fn = _AttnBackendRegistry.get_active_backend()
  else:
    backend_name = _AttnBackendRegistry.normalize_backend(backend)
    if not _AttnBackendRegistry.ensure_backend_registered(backend_name):
      raise ValueError(f"Backend {backend_name.value} is not registered.")
    backend_fn = _AttnBackendRegistry.get_backend(backend_name)
    if backend_fn is None:
      raise ValueError(f"Backend {backend_name.value} is not registered.")

  kwargs = {
    "query": query,
    "key": key,
    "value": value,
    "attn_mask": attn_mask,
    "dropout_p": dropout_p,
    "is_causal": is_causal,
    "scale": scale,
    **attention_kwargs,
    "cp_gqa_strategy": cp_gqa_strategy,
    "_cp_config": cp_config,
  }
  if _supports_enable_gqa():
    kwargs["enable_gqa"] = enable_gqa

  if _AttnBackendRegistry._checks_enabled:
    supported_arg_names = _AttnBackendRegistry.get_supported_arg_names(backend_name)
    removed_kwargs = set(kwargs) - supported_arg_names
    if removed_kwargs:
      logger.warning(
        f"Removing unsupported arguments for attention backend {backend_name.value}: {removed_kwargs}."
      )
    for check in _AttnBackendRegistry.get_constraints(backend_name):
      check(**kwargs)

  kwargs = {
    key: value
    for key, value in kwargs.items()
    if key in _AttnBackendRegistry.get_supported_arg_names(backend_name)
  }
  return backend_fn(**kwargs)
