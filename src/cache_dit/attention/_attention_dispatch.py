"""Adapted from diffusers' attention dispatch path and attention backend implementations.

The attention backend registry and dispatch logic are designed to be extensible for custom backends,
but the cache-dit runtime currently only registers its own native attention backend. The dispatch
function will route to the appropriate backend based on the active backend setting, which can be
configured globally or per-module. The templated context parallel attention implementations are
designed to be flexible for future support of more backends and features.
"""
import inspect
import math
import os
from enum import Enum
from typing import Any, Callable, Optional

import torch

try:
  from flash_attn import flash_attn_func
  from flash_attn.flash_attn_interface import _wrapped_flash_attn_backward, _wrapped_flash_attn_forward

  _flash_attn_available = True
except Exception:
  flash_attn_func = None
  _wrapped_flash_attn_backward = None
  _wrapped_flash_attn_forward = None
  _flash_attn_available = False

try:
  from flash_attn_interface import flash_attn_func as flash_attn_3_func

  _flash_attn_3_available = True
except ImportError:
  flash_attn_3_func = None
  _flash_attn_3_available = False

try:
  from torch_npu import npu_fusion_attention
except ImportError:
  npu_fusion_attention = None

try:
  from torch_npu import npu_fused_infer_attention_score
except ImportError:
  npu_fused_infer_attention_score = None

try:
  from sageattention import sageattn
except ImportError:
  sageattn = None

from ..logger import init_logger
from ..distributed.core import (
  _ContextParallelConfig,
  _normalize_parallel_config,
  RingAttention,
  UlyssesAttention,
  USPAttention,
)

logger = init_logger(__name__)
MAX_TOKEN = 2147483647

__all__ = [
  "_AttnBackend",
  "_AttnBackendRegistry",
  "_dispatch_attention_fn",
  "_native_attention",
  "_sdpa_cudnn_attention",
  "_sage_attention",
  "_flash_attention",
  "_flash_attention_3",
  "_native_npu_attention",
  "_npu_fused_infer_attention",
]

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
  """Cache-dit owned attention backend identifiers."""

  NATIVE = "native"
  FLASH = "flash"
  SAGE = "sage"
  _FLASH_3 = "_flash_3"
  _SDPA_CUDNN = "_sdpa_cudnn"
  _NATIVE_NPU = "_native_npu"
  _NPU_FIA = "_npu_fia"


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


# Cache-dit-owned attention backends are always registered because
# context parallel execution depends on their extended semantics.
_ATTENTION_OPS_ALLOW_ATTN_MASK = [
  "_native_attention_forward_op",
  "_sdpa_cudnn_attention_forward_op",
  "_npu_attention_forward_op",
  "_npu_fused_infer_attention_forward_op",
]


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
  *,
  forward_op,
  backward_op,
  _cp_config: Optional["_ContextParallelConfig"] = None,
):
  if attn_mask is not None:
    # NOTE(DefTruth): Check if forward_op is native attention forward op
    forward_op_name = forward_op.__name__
    if forward_op_name not in _ATTENTION_OPS_ALLOW_ATTN_MASK:
      raise ValueError("Templated context parallel attention with attn_mask "
                       "is only supported for native attention backend, "
                       f"but got forward_op: {forward_op_name}.")
  if is_causal:
    raise ValueError("Causal attention is not yet supported for templated attention.")
  if enable_gqa:
    raise ValueError("GQA is not yet supported for templated attention.")

  if _cp_config is None:
    raise ValueError("Context parallel config must be provided for templated attention.")

  if _cp_config.ring_degree > 1 and _cp_config.ulysses_degree > 1:
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
      forward_op,
      backward_op,
      _cp_config,
    )
  else:
    raise ValueError("Reaching this branch of code is unexpected. Please report a bug.")


def _native_attention_forward_op(
  ctx: torch.autograd.function.FunctionCtx,
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  _save_ctx: bool = True,
  _cp_config: Optional["_ContextParallelConfig"] = None,
):
  # used for backward pass
  if _save_ctx:
    ctx.save_for_backward(query, key, value)
    ctx.attn_mask = attn_mask
    ctx.dropout_p = dropout_p
    ctx.is_causal = is_causal
    ctx.scale = scale
    ctx.enable_gqa = enable_gqa

  if return_lse:
    # Use native flash attention to get lse if return_lse is True
    if attn_mask is not None:
      raise ValueError("`attn_mask` is not yet supported for native flash attention with lse.")
    out, lse = torch.ops.aten._scaled_dot_product_flash_attention(
      query.transpose(1, 2),
      key.transpose(1, 2),
      value.transpose(1, 2),
      dropout_p=dropout_p,
      is_causal=is_causal,
      scale=scale,
    )[:2]
    # [B, H, N, D] -> [B, N, H, D]
    out = out.transpose(1, 2)  # type: torch.Tensor
    lse = lse.transpose(1, 2)  # type: torch.Tensor
    if lse.dim() == 3:
      # [B, N, H] -> [B, N, H, 1]
      lse = lse.unsqueeze(-1)
    return out, lse

  query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
  out = torch.nn.functional.scaled_dot_product_attention(
    query=query,
    key=key,
    value=value,
    attn_mask=attn_mask,
    dropout_p=dropout_p,
    is_causal=is_causal,
    scale=scale,
    enable_gqa=enable_gqa,
  )
  out = out.permute(0, 2, 1, 3)

  return out


def _native_attention_backward_op(
  ctx: torch.autograd.function.FunctionCtx,
  grad_out: torch.Tensor,
  *args,
  **kwargs,
):
  query, key, value = ctx.saved_tensors

  query.requires_grad_(True)
  key.requires_grad_(True)
  value.requires_grad_(True)

  query_t, key_t, value_t = (x.permute(0, 2, 1, 3) for x in (query, key, value))
  out = torch.nn.functional.scaled_dot_product_attention(
    query=query_t,
    key=key_t,
    value=value_t,
    attn_mask=ctx.attn_mask,
    dropout_p=ctx.dropout_p,
    is_causal=ctx.is_causal,
    scale=ctx.scale,
    enable_gqa=ctx.enable_gqa,
  )
  out = out.permute(0, 2, 1, 3)

  grad_out_t = grad_out.permute(0, 2, 1, 3)
  grad_query_t, grad_key_t, grad_value_t = torch.autograd.grad(
    outputs=out,
    inputs=[query_t, key_t, value_t],
    grad_outputs=grad_out_t,
    retain_graph=False,
  )

  grad_query = grad_query_t.permute(0, 2, 1, 3)
  grad_key = grad_key_t.permute(0, 2, 1, 3)
  grad_value = grad_value_t.permute(0, 2, 1, 3)

  return grad_query, grad_key, grad_value


@_AttnBackendRegistry.register(
  _AttnBackend.NATIVE,
  constraints=[_check_device, _check_shape],
  supports_context_parallel=True,
)
def _native_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  _cp_config: Optional["_ContextParallelConfig"] = None,
) -> torch.Tensor:
  if return_lse:
    raise ValueError("Native attention backend does not support setting `return_lse=True`.")
  if _cp_config is None:
    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    out = torch.nn.functional.scaled_dot_product_attention(
      query=query,
      key=key,
      value=value,
      attn_mask=attn_mask,
      dropout_p=dropout_p,
      is_causal=is_causal,
      scale=scale,
      enable_gqa=enable_gqa,
    )
    out = out.permute(0, 2, 1, 3)
  else:
    out = _context_parallel_attention(
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      enable_gqa,
      return_lse,
      forward_op=_native_attention_forward_op,
      backward_op=_native_attention_backward_op,
      _cp_config=_cp_config,
    )
  return out


def _sdpa_cudnn_attention_forward_op(
  ctx: torch.autograd.function.FunctionCtx,
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  _save_ctx: bool = True,
  _cp_config: Optional["_ContextParallelConfig"] = None,
):
  # Native attention does not return_lse
  if return_lse:
    raise ValueError("cudnn attention with sdpa does not support return_lse=True")

  # used for backward pass
  if _save_ctx:
    ctx.save_for_backward(query, key, value)
    ctx.attn_mask = attn_mask
    ctx.dropout_p = dropout_p
    ctx.is_causal = is_causal
    ctx.scale = scale
    ctx.enable_gqa = enable_gqa

  query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
  with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
    out = torch.nn.functional.scaled_dot_product_attention(
      query=query,
      key=key,
      value=value,
      attn_mask=attn_mask,
      dropout_p=dropout_p,
      is_causal=is_causal,
      scale=scale,
      enable_gqa=enable_gqa,
    )
  out = out.permute(0, 2, 1, 3)

  return out


def _sdpa_cudnn_attention_backward_op(
  ctx: torch.autograd.function.FunctionCtx,
  grad_out: torch.Tensor,
  *args,
  **kwargs,
):
  raise NotImplementedError("Backward for cudnn attention with sdpa is not implemented yet.")


@_AttnBackendRegistry.register(
  _AttnBackend._SDPA_CUDNN,
  constraints=[_check_device, _check_shape],
  supports_context_parallel=True,
)
def _sdpa_cudnn_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  _cp_config: Optional["_ContextParallelConfig"] = None,
) -> torch.Tensor:
  lse = None
  if _cp_config is None and not return_lse:
    query, key, value = (x.permute(0, 2, 1, 3).contiguous() for x in (query, key, value))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
      out = torch.nn.functional.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
      )
    out = out.permute(0, 2, 1, 3)
  else:
    out = _context_parallel_attention(
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      enable_gqa,
      return_lse,
      forward_op=_sdpa_cudnn_attention_forward_op,
      backward_op=_sdpa_cudnn_attention_backward_op,
      _cp_config=_cp_config,
    )
    if return_lse:
      out, lse = out

  return (out, lse) if return_lse else out


def _sage_attention_forward_op(
  ctx: torch.autograd.function.FunctionCtx,
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  _save_ctx: bool = True,
  _cp_config: Optional["_ContextParallelConfig"] = None,
):
  if attn_mask is not None:
    raise ValueError("`attn_mask` is not yet supported for Sage attention.")
  if dropout_p > 0.0:
    raise ValueError("`dropout_p` is not yet supported for Sage attention.")
  if enable_gqa:
    raise ValueError("`enable_gqa` is not yet supported for Sage attention.")
  if sageattn is None:
    raise RuntimeError(
      "Sage attention backend is not available. Please install `sageattention` to use it.")

  out = sageattn(
    q=query,
    k=key,
    v=value,
    tensor_layout="NHD",
    is_causal=is_causal,
    sm_scale=scale,
    return_lse=return_lse,
  )
  lse = None
  if return_lse:
    out, lse, *_ = out
    lse = lse.permute(0, 2, 1)

  return (out, lse) if return_lse else out


def _sage_attention_backward_op(
  ctx: torch.autograd.function.FunctionCtx,
  grad_out: torch.Tensor,
  *args,
):
  raise NotImplementedError("Backward pass is not implemented for Sage attention.")


@_AttnBackendRegistry.register(
  _AttnBackend.SAGE,
  constraints=[_check_device_cuda, _check_qkv_dtype_bf16_or_fp16, _check_shape],
  supports_context_parallel=True,
)
def _sage_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  _cp_config: Optional["_ContextParallelConfig"] = None,
) -> torch.Tensor:
  if attn_mask is not None:
    raise ValueError("`attn_mask` is not yet supported for Sage attention.")
  if dropout_p > 0.0:
    raise ValueError("`dropout_p` is not yet supported for Sage attention.")
  if enable_gqa:
    raise ValueError("`enable_gqa` is not yet supported for Sage attention.")

  lse = None
  if _cp_config is None:
    if sageattn is None:
      raise RuntimeError(
        "Sage attention backend is not available. Please install `sageattention` to use it.")
    out = sageattn(
      q=query,
      k=key,
      v=value,
      tensor_layout="NHD",
      is_causal=is_causal,
      sm_scale=scale,
      return_lse=return_lse,
    )
    if return_lse:
      out, lse, *_ = out
  else:
    out = _context_parallel_attention(
      query,
      key,
      value,
      None,
      0.0,
      is_causal,
      scale,
      False,
      return_lse,
      forward_op=_sage_attention_forward_op,
      backward_op=_sage_attention_backward_op,
      _cp_config=_cp_config,
    )
    if return_lse:
      out, lse = out

  return (out, lse) if return_lse else out


def _flash_attention_forward_op(
  ctx: torch.autograd.function.FunctionCtx,
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  _save_ctx: bool = True,
  _cp_config: Optional["_ContextParallelConfig"] = None,
):
  if attn_mask is not None:
    raise ValueError("`attn_mask` is not yet supported for flash-attn 2.")
  if enable_gqa:
    raise ValueError("`enable_gqa` is not yet supported for flash-attn 2.")
  if flash_attn_func is None or _wrapped_flash_attn_forward is None:
    raise RuntimeError("Flash attention backend is not available.")

  window_size = (-1, -1)
  softcap = 0.0
  alibi_slopes = None
  deterministic = False
  grad_enabled = any(x.requires_grad for x in (query, key, value))

  if scale is None:
    scale = query.shape[-1] ** (-0.5)

  if grad_enabled or (_cp_config is not None and getattr(_cp_config, "_world_size", 1) > 1):
    dropout_p = dropout_p if dropout_p > 0 else 1e-30

  with torch.set_grad_enabled(grad_enabled):
    out, lse, _, rng_state = _wrapped_flash_attn_forward(
      query,
      key,
      value,
      dropout_p,
      scale,
      is_causal,
      window_size[0],
      window_size[1],
      softcap,
      alibi_slopes,
      return_lse,
    )
    lse = lse.permute(0, 2, 1)

  if _save_ctx:
    ctx.save_for_backward(query, key, value, out, lse, rng_state)
    ctx.dropout_p = dropout_p
    ctx.scale = scale
    ctx.is_causal = is_causal
    ctx.window_size = window_size
    ctx.softcap = softcap
    ctx.alibi_slopes = alibi_slopes
    ctx.deterministic = deterministic

  return (out, lse) if return_lse else out


def _flash_attention_backward_op(
  ctx: torch.autograd.function.FunctionCtx,
  grad_out: torch.Tensor,
  *args,
  **kwargs,
):
  if _wrapped_flash_attn_backward is None:
    raise RuntimeError("Flash attention backend is not available.")

  query, key, value, out, lse, rng_state = ctx.saved_tensors
  grad_query = torch.empty_like(query)
  grad_key = torch.empty_like(key)
  grad_value = torch.empty_like(value)

  _wrapped_flash_attn_backward(
    grad_out,
    query,
    key,
    value,
    out,
    lse,
    grad_query,
    grad_key,
    grad_value,
    ctx.dropout_p,
    ctx.scale,
    ctx.is_causal,
    ctx.window_size[0],
    ctx.window_size[1],
    ctx.softcap,
    ctx.alibi_slopes,
    ctx.deterministic,
    rng_state,
  )

  grad_query = grad_query[..., :grad_out.shape[-1]]
  grad_key = grad_key[..., :grad_out.shape[-1]]
  grad_value = grad_value[..., :grad_out.shape[-1]]

  return grad_query, grad_key, grad_value


if _flash_attn_available:

  @_AttnBackendRegistry.register(
    _AttnBackend.FLASH,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
    supports_context_parallel=True,
  )
  def _flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _cp_config: Optional["_ContextParallelConfig"] = None,
  ) -> torch.Tensor:
    lse = None
    if attn_mask is not None:
      raise ValueError("`attn_mask` is not supported for flash-attn 2.")
    if enable_gqa:
      raise ValueError("`enable_gqa` is not yet supported for flash-attn 2.")

    if _cp_config is None:
      out = flash_attn_func(
        q=query,
        k=key,
        v=value,
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=is_causal,
        return_attn_probs=return_lse,
      )
      if return_lse:
        out, lse, *_ = out
    else:
      out = _context_parallel_attention(
        query,
        key,
        value,
        None,
        dropout_p,
        is_causal,
        scale,
        False,
        return_lse,
        forward_op=_flash_attention_forward_op,
        backward_op=_flash_attention_backward_op,
        _cp_config=_cp_config,
      )
      if return_lse:
        out, lse = out

    return (out, lse) if return_lse else out

else:
  _flash_attention = None  # type: ignore[assignment]


def _flash_attention_3_forward_op(
  ctx: torch.autograd.function.FunctionCtx,
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  _save_ctx: bool = True,
  _cp_config: Optional["_ContextParallelConfig"] = None,
):
  """Flash Attention 3 forward operation for cache-dit (inference only)."""
  if attn_mask is not None:
    raise ValueError("`attn_mask` is not yet supported for flash-attn 3.")
  if enable_gqa:
    raise ValueError("`enable_gqa` is not yet supported for flash-attn 3.")
  if dropout_p > 0.0:
    raise ValueError("`dropout_p` > 0 is not yet supported for flash-attn 3.")

  if scale is None:
    scale = query.shape[-1] ** (-0.5)

  if _save_ctx:
    logger.warning(
      "Flash Attention 3 is configured for inference only, but _save_ctx=True was passed. "
      "Context will not be saved.")

  # Hardcoded parameters for FA3
  window_size = (-1, -1)
  softcap = 0.0
  deterministic = False

  out = flash_attn_3_func(
    q=query,
    k=key,
    v=value,
    softmax_scale=scale,
    causal=is_causal,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=window_size,
    attention_chunk=0,
    softcap=softcap,
    num_splits=1,
    pack_gqa=None,
    deterministic=deterministic,
    sm_margin=0,
    return_attn_probs=return_lse,
  )
  if return_lse:
    out, lse = out
    lse = lse.permute(0, 2, 1)
    return out, lse
  else:
    return out


def _flash_attention_3_backward_op(
  ctx: torch.autograd.function.FunctionCtx,
  grad_out: torch.Tensor,
  *args,
  **kwargs,
):
  """Flash Attention 3 backward operation for cache-dit."""
  raise NotImplementedError(
    "Backward pass for Flash Attention 3 with context parallelism is not implemented yet in cache-dit."
  )


# Re-register Flash Attention 3 backend
if _flash_attn_3_available:

  @_AttnBackendRegistry.register(
    _AttnBackend._FLASH_3,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
    supports_context_parallel=True,
  )
  def _flash_attention_3(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = False,
    return_lse: bool = False,
    _cp_config: Optional["_ContextParallelConfig"] = None,
  ) -> torch.Tensor:
    lse = None
    if _cp_config is None:
      # Non-parallel: use native flash-attn-3
      window_size = (-1, -1)
      softcap = 0.0
      deterministic = False
      out = flash_attn_3_func(
        q=query,
        k=key,
        v=value,
        softmax_scale=scale,
        causal=is_causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=window_size,
        attention_chunk=0,
        softcap=softcap,
        num_splits=1,
        pack_gqa=None,
        deterministic=deterministic,
        sm_margin=0,
        return_attn_probs=return_lse,
      )
      if return_lse:
        out, lse = out
        lse = lse.permute(0, 2, 1)
    else:
      # Parallel: use cache-dit's optimized implementation
      out = _context_parallel_attention(
        query,
        key,
        value,
        None,
        0.0,
        is_causal,
        scale,
        False,
        return_lse,
        forward_op=_flash_attention_3_forward_op,
        backward_op=_flash_attention_3_backward_op,
        _cp_config=_cp_config,
      )
      if return_lse:
        out, lse = out

    return (out, lse) if return_lse else out

else:
  _flash_attention_3 = None  # type: ignore[assignment]


def _maybe_modify_attn_mask_npu(query: torch.Tensor,
                                key: torch.Tensor,
                                attn_mask: Optional[torch.Tensor] = None):
  # Skip Attention Mask if all values are 1, `None` mask can speedup the computation
  if attn_mask is not None and torch.all(attn_mask != 0):
    attn_mask = None

  # Reshape Attention Mask: [batch_size, seq_len_k] -> [batch_size, 1, sqe_len_q, seq_len_k]
  # https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_fusion_attention.md
  if (attn_mask is not None and attn_mask.ndim == 2 and attn_mask.shape[0] == query.shape[0]
      and attn_mask.shape[1] == key.shape[1]):
    B, Sq, Skv = attn_mask.shape[0], query.shape[1], key.shape[1]
    attn_mask = ~attn_mask.to(torch.bool)
    attn_mask = attn_mask.unsqueeze(1).expand(B, Sq, Skv).unsqueeze(1).contiguous()

  return attn_mask


@_AttnBackendRegistry.register(
  _AttnBackend._NATIVE_NPU,
  constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
  supports_context_parallel=True,
)
def _native_npu_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  scale: Optional[float] = None,
  return_lse: bool = False,
  _cp_config: Optional["_ContextParallelConfig"] = None,
) -> torch.Tensor:
  if return_lse:
    raise ValueError("NPU attention backend does not support setting `return_lse=True`.")
  if _cp_config is None:
    attn_mask = _maybe_modify_attn_mask_npu(query, key, attn_mask)
    out = npu_fusion_attention(
      query,
      key,
      value,
      atten_mask=attn_mask,
      input_layout="BSND",
      scale=1.0 / math.sqrt(query.shape[-1]) if scale is None else scale,
      pre_tockens=MAX_TOKEN,
      next_tockens=MAX_TOKEN,
      head_num=query.size(2),
    )[0]
  else:
    out = _context_parallel_attention(
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      None,
      1.0 / math.sqrt(query.shape[-1]) if scale is None else scale,
      None,
      return_lse,
      forward_op=_npu_attention_forward_op,
      backward_op=_npu_attention_backward_op,
      _cp_config=_cp_config,
    )
  return out


def _npu_attention_forward_op(
  ctx: torch.autograd.function.FunctionCtx,
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  _save_ctx: bool = True,
  _cp_config: Optional["_ContextParallelConfig"] = None,
):
  if return_lse:
    raise ValueError("NPU attention backend does not support setting `return_lse=True`.")

  attn_mask = _maybe_modify_attn_mask_npu(query, key, attn_mask)
  out = npu_fusion_attention(
    query,
    key,
    value,
    atten_mask=attn_mask,
    input_layout="BSND",
    scale=scale,
    pre_tockens=MAX_TOKEN,
    next_tockens=MAX_TOKEN,
    head_num=query.size(2),
  )[0]

  return out


@_AttnBackendRegistry.register(
  _AttnBackend._NPU_FIA,
  constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
  supports_context_parallel=True,
)
def _npu_fused_infer_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  scale: Optional[float] = None,
  return_lse: bool = False,
  _cp_config: Optional["_ContextParallelConfig"] = None,
) -> torch.Tensor:
  if _cp_config is None:
    attn_mask = _maybe_modify_attn_mask_npu(query, key, attn_mask)
    out = npu_fused_infer_attention_score(
      query,
      key,
      value,
      atten_mask=attn_mask,
      input_layout="BSND",
      scale=1.0 / math.sqrt(query.shape[-1]) if scale is None else scale,
      pre_tokens=MAX_TOKEN,
      next_tokens=MAX_TOKEN,
      num_heads=query.size(2),
    )
  else:
    out = _context_parallel_attention(
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      None,
      1.0 / math.sqrt(query.shape[-1]) if scale is None else scale,
      None,
      return_lse,
      forward_op=_npu_fused_infer_attention_forward_op,
      backward_op=_npu_attention_backward_op,
      _cp_config=_cp_config,
    )
  return out


def _npu_fused_infer_attention_forward_op(
  ctx: torch.autograd.function.FunctionCtx,
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  _save_ctx: bool = True,
  _cp_config: Optional["_ContextParallelConfig"] = None,
):
  lse = None
  attn_mask = _maybe_modify_attn_mask_npu(query, key, attn_mask)

  result = npu_fused_infer_attention_score(
    query,
    key,
    value,
    atten_mask=attn_mask,
    input_layout="BSND",
    scale=1.0 / math.sqrt(query.shape[-1]) if scale is None else scale,
    pre_tokens=MAX_TOKEN,
    next_tokens=MAX_TOKEN,
    num_heads=query.size(2),
    softmax_lse_flag=return_lse,
  )

  if isinstance(result, tuple):
    out, raw_lse = result
  else:
    out, raw_lse = result, None

  if return_lse:
    if raw_lse is None:
      raise RuntimeError("return_lse=True but kernel did not return LSE tensor")
    lse = raw_lse.permute(0, 2, 1, 3)
    return out, lse

  return out


def _npu_attention_backward_op(
  ctx: torch.autograd.function.FunctionCtx,
  grad_out: torch.Tensor,
  *args,
  **kwargs,
):
  raise NotImplementedError("Backward pass is not implemented for Npu Fusion Attention.")
