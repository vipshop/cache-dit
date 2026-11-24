import importlib.metadata
import math
from typing import Optional

import torch
from packaging.version import Version
from torch.distributed.tensor import DTensor

from ...logger import init_logger
from .native import _native_attention
from .register import (
  _AttnBackend,
  _AttnBackendRegistry,
  _context_parallel_attention,
  _ContextParallelConfig,
)

try:
  import sol_attn as _sol_attn
except ImportError:
  _sol_attn = None

logger = init_logger(__name__)

SOL_ATTN_DEFAULT_TAU = 1.0
SOL_ATTN_DEFAULT_THRESH_TYPE = "diag"
SOL_ATTN_DEFAULT_STRICT = False

_SOL_ATTN_INSTALL = (
  "pip install \"git+https://github.com/NVlabs/Sana.git@"
  "cf67a3acad7511acb85053e56a05e5dd24c82ba0#subdirectory=techniques/sparse_backends\"")
_SOL_ATTN_OP_NAME = "cache_dit_sol_attn_ops::forward"
_SOL_ATTN_OP_DEFINED = False
_SOL_RUNTIME_VALIDATED = False
_SOL_DEVICE_BACKENDS: dict[int, str] = {}
_SOL_CONFIG = {
  "tau": SOL_ATTN_DEFAULT_TAU,
  "thresh_type": SOL_ATTN_DEFAULT_THRESH_TYPE,
  "strict": SOL_ATTN_DEFAULT_STRICT,
}
_SOL_STATS = {
  "kernel_calls": 0,
  "fallback_calls": 0,
  "contiguous_copies": 0,
  "fallback_reasons": {},
}
_WARNED_FALLBACK_REASONS: set[str] = set()
_THRESH_TYPE_TO_INT = {"diag": 0, "exact": 1}


def _runtime_error(message: str) -> RuntimeError:
  return RuntimeError(f"{message} Install the optional dependency with `{_SOL_ATTN_INSTALL}`.")


def _validate_package_version(package: str, minimum: Optional[str] = None) -> None:
  try:
    version = importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError as exc:
    raise _runtime_error(f"Sol-Attn requires `{package}`.") from exc
  if minimum is not None and Version(version) < Version(minimum):
    raise _runtime_error(f"Sol-Attn requires `{package}>={minimum}`, found `{version}`.")


def _validate_sol_runtime() -> None:
  global _SOL_RUNTIME_VALIDATED
  if _SOL_RUNTIME_VALIDATED:
    return
  if _sol_attn is None:
    raise _runtime_error("The `sol_attn` attention backend is not installed.")
  _validate_package_version("sol-attn")
  if Version(torch.__version__.split("+")[0]) < Version("2.10"):
    raise _runtime_error(f"Sol-Attn requires `torch>=2.10`, found `{torch.__version__}`.")
  if torch.version.cuda is None or Version(torch.version.cuda) < Version("12.8"):
    raise _runtime_error(f"Sol-Attn requires CUDA>=12.8, found `{torch.version.cuda}`.")
  _validate_package_version("triton", "3.6")
  _SOL_RUNTIME_VALIDATED = True


def _validate_sol_device(device: torch.device) -> None:
  device_index = device.index if device.index is not None else torch.cuda.current_device()
  if device_index in _SOL_DEVICE_BACKENDS:
    return
  capability = torch.cuda.get_device_capability(device)
  if capability < (8, 0):
    raise RuntimeError(f"Sol-Attn requires NVIDIA compute capability >= 8.0, got {capability}.")
  backend = _sol_attn.get_sol_attn_backend(device)
  if backend.startswith("cute_"):
    _validate_package_version("nvidia-cutlass-dsl", "4.5")
    _validate_package_version("cuda-python")
  _SOL_DEVICE_BACKENDS[device_index] = backend
  logger.info(f"Sol-Attn upstream backend on CUDA:{device_index}: {backend}.")


def set_sol_attn_config(
  *,
  tau: float = SOL_ATTN_DEFAULT_TAU,
  thresh_type: str = SOL_ATTN_DEFAULT_THRESH_TYPE,
  strict: bool = SOL_ATTN_DEFAULT_STRICT,
) -> None:
  """Configure the process-wide Sol-Attn backend before inference or compilation.

  :param tau: Routing threshold multiplier; larger values select fewer exact KV blocks.
  :param thresh_type: Threshold estimator, either ``diag`` or ``exact``.
  :param strict: Raise on ineligible calls instead of using observable native fallback.
  """
  if not math.isfinite(tau):
    raise ValueError(f"Sol-Attn `tau` must be finite, got {tau}.")
  if thresh_type not in _THRESH_TYPE_TO_INT:
    raise ValueError(f"Sol-Attn `thresh_type` must be 'diag' or 'exact', got {thresh_type!r}.")
  _validate_sol_runtime()
  _ensure_sol_attn_op_defined()
  _SOL_CONFIG.update(tau=float(tau), thresh_type=thresh_type, strict=bool(strict))
  logger.info(f"Sol-Attn config: tau={tau}, thresh_type={thresh_type}, strict={bool(strict)}.")


def _get_sol_attn_stats() -> dict:
  """Return internal call statistics for validation and performance reports."""
  return {
    "kernel_calls": _SOL_STATS["kernel_calls"],
    "fallback_calls": _SOL_STATS["fallback_calls"],
    "contiguous_copies": _SOL_STATS["contiguous_copies"],
    "fallback_reasons": dict(_SOL_STATS["fallback_reasons"]),
    "backends": dict(_SOL_DEVICE_BACKENDS),
  }


def _reset_sol_attn_stats() -> None:
  _SOL_STATS.update(kernel_calls=0, fallback_calls=0, contiguous_copies=0, fallback_reasons={})
  _WARNED_FALLBACK_REASONS.clear()


def _record_fallback(reason: str) -> None:
  _SOL_STATS["fallback_calls"] += 1
  reasons = _SOL_STATS["fallback_reasons"]
  reasons[reason] = reasons.get(reason, 0) + 1
  if reason not in _WARNED_FALLBACK_REASONS:
    _WARNED_FALLBACK_REASONS.add(reason)
    logger.warning(f"Sol-Attn native fallback: {reason}.")


def _ineligible_reason(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor],
  dropout_p: float,
  is_causal: bool,
  scale: Optional[float],
  enable_gqa: bool,
  return_lse: bool,
  cp_gqa_strategy: Optional[str],
) -> Optional[str]:
  tensors = (query, key, value)
  if any(isinstance(x, DTensor) for x in tensors):
    return "DTensor inputs require conversion to local torch.Tensor first"
  if any(not isinstance(x, torch.Tensor) for x in tensors):
    return "Q/K/V must be local torch.Tensor objects"
  if any(x.ndim != 4 for x in tensors):
    return "Q/K/V must be rank-4 BTHD tensors"
  if query.shape != key.shape or query.shape != value.shape:
    return "Sol-Attn requires equal Q/K/V shapes (no cross-attention or GQA)"
  if any(x.device != query.device for x in (key, value)) or query.device.type != "cuda":
    return "Q/K/V must be on the same CUDA device"
  if query.shape[1] == 0:
    return "Sol-Attn requires a non-empty sequence"
  if query.shape[-1] != 128:
    return f"Sol-Attn requires head_dim=128, got {query.shape[-1]}"
  if any(x.dtype != torch.bfloat16 for x in tensors):
    return "Sol-Attn requires BF16 Q/K/V"
  if attn_mask is not None:
    return "attn_mask is not supported"
  if dropout_p != 0.0:
    return "dropout is not supported"
  if is_causal:
    return "causal attention is not supported"
  if enable_gqa or cp_gqa_strategy is not None:
    return "GQA/MQA and CP GQA strategies are not supported"
  if return_lse:
    return "return_lse is not supported outside Ring, which is disabled"
  default_scale = query.shape[-1] ** -0.5
  if scale is not None and (not math.isfinite(scale)
                            or not math.isclose(scale, default_scale, rel_tol=1e-6, abs_tol=1e-12)):
    return f"only the default scale {default_scale} is currently supported, got {scale}"
  return None


def _ensure_sol_attn_op_defined() -> None:
  global _SOL_ATTN_OP_DEFINED
  if _SOL_ATTN_OP_DEFINED:
    return

  torch.library.define(
    _SOL_ATTN_OP_NAME,
    "(Tensor q, Tensor k, Tensor v, float scale, float tau, int thresh_type) -> Tensor",
  )

  @torch.library.impl(_SOL_ATTN_OP_NAME, "CUDA")
  def _sol_attn_op_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    tau: float,
    thresh_type: int,
  ) -> torch.Tensor:
    _validate_sol_device(q.device)
    inputs = (q, k, v)
    contiguous = tuple(x.contiguous() for x in inputs)
    _SOL_STATS["contiguous_copies"] += sum(new is not old for old, new in zip(inputs, contiguous))
    q, k, v = contiguous
    output = _sol_attn.sol_attn(
      q,
      k,
      v,
      scale=scale,
      tau=tau,
      thresh_type="exact" if thresh_type else "diag",
      kv_splits=1,
      sink_tokens=0,
      sink_start=None,
    )
    _SOL_STATS["kernel_calls"] += 1
    return output.contiguous()

  @torch.library.register_fake(_SOL_ATTN_OP_NAME)
  def _sol_attn_op_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    tau: float,
    thresh_type: int,
  ) -> torch.Tensor:
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)

  _SOL_ATTN_OP_DEFINED = True


def _sol_attention_op(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  scale: float,
) -> torch.Tensor:
  _ensure_sol_attn_op_defined()
  return torch.ops.cache_dit_sol_attn_ops.forward(
    query,
    key,
    value,
    scale,
    _SOL_CONFIG["tau"],
    _THRESH_TYPE_TO_INT[_SOL_CONFIG["thresh_type"]],
  )


def _sol_attention_forward_op(
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
    raise ValueError("Sol-Attn does not return LSE; Ring attention is not supported.")
  return _sol_attention_op(query, key, value, query.shape[-1] ** -0.5 if scale is None else scale)


def _sol_attention_backward_op(
  ctx: torch.autograd.function.FunctionCtx,
  grad_out: torch.Tensor,
  *args,
):
  raise NotImplementedError("Sol-Attn is inference-only; backward is not implemented.")


@_AttnBackendRegistry.register(
  _AttnBackend.SOL_ATTN,
  constraints=[],
  supports_context_parallel=True,
)
def _sol_attention(
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
  _cp_config: Optional["_ContextParallelConfig"] = None,
) -> torch.Tensor:
  """Run approximate Sol-Attn on eligible BF16 NHD self-attention inputs."""
  _validate_sol_runtime()
  if _cp_config is not None and _cp_config.ring_degree > 1:
    raise NotImplementedError(
      "Sol-Attn does not support Ring or USP with Ring. Use pure Ulysses context parallelism.")
  reason = _ineligible_reason(
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
  )
  if reason is not None:
    if _SOL_CONFIG["strict"]:
      raise ValueError(f"Sol-Attn is ineligible: {reason}.")
    _record_fallback(reason)
    return _native_attention(
      query=query,
      key=key,
      value=value,
      attn_mask=attn_mask,
      dropout_p=dropout_p,
      is_causal=is_causal,
      scale=scale,
      enable_gqa=enable_gqa,
      return_lse=return_lse,
      cp_gqa_strategy=cp_gqa_strategy,
      _cp_config=_cp_config,
    )

  if _cp_config is None:
    return _sol_attention_op(query, key, value, query.shape[-1] ** -0.5 if scale is None else scale)
  return _context_parallel_attention(
    query,
    key,
    value,
    None,
    0.0,
    False,
    scale,
    False,
    False,
    None,
    forward_op=_sol_attention_forward_op,
    backward_op=_sol_attention_backward_op,
    _cp_config=_cp_config,
  )


__all__ = ["set_sol_attn_config"]
