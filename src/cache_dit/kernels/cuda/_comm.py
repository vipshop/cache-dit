import importlib
from types import ModuleType

import torch

_EXTENSION_MODULE_NAME = "cache_dit._C_comm"
_cached_extension_module: ModuleType | None = None
_cached_load_error: Exception | None = None


def _load_extension_module() -> ModuleType | None:
  """Import the optional communication CUDA extension at most once.

  :returns: The loaded extension module, or `None` when the import failed and the
    failure was cached.
  """

  global _cached_extension_module, _cached_load_error

  if _cached_extension_module is not None:
    return _cached_extension_module
  if _cached_load_error is not None:
    return None

  try:
    _cached_extension_module = importlib.import_module(_EXTENSION_MODULE_NAME)
  except Exception as exc:  # pragma: no cover - exercised in environments without the extension
    _cached_load_error = exc
    return None

  return _cached_extension_module


def comm_extension_is_available() -> bool:
  """Return whether the optional communication CUDA extension can be imported.

  :returns: `True` when the compiled communication extension module is available.
  """

  return _load_extension_module() is not None


def comm_get_load_error() -> Exception | None:
  """Return the cached import error from the optional communication CUDA extension.

  :returns: The import error raised while loading the extension, or `None` when the
    extension loaded successfully.
  """

  _load_extension_module()
  return _cached_load_error


def _get_required_ops_module() -> ModuleType:
  """Return the extension's `ops` submodule used by the Python wrappers.

  :returns: The `ops` submodule exported by the loaded communication extension.
  :raises RuntimeError: If the extension is unavailable or does not expose `ops`.
  """

  extension_module = _load_extension_module()
  if extension_module is None:
    raise RuntimeError(
      "The optional Cache-DiT communication CUDA extension is not available. Build it with "
      "`CACHE_DIT_BUILD_COMM=1 python -m pip install -e . --no-deps --no-build-isolation` "
      "after `conda activate cdit`.") from comm_get_load_error()

  ops_module = getattr(extension_module, "ops", None)
  if ops_module is None:
    raise RuntimeError(
      "The loaded Cache-DiT communication CUDA extension does not expose an `ops` submodule.")
  return ops_module


def _call_fp8_comm_per_token_quant(x: torch.Tensor, y: torch.Tensor) -> None:
  """Bridge the per-token FP8 quantizer to the compiled extension.

  :param x: Contiguous 2D BF16 activation `[M, D]` to quantize.
  :param y: Contiguous 2D FP8 destination `[M, D + 2]`. The trailing bytes of every row keep the
    per-token scale.
  """

  _get_required_ops_module().fp8_comm_per_token_quant(x, y)


def _call_fp8_comm_per_token_dequant(x: torch.Tensor, y: torch.Tensor) -> None:
  """Bridge the per-token FP8 dequantizer to the compiled extension.

  :param x: Contiguous 2D FP8 input `[M, D + 2]` packed by the per-token quantizer.
  :param y: Contiguous 2D BF16 destination `[M, D]`.
  """

  _get_required_ops_module().fp8_comm_per_token_dequant(x, y)


def _call_fp8_comm_qkv_permute_quant(x: torch.Tensor, y: torch.Tensor) -> None:
  """Bridge the QKV permuting FP8 quantizer to the compiled extension.

  :param x: 5D BF16 QKV activation `[B, S, P, N, D]` whose last dimension has stride 1.
  :param y: Contiguous 5D FP8 destination `[P, S, B, N, D + 4]`.
  """

  _get_required_ops_module().fp8_comm_qkv_permute_quant(x, y)


def _call_fp8_comm_qkv_permute_dequant(quant_x: torch.Tensor, x: torch.Tensor) -> None:
  """Bridge the QKV restoring FP8 dequantizer to the compiled extension.

  :param quant_x: 4D FP8 quantized activation `[S, B, N, D + 4]` whose last dimension has stride 1.
  :param x: Contiguous 4D BF16 destination `[B, S, N, D]` restored to the original QKV layout.
  """

  _get_required_ops_module().fp8_comm_qkv_permute_dequant(quant_x, x)


def _call_fused_merge_attn_states(
  prev_out: torch.Tensor,
  prev_lse: torch.Tensor,
  suff_out: torch.Tensor,
  suff_lse: torch.Tensor,
  out: torch.Tensor,
  lse: torch.Tensor,
) -> None:
  """Bridge the attention-state merge operator to the compiled extension.

  :param prev_out: Contiguous 3D previous attention output `[B * N, H, D]`.
  :param prev_lse: Contiguous 2D previous log-sum-exp `[B * N, H]`.
  :param suff_out: Contiguous 3D suffix attention output `[B * N, H, D]`.
  :param suff_lse: Contiguous 2D suffix log-sum-exp `[B * N, H]`.
  :param out: Contiguous 3D merged attention output `[B * N, H, D]`, written in place.
  :param lse: Contiguous 2D merged log-sum-exp `[B * N, H]`, written in place.
  """

  _get_required_ops_module().fused_merge_attn_states(
    prev_out,
    prev_lse,
    suff_out,
    suff_lse,
    out,
    lse,
  )


__all__ = [
  "_call_fp8_comm_per_token_dequant",
  "_call_fp8_comm_per_token_quant",
  "_call_fp8_comm_qkv_permute_dequant",
  "_call_fp8_comm_qkv_permute_quant",
  "_call_fused_merge_attn_states",
  "_get_required_ops_module",
  "comm_extension_is_available",
  "comm_get_load_error",
]
