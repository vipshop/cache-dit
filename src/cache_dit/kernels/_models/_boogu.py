"""Boogu-Image fused CUDA operators.

The three operators implement the hot normalization path of the Boogu-Image
transformer:

* ``boogu_scaled_rms_norm`` — ``rms_norm(x) * weight * (1 + scale)``
* ``boogu_gated_rms_norm`` — ``residual + tanh(gate) * rms_norm(x)``
* ``boogu_fused_qk_norm_rope`` — per-head RMSNorm followed by interleaved RoPE

They are optional: when ``cache_dit._C_models`` is missing the model code falls
back to the bit-equivalent eager chain in
``cache_dit.models.pipelines.boogu_image.fused_ops``. This module only owns the
operator layer (schema, fake, launch), not the call-site gating.
"""

import importlib
from types import ModuleType
from typing import Optional

import torch

_EXTENSION_MODULE_NAME = "cache_dit._C_models"
_cached_extension_module: Optional[ModuleType] = None
_cached_load_error: Optional[Exception] = None


def _load_extension_module() -> Optional[ModuleType]:
  """Import the optional model-specific CUDA extension at most once.

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
  except Exception as exc:  # pragma: no cover - exercised without the extension
    _cached_load_error = exc
    return None

  return _cached_extension_module


def models_extension_is_available() -> bool:
  """Return whether the optional model-specific CUDA extension can be imported.

  :returns: `True` when `cache_dit._C_models` is importable.
  """

  return _load_extension_module() is not None


def models_get_load_error() -> Optional[Exception]:
  """Return the cached import error of the optional model-specific extension.

  :returns: The import error raised while loading the extension, or `None` when
    the extension loaded successfully.
  """

  _load_extension_module()
  return _cached_load_error


def _get_required_ops_module() -> ModuleType:
  """Return the extension's `ops` submodule used by the Python wrappers.

  :returns: The `ops` submodule exported by the loaded extension.
  :raises RuntimeError: If the extension is unavailable or does not expose `ops`.
  """

  extension_module = _load_extension_module()
  if extension_module is None:
    raise RuntimeError(
      "The optional Cache-DiT model-specific CUDA extension is not available. Build the "
      "CUDA extensions with "
      "`CACHE_DIT_BUILD_CUDA=1 python -m pip install -e . --no-build-isolation` after "
      "`conda activate cdit`. The model kernels need sm80+, so a build that printed a "
      "skipped-SM warning omits this extension.") from models_get_load_error()

  ops_module = getattr(extension_module, "ops", None)
  if ops_module is None:
    raise RuntimeError(
      "The loaded Cache-DiT model-specific CUDA extension does not expose an `ops` "
      "submodule.")
  return ops_module


def _call_boogu_scaled_rms_norm(
  x: torch.Tensor,
  weight: torch.Tensor,
  eps: float,
  scale: Optional[torch.Tensor],
  zero_centered_weight: bool,
  out: torch.Tensor,
) -> None:
  """Bridge the fused scaled RMSNorm operator to the compiled extension.

  :param x: Contiguous 3D input `[B, S, N]` in bf16.
  :param weight: Contiguous per-channel weight `[N]` in bf16.
  :param eps: RMSNorm epsilon.
  :param scale: Optional per-channel modulation scale `[N]` in bf16.
  :param zero_centered_weight: Whether the weight uses the zero-centered convention.
  :param out: Pre-allocated output buffer matching `x`, written in place.
  """

  _get_required_ops_module().boogu_scaled_rms_norm(
    x,
    weight,
    eps,
    scale,
    zero_centered_weight,
    out,
  )


def _call_boogu_gated_rms_norm(
  x: torch.Tensor,
  weight: torch.Tensor,
  eps: float,
  gate: torch.Tensor,
  residual: torch.Tensor,
  zero_centered_weight: bool,
  out: torch.Tensor,
) -> None:
  """Bridge the fused gated residual RMSNorm operator to the extension.

  :param x: Contiguous 3D input `[B, S, N]` in bf16.
  :param weight: Contiguous per-channel weight `[N]` in bf16.
  :param eps: RMSNorm epsilon.
  :param gate: Contiguous per-channel gate `[N]` in bf16.
  :param residual: Contiguous residual stream matching `x` in bf16.
  :param zero_centered_weight: Whether the weight uses the zero-centered convention.
  :param out: Pre-allocated output buffer matching `x`, written in place.
  """

  _get_required_ops_module().boogu_gated_rms_norm(
    x,
    weight,
    eps,
    gate,
    residual,
    zero_centered_weight,
    out,
  )


def _call_boogu_fused_qk_norm_rope(
  x: torch.Tensor,
  freqs_cis: torch.Tensor,
  weight: torch.Tensor,
  eps: float,
  out: torch.Tensor,
) -> None:
  """Bridge the fused QK-norm + RoPE operator to the compiled extension.

  :param x: Contiguous 4D input `[B, S, H, D]` in bf16.
  :param freqs_cis: Contiguous complex64/complex128 rotary table `[B, S, D/2]`.
  :param weight: Contiguous per-channel weight `[D]` in bf16.
  :param eps: RMSNorm epsilon.
  :param out: Pre-allocated output buffer matching `x`, written in place.
  """

  _get_required_ops_module().boogu_fused_qk_norm_rope(x, freqs_cis, weight, eps, out)


torch.library.define(
  "cache_dit_cuda_ops::boogu_scaled_rms_norm",
  "(Tensor x, Tensor weight, float eps, Tensor? scale, bool zero_centered_weight) "
  "-> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::boogu_scaled_rms_norm", "CUDA")
def _scaled_rms_norm_impl(
  x: torch.Tensor,
  weight: torch.Tensor,
  eps: float,
  scale: Optional[torch.Tensor],
  zero_centered_weight: bool,
) -> torch.Tensor:
  if x.dim() != 3:
    raise ValueError(f"boogu_scaled_rms_norm expects a 3D input, got {tuple(x.shape)}.")
  out = torch.empty_like(x)
  _call_boogu_scaled_rms_norm(x, weight, eps, scale, zero_centered_weight, out)
  return out


@torch.library.register_fake("cache_dit_cuda_ops::boogu_scaled_rms_norm")
def _fake_scaled_rms_norm(
  x: torch.Tensor,
  weight: torch.Tensor,
  eps: float,
  scale: Optional[torch.Tensor],
  zero_centered_weight: bool,
) -> torch.Tensor:
  del weight, eps, scale, zero_centered_weight
  return torch.empty_like(x)


torch.library.define(
  "cache_dit_cuda_ops::boogu_gated_rms_norm",
  "(Tensor x, Tensor weight, float eps, Tensor gate, Tensor residual, "
  "bool zero_centered_weight) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::boogu_gated_rms_norm", "CUDA")
def _gated_rms_norm_impl(
  x: torch.Tensor,
  weight: torch.Tensor,
  eps: float,
  gate: torch.Tensor,
  residual: torch.Tensor,
  zero_centered_weight: bool,
) -> torch.Tensor:
  if x.dim() != 3:
    raise ValueError(f"boogu_gated_rms_norm expects a 3D input, got {tuple(x.shape)}.")
  if residual.shape != x.shape:
    raise ValueError("boogu_gated_rms_norm expects residual to match x, got "
                     f"{tuple(residual.shape)} vs {tuple(x.shape)}.")
  out = torch.empty_like(x)
  _call_boogu_gated_rms_norm(x, weight, eps, gate, residual, zero_centered_weight, out)
  return out


@torch.library.register_fake("cache_dit_cuda_ops::boogu_gated_rms_norm")
def _fake_gated_rms_norm(
  x: torch.Tensor,
  weight: torch.Tensor,
  eps: float,
  gate: torch.Tensor,
  residual: torch.Tensor,
  zero_centered_weight: bool,
) -> torch.Tensor:
  del weight, eps, gate, residual, zero_centered_weight
  return torch.empty_like(x)


torch.library.define(
  "cache_dit_cuda_ops::boogu_fused_qk_norm_rope",
  "(Tensor x, Tensor freqs_cis, Tensor weight, float eps) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::boogu_fused_qk_norm_rope", "CUDA")
def _fused_qk_norm_rope_impl(
  x: torch.Tensor,
  freqs_cis: torch.Tensor,
  weight: torch.Tensor,
  eps: float,
) -> torch.Tensor:
  if x.dim() != 4:
    raise ValueError(f"boogu_fused_qk_norm_rope expects a 4D input, got {tuple(x.shape)}.")
  out = torch.empty_like(x)
  _call_boogu_fused_qk_norm_rope(x, freqs_cis, weight, eps, out)
  return out


@torch.library.register_fake("cache_dit_cuda_ops::boogu_fused_qk_norm_rope")
def _fake_fused_qk_norm_rope(
  x: torch.Tensor,
  freqs_cis: torch.Tensor,
  weight: torch.Tensor,
  eps: float,
) -> torch.Tensor:
  del freqs_cis, weight, eps
  return torch.empty_like(x)


def boogu_scaled_rms_norm(
  x: torch.Tensor,
  weight: torch.Tensor,
  eps: float,
  scale: Optional[torch.Tensor] = None,
  zero_centered_weight: bool = False,
) -> torch.Tensor:
  """Fuse ``rms_norm(x) * weight * (1 + scale)`` into one launch.

  Call sites must check the bf16 / contiguity / batch preconditions first; this
  wrapper performs no gating and will raise (via `TORCH_CHECK`) on violation.

  :param x: Input activations `[B, S, N]` in bf16 on CUDA.
  :param weight: Per-channel norm weight `[N]` in bf16.
  :param eps: RMSNorm epsilon.
  :param scale: Optional per-channel modulation scale `[N]` in bf16.
  :param zero_centered_weight: Whether `weight` is zero-centered (adds 1).
  :returns: The fused output `[B, S, N]`.
  """

  return torch.ops.cache_dit_cuda_ops.boogu_scaled_rms_norm(
    x,
    weight,
    eps,
    scale,
    zero_centered_weight,
  )


def boogu_gated_rms_norm(
  x: torch.Tensor,
  weight: torch.Tensor,
  eps: float,
  gate: torch.Tensor,
  residual: torch.Tensor,
  zero_centered_weight: bool = False,
) -> torch.Tensor:
  """Fuse ``residual + tanh(gate) * rms_norm(x)`` into one launch.

  :param x: Input activations `[B, S, N]` in bf16 on CUDA.
  :param weight: Per-channel norm weight `[N]` in bf16.
  :param eps: RMSNorm epsilon.
  :param gate: Per-channel gate `[N]` in bf16 (the call site applies `tanh`).
  :param residual: Residual stream matching `x` in bf16.
  :param zero_centered_weight: Whether `weight` is zero-centered (adds 1).
  :returns: The gated residual output `[B, S, N]`.
  """

  return torch.ops.cache_dit_cuda_ops.boogu_gated_rms_norm(
    x,
    weight,
    eps,
    gate,
    residual,
    zero_centered_weight,
  )


def boogu_fused_qk_norm_rope(
  x: torch.Tensor,
  freqs_cis: torch.Tensor,
  weight: torch.Tensor,
  eps: float,
) -> torch.Tensor:
  """Fuse per-head RMSNorm with interleaved RoPE.

  :param x: Query or key states `[B, S, H, D]` in bf16 on CUDA.
  :param freqs_cis: Rotary table `[B, S, D/2]` in complex64 or complex128.
  :param weight: Per-channel norm weight `[D]` in bf16.
  :param eps: RMSNorm epsilon.
  :returns: The normalized and rotated states `[B, S, H, D]`.
  """

  return torch.ops.cache_dit_cuda_ops.boogu_fused_qk_norm_rope(
    x,
    freqs_cis,
    weight,
    eps,
  )


__all__ = [
  "_get_required_ops_module",
  "models_extension_is_available",
  "models_get_load_error",
  "boogu_scaled_rms_norm",
  "boogu_gated_rms_norm",
  "boogu_fused_qk_norm_rope",
]
