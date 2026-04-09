# Adapted from deepcompressor's implementation of low-rank decomposition
# for the residual weight matrix in SVDQuant.
from __future__ import annotations

import torch
import torch.linalg

__all__ = ["decompose_lowrank_residual"]

_SVD_PRECISIONS = ("low", "medium", "high")
_SVD_LOWRANK_BUFFER = 10
_SVD_LOWRANK_NITER = 4
_SVD_LOWRANK_SEED = 0


def _normalize_svd_precision(svd_precision: str) -> str:
  if not isinstance(svd_precision, str):
    raise TypeError(f"svd_precision must be a str, got {type(svd_precision)}.")
  normalized = svd_precision.lower()
  if normalized not in _SVD_PRECISIONS:
    raise ValueError(f"svd_precision must be one of {_SVD_PRECISIONS}, got {svd_precision!r}.")
  return normalized


def _assert_finite_factor(name: str, tensor: torch.Tensor) -> None:
  if torch.isnan(tensor).any():
    raise ValueError(f"NaN in {name} while computing the low-rank factors.")
  if torch.isinf(tensor).any():
    raise ValueError(f"Inf in {name} while computing the low-rank factors.")


def _svd_lowrank_rng_devices(weight: torch.Tensor) -> list[int]:
  if weight.device.type != "cuda":
    return []
  device_index = weight.device.index
  if device_index is None:
    device_index = torch.cuda.current_device()
  return [device_index]


def _run_seeded_svd_lowrank(
  weight: torch.Tensor,
  *,
  svd_dtype: torch.dtype,
  q: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  # Low precision defaults to a randomized algorithm. Keep it deterministic for
  # reproducible PTQ runs while restoring the caller's RNG state afterwards.
  with torch.random.fork_rng(devices=_svd_lowrank_rng_devices(weight)):
    torch.manual_seed(_SVD_LOWRANK_SEED)
    return torch.svd_lowrank(weight.to(dtype=svd_dtype), q=q, niter=_SVD_LOWRANK_NITER)


def _run_randomized_lowrank_svd(
  weight: torch.Tensor,
  rank: int,
  *,
  svd_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.dtype]:
  max_rank = min(weight.shape)
  q = min(max_rank, rank + _SVD_LOWRANK_BUFFER)

  try:
    u, s, v = _run_seeded_svd_lowrank(weight, svd_dtype=svd_dtype, q=q)
    factor_dtype = svd_dtype
  except (NotImplementedError, RuntimeError):
    if svd_dtype not in (torch.float16, torch.bfloat16):
      raise
    u, s, v = _run_seeded_svd_lowrank(weight, svd_dtype=torch.float32, q=q)
    factor_dtype = torch.float32

  up = u[:, :rank] * s[:rank]
  down = v[:, :rank].transpose(0, 1)
  _assert_finite_factor("U * S", up)
  _assert_finite_factor("V^T", down)
  return up, down, factor_dtype


def _run_full_svd(
  weight: torch.Tensor,
  rank: int,
  *,
  svd_dtype: torch.dtype,
  driver: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.dtype]:
  svd_kwargs = {"full_matrices": False}
  if driver is not None:
    svd_kwargs["driver"] = driver

  try:
    u, s, vh = torch.linalg.svd(weight.to(dtype=svd_dtype), **svd_kwargs)
    factor_dtype = svd_dtype
  except (NotImplementedError, RuntimeError):
    if svd_dtype not in (torch.float16, torch.bfloat16):
      raise
    u, s, vh = torch.linalg.svd(weight.to(dtype=torch.float32), full_matrices=False)
    factor_dtype = torch.float32

  up = u[:, :rank] * s[:rank]
  down = vh[:rank]
  _assert_finite_factor("U * S", up)
  _assert_finite_factor("V^T", down)
  return up, down, factor_dtype


@torch.inference_mode()
def decompose_lowrank_residual(
  weight: torch.Tensor,
  rank: int,
  *,
  output_dtype: torch.dtype | None = None,
  svd_precision: str = "low",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Factor a smoothed weight matrix into low-rank correction and residual.

  :param weight: Smoothed weight matrix with shape `[out_features, in_features]`.
  :param rank: Target rank for the low-rank approximation.
  :param output_dtype: Output dtype for the returned tensors. Defaults to the
    input `weight.dtype`.
  :param svd_precision: Precision mode for the low-rank decomposition.
    `"low"` uses randomized `torch.svd_lowrank`, `"medium"` uses the default
    full SVD route, and `"high"` uses float64 SVD with CUDA `gesvd`.

  :returns: A tuple `(down, up, residual)` where `up @ down` is the rank-`rank`
  approximation of `weight`, `down` has shape `[rank, in_features]`, `up` has
  shape `[out_features, rank]`, and `residual` is the remaining matrix to be
  quantized into the W4A4 path.
  """

  if weight.ndim != 2:
    raise ValueError("Weight tensor must be 2D.")
  if rank < 0:
    raise ValueError(f"rank must be non-negative, got {rank}.")

  out_features, in_features = weight.shape
  max_rank = min(out_features, in_features)
  if rank > max_rank:
    raise ValueError(f"rank {rank} exceeds the maximum supported rank {max_rank}.")

  svd_precision = _normalize_svd_precision(svd_precision)
  output_dtype = output_dtype or weight.dtype
  if rank == 0:
    down = torch.empty((0, in_features), dtype=output_dtype, device=weight.device)
    up = torch.empty((out_features, 0), dtype=output_dtype, device=weight.device)
    return down, up, weight.to(dtype=output_dtype)

  if svd_precision == "low":
    up, down, factor_dtype = _run_randomized_lowrank_svd(
      weight,
      rank,
      svd_dtype=weight.dtype,
    )
  elif svd_precision == "medium":
    up, down, factor_dtype = _run_full_svd(
      weight,
      rank,
      svd_dtype=weight.dtype,
    )
  else:
    up, down, factor_dtype = _run_full_svd(
      weight,
      rank,
      svd_dtype=torch.float64,
      driver="gesvd" if weight.device.type == "cuda" else None,
    )

  up = up.to(dtype=output_dtype)
  down = down.to(dtype=output_dtype)
  if (torch.isnan(up).any() or torch.isnan(down).any() or torch.isinf(up).any()
      or torch.isinf(down).any()):
    raise ValueError("Encountered NaN/Inf while computing the low-rank factors.")

  reconstructed = up.to(dtype=factor_dtype) @ down.to(dtype=factor_dtype)
  residual = (weight.to(dtype=factor_dtype) - reconstructed).to(dtype=output_dtype)
  return down, up, residual
