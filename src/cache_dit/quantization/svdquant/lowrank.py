# Adapted from deepcompressor's implementation of low-rank decomposition of the residual weight matrix in SVDQuant.
from __future__ import annotations
import torch
import torch.linalg

__all__ = ["decompose_lowrank_residual"]


@torch.inference_mode()
def decompose_lowrank_residual(
    weight: torch.Tensor,
    rank: int,
    *,
    output_dtype: torch.dtype | None = None,
    high_precision: bool = False,
    fp32_fallback: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError("Weight tensor must be 2D.")
    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}.")

    out_features, in_features = weight.shape
    max_rank = min(out_features, in_features)
    if rank > max_rank:
        raise ValueError(f"rank {rank} exceeds the maximum supported rank {max_rank}.")

    output_dtype = output_dtype or weight.dtype
    if rank == 0:
        down = torch.empty((0, in_features), dtype=output_dtype, device=weight.device)
        up = torch.empty((out_features, 0), dtype=output_dtype, device=weight.device)
        return down, up, weight.to(dtype=output_dtype)

    if high_precision:
        svd_dtype = torch.float64
    elif fp32_fallback:
        svd_dtype = torch.float32
    else:
        svd_dtype = weight.dtype

    try:
        u, s, vh = torch.linalg.svd(weight.to(dtype=svd_dtype), full_matrices=False)
    except NotImplementedError as exc:
        if (
            not high_precision
            and not fp32_fallback
            and svd_dtype in (torch.float16, torch.bfloat16)
        ):
            raise RuntimeError(
                "Low-precision SVD is not supported on this backend. "
                "Re-run with fp32_fallback=True or high_precision=True."
            ) from exc
        raise
    up = (u[:, :rank] * s[:rank]).to(dtype=output_dtype)
    down = vh[:rank].to(dtype=output_dtype)
    if (
        torch.isnan(up).any()
        or torch.isnan(down).any()
        or torch.isinf(up).any()
        or torch.isinf(down).any()
    ):
        raise ValueError("Encountered NaN/Inf while computing the low-rank factors.")

    reconstructed = up.to(dtype=svd_dtype) @ down.to(dtype=svd_dtype)
    residual = (weight.to(dtype=svd_dtype) - reconstructed).to(dtype=output_dtype)
    return down, up, residual
