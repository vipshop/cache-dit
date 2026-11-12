"""Lazy, optional adapter for FlashInfer's complete SVDQuant linear chain.

The adapter runs FlashInfer's `svdquant_linear` on the derived pack described in
`cache_dit.quantization.svdquant.packing.convert_svdq_nvfp4_pack_to_flashinfer`, which folds
cache-dit's per-output-channel subscale into FlashInfer's per-block FP8 scale. The derived pack
costs one `e4m3` re-encode of the weight scale, so this runtime is a numerical approximation of
the native/cute runtimes, not their equivalent: use it to compare end-to-end performance against
FlashInfer's own SVDQuant chain, not as the default runtime.
"""

from __future__ import annotations

import torch

# FlashInfer's `svdquant_linear` autotunes across several SM120 chains (`cute-dsl`,
# `cute-dsl-unfused`, `cutlass-sm120`, fused/separate preprocessing). The runtime pins the CuTe
# chain so a layer keeps one implementation per shape, matching the profile this runtime exists
# to compare against.
SVDQUANT_BACKEND = "cute-dsl"

# `448 * 6` maps the FP8-E4M3 block scale maximum (448) onto the E2M1 mantissa maximum (6).
NVFP4_GLOBAL_SCALE = 448.0 * 6.0

_AMAX_CHUNK_ROWS = 1024

_INSTALL_HINT = (
  "FlashInfer is optional. Install it with `pip install flashinfer-python` before selecting "
  "runtime_kernel='flashinfer'.")


def flashinfer_available() -> bool:
  """Return whether the optional FlashInfer package can be imported."""
  try:
    import flashinfer  # noqa: F401
  except Exception:
    return False
  return True


_SVDQUANT_LINEAR = None


def _load_svdquant_linear():
  """Resolve FlashInfer's `svdquant_linear` once and reuse the callable."""
  global _SVDQUANT_LINEAR
  if _SVDQUANT_LINEAR is not None:
    return _SVDQUANT_LINEAR
  try:
    from flashinfer.gemm.gemm_svdquant import svdquant_linear
  except Exception as exc:
    raise ImportError(f"{_INSTALL_HINT} Original error: {exc}") from exc
  _SVDQUANT_LINEAR = svdquant_linear
  return _SVDQUANT_LINEAR


def activation_global_scale(x: torch.Tensor, pre_quant_scale: torch.Tensor) -> torch.Tensor:
  """Compute FlashInfer's FP32 activation global scale for one forward.

  The scale only has to keep the per-block FP8 scales inside the E4M3 range, but it is a per-tensor
  value, so it must be recomputed for every activation batch.

  :param x: BF16 activation `[M, K]`.
  :param pre_quant_scale: Per-input-channel smoothing scale `[K]`, BF16.
  :returns: FP32 device tensor of shape `[1]` holding `448 * 6 / max|x * pre_quant_scale|`.
  """
  peak = torch.zeros((), dtype=torch.float32, device=x.device)
  for start in range(0, x.shape[0], _AMAX_CHUNK_ROWS):
    chunk = x[start:start + _AMAX_CHUNK_ROWS]
    peak = torch.maximum(peak, (chunk * pre_quant_scale).abs().amax().float())
  return (NVFP4_GLOBAL_SCALE / peak.clamp_min(1e-12)).reshape(1)


def flashinfer_svdquant_linear(
  x: torch.Tensor,
  weight_fp4: torch.Tensor,
  weight_sf: torch.Tensor,
  l2t_smoothed: torch.Tensor,
  l1_weight: torch.Tensor,
  pre_quant_scale: torch.Tensor,
  bias: torch.Tensor | None = None,
) -> torch.Tensor:
  """Run FlashInfer's fused SVDQuant linear with cache-dit's derived pack.

  Mirrors `flashinfer.gemm.svdquant_linear` but takes the un-divided LoRA-up weight and derives
  the per-forward `global_scale` / `alpha` pair itself, so callers never have to keep the two
  mutually inverse by hand.

  :param x: BF16 activation `[M, K]`.
  :param weight_fp4: Row-major packed E2M1 weight `[N, K / 2]`, `uint8`.
  :param weight_sf: Flat 128x4 swizzled weight block scales, `uint8`.
  :param l2t_smoothed: LoRA-down matrix `[K, R]`, BF16. cache-dit stores `proj_down` with the
    smoothing compensation already folded in, so `x @ proj_down` is the LoRA-down activation the
    native runtime produces; pass it unscaled. (`smooth` applies to the residual GEMM only.)
  :param l1_weight: LoRA-up weight `[N, R]`, BF16, at its own scale (not pre-divided by alpha).
  :param pre_quant_scale: `1 / smooth_factor` `[K]`, BF16.
  :param bias: Optional BF16 bias `[N]`.
  :returns: BF16 output `[M, N]`.
  """
  global_scale = activation_global_scale(x, pre_quant_scale)
  # The fused residual epilogue emits `alpha * (a @ b^T + d @ l1^T)`, and `l1` must carry
  # `1 / alpha` so the LoRA term lands at its original scale.
  alpha = (1.0 / global_scale).reshape(1)
  l1_scaled = (l1_weight.float() * global_scale).to(l1_weight.dtype).contiguous()
  return _load_svdquant_linear()(
    x,
    weight_fp4,
    weight_sf,
    alpha,
    pre_quant_scale,
    l2t_smoothed,
    l1_scaled,
    global_scale,
    bias,
    backend=SVDQUANT_BACKEND,
  )


__all__ = [
  "NVFP4_GLOBAL_SCALE",
  "SVDQUANT_BACKEND",
  "activation_global_scale",
  "flashinfer_available",
  "flashinfer_svdquant_linear",
]
