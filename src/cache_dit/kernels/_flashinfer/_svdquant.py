"""Lazy, optional adapter for FlashInfer's complete SVDQuant linear path.

This module currently exposes only an operator-level adapter. The full `SVDQW4A4Linear`
backend integration is intentionally deferred; see the future-contract note below before
wiring it back into the runtime.
"""

from __future__ import annotations

import torch

FLASHINFER_BACKENDS = {
  "flashinfer+cute-dsl": "cute-dsl",
  "flashinfer+cutlass-sm120": "cutlass-sm120",
}

_INSTALL_HINT = (
  "FlashInfer is optional. Install it with `pip install flashinfer-python` (or the matching "
  "source tree) before selecting runtime_kernel='flashinfer+cute-dsl' or "
  "'flashinfer+cutlass-sm120'.")

# -------------------------------------------------------------------------------------------------
# Future SVDQuant backend contract (design note, not implemented yet)
# -------------------------------------------------------------------------------------------------
# `runtime_kernel` must be selected at quantisation/export time, not only at inference dispatch.
# The checkpoint stores exactly one weight pack and a lightweight format tag; never store both a
# cache-dit native/cute pack and a FlashInfer pack, and never store the original BF16/FP32 weight.
#
# pseudocode:
#
#   def quantize_linear_for_runtime(linear, calibration, runtime_kernel):
#       shared = {
#           "pack_format": runtime_kernel,
#           "proj_up": ...,          # [N, R] BF16
#           "proj_down": ...,        # [K, R] BF16
#           "smooth_factor": ...,    # [K] BF16
#           "bias": ...,             # [N] BF16 or None
#           "rank": R,
#       }
#       if runtime_kernel in {"native", "cute"}:
#           pack = cache_dit_native_nvfp4_pack(linear.weight)
#           return shared | {
#               "qweight": pack.qweight,
#               "wscales": pack.wscales,
#               "wcscales": pack.per_output_channel_subscale,
#           }
#       if runtime_kernel in FLASHINFER_BACKENDS:
#           gw = (448 * 6) / linear.weight.abs().amax()
#           wq, w_sf = flashinfer.nvfp4_quantize(
#               linear.weight, gw, sfLayout=layout_128x4, do_shuffle=False)
#           return shared | {
#               "flashinfer_qweight": wq,       # [N, K/2] uint8
#               "flashinfer_wscales": w_sf,     # 128x4 FP8
#               "flashinfer_weight_scale": gw,  # scalar, needed by alpha
#           }
#       raise ValueError(runtime_kernel)
#
#   def forward(x, state, runtime_kernel):
#       assert state["pack_format"] == runtime_kernel
#       if runtime_kernel in {"native", "cute"}:
#           return cache_dit_pack_forward(x, state)
#       if runtime_kernel in FLASHINFER_BACKENDS:
#           gs = (448 * 6) / (x * state.smooth).abs().amax()
#           alpha = 1 / (gs * state.flashinfer_weight_scale)
#           return flashinfer.svdquant_linear(
#               x, state.flashinfer_qweight, state.flashinfer_wscales, alpha,
#               state.smooth, state.smooth[:, None] * state.proj_down,
#               state.proj_up / alpha, gs, state.bias,
#               backend=FLASHINFER_BACKENDS[runtime_kernel])
#
#   def load_checkpoint(state, requested_runtime):
#       if state["pack_format"] != requested_runtime:
#           raise ValueError(
#               "checkpoint pack does not match runtime_kernel; re-quantize from the "
#               "original BF16 model with the requested backend")
#
# Consequences:
#   * `wcscales` belongs only to the native/cute pack. A FlashInfer pack quantises the BF16
#     weight directly, so per-output-channel magnitude is already absorbed into `gw`/`w_sf`.
#   * Switching an existing cache-dit checkpoint to FlashInfer is not supported; re-quantize
#     from the source BF16 model instead. No lossy native-pack -> FlashInfer conversion.
#   * FlashInfer remains an optional dependency and is imported only when its pack/runtime is
#     actually requested.


def flashinfer_available() -> bool:
  """Return whether the optional FlashInfer package can be imported."""
  try:
    import flashinfer  # noqa: F401
  except Exception:
    return False
  return True


def _load_svdquant_linear():
  try:
    from flashinfer.gemm.gemm_svdquant import svdquant_linear
  except Exception as exc:
    raise ImportError(_INSTALL_HINT) from exc
  return svdquant_linear


def validate_unit_wcscales(wcscales: torch.Tensor | None) -> None:
  """Reject per-column weight scales that FlashInfer's public path cannot represent."""
  if wcscales is None:
    return
  if not torch.allclose(wcscales, torch.ones_like(wcscales), rtol=0.0, atol=0.0):
    raise ValueError("FlashInfer SVDQuant does not support non-unit per-column wcscales; "
                     "use runtime_kernel='native' or 'cute' for this checkpoint.")


def flashinfer_svdquant_linear(
  x: torch.Tensor,
  qweight: torch.Tensor,
  wscales: torch.Tensor,
  proj_up: torch.Tensor,
  proj_down: torch.Tensor,
  bias: torch.Tensor | None,
  wcscales: torch.Tensor | None,
  smooth: torch.Tensor,
  alpha: float,
  backend: str,
) -> torch.Tensor:
  """Run FlashInfer's full SVDQuant chain on standard row-major packs.

  :param x: BF16 activation `[M, K]`.
  :param qweight: standard row-major E2M1 weight `[N, K/2]`, uint8.
  :param wscales: standard 128x4 weight scales, flat uint8.
  :param proj_up: row-major LoRA-up weight `[N, R]`, BF16.
  :param proj_down: row-major LoRA-down weight `[K, R]`, BF16.
  :param bias: optional BF16 bias `[N]`.
  :param wcscales: optional per-column scales; must be unit or None.
  :param smooth: per-input-channel smoothing scale `[K]`, BF16.
  :param alpha: FP4 residual dequantization scale.
  :param backend: cache-dit runtime string.
  :returns: BF16 output `[M, N]`.
  """
  if backend not in FLASHINFER_BACKENDS:
    raise ValueError(f"Unknown FlashInfer SVDQuant backend {backend!r}.")
  validate_unit_wcscales(wcscales)
  svdquant_linear = _load_svdquant_linear()

  device = x.device
  # FlashInfer uses an activation global scale to map the smoothed input into E2M1. The
  # checkpoint's wtscale is the weight-side dequant scale, so the fused residual alpha is the
  # ratio of the two. l1 must use the same effective alpha to avoid scaling the LoRA term twice.
  amax = (x.float() * smooth.float().unsqueeze(0)).abs().amax().clamp_min(1e-12)
  global_scale = (448.0 * 6.0 / amax).to(torch.float32)
  alpha_eff = float(alpha) / float(global_scale.item())
  alpha_tensor = torch.tensor([alpha_eff], dtype=torch.float32, device=device)
  l2t_smoothed = (smooth.float().unsqueeze(1) * proj_down.float()).to(torch.bfloat16)
  l1_scaled = (proj_up.float() / alpha_eff).to(torch.bfloat16)
  return svdquant_linear(
    x,
    qweight,
    wscales,
    alpha_tensor,
    smooth,
    l2t_smoothed,
    l1_scaled,
    global_scale,
    bias,
    backend=FLASHINFER_BACKENDS[backend],
  )
