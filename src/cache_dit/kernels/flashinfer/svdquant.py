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

from ...logger import init_logger

logger = init_logger(__name__)

# FlashInfer's `svdquant_linear` autotunes across several SM120 chains (`cute-dsl`,
# `cute-dsl-unfused`, `cutlass-sm120`, fused/separate preprocessing). The runtime names the chain
# in the same string as the runtime family, so a layer always knows which chain it runs:
# - `flashinfer+cute-dsl`: one implementation per shape, the profile this runtime compares against.
# - `flashinfer+cutlass-sm120`: FlashInfer's fused quantize + LoRA-down prefix, ranks 32/64 only.
# Bare `flashinfer` resolves to `flashinfer+cutlass-sm120`.
SVDQUANT_BACKEND = "cute-dsl"
CUTLASS_SM120_BACKEND = "cutlass-sm120"

FLASHINFER_RUNTIME = "flashinfer"
CUTE_DSL_RUNTIME = f"{FLASHINFER_RUNTIME}+{SVDQUANT_BACKEND}"
CUTLASS_SM120_RUNTIME = f"{FLASHINFER_RUNTIME}+{CUTLASS_SM120_BACKEND}"
DEFAULT_FLASHINFER_RUNTIME = CUTLASS_SM120_RUNTIME
FLASHINFER_RUNTIMES = (CUTE_DSL_RUNTIME, CUTLASS_SM120_RUNTIME)

# FlashInfer's SM120 CUTLASS chain only instantiates these LoRA ranks.
CUTLASS_SM120_LORA_RANKS = (32, 64)

# `448 * 6` maps the FP8-E4M3 block scale maximum (448) onto the E2M1 mantissa maximum (6).
NVFP4_GLOBAL_SCALE = 448.0 * 6.0

_INSTALL_HINT = (
  "FlashInfer is optional. Install it with `pip install flashinfer-python` before selecting "
  "runtime_kernel='flashinfer'.")

_REPORTED_CHAINS: set[str] = set()


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


def resolve_runtime(runtime_kernel: str) -> str:
  """Normalize a FlashInfer runtime spelling and return the chain it names.

  :param runtime_kernel: `flashinfer`, `flashinfer+cute-dsl`, or `flashinfer+cutlass-sm120`.
  :returns: The explicit `flashinfer+<chain>` runtime string.
  :raises ValueError: If the string is not a FlashInfer runtime.
  """
  if runtime_kernel == FLASHINFER_RUNTIME:
    return DEFAULT_FLASHINFER_RUNTIME
  if runtime_kernel not in FLASHINFER_RUNTIMES:
    raise ValueError(f"{runtime_kernel!r} is not a FlashInfer runtime; expected one of "
                     f"{(FLASHINFER_RUNTIME, *FLASHINFER_RUNTIMES)}.")
  return runtime_kernel


def chain_for_runtime(runtime_kernel: str) -> str:
  """Return the FlashInfer chain backend named by a normalized runtime string."""
  return resolve_runtime(runtime_kernel).removeprefix(f"{FLASHINFER_RUNTIME}+")


def check_chain_support(chain: str, rank: int, device: torch.device | None = None) -> None:
  """Reject a chain that cannot run this rank on this device, before any kernel is built.

  :param chain: FlashInfer chain name, `cute-dsl` or `cutlass-sm120`.
  :param rank: LoRA rank of the layer.
  :param device: Device the layer runs on; the capability check is skipped when omitted or CPU.
  :raises ValueError: If the chain does not cover this rank or device.
  """
  if chain != CUTLASS_SM120_BACKEND:
    return
  if rank not in CUTLASS_SM120_LORA_RANKS:
    raise ValueError(f"runtime_kernel='{CUTLASS_SM120_RUNTIME}' only supports LoRA ranks "
                     f"{CUTLASS_SM120_LORA_RANKS}, got {rank}; use '{CUTE_DSL_RUNTIME}' instead.")
  if device is None or device.type != "cuda":
    return
  capability = torch.cuda.get_device_capability(device)
  if capability != (12, 0):
    raise ValueError(f"runtime_kernel='{CUTLASS_SM120_RUNTIME}' requires an SM120 (12.0) device, "
                     f"got {capability}; use '{CUTE_DSL_RUNTIME}' instead.")
  if chain not in _REPORTED_CHAINS:
    _REPORTED_CHAINS.add(chain)
    logger.info(
      "Selected runtime_kernel='%s'; the first use of each LoRA rank JIT-builds a "
      "FlashInfer CUTLASS module and autotunes a fused prefix, which can take a couple "
      "of minutes.", CUTLASS_SM120_RUNTIME)


def activation_global_scale(x: torch.Tensor, pre_quant_scale: torch.Tensor) -> torch.Tensor:
  """Compute FlashInfer's FP32 activation global scale for one forward.

  The scale only has to keep the per-block FP8 scales inside the E4M3 range, and it is a per-tensor
  value, so it must be recomputed for every activation batch. `max|x| * max|scale|` is an upper
  bound of `max|x * scale|`: it can only shrink the FP8 block scales (never overflow them), and it
  costs one fused reduction instead of a full elementwise smoothing pass.

  :param x: BF16 activation `[M, K]`.
  :param pre_quant_scale: Per-input-channel smoothing scale `[K]`, BF16.
  :returns: FP32 device tensor of shape `[1]` holding at most `448 * 6 / max|x * pre_quant_scale|`.
  """
  peak = (torch.linalg.vector_norm(x, float("inf")).float() * pre_quant_scale.abs().amax().float())
  return (NVFP4_GLOBAL_SCALE / peak.clamp_min(1e-12)).reshape(1)


def flashinfer_svdquant_linear(
  x: torch.Tensor,
  weight_fp4: torch.Tensor,
  weight_sf: torch.Tensor,
  l2t_smoothed: torch.Tensor,
  l1_weight: torch.Tensor,
  pre_quant_scale: torch.Tensor,
  bias: torch.Tensor | None = None,
  runtime_kernel: str | None = None,
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
  :param runtime_kernel: FlashInfer runtime spelling; `None` uses `flashinfer+cutlass-sm120`.
  :returns: BF16 output `[M, N]`.
  """
  chain = chain_for_runtime(runtime_kernel or DEFAULT_FLASHINFER_RUNTIME)
  check_chain_support(chain, l1_weight.shape[1], x.device)
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
    backend=chain,
  )


__all__ = [
  "CUTE_DSL_RUNTIME",
  "CUTLASS_SM120_BACKEND",
  "CUTLASS_SM120_LORA_RANKS",
  "CUTLASS_SM120_RUNTIME",
  "DEFAULT_FLASHINFER_RUNTIME",
  "FLASHINFER_RUNTIME",
  "FLASHINFER_RUNTIMES",
  "NVFP4_GLOBAL_SCALE",
  "SVDQUANT_BACKEND",
  "activation_global_scale",
  "chain_for_runtime",
  "check_chain_support",
  "flashinfer_available",
  "flashinfer_svdquant_linear",
  "resolve_runtime",
]
