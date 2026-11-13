"""Optional FlashInfer SVDQuant runtime.

This package intentionally does not import FlashInfer at import time. Cache-DiT remains usable
without the optional dependency; selecting a FlashInfer runtime raises a clear installation error at
the first forward.
"""

from .svdquant import CUTE_DSL_RUNTIME
from .svdquant import CUTLASS_SM120_BACKEND
from .svdquant import CUTLASS_SM120_LORA_RANKS
from .svdquant import CUTLASS_SM120_RUNTIME
from .svdquant import DEFAULT_FLASHINFER_RUNTIME
from .svdquant import FLASHINFER_RUNTIME
from .svdquant import FLASHINFER_RUNTIMES
from .svdquant import SVDQUANT_BACKEND
from .svdquant import activation_global_scale
from .svdquant import chain_for_runtime
from .svdquant import check_chain_support
from .svdquant import flashinfer_available
from .svdquant import flashinfer_svdquant_linear
from .svdquant import resolve_runtime

__all__ = [
  "CUTE_DSL_RUNTIME",
  "CUTLASS_SM120_BACKEND",
  "CUTLASS_SM120_LORA_RANKS",
  "CUTLASS_SM120_RUNTIME",
  "DEFAULT_FLASHINFER_RUNTIME",
  "FLASHINFER_RUNTIME",
  "FLASHINFER_RUNTIMES",
  "SVDQUANT_BACKEND",
  "activation_global_scale",
  "chain_for_runtime",
  "check_chain_support",
  "flashinfer_available",
  "flashinfer_svdquant_linear",
  "resolve_runtime",
]
