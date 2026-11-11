"""Optional FlashInfer SVDQuant backend.

This package intentionally does not import FlashInfer at import time. Cache-DiT remains usable
without the optional dependency; selecting a FlashInfer runtime raises a clear installation error.
"""

from .svdquant import FLASHINFER_BACKENDS
from .svdquant import flashinfer_available
from .svdquant import flashinfer_svdquant_linear
from .svdquant import validate_unit_wcscales

__all__ = [
  "FLASHINFER_BACKENDS",
  "flashinfer_available",
  "flashinfer_svdquant_linear",
  "validate_unit_wcscales",
]
