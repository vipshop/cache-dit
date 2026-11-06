"""Optional FlashInfer SVDQuant backend.

This package intentionally does not import FlashInfer at import time. Cache-DiT remains usable
without the optional dependency; selecting a FlashInfer runtime raises a clear installation error.
"""

from ._svdquant import FLASHINFER_BACKENDS
from ._svdquant import flashinfer_available
from ._svdquant import flashinfer_svdquant_linear
from ._svdquant import validate_unit_wcscales

__all__ = [
  "FLASHINFER_BACKENDS",
  "flashinfer_available",
  "flashinfer_svdquant_linear",
  "validate_unit_wcscales",
]
