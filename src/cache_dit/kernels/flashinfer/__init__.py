"""Optional FlashInfer SVDQuant runtime.

This package intentionally does not import FlashInfer at import time. Cache-DiT remains usable
without the optional dependency; selecting the `flashinfer` runtime raises a clear installation
error at the first forward.
"""

from .svdquant import SVDQUANT_BACKEND
from .svdquant import activation_global_scale
from .svdquant import flashinfer_available
from .svdquant import flashinfer_svdquant_linear

__all__ = [
  "SVDQUANT_BACKEND",
  "activation_global_scale",
  "flashinfer_available",
  "flashinfer_svdquant_linear",
]
