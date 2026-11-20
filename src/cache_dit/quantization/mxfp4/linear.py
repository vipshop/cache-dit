# coding=utf-8
# Copyright (c) Cache-DiT Contributors. All rights reserved.
"""MXFP4 Linear: weights stored as MXFP4 (E2M1 + E8M0 group scales).

The default compute path dequantizes weights to the activation dtype and calls
``torch.nn.functional.linear``. The dequant element-wise ops are fully
fusible, so under ``torch.compile`` the overhead collapses into a single
bandwidth-bound pass over the (4x smaller) packed weights, which makes this
path useful even on platforms without a fused MXFP4 GEMM.

MXFP4 is a weight-only format here: activations stay in the model dtype, which
is why the public quant type is ``mxfp4_weight_only``.

Environment controls:
  * ``CACHE_DIT_MXFP4_CACHE_DEQUANT=1``  cache the dequantized weight per layer
    (faster steady-state, but loses the runtime memory advantage).
"""

import os
from typing import Optional

import torch
import torch.nn.functional as F

from .tensor import MXFP4_BLOCK_SIZE, MXFP4Tensor

_CACHE_DEQUANT = os.environ.get("CACHE_DIT_MXFP4_CACHE_DEQUANT", "0") == "1"


class MXFP4Linear(torch.nn.Module):
  """Drop-in replacement of ``torch.nn.Linear`` with MXFP4 weight storage.

  ``quantize_device`` only selects where the one-time packing math runs; the
  packed buffers and bias always stay on the source linear's device, so
  quantizing a CPU-resident module never moves it.
  """

  def __init__(self,
               linear: torch.nn.Linear,
               block_size: int = MXFP4_BLOCK_SIZE,
               quantize_device: Optional[torch.device] = None):
    super().__init__()
    out_f, in_f = linear.weight.shape
    if in_f % block_size != 0 or in_f % 2 != 0:
      raise ValueError(f"Linear in_features={in_f} must be a multiple of block_size="
                       f"{block_size} and even (two E2M1 codes per byte), cannot "
                       "MXFP4-pack along K.")

    mx = MXFP4Tensor.from_weight(linear.weight.detach(),
                                 block_size=block_size,
                                 device=quantize_device)
    self.register_buffer("weight_packed", mx.packed)
    self.register_buffer("weight_scales", mx.scales)
    if linear.bias is not None:
      self.register_buffer("bias", linear.bias.detach().clone())
    else:
      self.bias = None

    self.in_features = in_f
    self.out_features = out_f
    self.block_size = block_size
    self._dequant_cache = None
    # Keep the original weight dtype for exact dequant output dtype.
    self._orig_dtype = linear.weight.dtype
    # Shared marker with the native FP8 module: tells the observer that this is
    # a quantized linear even though it carries no ``weight`` parameter.
    self._is_inner_quantized = True

  def _apply(self, fn, recurse: bool = True):
    """Move the packed payload/scales without ever dtype-casting them.

    ``float4_e2m1fn_x2`` and ``float8_e8m0fnu`` have no ``copy_`` kernel, so a
    plain ``module.half()`` / ``module.to(dtype)`` would raise. They are storage
    formats, not floats to cast: keep them verbatim, honor device moves, and let
    the bias cast (done by ``super()._apply``) drive the dequant output dtype.

    :param fn: Tensor transform installed by ``Module.to``/``cuda``/``half``.
    :param recurse: Whether children are transformed too.
    :returns: The transformed module.
    """
    packed, scales = self.weight_packed, self.weight_scales
    self.weight_packed = None
    self.weight_scales = None
    try:
      module = super()._apply(fn, recurse)
    except Exception:
      # Never lose the payload if the transform rejects the module.
      self.weight_packed, self.weight_scales = packed, scales
      raise
    if packed is not None:
      # One float probe yields both the target device and the target dtype: a
      # zero-sized tensor makes the copy free, and the fp4/e8m0 buffers are never
      # asked to dtype-cast (torch has no ``copy_`` kernel for them).
      probe = fn(torch.empty(0, dtype=self._orig_dtype, device=packed.device))
      module.weight_packed = packed.to(probe.device)
      module.weight_scales = scales.to(probe.device)
      module._orig_dtype = probe.dtype
    module._dequant_cache = None
    return module

  def dequantize_weight(self) -> torch.Tensor:
    """Materialize the dense weight in the current dequant output dtype.

    :returns: Dense ``[out_features, in_features]`` weight.
    """
    if _CACHE_DEQUANT and self._dequant_cache is not None:
      return self._dequant_cache
    w = MXFP4Tensor(self.weight_packed, self.weight_scales, self.out_features, self.in_features,
                    self.block_size).dequantize(dtype=self._orig_dtype)
    if _CACHE_DEQUANT:
      self._dequant_cache = w
    return w

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    w = self.dequantize_weight()
    return F.linear(x, w, self.bias)

  def extra_repr(self) -> str:
    return (f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, mxfp4(block={self.block_size})")
