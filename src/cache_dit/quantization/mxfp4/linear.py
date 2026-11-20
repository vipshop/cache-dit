# coding=utf-8
# Copyright (c) Cache-DiT Contributors. All rights reserved.
"""MXFP4 Linear: weights stored as MXFP4 (E2M1 + E8M0 group scales).

The default compute path dequantizes weights to the activation dtype and calls
``torch.nn.functional.linear``. The dequant element-wise ops are fully
fusible, so under ``torch.compile`` the overhead collapses into a single
bandwidth-bound pass over the (4x smaller) packed weights, which makes this
path useful even on platforms without a fused MXFP4 GEMM.

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
               weight_only: bool = False,
               quantize_device: Optional[torch.device] = None):
    super().__init__()
    out_f, in_f = linear.weight.shape
    if in_f % (2 * block_size) != 0:
      raise ValueError(f"Linear in_features={in_f} is not divisible by {2 * block_size}, "
                       "cannot MXFP4-pack along K.")

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
    self.weight_only = weight_only  # metadata for summary/serialization only
    self._dequant_cache = None
    # Keep the original weight dtype for exact dequant output dtype.
    self._orig_dtype = linear.weight.dtype

  def dequant_weight(self) -> torch.Tensor:
    if _CACHE_DEQUANT and self._dequant_cache is not None:
      return self._dequant_cache
    w = MXFP4Tensor(self.weight_packed, self.weight_scales, self.out_features, self.in_features,
                    self.block_size).dequantize(dtype=self._orig_dtype)
    if _CACHE_DEQUANT:
      self._dequant_cache = w
    return w

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    w = self.dequant_weight()
    return F.linear(x, w, self.bias)

  def extra_repr(self) -> str:
    return (f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, mxfp4(block={self.block_size})")
