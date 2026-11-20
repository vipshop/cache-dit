# coding=utf-8
# Copyright (c) Cache-DiT Contributors. All rights reserved.
#
# MXFP4 (OCP Microscaling FP4: E2M1 payload + E8M0 group-of-32 scale) native
# quantization support for Cache-DiT.
#
# Why native: torchao's MX path (MXLinear) lowers to torch._scaled_mm with
# BlockWise 1x32 scaling, which is not functional on all accelerators (e.g. it
# faults or returns NaN on PPU devices), while NVFP4 (E4M3 group-of-16 scales)
# requires Blackwell-class tensor cores. This package provides a pure-PyTorch
# MXFP4 weight representation + Linear module so that memory-compact DiT
# inference works on any CUDA-like backend. Quantization is online only (weights
# are packed at load time); there is no MXFP4 checkpoint load/serialize path yet,
# and a fused MXFP4 GEMM can be added later without changing the storage format.

from .linear import MXFP4Linear
from .ptq import quantize_mxfp4
from .tensor import (MXFP4_BLOCK_SIZE, MXFP4Tensor, dequantize_mxfp4, mxfp4_is_available,
                     quantize_mxfp4_weight, resolve_mxfp4_quantize_device)

__all__ = [
  "MXFP4_BLOCK_SIZE",
  "MXFP4Tensor",
  "MXFP4Linear",
  "quantize_mxfp4_weight",
  "dequantize_mxfp4",
  "quantize_mxfp4",
  "mxfp4_is_available",
  "resolve_mxfp4_quantize_device",
]
