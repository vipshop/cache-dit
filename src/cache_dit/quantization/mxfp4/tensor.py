# coding=utf-8
# Copyright (c) Cache-DiT Contributors. All rights reserved.
"""MXFP4 tensor packing / unpacking (pure PyTorch, no custom kernels).

Layout (OCP Microscaling Specs: version 1.0):
  * payload: ``torch.float4_e2m1fn_x2`` — two E2M1 codes packed per byte,
             shape ``[out_features, in_features // 2]`` (contiguous, K-packed).
  * scale:   ``torch.float8_e8m0fnu`` — one biased-exponent per 32-element
             group along K, shape ``[out_features, in_features // 32]``.
"""

from typing import Optional, Tuple

import torch

from ...platforms import current_platform

MXFP4_BLOCK_SIZE = 32
_E2M1_MAX = 6.0
_E8M0_BIAS = 127.0

# Decode table for the 4-bit E2M1(FN) code: [0..7] -> magnitudes, [8..15] -> -magnitudes.
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
                -6.0)

# Midpoints between consecutive representable positive values, used for
# round-to-nearest-even style quantization (ties resolved toward the even code).
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def mxfp4_is_available() -> bool:
  """Whether this build can use cache-dit's native MXFP4 backend."""
  return (getattr(torch, "float4_e2m1fn_x2", None) is not None
          and getattr(torch, "float8_e8m0fnu", None) is not None)


def _require_mx_dtypes():
  fp4 = getattr(torch, "float4_e2m1fn_x2", None)
  e8m0 = getattr(torch, "float8_e8m0fnu", None)
  if fp4 is None or e8m0 is None:
    raise RuntimeError("MXFP4 requires torch.float4_e2m1fn_x2 and torch.float8_e8m0fnu, "
                       "which are available in PyTorch >= 2.6. Please upgrade PyTorch or "
                       "choose another quantization type.")
  return fp4, e8m0


def _lut(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
  return torch.tensor(_E2M1_VALUES, device=device, dtype=dtype)


def resolve_mxfp4_quantize_device(weight: torch.Tensor, requested: str = "auto") -> torch.device:
  """Resolve the device used for the one-time MXFP4 packing.

  Packing is pure element-wise math, so the accelerator path is far faster than
  the CPU path (see ``bench/kernels/bench_mxfp4.py``); payload and scales are
  copied back to the weight's original device afterwards, so module placement
  never changes.

  :param weight: Source weight; an already-accelerated weight keeps its device.
  :param requested: ``auto`` (the current accelerator when available, else the
    weight device), ``cpu`` or ``cuda``.
  :returns: Device the packing math runs on.
  """
  requested = str(requested or "auto").lower()
  if requested == "auto":
    if weight.device.type != "cpu":
      return weight.device
    if current_platform.is_accelerator_available():
      return torch.device(current_platform.device_type, current_platform.current_device())
    return weight.device
  if requested == "cpu":
    return torch.device("cpu")
  if requested == "cuda":
    if not torch.cuda.is_available():
      raise RuntimeError("mxfp4_quantize_device='cuda' was requested, but CUDA is unavailable. "
                         "Use 'auto' or 'cpu' instead.")
    return torch.device("cuda", torch.cuda.current_device())
  raise ValueError(f"Unsupported MXFP4 quantize_device: {requested!r}.")


def _quant_codes(w: torch.Tensor) -> torch.Tensor:
  """Map a float tensor with |w| <= 6.0 to 4-bit E2M1 codes via nearest search."""
  mids = torch.tensor(_E2M1_MIDPOINTS, device=w.device, dtype=torch.float32)
  mag = w.abs().clamp(max=_E2M1_MAX).to(torch.float32)
  # bucketize(right=False) keeps the *lower* index on an exact hit, i.e. a value
  # equal to mids[k] lands on code k with k + 1 as the other tie candidate.
  idx = torch.bucketize(mag, mids)  # 0..7, nearest-magnitude bucket
  # Round-to-nearest-even on ties: an odd (odd code) tie bumps up to the even
  # code above it, an even tie already is the even code. Gathering the candidate
  # midpoint keeps this to one extra bool tensor instead of a [..., 7] broadcast.
  n_mid = len(_E2M1_MIDPOINTS)
  on_mid = (idx < n_mid) & (mag == mids.gather(0,
                                               idx.clamp(max=n_mid - 1).reshape(-1)).view_as(idx))
  even_idx = (idx % 2 == 0)
  idx = torch.where(on_mid & ~even_idx, idx + 1, idx)
  codes = idx.to(torch.uint8)
  codes = torch.where(w < 0, codes | 0x08, codes)
  return codes


def quantize_mxfp4_weight(
    weight: torch.Tensor,
    block_size: int = MXFP4_BLOCK_SIZE,
    device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
  """Quantize a 2-D ``[out, in]`` weight to MXFP4 along K (the ``in`` dim).

  :param weight: Source ``[out, in]`` weight; the result is returned on its device.
  :param block_size: Elements per E8M0 scale group (32 for OCP MX).
  :param device: Device the packing math runs on; ``None`` keeps the weight
    device. CPU->accelerator->CPU is cheap because only the 1/4-sized packed
    payload travels back.
  :return: (packed fp4 tensor ``[out, in // 2]`` viewed as float4_e2m1fn_x2,
            e8m0 scales ``[out, in // block_size]`` viewed as float8_e8m0fnu)
  """
  fp4_dtype, e8m0_dtype = _require_mx_dtypes()
  if weight.ndim != 2:
    raise ValueError(
      f"MXFP4 quantization expects 2-D Linear weights, got shape {tuple(weight.shape)}")
  out_f, in_f = weight.shape
  # E8M0 scales are one per block_size elements along K; byte packing needs an
  # even K, which an even block_size (32) already implies.
  if in_f % block_size != 0 or in_f % 2 != 0:
    raise ValueError(f"in_features={in_f} must be a multiple of block_size={block_size} and even "
                     "for MXFP4 packing along K.")

  storage_device = weight.device
  compute_device = storage_device if device is None else torch.device(device)
  # Move first, cast second: the weight travels in its original (bf16) dtype and
  # the fp32 promotion happens on the packing device. Casting on a CPU-resident
  # weight before the copy costs ~45 ms per 4096x4096 layer in the pipeline.
  w = weight.detach()
  if w.device != compute_device:
    w = w.to(device=compute_device)
  w = w.to(torch.float32)
  wg = w.reshape(out_f, in_f // block_size, block_size)

  amax = wg.abs().amax(dim=-1, keepdim=True)
  # Biased exponent via RCEIL (matches the OCP reference and torchao's rceil mode);
  # all-zero groups take exponent 0 (scale = 1.0) so that dequant keeps them exactly 0.
  descale = amax / _E2M1_MAX
  exponent = torch.where(
    descale > 0, torch.clamp(torch.ceil(torch.log2(descale)), min=-_E8M0_BIAS, max=_E8M0_BIAS),
    torch.zeros_like(descale))
  scale = torch.exp2(exponent)  # real (non-biased) scale value
  w_scaled = torch.clamp(wg / scale, -_E2M1_MAX, _E2M1_MAX).reshape(out_f, in_f)

  codes = _quant_codes(w_scaled)  # uint8 [out, in]
  packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
  scales_u8 = (exponent.squeeze(-1) + _E8M0_BIAS).to(torch.uint8).contiguous()
  if compute_device != storage_device:
    packed = packed.to(storage_device)
    scales_u8 = scales_u8.to(storage_device)
  return packed.view(fp4_dtype), scales_u8.view(e8m0_dtype)


def dequantize_mxfp4(packed: torch.Tensor,
                     scales: torch.Tensor,
                     out_features: int,
                     in_features: int,
                     dtype: torch.dtype = torch.bfloat16,
                     block_size: int = MXFP4_BLOCK_SIZE) -> torch.Tensor:
  fp4, e8m0 = _require_mx_dtypes()
  codes = packed.view(torch.uint8)
  lo = (codes & 0x0F).to(torch.long)
  hi = (codes >> 4).to(torch.long)
  lut = _lut(codes.device, torch.float32)
  vals = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(out_features, in_features)
  s = scales.view(e8m0).view(torch.uint8).to(torch.float32)
  s = torch.exp2(s - _E8M0_BIAS)  # [out, in // block]
  s = s.repeat_interleave(block_size, dim=1)[:, :in_features]
  return (vals * s).to(dtype)


class MXFP4Tensor:
  """Lightweight holder for MXFP4 packed weights + E8M0 group scales."""

  def __init__(self,
               packed: torch.Tensor,
               scales: torch.Tensor,
               out_features: int,
               in_features: int,
               block_size: int = MXFP4_BLOCK_SIZE):
    self.packed = packed
    self.scales = scales
    self.out_features = out_features
    self.in_features = in_features
    self.block_size = block_size

  @classmethod
  def from_weight(cls,
                  weight: torch.Tensor,
                  block_size: int = MXFP4_BLOCK_SIZE,
                  device: Optional[torch.device] = None) -> "MXFP4Tensor":
    packed, scales = quantize_mxfp4_weight(weight, block_size=block_size, device=device)
    return cls(packed, scales, weight.shape[0], weight.shape[1], block_size=block_size)

  def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return dequantize_mxfp4(self.packed,
                            self.scales,
                            self.out_features,
                            self.in_features,
                            dtype=dtype,
                            block_size=self.block_size)

  def nbytes(self) -> int:
    return self.packed.numel() * self.packed.element_size() + self.scales.numel()
