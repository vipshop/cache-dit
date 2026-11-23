"""SVDQ A4 runtimes, single-layer quantizers and module quantization flows.

Two-stage NVFP4 weight quantization (the standard W4A4/W3A4 codec)
-------------------------------------------------------------------
For a residual matrix R[N,K], every 16 consecutive K elements form one scale
group g. The encoder writes two multiplicative scales and a code:

  A_g       = max_{k in g} |R[n,k]|
  s_c[n]    = BF16(max_g A_g / (448 * C_b))                 (coarse, per output channel)
  s_g[n,g]  = FP8_E4M3(A_g / (s_c[n] * C_b))                (micro, per 16 inputs)
  q[n,k]    = argmin_{c in Q_b} |R[n,k] / (s_c[n] * s_g[n,g]) - c|
  R_hat     = s_c[n] * s_g[n,g] * Q_b[q[n,k]]

C_b is the largest legal magnitude of the selected codebook Q_b: 6 for W3, 1.5
for W1/W2. Keeping 448 = FP8_E4M3 max in the coarse divisor guarantees
normalized values stay inside that range before the FP8 cast. The coarse scale
is rounded to BF16 first and the group scale/codes are derived from the value
the W4A4 kernel actually applies. This is the same two-level contract as
NVFP4; only Q_b and C_b change for the packed formats. W3 uses a two-bit
magnitude plane, while W1/W2 store sign plus zero/one magnitude bit.

Three-stage refinement (production default for W1A4/W2A4)
---------------------------------------------------------
Two-stage W1/W2 have too few reconstruction levels inside a 16-element group,
so the optional third stage partitions each group into four 4-element
subgroups j and stores a 2-bit exponent e[n,j]. The decoder maps
(e, q) through a table T_b directly to an E2M1 nibble:

  e = 0: W1 {0.5,1.5}                 W2 {0.5,1.5}
  e = 1: W1 {1.5} (same base)         W2 {1,3}
  e = 2: W1 {3}                       W2 {2,6}
  e = 3: W1 {6}                       W2 {2,6} (reserved duplicate)

  R_hat = s_c[n] * s_g[n,g] * T_b[e[n,k/4], q[n,k]]

Every table entry is a legal E2M1 value, so the MMA never sees a fourth scale
multiply: the exponent is absorbed into the nibble during expansion. Storage
adds four 2-bit exponents per byte, i.e. 0.5 bit/weight; W1 becomes 2 bit/weight
and W2 3 bit/weight before the per-16 scales. The outer two scales still follow
the NVFP4 equations above, which keeps stages=2 checkpoints and the standard
W4A4 path bit-exact.

Packed runtimes keep only the packed tensors in HBM. Forward expands the
current projection into a transient E2M1 workspace, overlaps that expansion
with the activation producer, then invokes the unmodified W4A4 GEMM.
"""
from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
from torch import nn

from ...kernels import expand_wna4
from ...kernels import svdq_gemm_w4a4
from ...kernels import svdq_quantize_w4a4
from ...logger import init_logger
from ..config import QuantizeConfig
from ..config import _FLASHINFER_LORA_RANK_GRANULARITY
from ..config import _SVDQ_PACKED_A4_RANKS
from ..config import _SVDQ_RUNTIME_KERNELS
from .lowrank import decompose_lowrank_residual
from .packing import fp_quantize
from .packing import swizzle_fp4_scales

_CODEBOOKS = {
  1: (1.5, -1.5),
  2: (0.5, 1.5, -0.5, -1.5),
  3: (0.5, 1.5, 3.0, 6.0, -0.5, -1.5, -3.0, -6.0),
}
_CODE_MAX = {1: 1.5, 2: 1.5, 3: 6.0}
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
                -6.0)
# Nibble mapping for the optional per-4 inner scale. W1 stores only a sign, so each inner exponent
# selects one E2M1 magnitude. W2 stores sign + one magnitude bit, so pidx shifts both base
# magnitudes by a power of two. The tables are indexed by (pidx, code).
_PACKED_A4_3STAGE_NIBBLES = {
  1: ((1, 9), (3, 11), (5, 13), (7, 15)),
  2: ((1, 3, 9, 11), (2, 5, 10, 13), (4, 7, 12, 15)),
}
_PACKED_A4_3STAGE_CODE_MAX = 6.0

logger = init_logger(__name__)


def _pad(value: int, multiple: int) -> int:
  return (value + multiple - 1) // multiple * multiple


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
  """Pack integer codes little-endian within each byte."""
  if bits == 1:
    if codes.shape[-1] % 8:
      raise ValueError("W1 packing requires a multiple-of-8 K extent.")
    out = torch.zeros(codes.shape[:-1] + (codes.shape[-1] // 8, ),
                      dtype=torch.uint8,
                      device=codes.device)
    for i in range(8):
      out |= (codes[..., i::8].to(torch.uint8) & 1) << i
    return out.contiguous()
  if bits == 2:
    if codes.shape[-1] % 4:
      raise ValueError("W2 packing requires a multiple-of-4 K extent.")
    out = torch.zeros(codes.shape[:-1] + (codes.shape[-1] // 4, ),
                      dtype=torch.uint8,
                      device=codes.device)
    for i in range(4):
      out |= (codes[..., i::4].to(torch.uint8) & 3) << (2 * i)
    return out.contiguous()
  if codes.shape[-1] % 4:
    raise ValueError("W3 magnitude packing requires a multiple-of-4 K extent.")
  low = torch.zeros(codes.shape[:-1] + (codes.shape[-1] // 4, ),
                    dtype=torch.uint8,
                    device=codes.device)
  for i in range(4):
    low |= (codes[..., i::4].to(torch.uint8) & 3) << (2 * i)
  high = torch.zeros(codes.shape[:-1] + (codes.shape[-1] // 8, ),
                     dtype=torch.uint8,
                     device=codes.device)
  for i in range(8):
    high |= ((codes[..., i::8] >> 2).to(torch.uint8) & 1) << i
  return low.contiguous(), high.contiguous()


def unpack_codes(packed: torch.Tensor, bits: int, high: torch.Tensor | None = None) -> torch.Tensor:
  """Diagnostic unpacker; returns one integer code per weight."""
  out = []
  if bits == 1:
    for i in range(8):
      out.append((packed >> i) & 1)
    return torch.stack(out, dim=-1).flatten(-2)
  groups = 4
  for i in range(groups):
    out.append((packed >> (2 * i)) & 3)
  low = torch.stack(out, dim=-1).flatten(-2)
  if bits == 2:
    return low
  if high is None:
    raise ValueError("W3 requires its sign plane.")
  signs = []
  for i in range(8):
    signs.append((high >> i) & 1)
  sign = torch.stack(signs, dim=-1).flatten(-2)
  return low + (sign << 2)


def _validate_inner_scales(inner: torch.Tensor, bits: int) -> None:
  """Reject reserved 2-bit exponents before they reach a table lookup.

  W1 defines all four exponent values. W2 currently defines three; the encoder never emits 3, and a
  corrupted checkpoint must fail loudly instead of silently aliasing to the last table.
  """
  allowed = 4 if bits == 1 else 3
  if inner.numel():
    for shift in (0, 2, 4, 6):
      if torch.any(((inner >> shift) & 3) >= allowed).item():
        raise ValueError(f"W{bits}A4 three-stage inner scale contains a reserved exponent "
                         f"(allowed 0..{allowed - 1}).")


def dequantize_packed(packed: torch.Tensor,
                      scale: torch.Tensor,
                      bits: int,
                      high: torch.Tensor | None = None,
                      code_max: float | None = None) -> torch.Tensor:
  """Return FP32 packed weights with a per-16 scale."""
  codes = unpack_codes(packed, bits, high).long()
  codebook = torch.tensor(_CODEBOOKS[bits], dtype=torch.float32, device=packed.device)
  values = codebook[codes]
  groups = values.shape[-1] // 16
  values = values.view(*values.shape[:-1], groups, 16)
  return (values * scale.float().unsqueeze(-1)).flatten(-2)


def quantize_residual(residual: torch.Tensor, bits: int) -> tuple[torch.Tensor, ...]:
  """Quantize ``[N,K]`` with the standard two-stage NVFP4 scale hierarchy.

  The encoder mirrors the arithmetic of the W4A4 kernel exactly:

  1. Split K into 16-wide groups and measure ``A_g = max(abs(group))``.
  2. Derive the BF16 coarse scale ``s_c = max_g(A_g) / (448 * C_b)`` and
     round it to BF16 *before* using it, because that is the value the kernel
     multiplies after the MMA.
  3. Derive the per-group FP8 E4M3 scale ``s_g = A_g / (s_c * C_b)``.
  4. Normalize by both scales and pick the nearest code in ``Q_b``.
  5. Store ``q`` and the 128x4-swizzled ``s_g``; ``s_c`` stays per channel.

  ``448`` is the largest finite E4M3 value, so the normalized group scale can
  always be represented. A zero or FP8-underflowed group is forced to zero
  before the argmin, avoiding ``0/0``; its scale already reconstructs zero.

  :param residual: FP32/FP16/BF16 residual ``[N,K]`` after the low-rank split.
  :param bits: Packed code width: 1 (W1), 2 (W2) or 3 (W3).
  :returns: ``(packed, group_scales, channel_scales)``. ``group_scales`` is the
    FP8 buffer in the kernel's 128x4 swizzle; ``channel_scales`` is BF16 ``[N]``.
  """
  residual = residual.float()
  rows, channels = residual.shape
  if channels % 16:
    raise ValueError("Packed W1/W2/W3 require K divisible by 16.")
  # A singleton axis keeps the reductions aligned with the kernel's per-channel x per-16 contract.
  grouped = residual.view(rows, 1, channels // 16, 16)
  group_absmax = grouped.abs().amax(dim=-1, keepdim=True)
  code_max = _CODE_MAX[bits]
  # Reserve the FP8 dynamic range for s_g and cap every normalized group at code_max.
  channel_scale = (group_absmax.amax(dim=2, keepdim=True) / (448.0 * code_max)).clamp_min(1e-12)
  # Match the coarse scale actually applied by the W4A4 kernel before deriving group scales/codes.
  channel_scale = channel_scale.to(torch.bfloat16).float()
  group_scale = (group_absmax / channel_scale / code_max).clamp_min(1e-12)
  group_scale_fp8 = group_scale.to(torch.float8_e4m3fn)
  codebook = torch.tensor(_CODEBOOKS[bits], dtype=torch.float32, device=residual.device)
  # The W4A4 epilogue computes s_c * s_g * q. Using those exact FP8/BF16 values here makes the
  # two-stage codebook a drop-in replacement for NVFP4 instead of an independent approximation.
  normalized = grouped / channel_scale / group_scale_fp8.float()
  # An all-zero or underflowed group has a zero FP8 scale. Its payload is exactly zero regardless
  # of the code; avoid feeding 0/0 = NaN into the codebook argmin.
  normalized = torch.where(group_scale_fp8.float() > 0, normalized, torch.zeros_like(normalized))
  codes = fp_quantize(normalized, codebook=codebook)
  packed = pack_codes(codes.view(rows, channels), bits)
  flat_scales = swizzle_fp4_scales(group_scale_fp8.squeeze(1).squeeze(-1)).view(torch.float8_e4m3fn)
  coarse = channel_scale.reshape(-1).to(torch.bfloat16)
  return packed, flat_scales, coarse


def quantize_residual_3stage(residual: torch.Tensor, bits: int) -> tuple[torch.Tensor, ...]:
  """Quantize a residual with a per-4 inner E2M1 exponent on top of NVFP4 scales.

  The first two stages are identical to :func:`quantize_residual`: BF16
  per-channel ``s_c`` and FP8_E4M3 per-16 ``s_g`` still normalize the residual,
  so old stages=2 checkpoints and the W4A4 kernel contract remain valid. The
  encoder then splits each 16-wide group into four 4-wide subgroups and chooses
  one exponent ``e`` per subgroup:

  .. code-block:: text

    e=0: W1 {0.5,1.5}              W2 {0.5,1.5}
    e=1: W1 {1.5}                  W2 {1,3}
    e=2: W1 {3}                    W2 {2,6}
    e=3: W1 {6}                    W2 {2,6} (reserved duplicate)

  For each subgroup we test all four tables, pick the table+code with the
  smallest squared error, and emit a 2-bit exponent. The exponent is stored as
  four fields per uint8, costing 0.5 bit per weight. Because every table value
  is already a legal E2M1 nibble, the runtime expansion can write the final
  nibble directly; no extra multiplication or fourth scale is needed by MMA.

  :param residual: Floating-point residual ``[N,K]`` after the low-rank split.
  :param bits: ``1`` for sign-only W1 or ``2`` for W2.
  :returns: ``(packed, group_scales, channel_scales, inner_scales)`` where ``inner_scales`` is
    ``uint8 [N,K/16]`` and packs four per-4 exponents little-endian.
  """
  if bits not in (1, 2):
    raise ValueError(f"Three-stage quantization supports W1/W2 only, got bits={bits}.")
  residual = residual.float()
  rows, channels = residual.shape
  if channels % 16:
    raise ValueError("Packed W1/W2 three-stage quantization requires K divisible by 16.")
  groups = channels // 16
  grouped = residual.view(rows, groups, 16)
  # Stage 1/2: use code_max=6 so the largest inner-table value (W1 e=3 -> 6,
  # W2 e=2/3 -> 6) fits inside the same FP8-normalized group as NVFP4.
  group_absmax = grouped.abs().amax(dim=-1, keepdim=True)
  channel_scale = (group_absmax.amax(dim=1, keepdim=True) /
                   (448.0 * _PACKED_A4_3STAGE_CODE_MAX)).clamp_min(1e-12)
  channel_scale = channel_scale.to(torch.bfloat16).float()
  group_scale = (group_absmax / channel_scale / _PACKED_A4_3STAGE_CODE_MAX).clamp_min(1e-12)
  group_scale_fp8 = group_scale.to(torch.float8_e4m3fn)
  normalized = grouped / channel_scale / group_scale_fp8.float()
  normalized = torch.where(group_scale_fp8.float() > 0, normalized, torch.zeros_like(normalized))
  # Stage 3: pick one power-of-two table per four consecutive K elements. The
  # outer scales are fixed first, so the inner exponent only refines precision
  # and cannot perturb the already-validated NVFP4 scale path.
  sub = normalized.view(rows, groups, 4, 4)
  best_error = None
  best_codes = None
  best_pidx = None
  for pidx, nibbles in enumerate(_PACKED_A4_3STAGE_NIBBLES[bits]):
    codebook = torch.tensor([_E2M1_VALUES[index] for index in nibbles],
                            dtype=sub.dtype,
                            device=sub.device)
    codes = fp_quantize(sub, codebook=codebook)
    error = (sub - codebook[codes]).square().sum(dim=-1)
    if best_error is None:
      best_error = error
      best_codes = codes
      best_pidx = torch.full_like(error, pidx, dtype=torch.uint8)
    else:
      better = error < best_error
      best_error = torch.where(better, error, best_error)
      best_codes = torch.where(better.unsqueeze(-1), codes, best_codes)
      best_pidx = torch.where(better, pidx, best_pidx)
  packed = pack_codes(best_codes.reshape(rows, channels), bits)
  # The checkpoint stores 4 x 2-bit exponents in one byte; the C++/CUDA
  # decoders read the same little-endian fields before every 4-weight decode.
  micro = torch.zeros((rows, groups), dtype=torch.uint8, device=residual.device)
  for index in range(4):
    micro |= best_pidx[..., index].to(torch.uint8) << (2 * index)
  flat_scales = swizzle_fp4_scales(group_scale_fp8.squeeze(-1).view(torch.float8_e4m3fn))
  coarse = channel_scale.reshape(-1).to(torch.bfloat16)
  return packed, flat_scales, coarse, micro


def dequantize_packed_3stage(packed: torch.Tensor,
                             scale: torch.Tensor,
                             bits: int,
                             inner: torch.Tensor,
                             high: torch.Tensor | None = None) -> torch.Tensor:
  """Diagnostic FP32 reconstruction for a three-stage W1/W2 residual."""
  if bits not in (1, 2):
    raise ValueError(f"Three-stage dequantization supports W1/W2 only, got bits={bits}.")
  _validate_inner_scales(inner, bits)
  codes = unpack_codes(packed, bits, high).long()
  pidx = torch.stack([(inner >> (2 * index)) & 3 for index in range(4)], dim=-1).flatten(-2).long()
  pidx = pidx.repeat_interleave(4, dim=-1)
  if pidx.shape != codes.shape:
    raise ValueError(
      f"inner scale shape implies {pidx.shape[1]} weights, expected {codes.shape[1]}.")
  table = torch.tensor(_PACKED_A4_3STAGE_NIBBLES[bits], dtype=torch.long, device=packed.device)
  nibbles = table[pidx, codes]
  values = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)[nibbles]
  groups = values.shape[-1] // 16
  values = values.view(*values.shape[:-1], groups, 16)
  return (values * scale.float().unsqueeze(-1)).flatten(-2)


def _packed_expand_stream(device: torch.device) -> torch.cuda.Stream:
  """Return the per-device side stream used to overlap expansion with activation production."""
  return _packed_expand_stream_cached(device.type, device.index)


@functools.lru_cache(maxsize=16)
def _packed_expand_stream_cached(device_type: str, device_index: int | None) -> torch.cuda.Stream:
  return torch.cuda.Stream(device=torch.device(device_type, device_index))


# Standard SVDQ algorithm implementation. This is the most complete runtime path, including
# fused MLP and native/CuTe/FlashInfer backends.
class SVDQW4A4Linear(nn.Module):
  """Standard SVDQ W4A4 runtime with the most complete feature and backend support."""

  def __init__(
    self,
    in_features: int,
    out_features: int,
    rank: int = 32,
    bias: bool = True,
    precision: str = "int4",
    act_unsigned: bool = False,
    runtime_kernel: str | None = None,
    lora_layout: str | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
  ) -> None:
    """Initialize the packed W4A4 runtime module."""
    super().__init__()
    if device is None:
      device = torch.device("cpu")
    if rank < 0:
      raise ValueError(f"rank must be non-negative, got {rank}.")
    if runtime_kernel is not None and runtime_kernel not in _SVDQ_RUNTIME_KERNELS:
      raise ValueError(f"runtime_kernel must be one of {_SVDQ_RUNTIME_KERNELS} or None; "
                       f"got {runtime_kernel!r}.")
    resolved_runtime_kernel = runtime_kernel or "native"
    is_flashinfer = resolved_runtime_kernel.startswith("flashinfer")
    if is_flashinfer:
      from ...kernels.flashinfer import chain_for_runtime
      from ...kernels.flashinfer import check_chain_support
      from ...kernels.flashinfer import resolve_runtime
      resolved_runtime_kernel = resolve_runtime(resolved_runtime_kernel)
      check_chain_support(chain_for_runtime(resolved_runtime_kernel), rank)
    if (resolved_runtime_kernel == "cute" or is_flashinfer) and precision != "nvfp4":
      raise ValueError(f"runtime_kernel={resolved_runtime_kernel!r} is only supported for "
                       "precision='nvfp4'.")
    if (resolved_runtime_kernel == "cute" or is_flashinfer) and torch_dtype != torch.bfloat16:
      raise ValueError(f"runtime_kernel={resolved_runtime_kernel!r} currently requires "
                       "torch_dtype=torch.bfloat16.")
    if is_flashinfer and (rank == 0 or rank % _FLASHINFER_LORA_RANK_GRANULARITY != 0):
      raise ValueError(f"FlashInfer requires rank > 0 and a multiple of "
                       f"{_FLASHINFER_LORA_RANK_GRANULARITY}.")
    if lora_layout is not None and lora_layout not in {"native", "row_major"}:
      raise ValueError("lora_layout must be 'native', 'row_major', or None.")
    resolved_lora_layout = lora_layout
    if resolved_lora_layout is None:
      resolved_lora_layout = "row_major" if resolved_runtime_kernel == "cute" else "native"
    if resolved_lora_layout == "row_major" and resolved_runtime_kernel != "cute":
      raise ValueError("lora_layout='row_major' is only supported with runtime_kernel='cute'; "
                       "it selects the CuTe producer's row-major BF16 D.")
    if precision == "nvfp4":
      self.group_size = 16
    elif precision == "int4":
      self.group_size = 64
    else:
      raise ValueError(f"Invalid precision: {precision}")
    self.in_features = in_features
    self.out_features = out_features
    self.rank = rank
    self.precision = precision
    self.torch_dtype = torch_dtype
    self.runtime_kernel = resolved_runtime_kernel
    self.lora_layout = resolved_lora_layout
    self._cute_pack_ready = False
    self._cute_pack_version = 1
    self._cute_qweight = None
    self._cute_wscales = None
    self._cute_proj_up = None
    self._cute_proj_down = None
    self._cute_smooth = None
    self._cute_bias = None
    self._cute_wcscales = None
    self._flashinfer_pack_ready = False
    self._fi_qweight = None
    self._fi_w_sf = None
    self._fi_proj_up = None
    self._fi_proj_down = None
    self._fi_bias = None
    self._fi_pre_quant_scale = None
    padded_in = ((in_features + 127) // 128) * 128
    padded_out = ((out_features + 127) // 128) * 128
    self._padded_in_features = padded_in
    self._padded_out_features = padded_out
    self._needs_k_pad = padded_in != in_features
    self._needs_n_strip = padded_out != out_features
    if in_features % 2 != 0:
      raise ValueError(f"in_features must be divisible by 2, got {in_features}.")
    if padded_in % self.group_size != 0:
      raise ValueError(f"Padded in_features must be divisible by {self.group_size}.")
    self.qweight = nn.Parameter(torch.empty(padded_out,
                                            padded_in // 2,
                                            dtype=torch.int8,
                                            device=device),
                                requires_grad=False)
    self.bias = nn.Parameter(torch.empty(padded_out, dtype=torch_dtype, device=device),
                             requires_grad=True) if bias else None
    self.wscales = nn.Parameter(torch.empty(
      padded_in // self.group_size,
      padded_out,
      dtype=torch_dtype if precision == "int4" else torch.float8_e4m3fn,
      device=device),
                                requires_grad=False)
    self.smooth_factor = nn.Parameter(torch.empty(padded_in, dtype=torch_dtype, device=device),
                                      requires_grad=False)
    self.smooth_factor_orig = nn.Parameter(torch.empty(padded_in, dtype=torch_dtype, device=device),
                                           requires_grad=False)
    self.proj_down = nn.Parameter(torch.empty(padded_in, rank, dtype=torch_dtype, device=device))
    self.proj_up = nn.Parameter(torch.empty(padded_out, rank, dtype=torch_dtype, device=device))
    if precision == "nvfp4":
      self.wcscales = nn.Parameter(torch.ones(padded_out, dtype=torch_dtype, device=device),
                                   requires_grad=False)
      self.wtscale = 1.0
    else:
      self.wtscale = None
      self.wcscales = None
    self.act_unsigned = act_unsigned

  @classmethod
  def from_linear(cls, linear: nn.Linear, **kwargs) -> "SVDQW4A4Linear":
    """Allocate an uninitialized W4A4 shell matching a float Linear."""
    in_features = kwargs.pop("in_features", linear.in_features)
    torch_dtype = kwargs.pop("torch_dtype", linear.weight.dtype)
    device = kwargs.pop("device", linear.weight.device)
    return cls(in_features=in_features,
               out_features=linear.out_features,
               bias=linear.bias is not None,
               torch_dtype=torch_dtype,
               device=device,
               **kwargs)

  def forward(self, x: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
    """Quantize activations, run W4A4 GEMM, and restore the input rank."""
    return self._forward_plain(x, output)

  def _forward_plain(self, x: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
    """Run forward without DTensor awareness."""
    if x.ndim < 2:
      raise ValueError(f"input must have shape [..., in_features], got {tuple(x.shape)}.")
    *leading_shape, channels = x.shape
    token_count = 1
    for extent in leading_shape:
      token_count *= extent
    x = x.reshape(token_count, channels)
    if self._needs_k_pad:
      x = F.pad(x, (0, self._padded_in_features - self.in_features))
    if self.runtime_kernel.startswith("flashinfer"):
      use_direct_output = False
      padded_output = self._flashinfer_gemm(x)
    else:
      quantized_x, ascales, lora_act_out = self.quantize(x)
      use_direct_output = output is not None and output.shape == (quantized_x.shape[0],
                                                                  self._padded_out_features)
      padded_output = self.forward_quant(quantized_x, ascales, lora_act_out,
                                         output if use_direct_output else None)
    logical_output = padded_output[:token_count]
    if self._needs_n_strip:
      logical_output = logical_output[:, :self.out_features]
    if output is not None and not use_direct_output:
      expected_shape = (*leading_shape, self.out_features)
      if output.shape != expected_shape:
        raise ValueError(f"output must have shape {expected_shape}, got {tuple(output.shape)}.")
      reshaped_output = output.reshape(token_count, self.out_features)
      reshaped_output.copy_(logical_output)
      logical_output = reshaped_output
    return logical_output.reshape(*leading_shape, self.out_features)

  def quantize(self,
               x: torch.Tensor,
               pad_size: int = 256,
               fuse_glu: bool = False,
               gate: torch.Tensor | None = None,
               gate_act: str = "silu",
               gate_first: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize activations and compute the low-rank down projection."""
    if self.runtime_kernel.startswith("flashinfer"):
      raise RuntimeError("FlashInfer owns the full quantization/GEMM chain; call forward instead.")
    if self.lora_layout == "row_major":
      if not self._cute_pack_ready:
        self._prepare_cute_pack()
      return svdq_quantize_w4a4(input=x,
                                lora_down=self._cute_proj_down if self.rank > 0 else None,
                                smooth=self._cute_smooth,
                                gate=gate,
                                fuse_glu=fuse_glu,
                                lora_layout="row_major",
                                fp4=True,
                                pad_size=pad_size,
                                gate_act=gate_act,
                                gate_first=gate_first)
    return svdq_quantize_w4a4(input=x,
                              lora_down=self.proj_down,
                              smooth=self.smooth_factor,
                              fuse_glu=fuse_glu,
                              gate=gate,
                              lora_layout="native",
                              fp4=self.precision == "nvfp4",
                              pad_size=pad_size,
                              gate_act=gate_act,
                              gate_first=gate_first)

  def _load_from_state_dict(self, *args, **kwargs) -> None:
    super()._load_from_state_dict(*args, **kwargs)
    self._cute_pack_ready = False
    self._flashinfer_pack_ready = False
    if self.runtime_kernel == "cute":
      self._prepare_cute_pack()
    elif self.runtime_kernel.startswith("flashinfer"):
      self._prepare_flashinfer_pack()

  def _apply(self, fn, *args, **kwargs):
    """Rebuild cached packs after device or dtype conversion."""
    rebuild_cute = self._cute_pack_ready
    rebuild_flashinfer = self._flashinfer_pack_ready
    self._cute_pack_ready = False
    self._flashinfer_pack_ready = False
    module = super()._apply(fn, *args, **kwargs)
    if rebuild_cute and self.runtime_kernel == "cute":
      self._prepare_cute_pack()
    elif rebuild_flashinfer and self.runtime_kernel.startswith("flashinfer"):
      self._prepare_flashinfer_pack()
    return module

  def _prepare_cute_pack(self) -> None:
    """Build non-serialized CuTe tensors from the native checkpoint pack."""
    if self._cute_pack_ready:
      return
    from .packing import convert_svdq_nvfp4_pack_to_cute
    (self._cute_qweight, self._cute_wscales, self._cute_proj_up, self._cute_proj_down,
     self._cute_bias, self._cute_wcscales,
     self._cute_smooth) = convert_svdq_nvfp4_pack_to_cute(self.qweight, self.wscales, self.proj_up,
                                                          self.proj_down, self.bias, self.wcscales,
                                                          self.smooth_factor)
    self._cute_pack_ready = True

  def fused_gemm_args(self) -> dict[str, object]:
    """Return packed GEMM arguments matching this layer's runtime and D layout."""
    params: dict[str, object] = {
      "runtime": self.runtime_kernel,
      "lora_layout": self.lora_layout,
      "fp4": self.precision == "nvfp4",
      "alpha": 1.0 if self.wtscale is None else self.wtscale,
      "bias": self.bias,
      "lora_down": self.proj_down if self.rank > 0 else None,
      "smooth": self.smooth_factor,
      "wgt": self.qweight,
      "wscales": self.wscales,
      "lora_up": self.proj_up,
      "wcscales": self.wcscales,
    }
    if self.runtime_kernel == "cute":
      self._prepare_cute_pack()
      params.update({
        "wgt": self._cute_qweight,
        "wscales": self._cute_wscales,
        "lora_up": self._cute_proj_up,
        "bias": self._cute_bias,
        "wcscales": self._cute_wcscales,
        "lora_down": self._cute_proj_down if self.rank > 0 else None,
        "smooth": self._cute_smooth,
      })
    elif self.runtime_kernel != "native":
      raise RuntimeError("Fused MLP passes require native or cute runtime.")
    return params

  def _prepare_flashinfer_pack(self, force: bool = False) -> None:
    """Derive non-serialized FlashInfer tensors from the native checkpoint pack."""
    if self._flashinfer_pack_ready and not force:
      return
    from .packing import convert_svdq_nvfp4_pack_to_flashinfer
    (self._fi_qweight, self._fi_w_sf, self._fi_proj_up, self._fi_proj_down, self._fi_bias,
     self._fi_pre_quant_scale) = convert_svdq_nvfp4_pack_to_flashinfer(
       self.qweight, self.wscales, self.proj_up, self.proj_down, self.bias, self.wcscales,
       self.smooth_factor)
    self._flashinfer_pack_ready = True

  @torch.compiler.disable
  def _flashinfer_gemm(self, x: torch.Tensor) -> torch.Tensor:
    """Run FlashInfer's fused SVDQuant chain."""
    from ...kernels.flashinfer import flashinfer_svdquant_linear
    if not self._flashinfer_pack_ready or self._fi_qweight.device != x.device:
      self._prepare_flashinfer_pack(force=True)
    return flashinfer_svdquant_linear(x,
                                      weight_fp4=self._fi_qweight,
                                      weight_sf=self._fi_w_sf,
                                      l2t_smoothed=self._fi_proj_down,
                                      l1_weight=self._fi_proj_up,
                                      pre_quant_scale=self._fi_pre_quant_scale,
                                      bias=self._fi_bias,
                                      runtime_kernel=self.runtime_kernel)

  def forward_quant(
    self,
    quantized_x: torch.Tensor,
    ascales: torch.Tensor,
    lora_act: torch.Tensor,
    output: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Run packed GEMM from pre-quantized activation tensors."""
    if self.runtime_kernel.startswith("flashinfer"):
      raise RuntimeError("FlashInfer owns the full chain; call forward instead.")
    if self.runtime_kernel == "cute":
      if not self._cute_pack_ready:
        raise RuntimeError("The CuTe weight pack is not built.")
      use_large_k = (self.lora_layout == "row_major" and self.bias is not None
                     and self.wcscales is not None and quantized_x.shape[0] >= 4096
                     and self._padded_out_features <= 4096 and self._padded_in_features >= 8192)
      result = svdq_gemm_w4a4(
        act=quantized_x,
        wgt=self.qweight if use_large_k else self._cute_qweight,
        ascales=ascales,
        wscales=self.wscales if use_large_k else self._cute_wscales,
        lora_act_in=lora_act if self.rank > 0 else None,
        lora_up=(self.proj_up if use_large_k else self._cute_proj_up) if self.rank > 0 else None,
        bias=self.bias if use_large_k else self._cute_bias,
        wcscales=self.wcscales if use_large_k else self._cute_wcscales,
        runtime="cute",
        lora_layout=self.lora_layout,
        fp4=True,
        alpha=self.wtscale)
      if output is not None:
        output.copy_(result)
        return output
      return result
    if self.lora_layout != "native":
      raise RuntimeError("Native runtime requires native D layout.")
    result = svdq_gemm_w4a4(act=quantized_x,
                            wgt=self.qweight,
                            ascales=ascales,
                            wscales=self.wscales,
                            bias=self.bias,
                            runtime="native",
                            lora_layout="native",
                            fp4=self.precision == "nvfp4",
                            alpha=self.wtscale,
                            wcscales=self.wcscales,
                            act_unsigned=self.act_unsigned,
                            output_dtype=output.dtype if output is not None else self.proj_up.dtype,
                            lora_act_in=lora_act if self.rank > 0 else None,
                            lora_up=self.proj_up if self.rank > 0 else None)
    if output is not None:
      output.copy_(result)
      return output
    return result

  def __repr__(self) -> str:
    extra = ""
    if self._needs_k_pad or self._needs_n_strip:
      extra = (f", padded_in={self._padded_in_features}"
               f", padded_out={self._padded_out_features}")
    return (f"SVDQW4A4Linear(in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, precision={self.precision}, act_unsigned={self.act_unsigned}, "
            f"runtime_kernel={self.runtime_kernel}, lora_layout={self.lora_layout}{extra})")


# Specialized low-bit storage implementation. W1/W2/W3 codes are expanded to E2M1 on chip; the
# actual computation still uses the standard W4A4 FP4 block-scaled MMA, not an FP1/FP2/FP3 MMA.
class SVDQWNA4Linear(nn.Module):
  """Specialized W1/W2/W3 storage runtime whose computation remains W4A4 FP4 block-scaled MMA."""

  def __init__(self,
               in_features: int,
               out_features: int,
               bits: int,
               rank: int = 32,
               bias: bool = True,
               torch_dtype: torch.dtype = torch.bfloat16,
               device: str | torch.device | None = None,
               stages: int | None = None) -> None:
    super().__init__()
    if bits not in (1, 2, 3):
      raise ValueError(f"bits must be 1, 2 or 3, got {bits}.")
    if stages is None:
      stages = 3 if bits in (1, 2) else 2
    if stages not in (2, 3) or (bits == 3 and stages != 2):
      raise ValueError(f"W{bits}A4 stages must be 2, and W1/W2 also accept 3, got {stages}.")
    allowed_ranks = _SVDQ_PACKED_A4_RANKS[f"w{bits}a4"]
    if rank not in allowed_ranks:
      raise ValueError(f"W{bits}A4 rank must be one of {allowed_ranks}, got {rank}.")
    if torch_dtype != torch.bfloat16:
      raise ValueError("Packed SVDQ formats currently require BF16.")
    self.in_features = in_features
    self.out_features = out_features
    self.bits = bits
    self.stages = stages
    self.rank = rank
    self.quantized_in_features = _pad(in_features, 128)
    self.quantized_out_features = _pad(out_features, 256)
    kp, np_ = self.quantized_in_features, self.quantized_out_features
    if bits == 1:
      shape = (np_, kp // 8)
    elif bits == 2:
      shape = (np_, kp // 4)
    else:
      shape = (np_, kp // 4)
    self.register_buffer("qweight", torch.zeros(shape, dtype=torch.uint8, device=device))
    if bits == 3:
      self.register_buffer("qweight_hi",
                           torch.zeros((np_, kp // 8), dtype=torch.uint8, device=device))
    else:
      self.register_buffer("qweight_hi", None)
    if stages == 3:
      self.register_buffer("wmicroscales",
                           torch.zeros((np_, kp // 16), dtype=torch.uint8, device=device))
    else:
      self.register_buffer("wmicroscales", None)
    # The flat 128x4 SF buffer pads each 4-group atom to 512 B per 128 rows.
    groups = kp // 16
    sf_bytes = (np_ // 128) * ((groups + 3) // 4) * 512
    self.register_buffer("wscales", torch.zeros(sf_bytes, dtype=torch.uint8, device=device))
    self.register_buffer("wcscales", torch.ones(np_, dtype=torch.bfloat16, device=device))
    self.register_buffer("smooth_factor", torch.ones(kp, dtype=torch.bfloat16, device=device))
    self.register_buffer("proj_down", torch.zeros((kp, rank), dtype=torch.bfloat16, device=device))
    self.register_buffer("proj_up", torch.zeros((np_, rank), dtype=torch.bfloat16, device=device))
    self.bias = nn.Parameter(torch.zeros(np_, dtype=torch.bfloat16, device=device),
                             requires_grad=False) if bias else None

  @classmethod
  def from_linear(cls, linear: nn.Linear, rank: int = 32, **kwargs) -> "SVDQWNA4Linear":
    bits = kwargs.pop("bits")
    return _quantize_linear(linear, bits, rank=rank, **kwargs)

  def dequantize_weight(self) -> torch.Tensor:
    """Diagnostic FP32 reconstruction of the logical weight."""
    n, k = self.out_features, self.in_features
    kp = self.quantized_in_features
    # Undo the 128x4 SF swizzle with the shared helper.
    from .packing import unswizzle_fp4_scales
    flat = self.wscales.view(torch.uint8).reshape(-1)
    scales = unswizzle_fp4_scales(flat, self.quantized_out_features,
                                  kp // 16).float().view(self.quantized_out_features, -1)
    high = self.qweight_hi
    if self.stages == 3:
      values = dequantize_packed_3stage(self.qweight, scales, self.bits, self.wmicroscales)
    else:
      values = dequantize_packed(self.qweight, scales, self.bits, high)
    values = values * self.wcscales.float().unsqueeze(1)
    values = values[:n, :k] / self.smooth_factor[:k].float().unsqueeze(0)
    if self.rank:
      values = values + self.proj_up[:n].float() @ self.proj_down[:k].float().t()
    return values

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    rows = x.numel() // shape[-1]
    if rows == 0:
      return x.new_empty((*shape[:-1], self.out_features))
    flat = x.reshape(rows, shape[-1])
    if flat.shape[1] != self.quantized_in_features:
      flat = torch.nn.functional.pad(flat, (0, self.quantized_in_features - flat.shape[1]))
    flat = flat.contiguous()
    if not flat.is_cuda:
      raise ValueError("Packed W1/W2/W3 forward requires CUDA input on sm120+; "
                       "quantize_device='cpu' only controls offline quantization.")
    # Expansion only depends on the persistent weight, not the activation. Run it on a side stream
    # so its HBM traffic overlaps the activation producer. This keeps a single transient workspace
    # while hiding its latency instead of adding it to the critical path.
    current = torch.cuda.current_stream(flat.device)
    expand_stream = _packed_expand_stream(flat.device)
    expand_stream.wait_stream(current)
    with torch.cuda.stream(expand_stream):
      expanded = expand_wna4(self.qweight,
                             self.qweight_hi,
                             self.bits,
                             wgt_micro=self.wmicroscales if self.stages == 3 else None)
    qact, ascales, lora_act = svdq_quantize_w4a4(input=flat,
                                                 lora_down=self.proj_down if self.rank else None,
                                                 smooth=self.smooth_factor,
                                                 lora_layout="row_major",
                                                 fp4=True,
                                                 pad_size=256)
    current.wait_stream(expand_stream)
    expanded.record_stream(current)
    out = svdq_gemm_w4a4(qact,
                         expanded,
                         ascales,
                         self.wscales.view(torch.float8_e4m3fn),
                         lora_act_in=lora_act if self.rank else None,
                         lora_up=self.proj_up if self.rank else None,
                         bias=self.bias,
                         wcscales=self.wcscales,
                         lora_layout="row_major",
                         runtime="cute")
    return out[:rows, :self.out_features].reshape(*shape[:-1], self.out_features)

  def _load_from_state_dict(self, *args, **kwargs) -> None:
    super()._load_from_state_dict(*args, **kwargs)
    if self.stages == 3:
      _validate_inner_scales(self.wmicroscales, self.bits)


def _quantize_linear(linear: nn.Linear,
                     bits: int,
                     rank: int = 32,
                     smooth_factor: torch.Tensor | None = None,
                     alpha: float = 0.5,
                     torch_dtype: torch.dtype = torch.bfloat16,
                     device: str | torch.device | None = None,
                     calibrate_precision: str = "low",
                     stages: int | None = None) -> SVDQWNA4Linear:
  from .quantizer import _compute_weight_only_smooth_scale
  if stages is None:
    stages = 3 if bits in (1, 2) else 2
  device = torch.device(device or linear.weight.device)
  module_type = {1: SVDQW1A4Linear, 2: SVDQW2A4Linear, 3: SVDQW3A4Linear}[bits]
  module = module_type(linear.in_features,
                       linear.out_features,
                       rank=rank,
                       bias=linear.bias is not None,
                       torch_dtype=torch_dtype,
                       device=device,
                       stages=stages)
  weight = linear.weight.detach().to(device=device, dtype=torch.float32)
  if smooth_factor is None:
    smooth = _compute_weight_only_smooth_scale(weight.abs().amax(dim=0),
                                               alpha=alpha,
                                               math_dtype=torch.float32,
                                               output_dtype=torch.float32)
  else:
    smooth = smooth_factor.to(device=device, dtype=torch.float32)
  down, up, residual = decompose_lowrank_residual(weight * smooth.unsqueeze(0),
                                                  rank,
                                                  output_dtype=torch.float32,
                                                  svd_precision=calibrate_precision)
  n, k = weight.shape
  kp, np_ = module.quantized_in_features, module.quantized_out_features
  # Pad before the 128x4 scale swizzle. The SF offset depends on the padded K group count, so a
  # logical-shape swizzle followed by a buffer assignment is incorrect for ragged N/K.
  residual = torch.nn.functional.pad(residual, (0, kp - k, 0, np_ - n))
  if stages == 3:
    packed, scales, coarse, micro = quantize_residual_3stage(residual, bits)
    module.qweight.copy_(packed)
    module.wmicroscales.copy_(micro)
  else:
    packed, scales, coarse = quantize_residual(residual, bits)
    if bits == 3:
      low, high = packed
      module.qweight.copy_(low)
      module.qweight_hi.copy_(high)
    else:
      module.qweight.copy_(packed)
  module.wscales.copy_(scales.view(torch.uint8))
  module.wcscales.copy_(coarse)
  module.smooth_factor[:k] = smooth.to(torch_dtype)
  if rank:
    # The W4A4 producer consumes D @ proj_down with [K,R]; `decompose_lowrank_residual` returns
    # the transpose [R,K].
    module.proj_down[:k] = (down.double() / smooth.double().unsqueeze(0)).t().to(torch_dtype)
    module.proj_up[:n] = up.to(torch_dtype)
  if module.bias is not None:
    module.bias.zero_()
    module.bias[:n] = linear.bias.detach().to(device=device, dtype=torch_dtype)
  return module


class SVDQW1A4Linear(SVDQWNA4Linear):
  """Experimental one-bit sign-only SVDQ format."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, bits=1, **kwargs)


class SVDQW2A4Linear(SVDQWNA4Linear):
  """Two-bit SVDQ format for minimum persistent weight memory."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, bits=2, **kwargs)


class SVDQW3A4Linear(SVDQWNA4Linear):
  """Three-bit SVDQ format with a magnitude and sign plane."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, bits=3, **kwargs)


def _quantize_svdq_wNa4(
  linear: nn.Linear,
  bits: int,
  representative_activations=None,
  *,
  rank: int = 32,
  alpha: float = 0.5,
  smooth_strategy: str = "identity",
  torch_dtype: torch.dtype | None = None,
  device: str | torch.device | None = None,
  calibrate_precision: str = "low",
  stages: int | None = None,
) -> SVDQWNA4Linear:
  """Quantize one Linear into packed W1/W2/W3, with DQ or activation-calibrated PTQ.

  :param stages: Weight scale stages. ``None`` selects the production default:
    three stages for W1/W2 and the standard two-stage NVFP4 codec for W3. Pass
    ``2`` explicitly to reproduce the legacy packed path.
  """
  from .quantizer import _compute_weight_only_smooth_scale
  from .quantizer import compute_smooth_scale
  if not isinstance(linear, nn.Linear):
    raise TypeError(f"Expected nn.Linear, got {type(linear)}.")
  if type(linear.weight).__name__ == "DTensor":
    raise NotImplementedError("Packed W1/W2/W3 quantization does not support DTensor/TP weights.")
  if linear.weight.dtype != torch.bfloat16:
    raise ValueError(f"Packed W1/W2/W3 requires BF16 source weights, got {linear.weight.dtype}.")
  device = torch.device(device or linear.weight.device)
  torch_dtype = torch.bfloat16 if torch_dtype is None else torch_dtype
  k = linear.in_features
  if smooth_strategy == "activation":
    if representative_activations is None:
      raise ValueError("smooth_strategy='activation' requires representative activations.")
    acts = representative_activations
    if isinstance(acts, torch.Tensor):
      acts = [acts]
    span = None
    for tensor in acts:
      flat = tensor.detach().reshape(-1, tensor.shape[-1]).to(device=device, dtype=torch.float32)
      current = flat.abs().amax(dim=0)
      span = current if span is None else torch.maximum(span, current)
    weight_span = linear.weight.detach().to(device=device, dtype=torch.float32).abs().amax(dim=0)
    smooth = compute_smooth_scale(span,
                                  weight_span,
                                  alpha=alpha,
                                  math_dtype=torch.float32,
                                  output_dtype=torch.float32)
  elif smooth_strategy == "identity":
    smooth = torch.ones(k, dtype=torch.float32, device=device)
  elif smooth_strategy in ("weight", "weight_inv"):
    weight_span = linear.weight.detach().to(device=device, dtype=torch.float32).abs().amax(dim=0)
    smooth = _compute_weight_only_smooth_scale(weight_span,
                                               alpha=alpha,
                                               inverse=smooth_strategy == "weight_inv",
                                               math_dtype=torch.float32,
                                               output_dtype=torch.float32)
  else:
    raise ValueError(f"Unsupported packed SVDQ smooth strategy: {smooth_strategy!r}.")
  return _quantize_linear(linear,
                          bits,
                          rank=rank,
                          smooth_factor=smooth,
                          torch_dtype=torch_dtype,
                          device=device,
                          calibrate_precision=calibrate_precision,
                          stages=stages)


def _quantize_svdq_w1a4(linear, representative_activations=None, **kwargs):
  return _quantize_svdq_wNa4(linear, 1, representative_activations, **kwargs)


def _quantize_svdq_w2a4(linear, representative_activations=None, **kwargs):
  return _quantize_svdq_wNa4(linear, 2, representative_activations, **kwargs)


def _quantize_svdq_w3a4(linear, representative_activations=None, **kwargs):
  return _quantize_svdq_wNa4(linear, 3, representative_activations, **kwargs)


def _quantize_svdq_w4a4(module: nn.Module, quantize_config: QuantizeConfig) -> nn.Module:
  """Run the W4A4 module flow, selecting DQ or PTQ from ``quantize_config``.

  :param module: Root module containing the float `nn.Linear` layers to quantize.
  :param quantize_config: SVDQ `QuantizeConfig` describing mode, scope and serialization.
  :returns: The module with eligible layers replaced by `SVDQW4A4Linear` instances.
  """
  from ...utils import check_quantized
  from ..observers import QuantizeObserver
  from .ptq import SVDQFewShotRuntimeController
  from .ptq import SVDQPTQCalibrator
  from .ptq import SVDQPTQContext
  from .ptq import _attach_quantization_metadata
  from .ptq import _build_quant_config_snapshot
  from .ptq import _call_with_supported_kwargs
  from .ptq import _log_quantize_summary
  from .ptq import _maybe_collect_svdq_garbage
  from .ptq import _quantize_context_layers
  from .ptq import _resolve_svdq_quantize_device
  from .ptq import _save_quant_config_snapshot
  from .ptq import _save_quantized_module

  if not isinstance(module, nn.Module):
    raise TypeError(f"Expected nn.Module, got {type(module)}.")
  if not quantize_config.is_svdq():
    raise ValueError(f"_quantize_svdq_w4a4 only supports SVDQ types, got "
                     f"{quantize_config.quant_type}.")
  is_dq = quantize_config.is_svdq_dq()
  if check_quantized(module):
    logger.warning("Module %s is already quantized, skipping SVDQ W4A4 quantization.",
                   module.__class__.__name__)
    return module
  if is_dq and getattr(module, "_svdq_pending_quantization", False):
    logger.warning(
      "Module %s already has pending SVDQuant few-shot quantization, skipping re-arming.",
      module.__class__.__name__)
    return module

  context = SVDQPTQContext.from_config(module, quantize_config)
  quantize_device = _resolve_svdq_quantize_device(module, context.svdq_kwargs)
  if is_dq and quantize_config.is_svdq_dq_few_shot():
    controller = SVDQFewShotRuntimeController(
      context=context,
      quantize_device=quantize_device,
      serialize_to=quantize_config.serialize_to,
    )
    return controller.arm()

  observer = QuantizeObserver(getattr(quantize_config, "observer", None))
  observer_refs = observer.capture_references(module)
  was_training = module.training
  calibrator = None
  activation_spans = None
  observed_layer_names = None

  if not is_dq:
    calibrator = SVDQPTQCalibrator(context)
    calibrator.register(observation_device=quantize_device)
    module.eval()
    try:
      with torch.inference_mode():
        _call_with_supported_kwargs(
          quantize_config.calibrate_fn,
          model=module,
          quantize_config=quantize_config,
          calibrator=calibrator,
          component_name=getattr(module, "_actual_module_name", module.__class__.__name__),
          svdq_kwargs=context.svdq_kwargs,
        )
    finally:
      calibrator.remove()
      module.train(was_training)
    activation_spans = calibrator.activation_spans
    observed_layer_names = sorted(activation_spans)

  if is_dq:
    module.eval()
  try:
    module, quantized_layer_names = _quantize_context_layers(
      module,
      context,
      quantize_device=quantize_device,
      activation_spans=activation_spans,
    )
  finally:
    if is_dq:
      module.train(was_training)
  if observer_refs:
    observer.attach(module, observer_refs)

  if not quantized_layer_names:
    if is_dq:
      raise RuntimeError("SVDQ dynamic quantization found no layers to quantize. "
                         f"Skipped layers: {context.skipped_reasons}.")
    raise RuntimeError("SVDQ PTQ completed calibration but no layers were quantized. "
                       f"Skipped layers: {context.skipped_reasons}.")

  if context.svdq_kwargs.get("fused_mlp", False):
    from .passes import apply_passes
    apply_passes(module)

  _attach_quantization_metadata(
    module,
    quant_type=quantize_config.quant_type,
    quantized_layer_names=quantized_layer_names,
    exclude_layers=context.exclude_layers,
    svdq_kwargs=context.svdq_kwargs,
    checkpoint_path=quantize_config.serialize_to,
    quantize_config=quantize_config,
  )
  if not is_dq or quantize_config.serialize_to is not None:
    writer_rank = _save_quantized_module(
      module,
      serialize_to=quantize_config.serialize_to,
      quant_type=quantize_config.quant_type,
      rank=context.rank,
      precision=context.precision,
      quantized_layer_names=quantized_layer_names,
      svdq_kwargs=context.svdq_kwargs,
    )
    if writer_rank:
      _save_quant_config_snapshot(
        _build_quant_config_snapshot(
          checkpoint_path=quantize_config.serialize_to,
          quant_type=quantize_config.quant_type,
          rank=context.rank,
          exclude_layers=context.exclude_layers,
          regional_quantize=context.regional_quantize,
          repeated_blocks=context.repeated_blocks,
          verbose=context.verbose,
          svdq_kwargs=context.svdq_kwargs,
          backend=quantize_config.backend,
        ),
        checkpoint_path=quantize_config.serialize_to,
      )
  module.train(was_training)
  _log_quantize_summary(
    context,
    quantized_layer_names=quantized_layer_names,
    serialize_to=quantize_config.serialize_to,
    observed_layer_names=observed_layer_names,
  )
  if calibrator is not None:
    calibrator.activation_spans.clear()
  _maybe_collect_svdq_garbage(quantize_device)
  return module


def _get_svdq_wNa4_quantizer(precision: str):
  """Return the module-level quantizer for any A4 manifest precision or alias."""
  from .wNaN import _quantize_svdq_wNaN

  quantizers = {
    # INT4/NVFP4 are encoding aliases; W4A4 is the weight/activation semantic name.
    "int4":
    _quantize_svdq_w4a4,
    "nvfp4":
    _quantize_svdq_w4a4,
    "w4a4":
    _quantize_svdq_w4a4,
    "w1a4":
    functools.partial(_quantize_svdq_wNaN, quantize_linear=_quantize_svdq_w1a4, precision="w1a4"),
    "w2a4":
    functools.partial(_quantize_svdq_wNaN, quantize_linear=_quantize_svdq_w2a4, precision="w2a4"),
    "w3a4":
    functools.partial(_quantize_svdq_wNaN, quantize_linear=_quantize_svdq_w3a4, precision="w3a4"),
  }
  try:
    return quantizers[precision]
  except KeyError as exc:
    raise ValueError(f"Unsupported SVDQ A4 precision: {precision!r}.") from exc
