"""Cache-DiT native NVFP4 quantization and block-scaled GEMM operators.

Numerical contract
==================
For a BF16 tensor ``x[R,K]``, :func:`fp4_quantize` uses two levels of scaling.
The first level is one FP32 scale per row::

  global[r] = max_k(abs(x[r,k])) / 2688
  xn[r,k] = x[r,k] / global[r]

Here ``2688 = 448 * 6``: 448 is the maximum finite UE4M3 scale and 6 is the
maximum finite E2M1 value. Every consecutive group of 16 normalized values then
gets its own UE4M3 block scale::

  block[r,g] = UE4M3(max_{k in g}(abs(xn[r,k])) / 6)
  q[r,k] = E2M1_RNE(xn[r,k] / float(block[r,g]))

The reciprocal uses the already rounded UE4M3 value. A zero row or group yields
zero scale and zero qdata. ``level1=False`` skips the row scale and is intended
only for single-level operator comparisons, not for the production Linear.

Returned tensors
================
``qdata`` has shape ``[R,K/2]`` and dtype ``float4_e2m1fn_x2``. Each byte stores
two adjacent FP4 values: even K in the low nibble and odd K in the high nibble.
``block_scales`` is a canonical CUTLASS buffer with 512 bytes per 128 rows by
four K16 groups, including zero padding for partial row blocks. ``global_scales``
is FP32 ``[R]`` when level 1 is enabled and an empty tensor otherwise.

Typical two-level Linear call
=============================
>>> qa, sa, ga = fp4_quantize(activation)                 # [M,K] BF16
>>> qw, sw, gw = fp4_quantize(weight)                     # [N,K] BF16
>>> output = fp4_gemm(qa, qw, sa, sw, ga, gw, bias=bias) # [M,N] BF16

The block-scaled collective evaluates the K16 products in FP32. Its epilogue
then computes ``bias[n] + gw[n] * (ga[m] * accumulator[m,n])`` in FP32 and
converts once to BF16. ``use_native=False`` instead calls ``torch._scaled_mm``
and applies the two global scales as separate tensor operations; it is retained
as a reference implementation. On SM120, native GEMM pads logical K to 256 and
N to 128 when necessary, then slices the output back to the original N. The
scaled_mm path similarly pads N to 16. Packed qdata, canonical scales, global
scales and bias are padded with zero, so valid output elements are unchanged.
The quantization entry itself accepts any ``K%64==0``.

Both public functions are ordinary ``torch.library`` operators with CUDA and
fake implementations, so eager, ``torch.compile(fullgraph=True)`` and CUDA Graph
use the same shape/dtype contract. Non-contiguous GEMM operands are made
contiguous in the CUDA implementation; the fake implementation performs only
metadata validation and allocation.
"""

import torch
import torch.nn.functional as F

from .cuda_ext import _get_required_ops_module

_ACTIVATIONS = {"gelu": 1, "gelu_tanh": 2}


def _fp4_ops_module():
  """Return the compiled ops module after checking the FP4 entry points.

  :returns: The pybind ``ops`` submodule.
  """

  ops = _get_required_ops_module()
  missing = [
    name for name in ("fp4_quantize", "fp4_quantize_swiglu", "fp4_gemm", "fp4_gemm_activation")
    if not hasattr(ops, name)
  ]
  if missing:
    raise RuntimeError(
      f"The loaded cache_dit._C_ops extension is missing {missing}. Rebuild it with "
      "`CACHE_DIT_BUILD_CUDA=1 CACHE_DIT_CUDA_ARCH_LIST=120a bash tools/build_fast.sh "
      "install -e . --no-build-isolation --no-deps`.")
  return ops


def _scale_shape(rows: int, cols: int) -> tuple[int, int]:
  """Return the canonical swizzled block-scale shape.

  :param rows: Logical row count before 128-row padding.
  :param cols: Logical reduction width.
  :returns: Two-dimensional blocked scale shape.
  """

  return ((rows + 127) // 128 * 32, (cols + 63) // 64 * 16)


def _validate_quantize_metadata(x: torch.Tensor) -> None:
  """Validate metadata shared by eager and fake quantization.

  :param x: BF16 tensor to quantize.
  """

  if x.dtype != torch.bfloat16:
    raise ValueError(f"cache-dit FP4 quantization expects bfloat16, got {x.dtype}.")
  if x.dim() != 2:
    raise ValueError(f"cache-dit FP4 quantization expects a 2D tensor, got {tuple(x.shape)}.")
  if x.shape[0] == 0 or x.shape[1] == 0:
    raise ValueError("cache-dit FP4 quantization does not accept empty tensors.")
  if x.shape[1] % 64 != 0:
    raise ValueError("cache-dit FP4 quantization requires K % 64 == 0, "
                     f"got {x.shape[1]}.")


def _call_fp4_quantize(x: torch.Tensor, qdata: torch.Tensor, block_scales: torch.Tensor,
                       global_scales: torch.Tensor, level1: bool) -> None:
  """Bridge functional quantization to the out-param extension.

  :param x: Contiguous BF16 input.
  :param qdata: Packed FP4 destination.
  :param block_scales: Canonical UE4M3 block-scale destination.
  :param global_scales: Optional per-row FP32 global-scale destination.
  :param level1: Whether per-row global scales are produced.
  """

  _fp4_ops_module().fp4_quantize(x, qdata, block_scales, global_scales, level1)


torch.library.define(
  "cache_dit_cuda_ops::fp4_quantize",
  "(Tensor x, bool level1) -> (Tensor, Tensor, Tensor)",
)


@torch.library.impl("cache_dit_cuda_ops::fp4_quantize", "CUDA")
def _fp4_quantize(x: torch.Tensor, level1: bool):
  _validate_quantize_metadata(x)
  x = x.contiguous()
  rows, cols = x.shape
  qdata = torch.empty((rows, cols // 2), dtype=torch.float4_e2m1fn_x2, device=x.device)
  block_scales = torch.empty(_scale_shape(rows, cols), dtype=torch.float8_e4m3fn, device=x.device)
  global_scales = torch.empty((rows if level1 else 0, ), dtype=torch.float32, device=x.device)
  _call_fp4_quantize(x, qdata, block_scales, global_scales, level1)
  return qdata, block_scales, global_scales


@torch.library.register_fake("cache_dit_cuda_ops::fp4_quantize")
def _fake_fp4_quantize(x: torch.Tensor, level1: bool):
  _validate_quantize_metadata(x)
  rows, cols = x.shape
  return (
    x.new_empty((rows, cols // 2), dtype=torch.float4_e2m1fn_x2),
    x.new_empty(_scale_shape(rows, cols), dtype=torch.float8_e4m3fn),
    x.new_empty((rows if level1 else 0, ), dtype=torch.float32),
  )


def _validate_swiglu_metadata(value: torch.Tensor, gate: torch.Tensor | None) -> tuple[int, int]:
  """Validate gated producer metadata and return its logical output shape."""

  if value.dtype != torch.bfloat16 or value.dim() != 2:
    raise ValueError("FP4 SwiGLU quantization expects a 2D bfloat16 value tensor.")
  rows = value.shape[0]
  if rows == 0:
    raise ValueError("FP4 SwiGLU quantization does not accept empty tensors.")
  if gate is None:
    if value.shape[1] % 2 != 0:
      raise ValueError("Half-split FP4 SwiGLU expects value with an even 2H width.")
    cols = value.shape[1] // 2
  else:
    if gate.dtype != torch.bfloat16 or gate.dim() != 2:
      raise ValueError("FP4 SwiGLU gate must be a 2D bfloat16 tensor.")
    if gate.shape[0] != rows:
      raise ValueError("FP4 SwiGLU value and gate row counts must match.")
    cols = gate.shape[1]
    if value.shape[1] != cols:
      raise ValueError("A separate FP4 SwiGLU gate expects value and gate shapes [R,H].")
  if cols == 0 or cols % 64 != 0:
    raise ValueError(f"FP4 SwiGLU quantization requires a nonempty H divisible by 64, got {cols}.")
  return rows, cols


def _call_fp4_quantize_swiglu(value: torch.Tensor, gate: torch.Tensor | None, qdata: torch.Tensor,
                              block_scales: torch.Tensor, global_scales: torch.Tensor,
                              level1: bool) -> None:
  """Bridge functional SwiGLU quantization to the out-param extension."""

  _fp4_ops_module().fp4_quantize_swiglu(value, gate, qdata, block_scales, global_scales, level1)


torch.library.define(
  "cache_dit_cuda_ops::fp4_quantize_swiglu",
  "(Tensor value, Tensor? gate, bool level1) -> (Tensor, Tensor, Tensor)",
)


@torch.library.impl("cache_dit_cuda_ops::fp4_quantize_swiglu", "CUDA")
def _fp4_quantize_swiglu(value: torch.Tensor, gate: torch.Tensor | None, level1: bool):
  rows, cols = _validate_swiglu_metadata(value, gate)
  value = value.contiguous()
  gate = gate.contiguous() if gate is not None else None
  qdata = torch.empty((rows, cols // 2), dtype=torch.float4_e2m1fn_x2, device=value.device)
  block_scales = torch.empty(_scale_shape(rows, cols),
                             dtype=torch.float8_e4m3fn,
                             device=value.device)
  global_scales = torch.empty((rows if level1 else 0, ), dtype=torch.float32, device=value.device)
  _call_fp4_quantize_swiglu(value, gate, qdata, block_scales, global_scales, level1)
  return qdata, block_scales, global_scales


@torch.library.register_fake("cache_dit_cuda_ops::fp4_quantize_swiglu")
def _fake_fp4_quantize_swiglu(value: torch.Tensor, gate: torch.Tensor | None, level1: bool):
  rows, cols = _validate_swiglu_metadata(value, gate)
  return (
    value.new_empty((rows, cols // 2), dtype=torch.float4_e2m1fn_x2),
    value.new_empty(_scale_shape(rows, cols), dtype=torch.float8_e4m3fn),
    value.new_empty((rows if level1 else 0, ), dtype=torch.float32),
  )


def _validate_gemm_metadata(a: torch.Tensor, b: torch.Tensor, a_scales: torch.Tensor,
                            b_scales: torch.Tensor, a_global_scales: torch.Tensor,
                            b_global_scales: torch.Tensor, bias: torch.Tensor | None) -> None:
  """Validate metadata shared by eager and fake GEMM.

  :param a: Packed activation ``[M, K / 2]``.
  :param b: Packed weight ``[N, K / 2]``.
  :param a_scales: Activation block scales.
  :param b_scales: Weight block scales.
  :param a_global_scales: Empty or per-row activation scales.
  :param b_global_scales: Empty or per-output-channel weight scales.
  :param bias: Optional BF16 bias ``[N]``.
  """

  if a.dtype != torch.float4_e2m1fn_x2 or b.dtype != torch.float4_e2m1fn_x2:
    raise ValueError("cache-dit FP4 GEMM expects float4_e2m1fn_x2 A and B.")
  if a.dim() != 2 or b.dim() != 2:
    raise ValueError("cache-dit FP4 GEMM expects 2D A and B.")
  if a.shape[1] != b.shape[1]:
    raise ValueError(f"FP4 GEMM packed inner dimensions differ: {a.shape[1]} and {b.shape[1]}.")
  rows, cols = a.shape[0], a.shape[1] * 2
  out_channels = b.shape[0]
  if cols % 256 != 0 or out_channels % 128 != 0:
    raise ValueError("FP4 GEMM CUTLASS collective requires logical K % 256 == 0 and N % 128 == 0.")
  if a_scales.dtype != torch.float8_e4m3fn or b_scales.dtype != torch.float8_e4m3fn:
    raise ValueError("FP4 GEMM block scales must be float8_e4m3fn.")
  expected_a_scales = (rows + 127) // 128 * (cols // 64) * 512
  expected_b_scales = (out_channels + 127) // 128 * (cols // 64) * 512
  if a_scales.numel() != expected_a_scales or b_scales.numel() != expected_b_scales:
    raise ValueError("FP4 GEMM block scale sizes do not match the operand shapes.")
  if a_global_scales.dtype != torch.float32 or b_global_scales.dtype != torch.float32:
    raise ValueError("FP4 GEMM global scales must be float32.")
  if a_global_scales.numel() not in (0, rows) or b_global_scales.numel() not in (0, out_channels):
    raise ValueError("FP4 GEMM global scales must be empty or match the operand rows.")
  if bias is not None and (bias.dtype != torch.bfloat16 or bias.shape != (out_channels, )):
    raise ValueError("FP4 GEMM bias must be bfloat16 with shape [N].")


def _call_fp4_gemm(a: torch.Tensor, b: torch.Tensor, a_scales: torch.Tensor, b_scales: torch.Tensor,
                   a_global_scales: torch.Tensor, b_global_scales: torch.Tensor,
                   bias: torch.Tensor | None, out: torch.Tensor) -> None:
  """Bridge functional GEMM to the out-param extension.

  :param a: Packed activation.
  :param b: Packed weight.
  :param a_scales: Activation block scales.
  :param b_scales: Weight block scales.
  :param a_global_scales: Activation global scales.
  :param b_global_scales: Weight global scales.
  :param bias: Optional BF16 bias.
  :param out: BF16 destination.
  """

  _fp4_ops_module().fp4_gemm(a, b, a_scales, b_scales, a_global_scales, b_global_scales, bias, out)


torch.library.define(
  "cache_dit_cuda_ops::fp4_gemm",
  "(Tensor a, Tensor b, Tensor a_scales, Tensor b_scales, Tensor a_global_scales, "
  "Tensor b_global_scales, Tensor? bias) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::fp4_gemm", "CUDA")
def _fp4_gemm(a: torch.Tensor, b: torch.Tensor, a_scales: torch.Tensor, b_scales: torch.Tensor,
              a_global_scales: torch.Tensor, b_global_scales: torch.Tensor,
              bias: torch.Tensor | None) -> torch.Tensor:
  _validate_gemm_metadata(a, b, a_scales, b_scales, a_global_scales, b_global_scales, bias)
  out = torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device=a.device)
  _call_fp4_gemm(a.contiguous(), b.contiguous(), a_scales.contiguous(), b_scales.contiguous(),
                 a_global_scales.contiguous(), b_global_scales.contiguous(), bias, out)
  return out


@torch.library.register_fake("cache_dit_cuda_ops::fp4_gemm")
def _fake_fp4_gemm(a: torch.Tensor, b: torch.Tensor, a_scales: torch.Tensor, b_scales: torch.Tensor,
                   a_global_scales: torch.Tensor, b_global_scales: torch.Tensor,
                   bias: torch.Tensor | None) -> torch.Tensor:
  _validate_gemm_metadata(a, b, a_scales, b_scales, a_global_scales, b_global_scales, bias)
  return a.new_empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16)


def _call_fp4_gemm_activation(a: torch.Tensor, b: torch.Tensor, a_scales: torch.Tensor,
                              b_scales: torch.Tensor, a_global_scales: torch.Tensor,
                              b_global_scales: torch.Tensor, bias: torch.Tensor | None,
                              out: torch.Tensor, activation: int) -> None:
  """Bridge functional activation GEMM to the out-param extension."""

  _fp4_ops_module().fp4_gemm_activation(a, b, a_scales, b_scales, a_global_scales, b_global_scales,
                                        bias, out, activation)


torch.library.define(
  "cache_dit_cuda_ops::fp4_gemm_activation",
  "(Tensor a, Tensor b, Tensor a_scales, Tensor b_scales, Tensor a_global_scales, "
  "Tensor b_global_scales, Tensor? bias, int activation) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::fp4_gemm_activation", "CUDA")
def _fp4_gemm_activation(a: torch.Tensor, b: torch.Tensor, a_scales: torch.Tensor,
                         b_scales: torch.Tensor, a_global_scales: torch.Tensor,
                         b_global_scales: torch.Tensor, bias: torch.Tensor | None,
                         activation: int) -> torch.Tensor:
  _validate_gemm_metadata(a, b, a_scales, b_scales, a_global_scales, b_global_scales, bias)
  if activation not in _ACTIVATIONS.values():
    raise ValueError(f"Unsupported FP4 activation id: {activation}.")
  out = torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device=a.device)
  _call_fp4_gemm_activation(a.contiguous(), b.contiguous(), a_scales.contiguous(),
                            b_scales.contiguous(), a_global_scales.contiguous(),
                            b_global_scales.contiguous(), bias, out, activation)
  return out


@torch.library.register_fake("cache_dit_cuda_ops::fp4_gemm_activation")
def _fake_fp4_gemm_activation(a: torch.Tensor, b: torch.Tensor, a_scales: torch.Tensor,
                              b_scales: torch.Tensor, a_global_scales: torch.Tensor,
                              b_global_scales: torch.Tensor, bias: torch.Tensor | None,
                              activation: int) -> torch.Tensor:
  _validate_gemm_metadata(a, b, a_scales, b_scales, a_global_scales, b_global_scales, bias)
  if activation not in _ACTIVATIONS.values():
    raise ValueError(f"Unsupported FP4 activation id: {activation}.")
  return a.new_empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16)


def fp4_quantize(x: torch.Tensor,
                 level1: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Quantize a row-major BF16 tensor to packed NVFP4.

  With ``level1=True``, computes one FP32 scale per row, normalizes the row,
  then computes one UE4M3 scale per 16 normalized values. E2M1 conversion uses
  round-to-nearest-even and saturation. With ``level1=False``, only the per-16
  block scale is used and the global-scale output is empty.

  :param x: BF16 tensor ``[R, K]`` with ``K % 64 == 0``. Non-contiguous input is
    copied into a contiguous buffer by the CUDA implementation.
  :param level1: Produce per-row FP32 global scales in addition to per-16 UE4M3
    block scales. Disable only for isolated single-level GEMM comparisons.
  :returns: ``(qdata, block_scales, global_scales)``. Qdata has shape
    ``[R, K/2]`` and dtype ``float4_e2m1fn_x2``; adjacent even/odd K values are
    packed into the low/high nibbles. Block scales use the canonical CUTLASS
    128-row by four-K16-group layout with zero padding. Global scales are FP32
    ``[R]`` when enabled and empty otherwise.
  """

  return torch.ops.cache_dit_cuda_ops.fp4_quantize(x, level1)


def fp4_quantize_swiglu(
  value: torch.Tensor,
  gate: torch.Tensor | None = None,
  level1: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Quantize ``value * silu(gate)`` without materializing the BF16 product.

  :param value: BF16 ``[R,H]`` with a separate gate, or half-split ``[R,2H]``.
  :param gate: Optional BF16 ``[R,H]`` gate. If omitted, the second half of
    ``value`` is used and the first half supplies the value operand.
  :param level1: Produce per-row FP32 global scales in addition to block scales.
  :returns: The same ``(qdata, block_scales, global_scales)`` contract as
    :func:`fp4_quantize` applied to the explicitly gated BF16 product.
  """

  return torch.ops.cache_dit_cuda_ops.fp4_quantize_swiglu(value, gate, level1)


def _fp4_gemm_scaled_mm(a: torch.Tensor, b: torch.Tensor, a_scales: torch.Tensor,
                        b_scales: torch.Tensor, a_global_scales: torch.Tensor,
                        b_global_scales: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
  """Run block-scaled NVFP4 GEMM through PyTorch's cuBLAS-backed operator.

  This is the same GEMM operator TorchAO's NVFP4 linear dispatch uses. It is
  retained as a reference path, not the native default. The operator applies
  only the UE4M3 block scales; the optional FP32 row/column global scales are
  folded afterward in this wrapper.

  :param a: Packed activation ``[M, K / 2]``.
  :param b: Packed weight ``[N, K / 2]``.
  :param a_scales: Canonical activation block scales.
  :param b_scales: Canonical weight block scales.
  :param a_global_scales: Empty or per-row FP32 activation scales.
  :param b_global_scales: Empty or per-output-channel FP32 weight scales.
  :param bias: Optional BF16 bias ``[N]``.
  :returns: BF16 output ``[M, N]``.
  """

  scaled = a_global_scales.numel() != 0 or b_global_scales.numel() != 0
  out = torch._scaled_mm(
    a,
    b.t(),
    a_scales,
    b_scales,
    bias=None if scaled else bias,
    out_dtype=torch.bfloat16,
  )
  if scaled:
    if (torch.compiler.is_compiling() and a_global_scales.numel() != 0
        and b_global_scales.numel() != 0):
      # Inductor folds this outer product into one pointwise epilogue instead of
      # materializing two sequential BF16 scale passes.
      scale = (a_global_scales[:, None] * b_global_scales[None, :]).to(torch.bfloat16)
      out = out * scale
    else:
      if a_global_scales.numel() != 0:
        out = out * a_global_scales[:, None].to(torch.bfloat16)
      if b_global_scales.numel() != 0:
        out = out * b_global_scales[None, :].to(torch.bfloat16)
    if bias is not None:
      out = out + bias
  return out


def _pad_block_scales(scales: torch.Tensor, rows: int, cols: int, padded_rows: int,
                      padded_cols: int) -> torch.Tensor:
  """Zero-pad canonical scales without dequantizing the operands.

  A canonical buffer is viewed as ``[row_block, k64_block, 32, 16]``. Copying
  complete 512-byte atoms preserves every logical scale while new K64 atoms and
  row-block padding remain zero. Zero scales make the corresponding padded
  contributions exactly zero in the block-scaled GEMM.
  """

  old_row_blocks = (rows + 127) // 128
  new_row_blocks = (padded_rows + 127) // 128
  old_k_blocks = cols // 64
  new_k_blocks = padded_cols // 64
  padded = scales.new_zeros((new_row_blocks, new_k_blocks, 32, 16))
  padded[:old_row_blocks, :old_k_blocks].copy_(scales.reshape(old_row_blocks, old_k_blocks, 32, 16))
  return padded.reshape(_scale_shape(padded_rows, padded_cols))


def _pad_gemm_operands(a: torch.Tensor, b: torch.Tensor, a_scales: torch.Tensor,
                       b_scales: torch.Tensor, a_global_scales: torch.Tensor,
                       b_global_scales: torch.Tensor, bias: torch.Tensor | None, padded_k: int,
                       padded_n: int):
  """Pad packed GEMM operands with mathematically neutral zero values."""

  rows, cols = a.shape[0], a.shape[1] * 2
  out_channels = b.shape[0]
  if cols % 64 != 0:
    raise ValueError("FP4 GEMM padding requires the original logical K % 64 == 0.")
  if padded_k != cols or padded_n != out_channels:
    a = F.pad(a.view(torch.uint8), (0, (padded_k - cols) // 2)).view(a.dtype)
    b = F.pad(b.view(torch.uint8),
              (0, (padded_k - cols) // 2, 0, padded_n - out_channels)).view(b.dtype)
    a_scales = _pad_block_scales(a_scales, rows, cols, rows, padded_k)
    b_scales = _pad_block_scales(b_scales, out_channels, cols, padded_n, padded_k)
    if b_global_scales.numel() != 0:
      b_global_scales = F.pad(b_global_scales, (0, padded_n - out_channels))
    if bias is not None:
      bias = F.pad(bias, (0, padded_n - out_channels))
  return a, b, a_scales, b_scales, a_global_scales, b_global_scales, bias


def fp4_gemm(a: torch.Tensor,
             b: torch.Tensor,
             a_scales: torch.Tensor,
             b_scales: torch.Tensor,
             a_global_scales: torch.Tensor,
             b_global_scales: torch.Tensor,
             bias: torch.Tensor | None = None,
             use_native: bool = True,
             activation: str | None = None) -> torch.Tensor:
  """Compute packed NVFP4 ``a @ b.T`` with a BF16 output.

  The native SM120 CUTLASS collective consumes packed E2M1 data and canonical
  UE4M3 block scales directly. Its custom EVT (Epilogue Visitor Tree, the
  collective's per-output-element computation graph) evaluates::

    output[m,n] = BF16(b_global_scales[n] *
                       (a_global_scales[m] * accumulator[m,n]) + bias[n])

  All scale and bias arithmetic is FP32. If both global-scale tensors are empty,
  the single-level epilogue computes only ``accumulator + bias``. No explicit
  full-output scale tensor or separate BF16 scale pass is required.

  :param a: Packed activation ``[M,K/2]`` from :func:`fp4_quantize`.
  :param b: Packed weight ``[N,K/2]`` from :func:`fp4_quantize`. Its storage is
    row-major ``[N,K]`` like ``torch.nn.Linear.weight``; the collective consumes
    the mathematical transpose without a runtime repack.
  :param a_scales: Canonical activation UE4M3 block scales.
  :param b_scales: Canonical weight UE4M3 block scales.
  :param a_global_scales: Empty or per-row FP32 activation scales ``[M]``.
  :param b_global_scales: Empty or per-output-channel FP32 weight scales ``[N]``.
  :param bias: Optional BF16 bias ``[N]``.
  :param use_native: ``True`` uses cache-dit's collective and pads to logical
    ``K%256==0`` and ``N%128==0`` when needed. ``False`` selects
    ``torch._scaled_mm`` and pads N to 16. Both paths zero-pad operands and
    depad the output, so unsupported native tile shapes do not change results.
  :param activation: Optional native ``gelu`` or ``gelu_tanh`` epilogue. The
    ``torch._scaled_mm`` reference does not expose an activation epilogue.
  :returns: BF16 output ``[M,N]``.
  """

  activation_id = None
  if activation is not None:
    activation_id = _ACTIVATIONS.get(activation)
    if activation_id is None:
      raise ValueError(
        f"Unsupported FP4 activation {activation!r}; expected 'gelu' or 'gelu_tanh'.")
    if not use_native:
      raise ValueError("FP4 activation requires use_native=True; torch._scaled_mm exposes no "
                       "activation epilogue.")

  logical_k = a.shape[1] * 2
  out_channels = b.shape[0]
  k_multiple, n_multiple = (256, 128) if use_native else (64, 16)
  padded_k = (logical_k + k_multiple - 1) // k_multiple * k_multiple
  padded_n = (out_channels + n_multiple - 1) // n_multiple * n_multiple
  a, b, a_scales, b_scales, a_global_scales, b_global_scales, bias = _pad_gemm_operands(
    a, b, a_scales, b_scales, a_global_scales, b_global_scales, bias, padded_k, padded_n)
  if use_native:
    if activation_id is None:
      out = torch.ops.cache_dit_cuda_ops.fp4_gemm(a, b, a_scales, b_scales, a_global_scales,
                                                  b_global_scales, bias)
    else:
      out = torch.ops.cache_dit_cuda_ops.fp4_gemm_activation(a, b, a_scales, b_scales,
                                                             a_global_scales, b_global_scales, bias,
                                                             activation_id)
  else:
    out = _fp4_gemm_scaled_mm(a, b, a_scales, b_scales, a_global_scales, b_global_scales, bias)
  return out[:, :out_channels]


__all__ = ["fp4_gemm", "fp4_quantize", "fp4_quantize_swiglu"]
