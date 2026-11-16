"""Cache-DiT FP8 E4M3 quantization and GEMM operators.

Numerical convention
====================
For one quantization unit ``U``, with finite BF16 values ``X``::

    scale = max(abs(X[U])) / 448
    q = satfinite_e4m3(X / scale)
    x_hat = scale * q

``scale`` is an FP32 dequantization multiplier, not its reciprocal. An all-zero
unit returns ``scale=0, q=0`` and uses 1 as a safe division denominator. E4M3 is
the only FP8 storage format; there is no NVFP4-style second UE4M3 scale level.

For physical row-major activation ``A[M,K]`` and weight ``B[N,K]``, GEMM computes
``D = A @ B.T + bias`` in FP32 before writing BF16:

* ``per_row``: scales ``sa[M]`` and ``sb[N]``. The CUTLASS epilogue evaluates
  ``D[m,n] = bias[n] + sa[m] * sb[n] * sum_k(qA[m,k] * qB[n,k])``.
* ``per_tensor``: scalar ``sa`` and ``sb``. The same epilogue broadcasts the
  scalars with zero strides, without expanding them to M/N vectors.
* ``per_block``: activation scales ``[M,ceil(K/128)]`` and weight scales
  ``[ceil(N/128),ceil(K/128)]``. Each K128 mainloop block applies its activation
  and weight scales; the epilogue adds bias only. Weight scales correspond to
  128x128 blocks, whereas activation scales correspond to 1x128 blocks.

Typical low-level use (weight quantization is done once)::

    qa, sa = fp8_quantize(a, mode="per_row")
    qw, sw = fp8_quantize(w, mode="per_row", is_weight=True)
    d = fp8_gemm(qa, qw, sa, sw, bias=bias, mode="per_row")

API layers and shape handling
=============================
``fp8_quantize`` is the native custom-op quantizer. ``fp8_quantize_torch`` is the
mathematically equivalent pure-PyTorch implementation used by the high-level
Linear under ``torch.compile`` so Inductor can fuse surrounding operations.
``fp8_gemm`` defaults to the CUTLASS collective; ``use_native=False`` selects
``torch._scaled_mm`` only for per-row/per-tensor. Per-block always uses the
CUTLASS blockwise mainloop because ``_scaled_mm`` has no matching scale ABI.

The quantizer itself has explicit input requirements (K divisible by 16, and by
128 for per-block); it does not silently change the returned qdata shape. The
public GEMM separately zero-pads already quantized operands to the selected GEMM
backend's granularity and slices output N back. The high-level ``QW8A8Linear``
instead pads BF16 weights/activations *before* quantization, so its entire
quantize -> GEMM -> depad chain supports arbitrary Linear shapes. Zero qdata,
zero padding scales and zero padding bias make these transformations equivalent
on the original output region.

These are torch.library custom ops with matching CUDA and fake implementations.
There is intentionally no ``use_ws`` argument: CUTLASS selects the mainloop and
warp specialization rather than exposing the removed hand-written WS/non-WS
kernels as a public performance knob.
"""

import torch
import torch.nn.functional as F

from .cuda_ext import _get_required_ops_module

_MODE_TO_ID = {
  "per_row": 0,
  "per_tensor": 1,
  "per_block": 2,
}
_ID_TO_MODE = {value: key for key, value in _MODE_TO_ID.items()}


def _resolve_mode(mode: str | int) -> int:
  """Normalize a public granularity selector to the extension's integer id.

  :param mode: Granularity name or integer id.
  :returns: Integer id understood by the compiled extension.
  """

  if isinstance(mode, str):
    normalized = mode.lower()
    if normalized not in _MODE_TO_ID:
      raise ValueError(f"Unsupported FP8 granularity: {mode!r}; expected one of "
                       f"{sorted(_MODE_TO_ID)}.")
    return _MODE_TO_ID[normalized]
  if mode not in _ID_TO_MODE:
    raise ValueError(f"Unsupported FP8 granularity id: {mode!r}.")
  return int(mode)


def _fp8_ops_module():
  """Return the compiled ops module after checking this build contains FP8.

  :returns: The pybind ``ops`` submodule.
  """

  ops = _get_required_ops_module()
  missing = [
    name for name in ("fp8_quantize", "fp8_gemm", "fp8_quantize_gemm") if not hasattr(ops, name)
  ]
  if missing:
    raise RuntimeError(
      f"The loaded cache_dit._C_ops extension is missing {missing}. Rebuild it with "
      "`CACHE_DIT_BUILD_CUDA=1 CACHE_DIT_CUDA_ARCH_LIST=120a bash tools/build_fast.sh "
      "install -e . --no-build-isolation --no-deps`.")
  return ops


def _scale_shape(rows: int, cols: int, mode: int, is_weight: bool) -> tuple[int, ...]:
  """Return the scale shape for one quantization call.

  :param rows: Input row count.
  :param cols: Input column count.
  :param mode: Integer granularity id.
  :param is_weight: Whether the input is a ``[N, K]`` linear weight.
  :returns: Scale tensor shape.
  """

  if mode == _MODE_TO_ID["per_row"]:
    return (rows, )
  if mode == _MODE_TO_ID["per_tensor"]:
    return (1, )
  num_k_blocks = (cols + 127) // 128
  if is_weight:
    return ((rows + 127) // 128, num_k_blocks)
  return (rows, num_k_blocks)


def _validate_quantize_metadata(x: torch.Tensor, mode: int) -> None:
  """Validate metadata that is also available under FakeTensor tracing.

  :param x: Tensor to quantize.
  :param mode: Integer granularity id.
  """

  if x.dtype != torch.bfloat16:
    raise ValueError(f"cache-dit FP8 quantization expects bfloat16, got {x.dtype}.")
  if x.dim() != 2:
    raise ValueError(f"cache-dit FP8 quantization expects a 2D tensor, got {tuple(x.shape)}.")
  if x.shape[0] == 0 or x.shape[1] == 0:
    raise ValueError("cache-dit FP8 quantization does not accept empty tensors.")
  if x.shape[1] % 16 != 0:
    raise ValueError("cache-dit FP8 quantization requires the last dimension to be a multiple "
                     f"of 16, got {x.shape[1]}.")
  if mode == _MODE_TO_ID["per_block"] and x.shape[1] % 128 != 0:
    raise ValueError("cache-dit FP8 per-block quantization requires the last dimension to be "
                     f"a multiple of 128, got {x.shape[1]}.")


def _call_fp8_quantize(x: torch.Tensor, output: torch.Tensor, scale: torch.Tensor, mode: int,
                       is_weight: bool) -> None:
  """Bridge the Python quantizer to the compiled extension.

  :param x: Contiguous 2D BF16 input.
  :param output: Contiguous E4M3 destination.
  :param scale: FP32 scale destination.
  :param mode: Integer granularity id.
  :param is_weight: Whether ``x`` is a ``[N, K]`` linear weight.
  """

  _fp8_ops_module().fp8_quantize(x, output, scale, mode, is_weight)


torch.library.define(
  "cache_dit_cuda_ops::fp8_quantize",
  "(Tensor x, int mode, bool is_weight) -> (Tensor, Tensor)",
)


@torch.library.impl("cache_dit_cuda_ops::fp8_quantize", "CUDA")
def _fp8_quantize(x: torch.Tensor, mode: int, is_weight: bool):
  _validate_quantize_metadata(x, mode)
  x = x.contiguous()
  rows, cols = x.shape
  output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
  scale = torch.empty(_scale_shape(rows, cols, mode, is_weight),
                      dtype=torch.float32,
                      device=x.device)
  _call_fp8_quantize(x, output, scale, mode, is_weight)
  return output, scale


@torch.library.register_fake("cache_dit_cuda_ops::fp8_quantize")
def _fake_fp8_quantize(x: torch.Tensor, mode: int, is_weight: bool):
  _validate_quantize_metadata(x, mode)
  rows, cols = x.shape
  return (
    x.new_empty(x.shape, dtype=torch.float8_e4m3fn),
    x.new_empty(_scale_shape(rows, cols, mode, is_weight), dtype=torch.float32),
  )


def _validate_gemm_metadata(a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor,
                            b_scale: torch.Tensor, bias: torch.Tensor | None, mode: int) -> None:
  """Validate metadata shared by the eager and fake GEMM implementations.

  :param a: Quantized activation ``[M, K]``.
  :param b: Quantized weight ``[N, K]``.
  :param a_scale: Activation scale.
  :param b_scale: Weight scale.
  :param bias: Optional BF16 bias ``[N]``.
  :param mode: Integer granularity id.
  """

  if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
    raise ValueError("cache-dit FP8 GEMM expects float8_e4m3fn A and B.")
  if a.dim() != 2 or b.dim() != 2:
    raise ValueError("cache-dit FP8 GEMM expects 2D A and B.")
  if a.shape[1] != b.shape[1]:
    raise ValueError(f"FP8 GEMM inner dimensions differ: {a.shape[1]} and {b.shape[1]}.")
  if a.shape[1] % 16 != 0 or b.shape[0] % 8 != 0:
    raise ValueError("FP8 GEMM requires K % 16 == 0 and N % 8 == 0.")
  if mode == _MODE_TO_ID["per_block"] and a.shape[1] % 128 != 0:
    raise ValueError("FP8 per-block GEMM requires K % 128 == 0.")
  if a_scale.dtype != torch.float32 or b_scale.dtype != torch.float32:
    raise ValueError("FP8 GEMM scales must be float32.")
  rows, cols = a.shape
  num_k_blocks = (cols + 127) // 128
  expected_a = {
    _MODE_TO_ID["per_row"]: rows,
    _MODE_TO_ID["per_tensor"]: 1,
    _MODE_TO_ID["per_block"]: rows * num_k_blocks,
  }[mode]
  expected_b = {
    _MODE_TO_ID["per_row"]: b.shape[0],
    _MODE_TO_ID["per_tensor"]: 1,
    _MODE_TO_ID["per_block"]: ((b.shape[0] + 127) // 128) * num_k_blocks,
  }[mode]
  if a_scale.numel() != expected_a or b_scale.numel() != expected_b:
    raise ValueError(f"FP8 GEMM scale sizes do not match mode {mode}: got "
                     f"{a_scale.numel()} and {b_scale.numel()}, expected {expected_a} and "
                     f"{expected_b}.")
  if bias is not None and (bias.dtype != torch.bfloat16 or bias.shape != (b.shape[0], )):
    raise ValueError("FP8 GEMM bias must be bfloat16 with shape [N].")


def _call_fp8_gemm(a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor,
                   bias: torch.Tensor | None, out: torch.Tensor, mode: int) -> None:
  """Bridge the Python GEMM to the compiled extension.

  :param a: Quantized activation ``[M, K]``.
  :param b: Quantized weight ``[N, K]``.
  :param a_scale: Activation scale.
  :param b_scale: Weight scale.
  :param bias: Optional BF16 bias.
  :param out: BF16 destination ``[M, N]``.
  :param mode: Integer granularity id.
  """

  _fp8_ops_module().fp8_gemm(a, b, a_scale, b_scale, bias, out, mode)


torch.library.define(
  "cache_dit_cuda_ops::fp8_gemm",
  "(Tensor a, Tensor b, Tensor a_scale, Tensor b_scale, Tensor? bias, int mode) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::fp8_gemm", "CUDA")
def _fp8_gemm(a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor,
              bias: torch.Tensor | None, mode: int) -> torch.Tensor:
  _validate_gemm_metadata(a, b, a_scale, b_scale, bias, mode)
  a = a.contiguous()
  b = b.contiguous()
  out = torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device=a.device)
  _call_fp8_gemm(a, b, a_scale.contiguous(), b_scale.contiguous(), bias, out, mode)
  return out


@torch.library.register_fake("cache_dit_cuda_ops::fp8_gemm")
def _fake_fp8_gemm(a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor,
                   bias: torch.Tensor | None, mode: int) -> torch.Tensor:
  _validate_gemm_metadata(a, b, a_scale, b_scale, bias, mode)
  return a.new_empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16)


def _validate_quantize_gemm_metadata(x: torch.Tensor, b: torch.Tensor, b_scale: torch.Tensor,
                                     bias: torch.Tensor | None) -> None:
  """Validate the public per-row quantize+GEMM metadata.

  This convenience op is fixed to per-row quantization, so checking that recipe
  is sufficient; it does not accept or dispatch a separate mode argument.

  :param x: BF16 activation ``[M, K]``.
  :param b: Quantized weight ``[N, K]``.
  :param b_scale: Per-output-channel weight scale.
  :param bias: Optional BF16 bias ``[N]``.
  """

  _validate_quantize_metadata(x, _MODE_TO_ID["per_row"])
  if b.dtype != torch.float8_e4m3fn or b.dim() != 2 or b.shape[1] != x.shape[1]:
    raise ValueError("FP8 quantize+GEMM expects an E4M3 weight with shape [N, K].")
  if b.shape[0] % 8 != 0:
    raise ValueError("FP8 quantize+GEMM requires N % 8 == 0.")
  if b_scale.dtype != torch.float32 or b_scale.numel() != b.shape[0]:
    raise ValueError("FP8 quantize+GEMM expects one float32 scale per output channel.")
  if bias is not None and (bias.dtype != torch.bfloat16 or bias.shape != (b.shape[0], )):
    raise ValueError("FP8 quantize+GEMM bias must be bfloat16 with shape [N].")


torch.library.define(
  "cache_dit_cuda_ops::fp8_quantize_gemm",
  "(Tensor x, Tensor b, Tensor b_scale, Tensor? bias) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::fp8_quantize_gemm", "CUDA")
def _fp8_quantize_gemm(x: torch.Tensor, b: torch.Tensor, b_scale: torch.Tensor,
                       bias: torch.Tensor | None) -> torch.Tensor:
  _validate_quantize_gemm_metadata(x, b, b_scale, bias)
  x = x.contiguous()
  b = b.contiguous()
  out = torch.empty((x.shape[0], b.shape[0]), dtype=torch.bfloat16, device=x.device)
  _fp8_ops_module().fp8_quantize_gemm(x, b, b_scale.contiguous(), bias, out)
  return out


@torch.library.register_fake("cache_dit_cuda_ops::fp8_quantize_gemm")
def _fake_fp8_quantize_gemm(x: torch.Tensor, b: torch.Tensor, b_scale: torch.Tensor,
                            bias: torch.Tensor | None) -> torch.Tensor:
  _validate_quantize_gemm_metadata(x, b, b_scale, bias)
  return x.new_empty((x.shape[0], b.shape[0]), dtype=torch.bfloat16)


def fp8_quantize_torch(x: torch.Tensor,
                       mode: str = "per_row",
                       is_weight: bool = False,
                       safe_scale: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
  """Pure-PyTorch FP8 quantizer used when ``torch.compile`` traces the module.

  For finite inputs with nonzero amax, the math and qdata are bit-identical to
  :func:`fp8_quantize`: ``scale = amax / 448``. By default, zero scales are
  replaced by one for the division, preserving scale zero and quantized zero for
  all-zero quantization units. Setting ``safe_scale=False`` assumes nonzero amax
  and produces NaNs for zero units. NaN inputs differ by design: CUDA ``fmaxf``
  ignores NaNs, while torch ``amax`` propagates them. Keeping this path in
  ordinary pointwise/reduction ops lets Inductor fuse it around the FP8 GEMM
  instead of materializing an opaque custom-op boundary.

  :param x: 2D BF16 tensor ``[M, K]``; non-contiguous inputs are copied.
  :param mode: ``per_row``, ``per_tensor`` or ``per_block``.
  :param is_weight: Whether ``x`` is an ``[N, K]`` linear weight.
  :param safe_scale: Replace a zero scale by one for the division. Enabled by
    default; disable only when all rows, tensors, and blocks have nonzero amax.
  :returns: ``(quantized, scale)``.
  """

  mode_id = _resolve_mode(mode)
  _validate_quantize_metadata(x, mode_id)
  x = x.contiguous()
  rows, cols = x.shape
  fp32 = x.float()
  if mode_id == _MODE_TO_ID["per_row"]:
    scale = fp32.abs().amax(dim=1) / 448.0
    denominator = torch.where(scale == 0, torch.ones_like(scale), scale) if safe_scale else scale
    quantized = (fp32 / denominator[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized, scale
  if mode_id == _MODE_TO_ID["per_tensor"]:
    scale = fp32.abs().amax().reshape(1) / 448.0
    denominator = torch.where(scale == 0, torch.ones_like(scale), scale) if safe_scale else scale
    quantized = (fp32 / denominator[0]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized, scale

  num_k_blocks = (cols + 127) // 128
  if not is_weight:
    amax = fp32.reshape(rows, num_k_blocks, 128).abs().amax(dim=2)
    scale = amax / 448.0
    denominator = torch.where(scale == 0, torch.ones_like(scale), scale) if safe_scale else scale
    quantized = (fp32.reshape(rows, num_k_blocks, 128) / denominator[:, :, None]).clamp(
      -448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized.reshape(rows, cols), scale

  num_n_blocks = (rows + 127) // 128
  padded = torch.nn.functional.pad(fp32,
                                   (0, num_k_blocks * 128 - cols, 0, num_n_blocks * 128 - rows))
  blocks = padded.reshape(num_n_blocks, 128, num_k_blocks, 128)
  scale = blocks.abs().amax(dim=(1, 3)) / 448.0
  denominator = torch.where(scale == 0, torch.ones_like(scale), scale) if safe_scale else scale
  quantized = (blocks / denominator[:, None, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
  return quantized.reshape(num_n_blocks * 128, num_k_blocks * 128)[:rows, :cols], scale


def fp8_quantize(x: torch.Tensor,
                 mode: str = "per_row",
                 is_weight: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
  """Quantize a 2D BF16 tensor to E4M3 with the requested scale granularity.

  Each scale is ``amax / 448`` and each output value is
  ``satfinite_e4m3(input / scale)``. To reconstruct the approximate BF16 value,
  multiply qdata by the scale; do not divide by it. Zero units return zero scale
  and zero qdata. This custom op performs the reduction and conversion in CUDA;
  it does not fuse a following GEMM.

  This low-level quantizer does not pad its output. Callers must supply K % 16
  == 0, or K % 128 == 0 for per-block. :func:`fp8_gemm` can pad qdata later, and
  :class:`cache_dit.quantization.fp8.linear.QW8A8Linear` normally handles shape
  alignment before calling this function.

  :param x: Contiguous-capable 2D BF16 tensor ``[M, K]`` for activations or
    ``[N, K]`` for a weight. Per-row activation scaling is per token; per-row
    weight scaling is per output channel (per column of mathematical ``B.T``).
  :param mode: ``per_row`` (one scale per row), ``per_tensor`` (one scale for the
    whole tensor), or ``per_block`` (activation 1x128 / weight 128x128 blocks).
  :param is_weight: Only changes per-block scale-table layout. Set true for a
    weight ``[N,K]`` to return ``[ceil(N/128),ceil(K/128)]`` scales. Activation
    scales remain ``[M,ceil(K/128)]``. Per-row/per-tensor are unaffected.
  :returns: ``(qdata, scale)``. ``qdata`` has the same shape as ``x`` and dtype
    ``torch.float8_e4m3fn``; ``scale`` is FP32 with the mode-dependent shape
    above. Dequantization is approximately ``qdata.float() * scale``.
  """

  return torch.ops.cache_dit_cuda_ops.fp8_quantize(x, _resolve_mode(mode), is_weight)


def _fp8_gemm_scaled_mm(a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor,
                        b_scale: torch.Tensor, bias: torch.Tensor | None,
                        mode_id: int) -> torch.Tensor:
  """Run the per-row/per-tensor reference through ``torch._scaled_mm``.

  PyTorch expects the second matrix as ``[K,N]``; ``b.t()`` is a view of our
  row-major ``[N,K]`` weight and does not quantize or dequantize it. Per-row uses
  scale shapes ``[M,1]`` and ``[1,N]`` for row/column broadcast. Per-tensor passes
  the two scalar scales directly. The operator applies scales and bias according
  to its own API, so this helper does not multiply them again afterward.

  Per-block is intentionally absent: ``_scaled_mm`` has no 1x128-by-128x128
  block-scale contract. Its N % 16 requirement is handled by the public wrapper
  before this helper is called.
  """

  if mode_id == _MODE_TO_ID["per_row"]:
    return torch._scaled_mm(
      a,
      b.t(),
      a_scale[:, None].contiguous(),
      b_scale[None, :].contiguous(),
      bias=bias,
      out_dtype=torch.bfloat16,
    )
  return torch._scaled_mm(
    a,
    b.t(),
    a_scale,
    b_scale,
    bias=bias,
    out_dtype=torch.bfloat16,
  )


def _pad_fp8_gemm_operands(a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor,
                           b_scale: torch.Tensor, bias: torch.Tensor | None, mode: int,
                           padded_k: int, padded_n: int):
  """Zero-pad quantized operands to the selected GEMM's tile granularity.

  K padding appends zero E4M3 values to both A and B, so every added dot-product
  term is zero. N padding appends zero rows to B and zero bias entries, then the
  caller discards those output columns. Per-tensor needs no scale-table change:
  its scalar scale multiplied by padded zero qdata is still zero. Per-row pads
  b_scale[N] with zeros. Per-block pads whole 128-block rows/columns in both
  scale tables, so every newly introduced block also has zero scale.

  No amax is recomputed here: this helper only adapts an existing quantized
  representation to a backend layout. High-level Linear code normally performs
  the equivalent padding before quantization to avoid repeating weight work.
  """

  cols = a.shape[1]
  out_channels = b.shape[0]
  if cols != padded_k or out_channels != padded_n:
    a = F.pad(a, (0, padded_k - cols))
    b = F.pad(b, (0, padded_k - cols, 0, padded_n - out_channels))
    if mode == _MODE_TO_ID["per_row"]:
      b_scale = F.pad(b_scale, (0, padded_n - out_channels))
    elif mode == _MODE_TO_ID["per_block"]:
      old_k_blocks = a_scale.shape[1]
      new_k_blocks = padded_k // 128
      a_scale = F.pad(a_scale, (0, new_k_blocks - old_k_blocks))
      b_scale = F.pad(b_scale,
                      (0, new_k_blocks - old_k_blocks, 0, padded_n // 128 - b_scale.shape[0]))
    if bias is not None:
      bias = F.pad(bias, (0, padded_n - out_channels))
  return a, b, a_scale, b_scale, bias


def fp8_gemm(a: torch.Tensor,
             b: torch.Tensor,
             a_scale: torch.Tensor,
             b_scale: torch.Tensor,
             bias: torch.Tensor | None = None,
             mode: str = "per_row",
             use_native: bool = True) -> torch.Tensor:
  """Compute ``a @ b.T + bias`` from E4M3 inputs and FP32 scales.

  This function consumes the outputs of :func:`fp8_quantize`; it does not compute
  amax. Accumulation is FP32 and the public output is BF16. For per-row, the
  native EVT evaluates ``bias[n] + a_scale[m] * b_scale[n] * acc[m,n]`` in place
  of a separate scaled-output tensor. Per-tensor broadcasts scalar scales with
  zero strides. Per-block applies each 1x128/128x128 scale pair in the CUTLASS
  mainloop and uses a bias-only epilogue.

  Automatic shape handling is local to this wrapper. The native per-row/tensor
  path pads K/N to multiples of 16/8, the ``_scaled_mm`` reference pads to 16/16,
  and per-block pads to 128/128. It pads qdata and any necessary scale/bias
  entries with zero, then returns ``out[:, :N]``. These zeros contribute nothing
  to the original output. The underlying C++ ABI remains strict; padding happens
  before entering the custom op. For a full BF16 Linear chain, prefer
  ``QW8A8Linear``, which pads before quantization and avoids weight repacking on
  every forward.

  :param a: E4M3 activation ``[M,K]``.
  :param b: E4M3 weight ``[N,K]`` in ordinary row-major Linear layout. The
    collective consumes its mathematical transpose without a materialized copy.
  :param a_scale: FP32 activation scale from :func:`fp8_quantize`: ``[M]`` for
    per-row, scalar for per-tensor, or ``[M,ceil(K/128)]`` for per-block.
  :param b_scale: FP32 weight scale: ``[N]``, scalar, or
    ``[ceil(N/128),ceil(K/128)]`` according to the same mode.
  :param bias: Optional BF16 vector ``[N]``.
  :param mode: ``per_row``, ``per_tensor`` or ``per_block``.
  :param use_native: ``True`` (default) launches the CUTLASS collective.
    ``False`` selects the ``torch._scaled_mm`` reference for per-row/per-tensor;
    per-block still uses the native blockwise collective because scaled_mm does
    not support this scale layout.
  :returns: BF16 tensor ``[M,N]`` with the original, unpadded N.
  """

  mode_id = _resolve_mode(mode)
  out_channels = b.shape[0]
  logical_k = a.shape[1]
  use_scaled_mm = not use_native and mode_id != _MODE_TO_ID["per_block"]
  k_multiple, n_multiple = (128, 128) if mode_id == _MODE_TO_ID["per_block"] else (
    16, 16 if use_scaled_mm else 8)
  padded_k = (logical_k + k_multiple - 1) // k_multiple * k_multiple
  padded_n = (out_channels + n_multiple - 1) // n_multiple * n_multiple
  a, b, a_scale, b_scale, bias = _pad_fp8_gemm_operands(a, b, a_scale, b_scale, bias, mode_id,
                                                        padded_k, padded_n)
  if use_scaled_mm:
    out = _fp8_gemm_scaled_mm(a, b, a_scale, b_scale, bias, mode_id)
  else:
    out = torch.ops.cache_dit_cuda_ops.fp8_gemm(a, b, a_scale, b_scale, bias, mode_id)
  return out[:, :out_channels]


def fp8_quantize_gemm(x: torch.Tensor,
                      b: torch.Tensor,
                      b_scale: torch.Tensor,
                      bias: torch.Tensor | None = None) -> torch.Tensor:
  """Quantize a BF16 activation per row and run the CUTLASS FP8 GEMM.

  :param x: BF16 activation ``[M, K]``.
  :param b: Quantized weight ``[N, K]``.
  :param b_scale: Per-output-channel FP32 weight scale.
  :param bias: Optional BF16 bias ``[N]``.
  :returns: BF16 output ``[M, N]``.
  """

  return torch.ops.cache_dit_cuda_ops.fp8_quantize_gemm(x, b, b_scale, bias)


__all__ = ["fp8_gemm", "fp8_quantize", "fp8_quantize_gemm", "fp8_quantize_torch"]
