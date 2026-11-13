"""Cache-DiT native NVFP4 quantization and block-scaled GEMM operators."""

import torch

from .cuda_ext import _get_required_ops_module


def _fp4_ops_module():
  """Return the compiled ops module after checking the FP4 entry points.

  :returns: The pybind ``ops`` submodule.
  """

  ops = _get_required_ops_module()
  missing = [name for name in ("fp4_quantize", "fp4_gemm") if not hasattr(ops, name)]
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
  if cols % 64 != 0 or out_channels % 8 != 0:
    raise ValueError("FP4 GEMM requires logical K % 64 == 0 and N % 8 == 0.")
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
                   bias: torch.Tensor | None, out: torch.Tensor, use_ws: bool) -> None:
  """Bridge functional GEMM to the out-param extension.

  :param a: Packed activation.
  :param b: Packed weight.
  :param a_scales: Activation block scales.
  :param b_scales: Weight block scales.
  :param a_global_scales: Activation global scales.
  :param b_global_scales: Weight global scales.
  :param bias: Optional BF16 bias.
  :param out: BF16 destination.
  :param use_ws: Select the warp-specialized mainloop.
  """

  _fp4_ops_module().fp4_gemm(a, b, a_scales, b_scales, a_global_scales, b_global_scales, bias, out,
                             use_ws)


torch.library.define(
  "cache_dit_cuda_ops::fp4_gemm",
  "(Tensor a, Tensor b, Tensor a_scales, Tensor b_scales, Tensor a_global_scales, "
  "Tensor b_global_scales, Tensor? bias, bool use_ws) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::fp4_gemm", "CUDA")
def _fp4_gemm(a: torch.Tensor, b: torch.Tensor, a_scales: torch.Tensor, b_scales: torch.Tensor,
              a_global_scales: torch.Tensor, b_global_scales: torch.Tensor,
              bias: torch.Tensor | None, use_ws: bool) -> torch.Tensor:
  _validate_gemm_metadata(a, b, a_scales, b_scales, a_global_scales, b_global_scales, bias)
  out = torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device=a.device)
  _call_fp4_gemm(a.contiguous(), b.contiguous(), a_scales.contiguous(), b_scales.contiguous(),
                 a_global_scales.contiguous(), b_global_scales.contiguous(), bias, out, use_ws)
  return out


@torch.library.register_fake("cache_dit_cuda_ops::fp4_gemm")
def _fake_fp4_gemm(a: torch.Tensor, b: torch.Tensor, a_scales: torch.Tensor, b_scales: torch.Tensor,
                   a_global_scales: torch.Tensor, b_global_scales: torch.Tensor,
                   bias: torch.Tensor | None, use_ws: bool) -> torch.Tensor:
  del use_ws
  _validate_gemm_metadata(a, b, a_scales, b_scales, a_global_scales, b_global_scales, bias)
  return a.new_empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16)


def fp4_quantize(x: torch.Tensor,
                 level1: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Quantize a row-major BF16 tensor to packed NVFP4.

  :param x: BF16 tensor ``[R, K]`` with ``K % 64 == 0``.
  :param level1: Produce per-row FP32 global scales in addition to per-16 UE4M3
    block scales. Disable only for isolated single-level GEMM comparisons.
  :returns: ``(qdata, block_scales, global_scales)``. Qdata has shape
    ``[R, K / 2]`` and dtype ``float4_e2m1fn_x2``. Global scales are empty when
    ``level1=False``.
  """

  return torch.ops.cache_dit_cuda_ops.fp4_quantize(x, level1)


def _fp4_gemm_scaled_mm(a: torch.Tensor, b: torch.Tensor, a_scales: torch.Tensor,
                        b_scales: torch.Tensor, a_global_scales: torch.Tensor,
                        b_global_scales: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
  """Run block-scaled NVFP4 GEMM through PyTorch's cuBLAS-backed operator.

  This is the same GEMM operator TorchAO's NVFP4 linear dispatch uses. It is kept
  as the default because the CuTe kernel in this module is slower on the measured
  SM120 production shapes.

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


def fp4_gemm(a: torch.Tensor,
             b: torch.Tensor,
             a_scales: torch.Tensor,
             b_scales: torch.Tensor,
             a_global_scales: torch.Tensor,
             b_global_scales: torch.Tensor,
             bias: torch.Tensor | None = None,
             use_ws: bool = True,
             use_native: bool = True) -> torch.Tensor:
  """Compute packed NVFP4 ``a @ b.T`` with a BF16 output.

  By default this uses cache-dit's SM120 CUTLASS collective and a custom FP32
  epilogue that folds both global scales and bias into the GEMM. Set
  ``use_native=False`` for the ``torch._scaled_mm`` reference path. Shapes outside
  the collective domain use the bundled hand-written CuTe kernel.

  :param a: Packed activation ``[M, K / 2]``.
  :param b: Packed weight ``[N, K / 2]``.
  :param a_scales: Canonical activation block scales.
  :param b_scales: Canonical weight block scales.
  :param a_global_scales: Empty or per-row FP32 activation scales.
  :param b_global_scales: Empty or per-output-channel FP32 weight scales.
  :param bias: Optional BF16 bias ``[N]``.
  :param use_ws: Select the mainloop only for the hand-written tail fallback.
  :param use_native: ``True`` uses cache-dit's collective; ``False`` selects
    ``torch._scaled_mm`` when supported and the native tail fallback otherwise.
  :returns: BF16 output ``[M, N]``.
  """

  if not use_native and b.shape[0] % 16 == 0:
    return _fp4_gemm_scaled_mm(a, b, a_scales, b_scales, a_global_scales, b_global_scales, bias)
  return torch.ops.cache_dit_cuda_ops.fp4_gemm(a, b, a_scales, b_scales, a_global_scales,
                                               b_global_scales, bias, use_ws)


__all__ = ["fp4_gemm", "fp4_quantize"]
