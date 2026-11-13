"""Cache-DiT native FP8 quantization and GEMM operators.

The low-level API keeps quantization and GEMM separate so callers can reuse an
offline-quantized weight. Three scale granularities match TorchAO's public FP8
types:

* ``per_row``: activation scale ``[M]``, weight scale ``[N]``.
* ``per_tensor``: one activation scale and one weight scale.
* ``per_block``: activation ``[M, K / 128]`` and weight
  ``[ceil(N / 128), K / 128]`` scales; GEMM rescales each K block.
"""

import torch

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
    name for name in ("fp8_quantize", "fp8_gemm", "fp8_quantize_gemm_per_row")
    if not hasattr(ops, name)
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
                   bias: torch.Tensor | None, out: torch.Tensor, mode: int, use_ws: bool) -> None:
  """Bridge the Python GEMM to the compiled extension.

  :param a: Quantized activation ``[M, K]``.
  :param b: Quantized weight ``[N, K]``.
  :param a_scale: Activation scale.
  :param b_scale: Weight scale.
  :param bias: Optional BF16 bias.
  :param out: BF16 destination ``[M, N]``.
  :param mode: Integer granularity id.
  :param use_ws: Select the warp-specialized mainloop.
  """

  _fp8_ops_module().fp8_gemm(a, b, a_scale, b_scale, bias, out, mode, use_ws)


torch.library.define(
  "cache_dit_cuda_ops::fp8_gemm",
  "(Tensor a, Tensor b, Tensor a_scale, Tensor b_scale, Tensor? bias, int mode, bool use_ws) "
  "-> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::fp8_gemm", "CUDA")
def _fp8_gemm(a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor,
              bias: torch.Tensor | None, mode: int, use_ws: bool) -> torch.Tensor:
  _validate_gemm_metadata(a, b, a_scale, b_scale, bias, mode)
  a = a.contiguous()
  b = b.contiguous()
  out = torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device=a.device)
  _call_fp8_gemm(a, b, a_scale.contiguous(), b_scale.contiguous(), bias, out, mode, use_ws)
  return out


@torch.library.register_fake("cache_dit_cuda_ops::fp8_gemm")
def _fake_fp8_gemm(a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor,
                   bias: torch.Tensor | None, mode: int, use_ws: bool) -> torch.Tensor:
  del use_ws
  _validate_gemm_metadata(a, b, a_scale, b_scale, bias, mode)
  return a.new_empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16)


def _validate_quantize_gemm_metadata(x: torch.Tensor, b: torch.Tensor, b_scale: torch.Tensor,
                                     bias: torch.Tensor | None) -> None:
  """Validate the public per-row quantize+GEMM metadata.

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
  "cache_dit_cuda_ops::fp8_quantize_gemm_per_row",
  "(Tensor x, Tensor b, Tensor b_scale, Tensor? bias, bool use_ws) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::fp8_quantize_gemm_per_row", "CUDA")
def _fp8_quantize_gemm_per_row(x: torch.Tensor, b: torch.Tensor, b_scale: torch.Tensor,
                               bias: torch.Tensor | None, use_ws: bool) -> torch.Tensor:
  _validate_quantize_gemm_metadata(x, b, b_scale, bias)
  x = x.contiguous()
  b = b.contiguous()
  out = torch.empty((x.shape[0], b.shape[0]), dtype=torch.bfloat16, device=x.device)
  _fp8_ops_module().fp8_quantize_gemm_per_row(x, b, b_scale.contiguous(), bias, out, use_ws)
  return out


@torch.library.register_fake("cache_dit_cuda_ops::fp8_quantize_gemm_per_row")
def _fake_fp8_quantize_gemm_per_row(x: torch.Tensor, b: torch.Tensor, b_scale: torch.Tensor,
                                    bias: torch.Tensor | None, use_ws: bool) -> torch.Tensor:
  del use_ws
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

  :param x: Contiguous-capable 2D BF16 tensor ``[M, K]``. For a linear weight
    this is ``[N, K]`` and ``per_row`` means per-output-channel, i.e. per-col of
    the mathematical ``B`` matrix.
  :param mode: ``per_row``, ``per_tensor`` or ``per_block``.
  :param is_weight: Set for ``[N, K]`` weights so ``per_block`` uses the
    ``128 x 128`` weight-scale layout.
  :returns: ``(quantized, scale)`` where quantized keeps the input layout.
  """

  return torch.ops.cache_dit_cuda_ops.fp8_quantize(x, _resolve_mode(mode), is_weight)


def fp8_gemm(a: torch.Tensor,
             b: torch.Tensor,
             a_scale: torch.Tensor,
             b_scale: torch.Tensor,
             bias: torch.Tensor | None = None,
             mode: str = "per_row",
             use_ws: bool | None = None) -> torch.Tensor:
  """Compute ``a @ b.T`` with FP8 inputs and a BF16 output.

  :param a: Quantized activation ``[M, K]``.
  :param b: Quantized weight ``[N, K]``.
  :param a_scale: Activation scale produced by :func:`fp8_quantize`.
  :param b_scale: Weight scale produced by :func:`fp8_quantize`.
  :param bias: Optional BF16 bias ``[N]``.
  :param mode: ``per_row``, ``per_tensor`` or ``per_block``.
  :param use_ws: Select the warp-specialized mainloop. ``None`` uses the measured
    default: WS for per-block, all-warps for per-row/per-tensor because the former's
    epilogue spills 144 B/thread in the 384-thread WS kernel on sm120.
  :returns: BF16 output ``[M, N]``.
  """

  mode_id = _resolve_mode(mode)
  if use_ws is None:
    use_ws = mode_id == _MODE_TO_ID["per_block"]
  return torch.ops.cache_dit_cuda_ops.fp8_gemm(a, b, a_scale, b_scale, bias, mode_id, use_ws)


def fp8_quantize_gemm(x: torch.Tensor,
                      b: torch.Tensor,
                      b_scale: torch.Tensor,
                      bias: torch.Tensor | None = None,
                      use_ws: bool = False) -> torch.Tensor:
  """Quantize a BF16 activation per row and immediately run FP8 GEMM.

  :param x: BF16 activation ``[M, K]``.
  :param b: Quantized weight ``[N, K]``.
  :param b_scale: Per-output-channel FP32 weight scale.
  :param bias: Optional BF16 bias ``[N]``.
  :param use_ws: Select the warp-specialized GEMM mainloop.
  :returns: BF16 output ``[M, N]``.
  """

  return torch.ops.cache_dit_cuda_ops.fp8_quantize_gemm_per_row(x, b, b_scale, bias, use_ws)


__all__ = ["fp8_gemm", "fp8_quantize", "fp8_quantize_gemm", "fp8_quantize_torch"]
