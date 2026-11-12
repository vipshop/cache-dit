import torch

from .cuda_ext import _get_required_ops_module


def _call_ulysses_fp8_per_token_quant(x: torch.Tensor, y: torch.Tensor,
                                      packed_head_size: int) -> None:
  """Bridge the per-token FP8 quantizer to the compiled extension.

  :param x: Contiguous 2D BF16 activation `[M, D]` to quantize.
  :param y: Contiguous 2D FP8 destination `[M, packed_head_size]`.
  :param packed_head_size: Packed row width, including scale and alignment padding.
  """

  _get_required_ops_module().ulysses_fp8_per_token_quant(x, y, packed_head_size)


def _call_ulysses_fp8_per_token_dequant(x: torch.Tensor, y: torch.Tensor, head_size: int) -> None:
  """Bridge the per-token FP8 dequantizer to the compiled extension.

  :param x: Contiguous 2D FP8 input `[M, packed_head_size]` packed by the quantizer.
  :param y: Contiguous 2D BF16 destination `[M, D]`.
  :param head_size: Original attention head dimension `D`.
  """

  _get_required_ops_module().ulysses_fp8_per_token_dequant(x, y, head_size)


def _call_ulysses_fp8_qkv_permute_quant(x: torch.Tensor, y: torch.Tensor,
                                        packed_head_size: int) -> None:
  """Bridge the QKV permuting FP8 quantizer to the compiled extension.

  :param x: 5D BF16 QKV activation `[B, S, P, N, D]` whose last dimension has stride 1.
  :param y: Contiguous 5D FP8 destination `[P, S, B, N, packed_head_size]`.
  :param packed_head_size: Packed row width, including the FP32 scale and alignment padding.
  """

  _get_required_ops_module().ulysses_fp8_qkv_permute_quant(x, y, packed_head_size)


def _call_ulysses_fp8_qkv_permute_dequant(quant_x: torch.Tensor, x: torch.Tensor,
                                          head_size: int) -> None:
  """Bridge the QKV restoring FP8 dequantizer to the compiled extension.

  :param quant_x: 4D FP8 activation `[S, B, N, packed_head_size]`, last dimension stride 1.
  :param x: Contiguous 4D BF16 destination `[B, S, N, D]` restored to the original QKV layout.
  :param head_size: Original attention head dimension `D`.
  """

  _get_required_ops_module().ulysses_fp8_qkv_permute_dequant(quant_x, x, head_size)


def _call_ulysses_fp8_o_permute_quant(x: torch.Tensor, y: torch.Tensor, partitions: int,
                                      packed_head_size: int) -> None:
  """Bridge the O-direction permuting FP8 quantizer to the compiled extension.

  :param x: BF16 activation `[B, S_GLOBAL, H_LOCAL, D]` whose last dimension has stride 1.
  :param y: Contiguous FP8 destination `[P, H_LOCAL, B, S_LOCAL, packed_head_size]`.
  :param partitions: Number of sequence partitions `P`.
  :param packed_head_size: Padded row width, including the two-byte BF16 scale.
  """

  _get_required_ops_module().ulysses_fp8_o_permute_quant(x, y, partitions, packed_head_size)


def _call_ulysses_fp8_o_permute_dequant(quant_x: torch.Tensor, x: torch.Tensor,
                                        head_size: int) -> None:
  """Bridge the O-direction restoring FP8 dequantizer to the compiled extension.

  :param quant_x: FP8 activation `[H_GLOBAL, B, S_LOCAL, packed_head_size]` with stride 1 last.
  :param x: Contiguous BF16 destination `[B, S_LOCAL, H_GLOBAL, head_size]`.
  :param head_size: Original attention head dimension `D`.
  """

  _get_required_ops_module().ulysses_fp8_o_permute_dequant(quant_x, x, head_size)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp8_per_token_quant",
  "(Tensor x, int? packed_head_size=None) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp8_per_token_quant", "CUDA")
def _ulysses_fp8_per_token_quant(x: torch.Tensor,
                                 packed_head_size: int | None = None) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  *shape, head_size = x.shape
  packed_head_size = head_size + 2 if packed_head_size is None else packed_head_size
  x = x.reshape(-1, head_size).contiguous()
  rows, _ = x.shape
  quant_x = torch.empty((rows, packed_head_size), dtype=torch.float8_e4m3fn, device=x.device)
  _call_ulysses_fp8_per_token_quant(x, quant_x, packed_head_size)
  return quant_x.reshape(*shape, packed_head_size)


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp8_per_token_quant")
def _fake_ulysses_fp8_per_token_quant(x: torch.Tensor,
                                      packed_head_size: int | None = None) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  *shape, head_size = x.shape
  packed_head_size = head_size + 2 if packed_head_size is None else packed_head_size
  return x.new_empty((*shape, packed_head_size), dtype=torch.float8_e4m3fn)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp8_per_token_dequant",
  "(Tensor x, int? head_size=None) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp8_per_token_dequant", "CUDA")
def _ulysses_fp8_per_token_dequant(x: torch.Tensor, head_size: int | None = None) -> torch.Tensor:
  if x.dtype != torch.float8_e4m3fn:
    raise ValueError(f"Expected a float8_e4m3fn activation, got {x.dtype}.")
  *shape, packed_head_size = x.shape
  head_size = packed_head_size - 2 if head_size is None else head_size
  x = x.reshape(-1, packed_head_size).contiguous()
  rows, _ = x.shape
  dequant_x = torch.empty((rows, head_size), dtype=torch.bfloat16, device=x.device)
  _call_ulysses_fp8_per_token_dequant(x, dequant_x, head_size)
  return dequant_x.reshape(*shape, head_size)


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp8_per_token_dequant")
def _fake_ulysses_fp8_per_token_dequant(x: torch.Tensor,
                                        head_size: int | None = None) -> torch.Tensor:
  if x.dtype != torch.float8_e4m3fn:
    raise ValueError(f"Expected a float8_e4m3fn activation, got {x.dtype}.")
  *shape, packed_head_size = x.shape
  head_size = packed_head_size - 2 if head_size is None else head_size
  return x.new_empty((*shape, head_size), dtype=torch.bfloat16)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp8_qkv_permute_quant",
  "(Tensor x, int? packed_head_size=None) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp8_qkv_permute_quant", "CUDA")
def _ulysses_fp8_qkv_permute_quant(x: torch.Tensor,
                                   packed_head_size: int | None = None) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  batch, seq_len, partitions, num_heads, head_size = x.shape
  packed_head_size = head_size + 4 if packed_head_size is None else packed_head_size
  quant_x = torch.empty((partitions, seq_len, batch, num_heads, packed_head_size),
                        dtype=torch.float8_e4m3fn,
                        device=x.device)
  _call_ulysses_fp8_qkv_permute_quant(x, quant_x, packed_head_size)
  return quant_x


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp8_qkv_permute_quant")
def _fake_ulysses_fp8_qkv_permute_quant(x: torch.Tensor,
                                        packed_head_size: int | None = None) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  batch, seq_len, partitions, num_heads, head_size = x.shape
  packed_head_size = head_size + 4 if packed_head_size is None else packed_head_size
  return x.new_empty((partitions, seq_len, batch, num_heads, packed_head_size),
                     dtype=torch.float8_e4m3fn)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp8_qkv_permute_dequant",
  "(Tensor quant_x, int? head_size=None) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp8_qkv_permute_dequant", "CUDA")
def _ulysses_fp8_qkv_permute_dequant(quant_x: torch.Tensor,
                                     head_size: int | None = None) -> torch.Tensor:
  if quant_x.dtype != torch.float8_e4m3fn:
    raise ValueError(f"Expected a float8_e4m3fn activation, got {quant_x.dtype}.")
  seq_len, batch, num_heads, packed_head_size = quant_x.shape
  head_size = packed_head_size - 4 if head_size is None else head_size
  # The dequantized activation is consumed as a contiguous `[B, S, N, D]` buffer.
  x = torch.empty((batch, seq_len, num_heads, head_size),
                  dtype=torch.bfloat16,
                  device=quant_x.device)
  _call_ulysses_fp8_qkv_permute_dequant(quant_x, x, head_size)
  return x


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp8_qkv_permute_dequant")
def _fake_ulysses_fp8_qkv_permute_dequant(quant_x: torch.Tensor,
                                          head_size: int | None = None) -> torch.Tensor:
  if quant_x.dtype != torch.float8_e4m3fn:
    raise ValueError(f"Expected a float8_e4m3fn activation, got {quant_x.dtype}.")
  seq_len, batch, num_heads, packed_head_size = quant_x.shape
  head_size = packed_head_size - 4 if head_size is None else head_size
  return quant_x.new_empty((batch, seq_len, num_heads, head_size), dtype=torch.bfloat16)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp8_o_permute_quant",
  "(Tensor x, int partitions, int packed_head_size) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp8_o_permute_quant", "CUDA")
def _ulysses_fp8_o_permute_quant(x: torch.Tensor, partitions: int,
                                 packed_head_size: int) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  batch, seq_len, num_heads, _ = x.shape
  quant_x = torch.empty((partitions, num_heads, batch, seq_len // partitions, packed_head_size),
                        dtype=torch.float8_e4m3fn,
                        device=x.device)
  _call_ulysses_fp8_o_permute_quant(x, quant_x, partitions, packed_head_size)
  return quant_x


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp8_o_permute_quant")
def _fake_ulysses_fp8_o_permute_quant(x: torch.Tensor, partitions: int,
                                      packed_head_size: int) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  batch, seq_len, num_heads, _ = x.shape
  return x.new_empty((partitions, num_heads, batch, seq_len // partitions, packed_head_size),
                     dtype=torch.float8_e4m3fn)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp8_o_permute_dequant",
  "(Tensor quant_x, int head_size) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp8_o_permute_dequant", "CUDA")
def _ulysses_fp8_o_permute_dequant(quant_x: torch.Tensor, head_size: int) -> torch.Tensor:
  if quant_x.dtype != torch.float8_e4m3fn:
    raise ValueError(f"Expected a float8_e4m3fn activation, got {quant_x.dtype}.")
  num_heads, batch, seq_len, _ = quant_x.shape
  x = torch.empty((batch, seq_len, num_heads, head_size),
                  dtype=torch.bfloat16,
                  device=quant_x.device)
  _call_ulysses_fp8_o_permute_dequant(quant_x, x, head_size)
  return x


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp8_o_permute_dequant")
def _fake_ulysses_fp8_o_permute_dequant(quant_x: torch.Tensor, head_size: int) -> torch.Tensor:
  if quant_x.dtype != torch.float8_e4m3fn:
    raise ValueError(f"Expected a float8_e4m3fn activation, got {quant_x.dtype}.")
  num_heads, batch, seq_len, _ = quant_x.shape
  return quant_x.new_empty((batch, seq_len, num_heads, head_size), dtype=torch.bfloat16)


def ulysses_fp8_per_token_quant(x: torch.Tensor,
                                packed_head_size: int | None = None) -> torch.Tensor:
  """Quantize a floating-point tensor into the FP8 per-token communication layout.

  :param x: Dense BF16 activation to quantize.
  :param packed_head_size: Optional padded row width; defaults to `D + 2`. The BF16 scale remains
    at offset `D` and any trailing padding is zeroed.
  :returns: The quantized FP8 activation with the requested packed row width.
  """

  return torch.ops.cache_dit_cuda_ops.ulysses_fp8_per_token_quant(x, packed_head_size)


def ulysses_fp8_per_token_dequant(x: torch.Tensor, head_size: int | None = None) -> torch.Tensor:
  """Dequantize a FP8 per-token communication tensor back to floating-point format.

  :param x: FP8 activation packed by `ulysses_fp8_per_token_quant`.
  :param head_size: Original head dimension `D`; defaults to `packed_head_size - 2`. Padding beyond
    the scale is ignored.
  :returns: The dequantized BF16 activation with the original head dimension.
  """

  return torch.ops.cache_dit_cuda_ops.ulysses_fp8_per_token_dequant(x, head_size)


def ulysses_fp8_qkv_permute_quant(x: torch.Tensor,
                                  packed_head_size: int | None = None) -> torch.Tensor:
  """Quantize a QKV tensor into FP8 while permuting it for communication.

  :param x: Dense BF16 QKV activation shaped `[B, S, P, N, D]`, whose last dimension must have
    stride 1.
  :param packed_head_size: Optional padded row width; defaults to `D + 4`. The FP32 scale remains
    at offset `D` and any trailing padding is zeroed.
  :returns: The quantized FP8 activation shaped `[P, S, B, N, packed_head_size]`.
  """

  return torch.ops.cache_dit_cuda_ops.ulysses_fp8_qkv_permute_quant(x, packed_head_size)


def ulysses_fp8_qkv_permute_dequant(quant_x: torch.Tensor,
                                    head_size: int | None = None) -> torch.Tensor:
  """Dequantize a permuted FP8 QKV tensor and restore its original layout.

  :param quant_x: FP8 activation shaped `[S, B, N, packed_head_size]`, last dimension stride 1.
  :param head_size: Original head dimension `D`; defaults to `packed_head_size - 4`. Padding beyond
    the scale is ignored.
  :returns: The dequantized BF16 activation shaped `[B, S, N, D]` in a contiguous layout.
  """

  return torch.ops.cache_dit_cuda_ops.ulysses_fp8_qkv_permute_dequant(quant_x, head_size)


def ulysses_fp8_o_permute_quant(x: torch.Tensor, partitions: int,
                                packed_head_size: int) -> torch.Tensor:
  """Quantize an attention output into FP8 while permuting it for all-to-all.

  :param x: BF16 activation `[B, S_GLOBAL, H_LOCAL, D]`, whose last dimension has stride 1.
  :param partitions: Number of sequence partitions `P`.
  :param packed_head_size: Padded row width, at least `D + 2`; the BF16 scale stays at offset `D`.
  :returns: FP8 activation `[P, H_LOCAL, B, S_LOCAL, packed_head_size]` with zeroed padding.
  """

  return torch.ops.cache_dit_cuda_ops.ulysses_fp8_o_permute_quant(x, partitions, packed_head_size)


def ulysses_fp8_o_permute_dequant(quant_x: torch.Tensor, head_size: int) -> torch.Tensor:
  """Dequantize an all-to-all output and restore the attention output layout.

  :param quant_x: FP8 activation `[H_GLOBAL, B, S_LOCAL, packed_head_size]`, stride 1 last.
  :param head_size: Original attention head dimension `D`; trailing packed padding is ignored.
  :returns: Contiguous BF16 activation `[B, S_LOCAL, H_GLOBAL, D]`.
  """

  return torch.ops.cache_dit_cuda_ops.ulysses_fp8_o_permute_dequant(quant_x, head_size)


__all__ = [
  "_call_ulysses_fp8_o_permute_dequant",
  "_call_ulysses_fp8_o_permute_quant",
  "_call_ulysses_fp8_per_token_dequant",
  "_call_ulysses_fp8_per_token_quant",
  "_call_ulysses_fp8_qkv_permute_dequant",
  "_call_ulysses_fp8_qkv_permute_quant",
  "ulysses_fp8_o_permute_dequant",
  "ulysses_fp8_o_permute_quant",
  "ulysses_fp8_per_token_quant",
  "ulysses_fp8_per_token_dequant",
  "ulysses_fp8_qkv_permute_quant",
  "ulysses_fp8_qkv_permute_dequant",
]
