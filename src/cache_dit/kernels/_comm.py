import torch

from ._cuda_ext import _get_required_ops_module


def _call_fp8_comm_per_token_quant(x: torch.Tensor, y: torch.Tensor) -> None:
  """Bridge the per-token FP8 quantizer to the compiled extension.

  :param x: Contiguous 2D BF16 activation `[M, D]` to quantize.
  :param y: Contiguous 2D FP8 destination `[M, D + 2]`. The trailing bytes of every row keep the
    per-token scale.
  """

  _get_required_ops_module().fp8_comm_per_token_quant(x, y)


def _call_fp8_comm_per_token_dequant(x: torch.Tensor, y: torch.Tensor) -> None:
  """Bridge the per-token FP8 dequantizer to the compiled extension.

  :param x: Contiguous 2D FP8 input `[M, D + 2]` packed by the per-token quantizer.
  :param y: Contiguous 2D BF16 destination `[M, D]`.
  """

  _get_required_ops_module().fp8_comm_per_token_dequant(x, y)


def _call_fp8_comm_qkv_permute_quant(x: torch.Tensor, y: torch.Tensor) -> None:
  """Bridge the QKV permuting FP8 quantizer to the compiled extension.

  :param x: 5D BF16 QKV activation `[B, S, P, N, D]` whose last dimension has stride 1.
  :param y: Contiguous 5D FP8 destination `[P, S, B, N, D + 4]`.
  """

  _get_required_ops_module().fp8_comm_qkv_permute_quant(x, y)


def _call_fp8_comm_qkv_permute_dequant(quant_x: torch.Tensor, x: torch.Tensor) -> None:
  """Bridge the QKV restoring FP8 dequantizer to the compiled extension.

  :param quant_x: 4D FP8 quantized activation `[S, B, N, D + 4]` whose last dimension has stride 1.
  :param x: Contiguous 4D BF16 destination `[B, S, N, D]` restored to the original QKV layout.
  """

  _get_required_ops_module().fp8_comm_qkv_permute_dequant(quant_x, x)


torch.library.define(
  "cache_dit_cuda_ops::fp8_comm_per_token_quant",
  "(Tensor x) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::fp8_comm_per_token_quant", "CUDA")
def _fp8_comm_per_token_quant(x: torch.Tensor) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  *shape, head_size = x.shape
  x = x.reshape(-1, head_size).contiguous()
  rows, cols = x.shape
  quant_x = torch.empty((rows, cols + 2), dtype=torch.float8_e4m3fn, device=x.device)
  _call_fp8_comm_per_token_quant(x, quant_x)
  return quant_x.reshape(*shape, head_size + 2)


@torch.library.register_fake("cache_dit_cuda_ops::fp8_comm_per_token_quant")
def _fake_fp8_comm_per_token_quant(x: torch.Tensor) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  *shape, head_size = x.shape
  return x.new_empty((*shape, head_size + 2), dtype=torch.float8_e4m3fn)


torch.library.define(
  "cache_dit_cuda_ops::fp8_comm_per_token_dequant",
  "(Tensor x) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::fp8_comm_per_token_dequant", "CUDA")
def _fp8_comm_per_token_dequant(x: torch.Tensor) -> torch.Tensor:
  if x.dtype != torch.float8_e4m3fn:
    raise ValueError(f"Expected a float8_e4m3fn activation, got {x.dtype}.")
  *shape, packed_head_size = x.shape
  x = x.reshape(-1, packed_head_size).contiguous()
  rows, cols = x.shape
  dequant_x = torch.empty((rows, cols - 2), dtype=torch.bfloat16, device=x.device)
  _call_fp8_comm_per_token_dequant(x, dequant_x)
  return dequant_x.reshape(*shape, packed_head_size - 2)


@torch.library.register_fake("cache_dit_cuda_ops::fp8_comm_per_token_dequant")
def _fake_fp8_comm_per_token_dequant(x: torch.Tensor) -> torch.Tensor:
  if x.dtype != torch.float8_e4m3fn:
    raise ValueError(f"Expected a float8_e4m3fn activation, got {x.dtype}.")
  *shape, packed_head_size = x.shape
  return x.new_empty((*shape, packed_head_size - 2), dtype=torch.bfloat16)


torch.library.define(
  "cache_dit_cuda_ops::fp8_comm_qkv_permute_quant",
  "(Tensor x) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::fp8_comm_qkv_permute_quant", "CUDA")
def _fp8_comm_qkv_permute_quant(x: torch.Tensor) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  batch, seq_len, partitions, num_heads, head_size = x.shape
  quant_x = torch.empty((partitions, seq_len, batch, num_heads, head_size + 4),
                        dtype=torch.float8_e4m3fn,
                        device=x.device)
  _call_fp8_comm_qkv_permute_quant(x, quant_x)
  return quant_x


@torch.library.register_fake("cache_dit_cuda_ops::fp8_comm_qkv_permute_quant")
def _fake_fp8_comm_qkv_permute_quant(x: torch.Tensor) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  batch, seq_len, partitions, num_heads, head_size = x.shape
  return x.new_empty((partitions, seq_len, batch, num_heads, head_size + 4),
                     dtype=torch.float8_e4m3fn)


torch.library.define(
  "cache_dit_cuda_ops::fp8_comm_qkv_permute_dequant",
  "(Tensor quant_x) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::fp8_comm_qkv_permute_dequant", "CUDA")
def _fp8_comm_qkv_permute_dequant(quant_x: torch.Tensor) -> torch.Tensor:
  if quant_x.dtype != torch.float8_e4m3fn:
    raise ValueError(f"Expected a float8_e4m3fn activation, got {quant_x.dtype}.")
  seq_len, batch, num_heads, packed_head_size = quant_x.shape
  head_size = packed_head_size - 4
  # The dequantized activation is consumed as a contiguous `[B, S, N, D]` buffer.
  x = torch.empty((batch, seq_len, num_heads, head_size),
                  dtype=torch.bfloat16,
                  device=quant_x.device)
  _call_fp8_comm_qkv_permute_dequant(quant_x, x)
  return x


@torch.library.register_fake("cache_dit_cuda_ops::fp8_comm_qkv_permute_dequant")
def _fake_fp8_comm_qkv_permute_dequant(quant_x: torch.Tensor) -> torch.Tensor:
  if quant_x.dtype != torch.float8_e4m3fn:
    raise ValueError(f"Expected a float8_e4m3fn activation, got {quant_x.dtype}.")
  seq_len, batch, num_heads, packed_head_size = quant_x.shape
  return quant_x.new_empty((batch, seq_len, num_heads, packed_head_size - 4), dtype=torch.bfloat16)


def fp8_comm_per_token_quant(x: torch.Tensor) -> torch.Tensor:
  """Quantize a floating-point tensor into the FP8 per-token communication layout.

  :param x: Dense BF16 activation to quantize; the last dimension is the head dimension that gains
    two trailing scale columns.
  :returns: The quantized FP8 activation, keeping the leading dimensions and adding two scale
    columns to the last one.
  """

  return torch.ops.cache_dit_cuda_ops.fp8_comm_per_token_quant(x)


def fp8_comm_per_token_dequant(x: torch.Tensor) -> torch.Tensor:
  """Dequantize a FP8 per-token communication tensor back to floating-point format.

  :param x: FP8 activation packed by `fp8_comm_per_token_quant`.
  :returns: The dequantized BF16 activation, with the two trailing scale columns removed
    from the last dimension.
  """

  return torch.ops.cache_dit_cuda_ops.fp8_comm_per_token_dequant(x)


def fp8_comm_qkv_permute_quant(x: torch.Tensor) -> torch.Tensor:
  """Quantize a QKV tensor into FP8 while permuting it for communication.

  :param x: Dense BF16 QKV activation shaped `[B, S, P, N, D]`, whose last dimension must have
    stride 1.
  :returns: The quantized FP8 activation shaped `[P, S, B, N, D + 4]`.
  """

  return torch.ops.cache_dit_cuda_ops.fp8_comm_qkv_permute_quant(x)


def fp8_comm_qkv_permute_dequant(quant_x: torch.Tensor) -> torch.Tensor:
  """Dequantize a permuted FP8 QKV tensor and restore its original layout.

  :param quant_x: FP8 activation shaped `[S, B, N, D + 4]` packed by
    `fp8_comm_qkv_permute_quant`, whose last dimension must have stride 1.
  :returns: The dequantized BF16 activation shaped `[B, S, N, D]` in a contiguous layout.
  """

  return torch.ops.cache_dit_cuda_ops.fp8_comm_qkv_permute_dequant(quant_x)


__all__ = [
  "_call_fp8_comm_per_token_dequant",
  "_call_fp8_comm_per_token_quant",
  "_call_fp8_comm_qkv_permute_dequant",
  "_call_fp8_comm_qkv_permute_quant",
  "fp8_comm_per_token_quant",
  "fp8_comm_per_token_dequant",
  "fp8_comm_qkv_permute_quant",
  "fp8_comm_qkv_permute_dequant",
]
