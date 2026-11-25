import torch

from .cuda_ext import _get_required_ops_module


def _validate_fp8_group_size(group_size: int) -> None:
  if group_size not in (0, 16, 32):
    raise ValueError("group_size must be 0 (per-token), 16, or 32.")


def _fp8_packed_min_width(head_size: int, group_size: int, scale_bytes: int) -> int:
  _validate_fp8_group_size(group_size)
  groups = 1 if group_size == 0 else (head_size + group_size - 1) // group_size
  return head_size + groups * scale_bytes


def _fp8_resolve_head_size(packed_head_size: int, head_size: int | None, group_size: int,
                           scale_bytes: int) -> int:
  _validate_fp8_group_size(group_size)
  if head_size is not None:
    return head_size
  if group_size != 0:
    raise ValueError("head_size must be explicit for per-group dequantization.")
  return packed_head_size - scale_bytes


def _call_ulysses_fp8_per_token_quant(x: torch.Tensor, y: torch.Tensor, packed_head_size: int,
                                      group_size: int) -> None:
  """Bridge the FP8 quantizer to the compiled extension.

  :param x: Contiguous 2D BF16 activation `[M, D]` to quantize.
  :param y: Contiguous 2D FP8 destination `[M, packed_head_size]`.
  :param packed_head_size: Packed row width, including scales and alignment padding.
  :param group_size: Elements per scale group; zero selects per-token quantization.
  """

  ops = _get_required_ops_module()
  if group_size == 0:
    ops.ulysses_fp8_per_token_quant(x, y, packed_head_size)
  else:
    ops.ulysses_fp8_per_token_quant(x, y, packed_head_size, group_size)


def _call_ulysses_fp8_per_token_dequant(x: torch.Tensor, y: torch.Tensor, head_size: int,
                                        group_size: int) -> None:
  """Bridge the FP8 dequantizer to the compiled extension.

  :param x: Contiguous 2D FP8 input `[M, packed_head_size]` packed by the quantizer.
  :param y: Contiguous 2D BF16 destination `[M, D]`.
  :param head_size: Original attention head dimension `D`.
  :param group_size: Elements per scale group; zero selects per-token dequantization.
  """

  ops = _get_required_ops_module()
  if group_size == 0:
    ops.ulysses_fp8_per_token_dequant(x, y, head_size)
  else:
    ops.ulysses_fp8_per_token_dequant(x, y, head_size, group_size)


def _call_ulysses_fp8_qkv_permute_quant(x: torch.Tensor, y: torch.Tensor, packed_head_size: int,
                                        group_size: int) -> None:
  """Bridge the QKV permuting FP8 quantizer to the compiled extension.

  :param x: 5D BF16 QKV activation `[B, S, P, N, D]` whose last dimension has stride 1.
  :param y: Contiguous 5D FP8 destination `[P, S, B, N, packed_head_size]`.
  :param packed_head_size: Packed row width, including FP32 scales and alignment padding.
  :param group_size: Elements per scale group; zero selects per-token quantization.
  """

  ops = _get_required_ops_module()
  if group_size == 0:
    ops.ulysses_fp8_qkv_permute_quant(x, y, packed_head_size)
  else:
    ops.ulysses_fp8_qkv_permute_quant(x, y, packed_head_size, group_size)


def _call_ulysses_fp8_qkv_permute_dequant(quant_x: torch.Tensor, x: torch.Tensor, head_size: int,
                                          group_size: int) -> None:
  """Bridge the QKV restoring FP8 dequantizer to the compiled extension.

  :param quant_x: 4D FP8 activation `[S, B, N, packed_head_size]`, last dimension stride 1.
  :param x: Contiguous 4D BF16 destination `[B, S, N, D]` restored to the original QKV layout.
  :param head_size: Original attention head dimension `D`.
  :param group_size: Elements per scale group; zero selects per-token dequantization.
  """

  ops = _get_required_ops_module()
  if group_size == 0:
    ops.ulysses_fp8_qkv_permute_dequant(quant_x, x, head_size)
  else:
    ops.ulysses_fp8_qkv_permute_dequant(quant_x, x, head_size, group_size)


def _call_ulysses_fp8_o_permute_quant(x: torch.Tensor, y: torch.Tensor, partitions: int,
                                      packed_head_size: int, group_size: int) -> None:
  """Bridge the O-direction permuting FP8 quantizer to the compiled extension.

  :param x: BF16 activation `[B, S_GLOBAL, H_LOCAL, D]` whose last dimension has stride 1.
  :param y: Contiguous FP8 destination `[P, H_LOCAL, B, S_LOCAL, packed_head_size]`.
  :param partitions: Number of sequence partitions `P`.
  :param packed_head_size: Padded row width, including BF16 scales.
  :param group_size: Elements per scale group; zero selects per-token quantization.
  """

  ops = _get_required_ops_module()
  if group_size == 0:
    ops.ulysses_fp8_o_permute_quant(x, y, partitions, packed_head_size)
  else:
    ops.ulysses_fp8_o_permute_quant(x, y, partitions, packed_head_size, group_size)


def _call_ulysses_fp8_o_permute_dequant(quant_x: torch.Tensor, x: torch.Tensor, head_size: int,
                                        group_size: int) -> None:
  """Bridge the O-direction restoring FP8 dequantizer to the compiled extension.

  :param quant_x: FP8 activation `[H_GLOBAL, B, S_LOCAL, packed_head_size]` with stride 1 last.
  :param x: Contiguous BF16 destination `[B, S_LOCAL, H_GLOBAL, head_size]`.
  :param head_size: Original attention head dimension `D`.
  :param group_size: Elements per scale group; zero selects per-token dequantization.
  """

  ops = _get_required_ops_module()
  if group_size == 0:
    ops.ulysses_fp8_o_permute_dequant(quant_x, x, head_size)
  else:
    ops.ulysses_fp8_o_permute_dequant(quant_x, x, head_size, group_size)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp8_per_token_quant",
  "(Tensor x, int? packed_head_size=None, int group_size=0) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp8_per_token_quant", "CUDA")
def _ulysses_fp8_per_token_quant(x: torch.Tensor,
                                 packed_head_size: int | None = None,
                                 group_size: int = 0) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  *shape, head_size = x.shape
  if packed_head_size is None:
    packed_head_size = _fp8_packed_min_width(head_size, group_size, 2)
  x = x.reshape(-1, head_size).contiguous()
  rows, _ = x.shape
  quant_x = torch.empty((rows, packed_head_size), dtype=torch.float8_e4m3fn, device=x.device)
  _call_ulysses_fp8_per_token_quant(x, quant_x, packed_head_size, group_size)
  return quant_x.reshape(*shape, packed_head_size)


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp8_per_token_quant")
def _fake_ulysses_fp8_per_token_quant(x: torch.Tensor,
                                      packed_head_size: int | None = None,
                                      group_size: int = 0) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  *shape, head_size = x.shape
  if packed_head_size is None:
    packed_head_size = _fp8_packed_min_width(head_size, group_size, 2)
  return x.new_empty((*shape, packed_head_size), dtype=torch.float8_e4m3fn)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp8_per_token_dequant",
  "(Tensor x, int? head_size=None, int group_size=0) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp8_per_token_dequant", "CUDA")
def _ulysses_fp8_per_token_dequant(x: torch.Tensor,
                                   head_size: int | None = None,
                                   group_size: int = 0) -> torch.Tensor:
  if x.dtype != torch.float8_e4m3fn:
    raise ValueError(f"Expected a float8_e4m3fn activation, got {x.dtype}.")
  *shape, packed_head_size = x.shape
  head_size = _fp8_resolve_head_size(packed_head_size, head_size, group_size, 2)
  x = x.reshape(-1, packed_head_size).contiguous()
  rows, _ = x.shape
  dequant_x = torch.empty((rows, head_size), dtype=torch.bfloat16, device=x.device)
  _call_ulysses_fp8_per_token_dequant(x, dequant_x, head_size, group_size)
  return dequant_x.reshape(*shape, head_size)


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp8_per_token_dequant")
def _fake_ulysses_fp8_per_token_dequant(x: torch.Tensor,
                                        head_size: int | None = None,
                                        group_size: int = 0) -> torch.Tensor:
  if x.dtype != torch.float8_e4m3fn:
    raise ValueError(f"Expected a float8_e4m3fn activation, got {x.dtype}.")
  *shape, packed_head_size = x.shape
  head_size = _fp8_resolve_head_size(packed_head_size, head_size, group_size, 2)
  return x.new_empty((*shape, head_size), dtype=torch.bfloat16)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp8_qkv_permute_quant",
  "(Tensor x, int? packed_head_size=None, int group_size=0) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp8_qkv_permute_quant", "CUDA")
def _ulysses_fp8_qkv_permute_quant(x: torch.Tensor,
                                   packed_head_size: int | None = None,
                                   group_size: int = 0) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  batch, seq_len, partitions, num_heads, head_size = x.shape
  if packed_head_size is None:
    packed_head_size = _fp8_packed_min_width(head_size, group_size, 4)
  quant_x = torch.empty((partitions, seq_len, batch, num_heads, packed_head_size),
                        dtype=torch.float8_e4m3fn,
                        device=x.device)
  _call_ulysses_fp8_qkv_permute_quant(x, quant_x, packed_head_size, group_size)
  return quant_x


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp8_qkv_permute_quant")
def _fake_ulysses_fp8_qkv_permute_quant(x: torch.Tensor,
                                        packed_head_size: int | None = None,
                                        group_size: int = 0) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  batch, seq_len, partitions, num_heads, head_size = x.shape
  if packed_head_size is None:
    packed_head_size = _fp8_packed_min_width(head_size, group_size, 4)
  return x.new_empty((partitions, seq_len, batch, num_heads, packed_head_size),
                     dtype=torch.float8_e4m3fn)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp8_qkv_permute_dequant",
  "(Tensor quant_x, int? head_size=None, int group_size=0) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp8_qkv_permute_dequant", "CUDA")
def _ulysses_fp8_qkv_permute_dequant(quant_x: torch.Tensor,
                                     head_size: int | None = None,
                                     group_size: int = 0) -> torch.Tensor:
  if quant_x.dtype != torch.float8_e4m3fn:
    raise ValueError(f"Expected a float8_e4m3fn activation, got {quant_x.dtype}.")
  seq_len, batch, num_heads, packed_head_size = quant_x.shape
  head_size = _fp8_resolve_head_size(packed_head_size, head_size, group_size, 4)
  # The dequantized activation is consumed as a contiguous `[B, S, N, D]` buffer.
  x = torch.empty((batch, seq_len, num_heads, head_size),
                  dtype=torch.bfloat16,
                  device=quant_x.device)
  _call_ulysses_fp8_qkv_permute_dequant(quant_x, x, head_size, group_size)
  return x


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp8_qkv_permute_dequant")
def _fake_ulysses_fp8_qkv_permute_dequant(quant_x: torch.Tensor,
                                          head_size: int | None = None,
                                          group_size: int = 0) -> torch.Tensor:
  if quant_x.dtype != torch.float8_e4m3fn:
    raise ValueError(f"Expected a float8_e4m3fn activation, got {quant_x.dtype}.")
  seq_len, batch, num_heads, packed_head_size = quant_x.shape
  head_size = _fp8_resolve_head_size(packed_head_size, head_size, group_size, 4)
  return quant_x.new_empty((batch, seq_len, num_heads, head_size), dtype=torch.bfloat16)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp8_o_permute_quant",
  "(Tensor x, int partitions, int packed_head_size, int group_size=0) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp8_o_permute_quant", "CUDA")
def _ulysses_fp8_o_permute_quant(x: torch.Tensor,
                                 partitions: int,
                                 packed_head_size: int,
                                 group_size: int = 0) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  _validate_fp8_group_size(group_size)
  batch, seq_len, num_heads, _ = x.shape
  quant_x = torch.empty((partitions, num_heads, batch, seq_len // partitions, packed_head_size),
                        dtype=torch.float8_e4m3fn,
                        device=x.device)
  _call_ulysses_fp8_o_permute_quant(x, quant_x, partitions, packed_head_size, group_size)
  return quant_x


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp8_o_permute_quant")
def _fake_ulysses_fp8_o_permute_quant(x: torch.Tensor,
                                      partitions: int,
                                      packed_head_size: int,
                                      group_size: int = 0) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  _validate_fp8_group_size(group_size)
  batch, seq_len, num_heads, _ = x.shape
  return x.new_empty((partitions, num_heads, batch, seq_len // partitions, packed_head_size),
                     dtype=torch.float8_e4m3fn)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp8_o_permute_dequant",
  "(Tensor quant_x, int head_size, int group_size=0) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp8_o_permute_dequant", "CUDA")
def _ulysses_fp8_o_permute_dequant(quant_x: torch.Tensor,
                                   head_size: int,
                                   group_size: int = 0) -> torch.Tensor:
  if quant_x.dtype != torch.float8_e4m3fn:
    raise ValueError(f"Expected a float8_e4m3fn activation, got {quant_x.dtype}.")
  _validate_fp8_group_size(group_size)
  num_heads, batch, seq_len, _ = quant_x.shape
  x = torch.empty((batch, seq_len, num_heads, head_size),
                  dtype=torch.bfloat16,
                  device=quant_x.device)
  _call_ulysses_fp8_o_permute_dequant(quant_x, x, head_size, group_size)
  return x


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp8_o_permute_dequant")
def _fake_ulysses_fp8_o_permute_dequant(quant_x: torch.Tensor,
                                        head_size: int,
                                        group_size: int = 0) -> torch.Tensor:
  if quant_x.dtype != torch.float8_e4m3fn:
    raise ValueError(f"Expected a float8_e4m3fn activation, got {quant_x.dtype}.")
  _validate_fp8_group_size(group_size)
  num_heads, batch, seq_len, _ = quant_x.shape
  return quant_x.new_empty((batch, seq_len, num_heads, head_size), dtype=torch.bfloat16)


def ulysses_fp8_per_token_quant(x: torch.Tensor,
                                packed_head_size: int | None = None,
                                group_size: int = 0) -> torch.Tensor:
  """Quantize a BF16 tensor into the FP8 communication layout.

  :param x: Dense BF16 activation to quantize.
  :param packed_head_size: Optional padded row width; defaults to `D + ceil(D/G) * 2` bytes of
    payload and BF16 scales. Trailing alignment padding is zeroed.
  :param group_size: Elements per scale group, 16 or 32; zero selects per-token quantization.
  :returns: The quantized FP8 activation with the requested packed row width.
  """

  return torch.ops.cache_dit_cuda_ops.ulysses_fp8_per_token_quant(x, packed_head_size, group_size)


def ulysses_fp8_per_token_dequant(x: torch.Tensor,
                                  head_size: int | None = None,
                                  group_size: int = 0) -> torch.Tensor:
  """Dequantize an FP8 communication tensor back to BF16.

  :param x: FP8 activation packed by `ulysses_fp8_per_token_quant`.
  :param head_size: Original head dimension `D`; required for per-group input and padded layouts.
    The per-token unpadded default remains `packed_head_size - 2`.
  :param group_size: Elements per scale group, 16 or 32; zero selects per-token dequantization.
  :returns: The dequantized BF16 activation with the original head dimension.
  """

  return torch.ops.cache_dit_cuda_ops.ulysses_fp8_per_token_dequant(x, head_size, group_size)


def ulysses_fp8_qkv_permute_quant(x: torch.Tensor,
                                  packed_head_size: int | None = None,
                                  group_size: int = 0) -> torch.Tensor:
  """Quantize a QKV tensor into FP8 while permuting it for communication.

  :param x: Dense BF16 QKV activation shaped `[B, S, P, N, D]`, whose last dimension must have
    stride 1.
  :param packed_head_size: Optional padded row width; defaults to `D + ceil(D/G) * 4` bytes of
    payload and FP32 scales. Trailing alignment padding is zeroed.
  :param group_size: Elements per scale group, 16 or 32; zero selects per-token quantization.
  :returns: The quantized FP8 activation shaped `[P, S, B, N, packed_head_size]`.
  """

  return torch.ops.cache_dit_cuda_ops.ulysses_fp8_qkv_permute_quant(x, packed_head_size, group_size)


def ulysses_fp8_qkv_permute_dequant(quant_x: torch.Tensor,
                                    head_size: int | None = None,
                                    group_size: int = 0) -> torch.Tensor:
  """Dequantize a permuted FP8 QKV tensor and restore its original layout.

  :param quant_x: FP8 activation shaped `[S, B, N, packed_head_size]`, last dimension stride 1.
  :param head_size: Original head dimension `D`; required for per-group input and padded layouts.
    The per-token unpadded default remains `packed_head_size - 4`.
  :param group_size: Elements per scale group, 16 or 32; zero selects per-token dequantization.
  :returns: The dequantized BF16 activation shaped `[B, S, N, D]` in a contiguous layout.
  """

  return torch.ops.cache_dit_cuda_ops.ulysses_fp8_qkv_permute_dequant(quant_x, head_size,
                                                                      group_size)


def ulysses_fp8_o_permute_quant(x: torch.Tensor,
                                partitions: int,
                                packed_head_size: int,
                                group_size: int = 0) -> torch.Tensor:
  """Quantize an attention output into FP8 while permuting it for all-to-all.

  :param x: BF16 activation `[B, S_GLOBAL, H_LOCAL, D]`, whose last dimension has stride 1.
  :param partitions: Number of sequence partitions `P`.
  :param packed_head_size: Padded row width with space for all BF16 group scales.
  :param group_size: Elements per scale group, 16 or 32; zero selects per-token quantization.
  :returns: FP8 activation `[P, H_LOCAL, B, S_LOCAL, packed_head_size]` with zeroed padding.
  """

  return torch.ops.cache_dit_cuda_ops.ulysses_fp8_o_permute_quant(x, partitions, packed_head_size,
                                                                  group_size)


def ulysses_fp8_o_permute_dequant(quant_x: torch.Tensor,
                                  head_size: int,
                                  group_size: int = 0) -> torch.Tensor:
  """Dequantize an all-to-all output and restore the attention output layout.

  :param quant_x: FP8 activation `[H_GLOBAL, B, S_LOCAL, packed_head_size]`, stride 1 last.
  :param head_size: Original attention head dimension `D`; trailing packed padding is ignored.
  :param group_size: Elements per scale group, 16 or 32; zero selects per-token dequantization.
  :returns: Contiguous BF16 activation `[B, S_LOCAL, H_GLOBAL, D]`.
  """

  return torch.ops.cache_dit_cuda_ops.ulysses_fp8_o_permute_dequant(quant_x, head_size, group_size)


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
