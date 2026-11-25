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


def _validate_fp6_group_size(group_size: int) -> None:
  if group_size not in (0, 16, 32):
    raise ValueError("group_size must be 0 (per-token), 16, or 32.")


def _fp6_packed_min_width(head_size: int, group_size: int, scale_bytes: int) -> int:
  _validate_fp6_group_size(group_size)
  bitstream_bytes = ((head_size + 3) // 4) * 3
  payload_bytes = (bitstream_bytes + 3) // 4 * 4
  groups = 1 if group_size == 0 else (head_size + group_size - 1) // group_size
  return (payload_bytes + groups * scale_bytes + 3) // 4 * 4


def _validate_fp4_group_size(group_size: int) -> None:
  if group_size != 16:
    raise ValueError("FP4 communication supports only group_size=16.")


def _fp4_packed_min_width(head_size: int, group_size: int, scale_bytes: int) -> int:
  _validate_fp4_group_size(group_size)
  groups = (head_size + group_size - 1) // group_size
  return (head_size + 1) // 2 + groups * scale_bytes


def _call_ulysses_fp4_per_token_quant(x: torch.Tensor, y: torch.Tensor, packed_head_size: int,
                                      group_size: int) -> None:
  _get_required_ops_module().ulysses_fp4_per_token_quant(x, y, packed_head_size, group_size)


def _call_ulysses_fp4_per_token_dequant(x: torch.Tensor, y: torch.Tensor, head_size: int,
                                        group_size: int) -> None:
  _get_required_ops_module().ulysses_fp4_per_token_dequant(x, y, head_size, group_size)


def _call_ulysses_fp4_qkv_permute_quant(x: torch.Tensor, y: torch.Tensor, packed_head_size: int,
                                        group_size: int) -> None:
  _get_required_ops_module().ulysses_fp4_qkv_permute_quant(x, y, packed_head_size, group_size)


def _call_ulysses_fp4_qkv_permute_dequant(quant_x: torch.Tensor, x: torch.Tensor, head_size: int,
                                          group_size: int) -> None:
  _get_required_ops_module().ulysses_fp4_qkv_permute_dequant(quant_x, x, head_size, group_size)


def _call_ulysses_fp4_o_permute_quant(x: torch.Tensor, y: torch.Tensor, partitions: int,
                                      packed_head_size: int, group_size: int) -> None:
  _get_required_ops_module().ulysses_fp4_o_permute_quant(x, y, partitions, packed_head_size,
                                                         group_size)


def _call_ulysses_fp4_o_permute_dequant(quant_x: torch.Tensor, x: torch.Tensor, head_size: int,
                                        group_size: int) -> None:
  _get_required_ops_module().ulysses_fp4_o_permute_dequant(quant_x, x, head_size, group_size)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp4_per_token_quant",
  "(Tensor x, int? packed_head_size=None, int group_size=16) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp4_per_token_quant", "CUDA")
def _ulysses_fp4_per_token_quant(x: torch.Tensor,
                                 packed_head_size: int | None = None,
                                 group_size: int = 16) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  _validate_fp4_group_size(group_size)
  *shape, head_size = x.shape
  if packed_head_size is None:
    packed_head_size = _fp4_packed_min_width(head_size, group_size, 2)
  x = x.reshape(-1, head_size).contiguous()
  rows, _ = x.shape
  quant_x = torch.empty((rows, packed_head_size), dtype=torch.uint8, device=x.device)
  _call_ulysses_fp4_per_token_quant(x, quant_x, packed_head_size, group_size)
  return quant_x.reshape(*shape, packed_head_size)


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp4_per_token_quant")
def _fake_ulysses_fp4_per_token_quant(x: torch.Tensor,
                                      packed_head_size: int | None = None,
                                      group_size: int = 16) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  _validate_fp4_group_size(group_size)
  *shape, head_size = x.shape
  if packed_head_size is None:
    packed_head_size = _fp4_packed_min_width(head_size, group_size, 2)
  return x.new_empty((*shape, packed_head_size), dtype=torch.uint8)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp4_per_token_dequant",
  "(Tensor x, int head_size, int group_size) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp4_per_token_dequant", "CUDA")
def _ulysses_fp4_per_token_dequant(x: torch.Tensor, head_size: int,
                                   group_size: int) -> torch.Tensor:
  if x.dtype != torch.uint8:
    raise ValueError(f"Expected a packed uint8 activation, got {x.dtype}.")
  _validate_fp4_group_size(group_size)
  *shape, packed_head_size = x.shape
  x = x.reshape(-1, packed_head_size).contiguous()
  rows, _ = x.shape
  dequant_x = torch.empty((rows, head_size), dtype=torch.bfloat16, device=x.device)
  _call_ulysses_fp4_per_token_dequant(x, dequant_x, head_size, group_size)
  return dequant_x.reshape(*shape, head_size)


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp4_per_token_dequant")
def _fake_ulysses_fp4_per_token_dequant(x: torch.Tensor, head_size: int,
                                        group_size: int) -> torch.Tensor:
  if x.dtype != torch.uint8:
    raise ValueError(f"Expected a packed uint8 activation, got {x.dtype}.")
  _validate_fp4_group_size(group_size)
  *shape, _ = x.shape
  return x.new_empty((*shape, head_size), dtype=torch.bfloat16)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp4_qkv_permute_quant",
  "(Tensor x, int? packed_head_size=None, int group_size=16) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp4_qkv_permute_quant", "CUDA")
def _ulysses_fp4_qkv_permute_quant(x: torch.Tensor,
                                   packed_head_size: int | None = None,
                                   group_size: int = 16) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  _validate_fp4_group_size(group_size)
  batch, seq_len, partitions, num_heads, head_size = x.shape
  if packed_head_size is None:
    packed_head_size = _fp4_packed_min_width(head_size, group_size, 4)
  quant_x = torch.empty((partitions, seq_len, batch, num_heads, packed_head_size),
                        dtype=torch.uint8,
                        device=x.device)
  _call_ulysses_fp4_qkv_permute_quant(x, quant_x, packed_head_size, group_size)
  return quant_x


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp4_qkv_permute_quant")
def _fake_ulysses_fp4_qkv_permute_quant(x: torch.Tensor,
                                        packed_head_size: int | None = None,
                                        group_size: int = 16) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  _validate_fp4_group_size(group_size)
  batch, seq_len, partitions, num_heads, head_size = x.shape
  if packed_head_size is None:
    packed_head_size = _fp4_packed_min_width(head_size, group_size, 4)
  return x.new_empty((partitions, seq_len, batch, num_heads, packed_head_size), dtype=torch.uint8)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp4_qkv_permute_dequant",
  "(Tensor quant_x, int head_size, int group_size) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp4_qkv_permute_dequant", "CUDA")
def _ulysses_fp4_qkv_permute_dequant(quant_x: torch.Tensor, head_size: int,
                                     group_size: int) -> torch.Tensor:
  if quant_x.dtype != torch.uint8:
    raise ValueError(f"Expected a packed uint8 activation, got {quant_x.dtype}.")
  _validate_fp4_group_size(group_size)
  seq_len, batch, num_heads, _ = quant_x.shape
  x = torch.empty((batch, seq_len, num_heads, head_size),
                  dtype=torch.bfloat16,
                  device=quant_x.device)
  _call_ulysses_fp4_qkv_permute_dequant(quant_x, x, head_size, group_size)
  return x


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp4_qkv_permute_dequant")
def _fake_ulysses_fp4_qkv_permute_dequant(quant_x: torch.Tensor, head_size: int,
                                          group_size: int) -> torch.Tensor:
  if quant_x.dtype != torch.uint8:
    raise ValueError(f"Expected a packed uint8 activation, got {quant_x.dtype}.")
  _validate_fp4_group_size(group_size)
  seq_len, batch, num_heads, _ = quant_x.shape
  return quant_x.new_empty((batch, seq_len, num_heads, head_size), dtype=torch.bfloat16)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp4_o_permute_quant",
  "(Tensor x, int partitions, int packed_head_size, int group_size=16) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp4_o_permute_quant", "CUDA")
def _ulysses_fp4_o_permute_quant(x: torch.Tensor,
                                 partitions: int,
                                 packed_head_size: int,
                                 group_size: int = 16) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  _validate_fp4_group_size(group_size)
  batch, seq_len, num_heads, _ = x.shape
  quant_x = torch.empty((partitions, num_heads, batch, seq_len // partitions, packed_head_size),
                        dtype=torch.uint8,
                        device=x.device)
  _call_ulysses_fp4_o_permute_quant(x, quant_x, partitions, packed_head_size, group_size)
  return quant_x


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp4_o_permute_quant")
def _fake_ulysses_fp4_o_permute_quant(x: torch.Tensor,
                                      partitions: int,
                                      packed_head_size: int,
                                      group_size: int = 16) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  _validate_fp4_group_size(group_size)
  batch, seq_len, num_heads, _ = x.shape
  return x.new_empty((partitions, num_heads, batch, seq_len // partitions, packed_head_size),
                     dtype=torch.uint8)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp4_o_permute_dequant",
  "(Tensor quant_x, int head_size, int group_size) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp4_o_permute_dequant", "CUDA")
def _ulysses_fp4_o_permute_dequant(quant_x: torch.Tensor, head_size: int,
                                   group_size: int) -> torch.Tensor:
  if quant_x.dtype != torch.uint8:
    raise ValueError(f"Expected a packed uint8 activation, got {quant_x.dtype}.")
  _validate_fp4_group_size(group_size)
  num_heads, batch, seq_len, _ = quant_x.shape
  x = torch.empty((batch, seq_len, num_heads, head_size),
                  dtype=torch.bfloat16,
                  device=quant_x.device)
  _call_ulysses_fp4_o_permute_dequant(quant_x, x, head_size, group_size)
  return x


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp4_o_permute_dequant")
def _fake_ulysses_fp4_o_permute_dequant(quant_x: torch.Tensor, head_size: int,
                                        group_size: int) -> torch.Tensor:
  if quant_x.dtype != torch.uint8:
    raise ValueError(f"Expected a packed uint8 activation, got {quant_x.dtype}.")
  _validate_fp4_group_size(group_size)
  num_heads, batch, seq_len, _ = quant_x.shape
  return quant_x.new_empty((batch, seq_len, num_heads, head_size), dtype=torch.bfloat16)


def ulysses_fp4_per_token_quant(x: torch.Tensor,
                                packed_head_size: int | None = None,
                                group_size: int = 16) -> torch.Tensor:
  """Quantize BF16 into packed E2M1 rows with one BF16 scale per 16 values."""
  _validate_fp4_group_size(group_size)
  return torch.ops.cache_dit_cuda_ops.ulysses_fp4_per_token_quant(x, packed_head_size, group_size)


def ulysses_fp4_per_token_dequant(x: torch.Tensor, head_size: int, group_size: int) -> torch.Tensor:
  """Dequantize packed generic FP4 rows to BF16."""
  _validate_fp4_group_size(group_size)
  return torch.ops.cache_dit_cuda_ops.ulysses_fp4_per_token_dequant(x, head_size, group_size)


def ulysses_fp4_qkv_permute_quant(x: torch.Tensor,
                                  packed_head_size: int | None = None,
                                  group_size: int = 16) -> torch.Tensor:
  """Quantize and permute QKV into packed E2M1 rows with FP32 group scales."""
  _validate_fp4_group_size(group_size)
  return torch.ops.cache_dit_cuda_ops.ulysses_fp4_qkv_permute_quant(x, packed_head_size, group_size)


def ulysses_fp4_qkv_permute_dequant(quant_x: torch.Tensor, head_size: int,
                                    group_size: int) -> torch.Tensor:
  """Restore a packed QKV FP4 partition to contiguous BF16."""
  _validate_fp4_group_size(group_size)
  return torch.ops.cache_dit_cuda_ops.ulysses_fp4_qkv_permute_dequant(quant_x, head_size,
                                                                      group_size)


def ulysses_fp4_o_permute_quant(x: torch.Tensor,
                                partitions: int,
                                packed_head_size: int,
                                group_size: int = 16) -> torch.Tensor:
  """Quantize and permute attention output into packed E2M1 group-16 rows."""
  _validate_fp4_group_size(group_size)
  return torch.ops.cache_dit_cuda_ops.ulysses_fp4_o_permute_quant(x, partitions, packed_head_size,
                                                                  group_size)


def ulysses_fp4_o_permute_dequant(quant_x: torch.Tensor, head_size: int,
                                  group_size: int) -> torch.Tensor:
  """Restore packed O FP4 rows to contiguous BF16 attention layout."""
  _validate_fp4_group_size(group_size)
  return torch.ops.cache_dit_cuda_ops.ulysses_fp4_o_permute_dequant(quant_x, head_size, group_size)


def _resolve_fN_group_size(bits: int, group_size: int | None) -> int:
  if bits == 8:
    return 0 if group_size is None else group_size
  if bits == 6:
    return 32 if group_size is None else group_size
  if bits == 4:
    return 16 if group_size is None else group_size
  raise ValueError(f"bits must be 4, 6, or 8, got {bits}.")


def _call_ulysses_fp6_per_token_quant(x: torch.Tensor, y: torch.Tensor, packed_head_size: int,
                                      group_size: int) -> None:
  """Call the packed FP6 quantizer.

  :param x: Contiguous BF16 input.
  :param y: Contiguous uint8 packed output.
  :param packed_head_size: Packed row width in bytes.
  :param group_size: Elements per scale group; zero selects per-token quantization.
  """
  _get_required_ops_module().ulysses_fp6_per_token_quant(x, y, packed_head_size, group_size)


def _call_ulysses_fp6_per_token_dequant(x: torch.Tensor, y: torch.Tensor, head_size: int,
                                        group_size: int) -> None:
  """Call the packed FP6 dequantizer.

  :param x: Contiguous uint8 packed input.
  :param y: Contiguous BF16 output.
  :param head_size: Original head dimension `D`.
  :param group_size: Elements per scale group; zero selects per-token dequantization.
  """
  _get_required_ops_module().ulysses_fp6_per_token_dequant(x, y, head_size, group_size)


def _call_ulysses_fp6_qkv_permute_quant(x: torch.Tensor, y: torch.Tensor, packed_head_size: int,
                                        group_size: int) -> None:
  """Call the QKV permuting FP6 quantizer.

  :param x: BF16 QKV activation `[B, S, P, N, D]`.
  :param y: Contiguous uint8 packed output.
  :param packed_head_size: Packed row width in bytes.
  :param group_size: Elements per scale group; zero selects per-token quantization.
  """
  _get_required_ops_module().ulysses_fp6_qkv_permute_quant(x, y, packed_head_size, group_size)


def _call_ulysses_fp6_qkv_permute_dequant(quant_x: torch.Tensor, x: torch.Tensor, head_size: int,
                                          group_size: int) -> None:
  """Call the QKV restoring FP6 dequantizer.

  :param quant_x: Packed QKV activation `[S, B, N, packed_head_size]`.
  :param x: Contiguous BF16 output `[B, S, N, D]`.
  :param head_size: Original head dimension `D`.
  :param group_size: Elements per scale group; zero selects per-token dequantization.
  """
  _get_required_ops_module().ulysses_fp6_qkv_permute_dequant(quant_x, x, head_size, group_size)


def _call_ulysses_fp6_o_permute_quant(x: torch.Tensor, y: torch.Tensor, partitions: int,
                                      packed_head_size: int, group_size: int) -> None:
  """Call the O-direction permuting FP6 quantizer.

  :param x: BF16 activation `[B, S_GLOBAL, H_LOCAL, D]`.
  :param y: Contiguous uint8 packed output.
  :param partitions: Number of sequence partitions `P`.
  :param packed_head_size: Packed row width in bytes.
  :param group_size: Elements per scale group; zero selects per-token quantization.
  """
  _get_required_ops_module().ulysses_fp6_o_permute_quant(x, y, partitions, packed_head_size,
                                                         group_size)


def _call_ulysses_fp6_o_permute_dequant(quant_x: torch.Tensor, x: torch.Tensor, head_size: int,
                                        group_size: int) -> None:
  """Call the O-direction restoring FP6 dequantizer.

  :param quant_x: Packed O activation `[H_GLOBAL, B, S_LOCAL, packed_head_size]`.
  :param x: Contiguous BF16 output `[B, S_LOCAL, H_GLOBAL, D]`.
  :param head_size: Original head dimension `D`.
  :param group_size: Elements per scale group; zero selects per-token dequantization.
  """
  _get_required_ops_module().ulysses_fp6_o_permute_dequant(quant_x, x, head_size, group_size)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp6_per_token_quant",
  "(Tensor x, int? packed_head_size=None, int group_size=32) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp6_per_token_quant", "CUDA")
def _ulysses_fp6_per_token_quant(x: torch.Tensor,
                                 packed_head_size: int | None = None,
                                 group_size: int = 32) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  *shape, head_size = x.shape
  if packed_head_size is None:
    packed_head_size = _fp6_packed_min_width(head_size, group_size, 2)
  x = x.reshape(-1, head_size).contiguous()
  rows, _ = x.shape
  quant_x = torch.empty((rows, packed_head_size), dtype=torch.uint8, device=x.device)
  _call_ulysses_fp6_per_token_quant(x, quant_x, packed_head_size, group_size)
  return quant_x.reshape(*shape, packed_head_size)


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp6_per_token_quant")
def _fake_ulysses_fp6_per_token_quant(x: torch.Tensor,
                                      packed_head_size: int | None = None,
                                      group_size: int = 32) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  *shape, head_size = x.shape
  if packed_head_size is None:
    packed_head_size = _fp6_packed_min_width(head_size, group_size, 2)
  return x.new_empty((*shape, packed_head_size), dtype=torch.uint8)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp6_per_token_dequant",
  "(Tensor x, int head_size, int group_size) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp6_per_token_dequant", "CUDA")
def _ulysses_fp6_per_token_dequant(x: torch.Tensor, head_size: int,
                                   group_size: int) -> torch.Tensor:
  if x.dtype != torch.uint8:
    raise ValueError(f"Expected a packed uint8 activation, got {x.dtype}.")
  *shape, packed_head_size = x.shape
  x = x.reshape(-1, packed_head_size).contiguous()
  rows, _ = x.shape
  dequant_x = torch.empty((rows, head_size), dtype=torch.bfloat16, device=x.device)
  _call_ulysses_fp6_per_token_dequant(x, dequant_x, head_size, group_size)
  return dequant_x.reshape(*shape, head_size)


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp6_per_token_dequant")
def _fake_ulysses_fp6_per_token_dequant(x: torch.Tensor, head_size: int,
                                        group_size: int) -> torch.Tensor:
  if x.dtype != torch.uint8:
    raise ValueError(f"Expected a packed uint8 activation, got {x.dtype}.")
  *shape, _ = x.shape
  return x.new_empty((*shape, head_size), dtype=torch.bfloat16)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp6_qkv_permute_quant",
  "(Tensor x, int? packed_head_size=None, int group_size=32) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp6_qkv_permute_quant", "CUDA")
def _ulysses_fp6_qkv_permute_quant(x: torch.Tensor,
                                   packed_head_size: int | None = None,
                                   group_size: int = 32) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  batch, seq_len, partitions, num_heads, head_size = x.shape
  if packed_head_size is None:
    packed_head_size = _fp6_packed_min_width(head_size, group_size, 4)
  quant_x = torch.empty((partitions, seq_len, batch, num_heads, packed_head_size),
                        dtype=torch.uint8,
                        device=x.device)
  _call_ulysses_fp6_qkv_permute_quant(x, quant_x, packed_head_size, group_size)
  return quant_x


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp6_qkv_permute_quant")
def _fake_ulysses_fp6_qkv_permute_quant(x: torch.Tensor,
                                        packed_head_size: int | None = None,
                                        group_size: int = 32) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  batch, seq_len, partitions, num_heads, head_size = x.shape
  if packed_head_size is None:
    packed_head_size = _fp6_packed_min_width(head_size, group_size, 4)
  return x.new_empty((partitions, seq_len, batch, num_heads, packed_head_size), dtype=torch.uint8)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp6_qkv_permute_dequant",
  "(Tensor quant_x, int head_size, int group_size) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp6_qkv_permute_dequant", "CUDA")
def _ulysses_fp6_qkv_permute_dequant(quant_x: torch.Tensor, head_size: int,
                                     group_size: int) -> torch.Tensor:
  if quant_x.dtype != torch.uint8:
    raise ValueError(f"Expected a packed uint8 activation, got {quant_x.dtype}.")
  seq_len, batch, num_heads, _ = quant_x.shape
  x = torch.empty((batch, seq_len, num_heads, head_size),
                  dtype=torch.bfloat16,
                  device=quant_x.device)
  _call_ulysses_fp6_qkv_permute_dequant(quant_x, x, head_size, group_size)
  return x


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp6_qkv_permute_dequant")
def _fake_ulysses_fp6_qkv_permute_dequant(quant_x: torch.Tensor, head_size: int,
                                          group_size: int) -> torch.Tensor:
  if quant_x.dtype != torch.uint8:
    raise ValueError(f"Expected a packed uint8 activation, got {quant_x.dtype}.")
  seq_len, batch, num_heads, _ = quant_x.shape
  return quant_x.new_empty((batch, seq_len, num_heads, head_size), dtype=torch.bfloat16)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp6_o_permute_quant",
  "(Tensor x, int partitions, int packed_head_size, int group_size=32) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp6_o_permute_quant", "CUDA")
def _ulysses_fp6_o_permute_quant(x: torch.Tensor,
                                 partitions: int,
                                 packed_head_size: int,
                                 group_size: int = 32) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  _validate_fp6_group_size(group_size)
  batch, seq_len, num_heads, _ = x.shape
  quant_x = torch.empty((partitions, num_heads, batch, seq_len // partitions, packed_head_size),
                        dtype=torch.uint8,
                        device=x.device)
  _call_ulysses_fp6_o_permute_quant(x, quant_x, partitions, packed_head_size, group_size)
  return quant_x


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp6_o_permute_quant")
def _fake_ulysses_fp6_o_permute_quant(x: torch.Tensor,
                                      partitions: int,
                                      packed_head_size: int,
                                      group_size: int = 32) -> torch.Tensor:
  if x.dtype != torch.bfloat16:
    raise ValueError(f"Expected a bfloat16 activation, got {x.dtype}.")
  _validate_fp6_group_size(group_size)
  batch, seq_len, num_heads, _ = x.shape
  return x.new_empty((partitions, num_heads, batch, seq_len // partitions, packed_head_size),
                     dtype=torch.uint8)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fp6_o_permute_dequant",
  "(Tensor quant_x, int head_size, int group_size) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fp6_o_permute_dequant", "CUDA")
def _ulysses_fp6_o_permute_dequant(quant_x: torch.Tensor, head_size: int,
                                   group_size: int) -> torch.Tensor:
  if quant_x.dtype != torch.uint8:
    raise ValueError(f"Expected a packed uint8 activation, got {quant_x.dtype}.")
  _validate_fp6_group_size(group_size)
  num_heads, batch, seq_len, _ = quant_x.shape
  x = torch.empty((batch, seq_len, num_heads, head_size),
                  dtype=torch.bfloat16,
                  device=quant_x.device)
  _call_ulysses_fp6_o_permute_dequant(quant_x, x, head_size, group_size)
  return x


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fp6_o_permute_dequant")
def _fake_ulysses_fp6_o_permute_dequant(quant_x: torch.Tensor, head_size: int,
                                        group_size: int) -> torch.Tensor:
  if quant_x.dtype != torch.uint8:
    raise ValueError(f"Expected a packed uint8 activation, got {quant_x.dtype}.")
  _validate_fp6_group_size(group_size)
  num_heads, batch, seq_len, _ = quant_x.shape
  return quant_x.new_empty((batch, seq_len, num_heads, head_size), dtype=torch.bfloat16)


def ulysses_fp6_per_token_quant(x: torch.Tensor,
                                packed_head_size: int | None = None,
                                group_size: int = 32) -> torch.Tensor:
  """Quantize BF16 into packed E2M3 communication rows.

  :param x: Dense BF16 activation to quantize.
  :param packed_head_size: Optional packed row width in bytes.
  :param group_size: Elements per scale group; defaults to 32, zero selects per-token.
  :returns: Packed uint8 activation with four E2M3 codes per three bytes.
  """
  _validate_fp6_group_size(group_size)
  return torch.ops.cache_dit_cuda_ops.ulysses_fp6_per_token_quant(x, packed_head_size, group_size)


def ulysses_fp6_per_token_dequant(x: torch.Tensor, head_size: int, group_size: int) -> torch.Tensor:
  """Dequantize packed E2M3 communication rows to BF16.

  :param x: Packed uint8 activation.
  :param head_size: Original head dimension `D`.
  :param group_size: Elements per scale group; required to prevent scale-layout mismatches.
  :returns: Dequantized BF16 activation.
  """
  _validate_fp6_group_size(group_size)
  return torch.ops.cache_dit_cuda_ops.ulysses_fp6_per_token_dequant(x, head_size, group_size)


def ulysses_fp6_qkv_permute_quant(x: torch.Tensor,
                                  packed_head_size: int | None = None,
                                  group_size: int = 32) -> torch.Tensor:
  """Quantize and permute QKV into packed E2M3 communication rows.

  :param x: BF16 QKV activation `[B, S, P, N, D]`.
  :param packed_head_size: Optional packed row width in bytes.
  :param group_size: Elements per scale group; defaults to 32, zero selects per-token.
  :returns: Packed uint8 activation `[P, S, B, N, packed_head_size]`.
  """
  _validate_fp6_group_size(group_size)
  return torch.ops.cache_dit_cuda_ops.ulysses_fp6_qkv_permute_quant(x, packed_head_size, group_size)


def ulysses_fp6_qkv_permute_dequant(quant_x: torch.Tensor, head_size: int,
                                    group_size: int) -> torch.Tensor:
  """Dequantize packed QKV E2M3 rows and restore `[B, S, N, D]`.

  :param quant_x: Packed QKV activation `[S, B, N, packed_head_size]`.
  :param head_size: Original head dimension `D`.
  :param group_size: Elements per scale group; required to prevent scale-layout mismatches.
  :returns: Contiguous BF16 QKV activation.
  """
  _validate_fp6_group_size(group_size)
  return torch.ops.cache_dit_cuda_ops.ulysses_fp6_qkv_permute_dequant(quant_x, head_size,
                                                                      group_size)


def ulysses_fp6_o_permute_quant(x: torch.Tensor,
                                partitions: int,
                                packed_head_size: int,
                                group_size: int = 32) -> torch.Tensor:
  """Quantize and permute attention output into packed E2M3 rows.

  :param x: BF16 activation `[B, S_GLOBAL, H_LOCAL, D]`.
  :param partitions: Number of sequence partitions `P`.
  :param packed_head_size: Packed row width in bytes.
  :param group_size: Elements per scale group; defaults to 32, zero selects per-token.
  :returns: Packed uint8 activation `[P, H_LOCAL, B, S_LOCAL, packed_head_size]`.
  """
  _validate_fp6_group_size(group_size)
  return torch.ops.cache_dit_cuda_ops.ulysses_fp6_o_permute_quant(x, partitions, packed_head_size,
                                                                  group_size)


def ulysses_fp6_o_permute_dequant(quant_x: torch.Tensor, head_size: int,
                                  group_size: int) -> torch.Tensor:
  """Dequantize packed O E2M3 rows and restore the attention layout.

  :param quant_x: Packed O activation `[H_GLOBAL, B, S_LOCAL, packed_head_size]`.
  :param head_size: Original head dimension `D`.
  :param group_size: Elements per scale group; required to prevent scale-layout mismatches.
  :returns: Contiguous BF16 activation `[B, S_LOCAL, H_GLOBAL, D]`.
  """
  _validate_fp6_group_size(group_size)
  return torch.ops.cache_dit_cuda_ops.ulysses_fp6_o_permute_dequant(quant_x, head_size, group_size)


def ulysses_fN_per_token_quant(x: torch.Tensor,
                               packed_head_size: int | None = None,
                               group_size: int | None = None,
                               *,
                               bits: int = 8) -> torch.Tensor:
  """Dispatch generic FP4/FP6/FP8 quantization.

  FP8 defaults to per-token, FP6 to group-32, and FP4 only supports/defaults to group-16.
  """
  resolved_group = _resolve_fN_group_size(bits, group_size)
  if bits == 8:
    return ulysses_fp8_per_token_quant(x, packed_head_size, resolved_group)
  if bits == 4:
    return ulysses_fp4_per_token_quant(x, packed_head_size, resolved_group)
  return ulysses_fp6_per_token_quant(x, packed_head_size, resolved_group)


def ulysses_fN_per_token_dequant(x: torch.Tensor,
                                 head_size: int | None = None,
                                 group_size: int | None = None,
                                 *,
                                 bits: int = 8) -> torch.Tensor:
  """Dispatch generic FP4/FP6/FP8 dequantization to BF16."""
  resolved_group = _resolve_fN_group_size(bits, group_size)
  if bits == 8:
    return ulysses_fp8_per_token_dequant(x, head_size, resolved_group)
  if head_size is None:
    raise ValueError("head_size is required for FP4/FP6 dequantization.")
  if bits == 4:
    return ulysses_fp4_per_token_dequant(x, head_size, resolved_group)
  return ulysses_fp6_per_token_dequant(x, head_size, resolved_group)


def ulysses_fN_qkv_permute_quant(x: torch.Tensor,
                                 packed_head_size: int | None = None,
                                 group_size: int | None = None,
                                 *,
                                 bits: int = 8) -> torch.Tensor:
  """Dispatch QKV FP4/FP6/FP8 permute and quantization."""
  resolved_group = _resolve_fN_group_size(bits, group_size)
  if bits == 8:
    return ulysses_fp8_qkv_permute_quant(x, packed_head_size, resolved_group)
  if bits == 4:
    return ulysses_fp4_qkv_permute_quant(x, packed_head_size, resolved_group)
  return ulysses_fp6_qkv_permute_quant(x, packed_head_size, resolved_group)


def ulysses_fN_qkv_permute_dequant(quant_x: torch.Tensor,
                                   head_size: int | None = None,
                                   group_size: int | None = None,
                                   *,
                                   bits: int = 8) -> torch.Tensor:
  """Dispatch QKV FP4/FP6/FP8 dequantization and layout restoration."""
  resolved_group = _resolve_fN_group_size(bits, group_size)
  if bits == 8:
    return ulysses_fp8_qkv_permute_dequant(quant_x, head_size, resolved_group)
  if head_size is None:
    raise ValueError("head_size is required for FP4/FP6 dequantization.")
  if bits == 4:
    return ulysses_fp4_qkv_permute_dequant(quant_x, head_size, resolved_group)
  return ulysses_fp6_qkv_permute_dequant(quant_x, head_size, resolved_group)


def ulysses_fN_o_permute_quant(x: torch.Tensor,
                               partitions: int,
                               packed_head_size: int,
                               group_size: int | None = None,
                               *,
                               bits: int = 8) -> torch.Tensor:
  """Dispatch O-direction FP4/FP6/FP8 permute and quantization."""
  resolved_group = _resolve_fN_group_size(bits, group_size)
  if bits == 8:
    return ulysses_fp8_o_permute_quant(x, partitions, packed_head_size, resolved_group)
  if bits == 4:
    return ulysses_fp4_o_permute_quant(x, partitions, packed_head_size, resolved_group)
  return ulysses_fp6_o_permute_quant(x, partitions, packed_head_size, resolved_group)


def ulysses_fN_o_permute_dequant(quant_x: torch.Tensor,
                                 head_size: int,
                                 group_size: int | None = None,
                                 *,
                                 bits: int = 8) -> torch.Tensor:
  """Dispatch O-direction FP4/FP6/FP8 dequantization."""
  resolved_group = _resolve_fN_group_size(bits, group_size)
  if bits == 8:
    return ulysses_fp8_o_permute_dequant(quant_x, head_size, resolved_group)
  if bits == 4:
    return ulysses_fp4_o_permute_dequant(quant_x, head_size, resolved_group)
  return ulysses_fp6_o_permute_dequant(quant_x, head_size, resolved_group)


def _call_ulysses_fN_fused_qkv_quant(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                     y: torch.Tensor, partitions: int, q_heads: int, k_heads: int,
                                     v_heads: int, q_width: int, k_width: int, v_width: int,
                                     chunk_row_bytes: int, group_size: int, bits: int,
                                     quantize_k: bool) -> None:
  """Call the fused QKV out-param quantizer without exposing a new public export."""
  _get_required_ops_module().ulysses_fN_fused_qkv_quant(q, k, v, y, partitions, q_heads, k_heads,
                                                        v_heads, q_width, k_width, v_width,
                                                        chunk_row_bytes, group_size, bits,
                                                        quantize_k)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fN_fused_qkv_quant",
  "(Tensor q, Tensor k, Tensor v, int partitions, int q_heads, int k_heads, int v_heads, "
  "int q_width, int k_width, int v_width, int chunk_row_bytes, int group_size, int bits, "
  "bool quantize_k) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fN_fused_qkv_quant", "CUDA")
def _ulysses_fN_fused_qkv_quant(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, partitions: int,
                                q_heads: int, k_heads: int, v_heads: int, q_width: int,
                                k_width: int, v_width: int, chunk_row_bytes: int, group_size: int,
                                bits: int, quantize_k: bool) -> torch.Tensor:
  batch, seq_len, _, _ = q.shape
  y = torch.empty((partitions * seq_len, chunk_row_bytes), dtype=torch.uint8, device=q.device)
  _call_ulysses_fN_fused_qkv_quant(q, k, v, y, partitions, q_heads, k_heads, v_heads, q_width,
                                   k_width, v_width, chunk_row_bytes, group_size, bits, quantize_k)
  return y


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fN_fused_qkv_quant")
def _fake_ulysses_fN_fused_qkv_quant(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                     partitions: int, q_heads: int, k_heads: int, v_heads: int,
                                     q_width: int, k_width: int, v_width: int, chunk_row_bytes: int,
                                     group_size: int, bits: int, quantize_k: bool) -> torch.Tensor:
  _, seq_len, _, _ = q.shape
  return q.new_empty((partitions * seq_len, chunk_row_bytes), dtype=torch.uint8)


def ulysses_fN_fused_qkv_quant(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, partitions: int,
                               q_heads: int, k_heads: int, v_heads: int, q_width: int, k_width: int,
                               v_width: int, chunk_row_bytes: int, group_size: int, bits: int,
                               quantize_k: bool) -> torch.Tensor:
  """Quantize three QKV tensors into one partition-major uint8 communication buffer."""
  return torch.ops.cache_dit_cuda_ops.ulysses_fN_fused_qkv_quant(q, k, v, partitions, q_heads,
                                                                 k_heads, v_heads, q_width, k_width,
                                                                 v_width, chunk_row_bytes,
                                                                 group_size, bits, quantize_k)


def _call_ulysses_fN_fused_qkv_dequant(packed: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                                       v: torch.Tensor, q_heads: int, k_heads: int, v_heads: int,
                                       head_size: int, q_width: int, k_width: int, v_width: int,
                                       chunk_row_bytes: int, group_size: int, bits: int,
                                       quantize_k: bool) -> None:
  """Call the fused QKV out-param dequantizer."""
  _get_required_ops_module().ulysses_fN_fused_qkv_dequant(packed, q, k, v, q_heads, k_heads,
                                                          v_heads, head_size, q_width, k_width,
                                                          v_width, chunk_row_bytes, group_size,
                                                          bits, quantize_k)


torch.library.define(
  "cache_dit_cuda_ops::ulysses_fN_fused_qkv_dequant",
  "(Tensor packed, int q_heads, int k_heads, int v_heads, int head_size, int q_width, "
  "int k_width, int v_width, int chunk_row_bytes, int group_size, int bits, bool quantize_k) "
  "-> (Tensor, Tensor, Tensor)",
)


@torch.library.impl("cache_dit_cuda_ops::ulysses_fN_fused_qkv_dequant", "CUDA")
def _ulysses_fN_fused_qkv_dequant(
    packed: torch.Tensor, q_heads: int, k_heads: int, v_heads: int, head_size: int, q_width: int,
    k_width: int, v_width: int, chunk_row_bytes: int, group_size: int, bits: int,
    quantize_k: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  seq_len = packed.shape[0]
  batches = chunk_row_bytes // (q_heads * q_width + k_heads * k_width + v_heads * v_width)
  q = torch.empty((batches, seq_len, q_heads, head_size),
                  dtype=torch.bfloat16,
                  device=packed.device)
  k = torch.empty((batches, seq_len, k_heads, head_size),
                  dtype=torch.bfloat16,
                  device=packed.device)
  v = torch.empty((batches, seq_len, v_heads, head_size),
                  dtype=torch.bfloat16,
                  device=packed.device)
  _call_ulysses_fN_fused_qkv_dequant(packed, q, k, v, q_heads, k_heads, v_heads, head_size, q_width,
                                     k_width, v_width, chunk_row_bytes, group_size, bits,
                                     quantize_k)
  return q, k, v


@torch.library.register_fake("cache_dit_cuda_ops::ulysses_fN_fused_qkv_dequant")
def _fake_ulysses_fN_fused_qkv_dequant(
    packed: torch.Tensor, q_heads: int, k_heads: int, v_heads: int, head_size: int, q_width: int,
    k_width: int, v_width: int, chunk_row_bytes: int, group_size: int, bits: int,
    quantize_k: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  seq_len = packed.shape[0]
  batches = chunk_row_bytes // (q_heads * q_width + k_heads * k_width + v_heads * v_width)
  return (packed.new_empty((batches, seq_len, q_heads, head_size), dtype=torch.bfloat16),
          packed.new_empty((batches, seq_len, k_heads, head_size), dtype=torch.bfloat16),
          packed.new_empty((batches, seq_len, v_heads, head_size), dtype=torch.bfloat16))


def ulysses_fN_fused_qkv_dequant(
    packed: torch.Tensor, q_heads: int, k_heads: int, v_heads: int, head_size: int, q_width: int,
    k_width: int, v_width: int, chunk_row_bytes: int, group_size: int, bits: int,
    quantize_k: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Restore three BF16 tensors from one fused QKV communication buffer."""
  return torch.ops.cache_dit_cuda_ops.ulysses_fN_fused_qkv_dequant(packed, q_heads, k_heads,
                                                                   v_heads, head_size, q_width,
                                                                   k_width, v_width,
                                                                   chunk_row_bytes, group_size,
                                                                   bits, quantize_k)


__all__ = [
  "ulysses_fN_o_permute_dequant",
  "ulysses_fN_o_permute_quant",
  "ulysses_fN_per_token_dequant",
  "ulysses_fN_per_token_quant",
  "ulysses_fN_qkv_permute_dequant",
  "ulysses_fN_qkv_permute_quant",
  "ulysses_fp4_o_permute_dequant",
  "ulysses_fp4_o_permute_quant",
  "ulysses_fp4_per_token_dequant",
  "ulysses_fp4_per_token_quant",
  "ulysses_fp4_qkv_permute_dequant",
  "ulysses_fp4_qkv_permute_quant",
  "ulysses_fp6_o_permute_dequant",
  "ulysses_fp6_o_permute_quant",
  "ulysses_fp6_per_token_dequant",
  "ulysses_fp6_per_token_quant",
  "ulysses_fp6_qkv_permute_dequant",
  "ulysses_fp6_qkv_permute_quant",
  "ulysses_fp8_o_permute_dequant",
  "ulysses_fp8_o_permute_quant",
  "ulysses_fp8_per_token_quant",
  "ulysses_fp8_per_token_dequant",
  "ulysses_fp8_qkv_permute_quant",
  "ulysses_fp8_qkv_permute_dequant",
]
