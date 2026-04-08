import torch
from typing import Callable, Tuple
from .backend import KernelBackend

_KERNEL_BE_FN = Callable[..., KernelBackend]
_TRITON_BE_FN = lambda: KernelBackend.TRITON
_CUDA_BE_FN = lambda: KernelBackend.CUDA
_ERROR_TEMPLATE = "kernel backend: {} is not supported now!"


def _ensure_backend_supported(backend: KernelBackend) -> None:
  if not KernelBackend.is_supported(backend):
    raise ValueError(_ERROR_TEMPLATE.format(backend))


# Ulysses FP8 communication related ops
def _fp8_comm_per_token_quant_impl(
  x: torch.Tensor,
  backend_fn: _KERNEL_BE_FN = _TRITON_BE_FN,
) -> torch.Tensor:
  backend = backend_fn()
  _ensure_backend_supported(backend)
  if backend == KernelBackend.TRITON:
    from .triton import fp8_comm_per_token_quant

    return fp8_comm_per_token_quant(x)
  else:
    raise ValueError(_ERROR_TEMPLATE.format(backend))


def _fp8_comm_per_token_dequant_impl(
  x: torch.Tensor,
  backend_fn: _KERNEL_BE_FN = _TRITON_BE_FN,
) -> torch.Tensor:
  backend = backend_fn()
  _ensure_backend_supported(backend)
  if backend == KernelBackend.TRITON:
    from .triton import fp8_comm_per_token_dequant

    return fp8_comm_per_token_dequant(x)
  else:
    raise ValueError(_ERROR_TEMPLATE.format(backend))


def _fp8_comm_qkv_permute_quant_impl(
  x: torch.Tensor,
  backend_fn: _KERNEL_BE_FN = _TRITON_BE_FN,
) -> torch.Tensor:
  backend = backend_fn()
  _ensure_backend_supported(backend)
  if backend == KernelBackend.TRITON:
    from .triton import fp8_comm_qkv_permute_quant

    return fp8_comm_qkv_permute_quant(x)
  else:
    raise ValueError(_ERROR_TEMPLATE.format(backend))


def _fp8_comm_qkv_permute_dequant_impl(
  quant_x: torch.Tensor,
  backend_fn: _KERNEL_BE_FN = _TRITON_BE_FN,
) -> torch.Tensor:
  backend = backend_fn()
  _ensure_backend_supported(backend)
  if backend == KernelBackend.TRITON:
    from .triton import fp8_comm_qkv_permute_dequant

    return fp8_comm_qkv_permute_dequant(quant_x)
  else:
    raise ValueError(_ERROR_TEMPLATE.format(backend))


# Attention related ops, e.g, for Ring Attention
def _fused_merge_attn_states_impl(
  prev_out: torch.Tensor,
  prev_lse: torch.Tensor,
  suff_out: torch.Tensor,
  suff_lse: torch.Tensor,
  backend_fn: _KERNEL_BE_FN = _TRITON_BE_FN,
) -> Tuple[torch.Tensor, torch.Tensor]:
  backend = backend_fn()
  _ensure_backend_supported(backend)
  if backend == KernelBackend.TRITON:
    from .triton import fused_merge_attn_states

    return fused_merge_attn_states(
      prev_out,
      prev_lse,
      suff_out,
      suff_lse,
    )
  else:
    raise ValueError(_ERROR_TEMPLATE.format(backend))


# SVDQuant related ops, with CUDA implementations by default.
def _svdq_extension_is_available_impl() -> bool:
  from .cuda import svdq_extension_is_available

  return svdq_extension_is_available()


def _svdq_get_load_error_impl() -> Exception | None:
  from .cuda import svdq_get_load_error

  return svdq_get_load_error()


def _svdq_gemm_w4a4_impl(
  act: torch.Tensor,
  wgt: torch.Tensor,
  ascales: torch.Tensor,
  wscales: torch.Tensor,
  lora_act_in: torch.Tensor | None = None,
  lora_up: torch.Tensor | None = None,
  bias: torch.Tensor | None = None,
  fp4: bool = False,
  alpha: float | None = 1.0,
  wcscales: torch.Tensor | None = None,
  act_unsigned: bool = False,
  output_dtype: torch.dtype | None = None,
  backend_fn: _KERNEL_BE_FN = _CUDA_BE_FN,
) -> torch.Tensor:
  backend = backend_fn()
  _ensure_backend_supported(backend)
  if backend == KernelBackend.CUDA:
    from .cuda import svdq_gemm_w4a4

    return svdq_gemm_w4a4(
      act=act,
      wgt=wgt,
      ascales=ascales,
      wscales=wscales,
      lora_act_in=lora_act_in,
      lora_up=lora_up,
      bias=bias,
      fp4=fp4,
      alpha=alpha,
      wcscales=wcscales,
      act_unsigned=act_unsigned,
      output_dtype=output_dtype,
    )
  raise ValueError(_ERROR_TEMPLATE.format(backend))


def _svdq_gemm_w4a4_ext_impl(
  act: torch.Tensor,
  wgt: torch.Tensor,
  out: torch.Tensor | None = None,
  qout: torch.Tensor | None = None,
  ascales: torch.Tensor | None = None,
  wscales: torch.Tensor | None = None,
  oscales: torch.Tensor | None = None,
  poolout: torch.Tensor | None = None,
  lora_act_in: torch.Tensor | None = None,
  lora_up: torch.Tensor | None = None,
  lora_down: torch.Tensor | None = None,
  lora_act_out: torch.Tensor | None = None,
  norm_q: torch.Tensor | None = None,
  norm_k: torch.Tensor | None = None,
  rotary_emb: torch.Tensor | None = None,
  bias: torch.Tensor | None = None,
  smooth_factor: torch.Tensor | None = None,
  out_vk: torch.Tensor | None = None,
  out_linearattn: torch.Tensor | None = None,
  act_unsigned: bool = False,
  lora_scales: list[float] | None = None,
  fuse_silu: bool = False,
  fp4: bool = False,
  alpha: float | None = 1.0,
  wcscales: torch.Tensor | None = None,
  out_q: torch.Tensor | None = None,
  out_k: torch.Tensor | None = None,
  out_v: torch.Tensor | None = None,
  attn_tokens: int = 0,
  backend_fn: _KERNEL_BE_FN = _CUDA_BE_FN,
) -> torch.Tensor:
  backend = backend_fn()
  _ensure_backend_supported(backend)
  if backend == KernelBackend.CUDA:
    from .cuda import svdq_gemm_w4a4_ext

    return svdq_gemm_w4a4_ext(
      act=act,
      wgt=wgt,
      out=out,
      qout=qout,
      ascales=ascales,
      wscales=wscales,
      oscales=oscales,
      poolout=poolout,
      lora_act_in=lora_act_in,
      lora_up=lora_up,
      lora_down=lora_down,
      lora_act_out=lora_act_out,
      norm_q=norm_q,
      norm_k=norm_k,
      rotary_emb=rotary_emb,
      bias=bias,
      smooth_factor=smooth_factor,
      out_vk=out_vk,
      out_linearattn=out_linearattn,
      act_unsigned=act_unsigned,
      lora_scales=lora_scales,
      fuse_silu=fuse_silu,
      fp4=fp4,
      alpha=alpha,
      wcscales=wcscales,
      out_q=out_q,
      out_k=out_k,
      out_v=out_v,
      attn_tokens=attn_tokens,
    )
  raise ValueError(_ERROR_TEMPLATE.format(backend))


def _svdq_quantize_w4a4_act_fuse_lora_impl(
  input: torch.Tensor,
  lora_down: torch.Tensor | None = None,
  smooth: torch.Tensor | None = None,
  fuse_glu: bool = False,
  fp4: bool = False,
  pad_size: int = 256,
  backend_fn: _KERNEL_BE_FN = _CUDA_BE_FN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  backend = backend_fn()
  _ensure_backend_supported(backend)
  if backend == KernelBackend.CUDA:
    from .cuda import svdq_quantize_w4a4_act_fuse_lora

    return svdq_quantize_w4a4_act_fuse_lora(
      input=input,
      lora_down=lora_down,
      smooth=smooth,
      fuse_glu=fuse_glu,
      fp4=fp4,
      pad_size=pad_size,
    )
  raise ValueError(_ERROR_TEMPLATE.format(backend))


def _svdq_quantize_w4a4_wgt_impl(
  input: torch.Tensor,
  output: torch.Tensor | None = None,
  oscales: torch.Tensor | None = None,
  backend_fn: _KERNEL_BE_FN = _CUDA_BE_FN,
) -> tuple[torch.Tensor, torch.Tensor]:
  backend = backend_fn()
  _ensure_backend_supported(backend)
  if backend == KernelBackend.CUDA:
    from .cuda import svdq_quantize_w4a4_wgt

    return svdq_quantize_w4a4_wgt(input=input, output=output, oscales=oscales)
  raise ValueError(_ERROR_TEMPLATE.format(backend))


def _svdq_set_faster_i2f_mode_impl(
  mode: str,
  backend_fn: _KERNEL_BE_FN = _CUDA_BE_FN,
) -> None:
  backend = backend_fn()
  _ensure_backend_supported(backend)
  if backend == KernelBackend.CUDA:
    from .cuda import svdq_set_faster_i2f_mode

    svdq_set_faster_i2f_mode(mode)
    return
  raise ValueError(_ERROR_TEMPLATE.format(backend))


def _svdq_set_log_level_impl(
  level: str,
  backend_fn: _KERNEL_BE_FN = _CUDA_BE_FN,
) -> None:
  backend = backend_fn()
  _ensure_backend_supported(backend)
  if backend == KernelBackend.CUDA:
    from .cuda import svdq_set_log_level

    svdq_set_log_level(level)
    return
  raise ValueError(_ERROR_TEMPLATE.format(backend))


# Ulysses FP8 communication related ops
def fp8_comm_per_token_quant(x: torch.Tensor) -> torch.Tensor:
  """Quantize a floating-point tensor to FP8 per-token format.

  :param x: Input floating-point tensor to be quantized.
  :returns: Quantized tensor in FP8 format, where the quantization is performed on a per-token
    quantization scheme suitable for communication purposes.
  """

  return _fp8_comm_per_token_quant_impl(x=x, backend_fn=_TRITON_BE_FN)


def fp8_comm_per_token_dequant(x: torch.Tensor) -> torch.Tensor:
  """Dequantize a FP8 tensor to floating-point format using per-token method.

  :param x: Input FP8 tensor to be dequantized.
  :returns: Dequantized tensor in floating-point format, where the dequantization is performed on a
    per-token basis suitable for communication purposes.
  """
  return _fp8_comm_per_token_dequant_impl(x=x, backend_fn=_TRITON_BE_FN)


def fp8_comm_qkv_permute_quant(x: torch.Tensor) -> torch.Tensor:
  """Quantize a floating-point tensor to FP8 format with QKV permutation.

  :param x: Input floating-point tensor to be quantized.
  :returns: Quantized tensor in FP8 format with QKV permutation, suitable for communication
    purposes.
  """
  return _fp8_comm_qkv_permute_quant_impl(x=x, backend_fn=_TRITON_BE_FN)


def fp8_comm_qkv_permute_dequant(quant_x: torch.Tensor) -> torch.Tensor:
  """Dequantize a FP8 tensor with QKV permutation to floating-point format.

  :param quant_x: Input FP8 tensor with QKV permutation to be dequantized.
  :returns: Dequantized tensor in floating-point format, where the dequantization is performed on a
    per-token basis suitable for communication purposes.
  """
  return _fp8_comm_qkv_permute_dequant_impl(
    quant_x=quant_x,
    backend_fn=_TRITON_BE_FN,
  )


# Attention related ops, e.g, for Ring Attention
def fused_merge_attn_states(
  prev_out: torch.Tensor,
  prev_lse: torch.Tensor,
  suff_out: torch.Tensor,
  suff_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Fuse the attention states of two consecutive attention states, e.g., Ring Attention.

  :param prev_out: Previous output tensor.
  :param prev_lse: Previous log-sum-exp tensor.
  :param suff_out: Sufficient output tensor.
  :param suff_lse: Sufficient log-sum-exp tensor.
  :returns: Fused output and log-sum-exp tensors.
  """
  return _fused_merge_attn_states_impl(
    prev_out=prev_out,
    prev_lse=prev_lse,
    suff_out=suff_out,
    suff_lse=suff_lse,
    backend_fn=_TRITON_BE_FN,
  )


# SVDQuant related ops, with CUDA implementations by default.
# Parameters names scheme:
#  - act = activation, wgt = weight, ascales = activation scales,
#  - wscales = weight scales, oscales = output scales,
#  - lora_up = LoRA up-projection weights,
#  - lora_down = LoRA down-projection weights
#  - ...


def svdq_gemm_w4a4(
  act: torch.Tensor,
  wgt: torch.Tensor,
  ascales: torch.Tensor,
  wscales: torch.Tensor,
  lora_act_in: torch.Tensor | None = None,
  lora_up: torch.Tensor | None = None,
  bias: torch.Tensor | None = None,
  fp4: bool = False,
  alpha: float | None = 1.0,
  wcscales: torch.Tensor | None = None,
  act_unsigned: bool = False,
  output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
  """Convenience wrapper for the plain linear SVDQ W4A4 CUDA GEMM path.

  :param act: Packed activation tensor `[M, K / 2]`.
  :param wgt: Packed quantized weight tensor `[N, K / 2]`.
  :param ascales: Activation scales `[K / G, M]`, where `G` is 64 for INT4 and 16 for FP4.
  :param wscales: Weight scales `[K / G, N]`, where `G` is 64 for INT4 and 16 for FP4.
  :param lora_act_in: Optional LoRA activation input `[M, R]`.
  :param lora_up: Optional LoRA up-projection weights `[N, R]`.
  :param bias: Optional dense output bias `[N]`.
  :param fp4: Whether the packed tensors use FP4/NVFP4 instead of INT4.
  :param alpha: Optional per-tensor FP4 scaling factor. `None` defaults to `1.0`.
  :param wcscales: Optional per-channel FP4 scales `[N]`.
  :param act_unsigned: Whether INT4 activations are stored as unsigned values.
  :param output_dtype: Optional dtype for the allocated dense output. If
    omitted, it is inferred from `lora_up`, `bias`, or INT4 `wscales`.

  :returns: A newly allocated dense output tensor `[M, N]`.
  """

  return _svdq_gemm_w4a4_impl(
    act=act,
    wgt=wgt,
    ascales=ascales,
    wscales=wscales,
    lora_act_in=lora_act_in,
    lora_up=lora_up,
    bias=bias,
    fp4=fp4,
    alpha=alpha,
    wcscales=wcscales,
    act_unsigned=act_unsigned,
    output_dtype=output_dtype,
    backend_fn=_CUDA_BE_FN,
  )


def svdq_gemm_w4a4_ext(
  act: torch.Tensor,
  wgt: torch.Tensor,
  out: torch.Tensor | None = None,
  qout: torch.Tensor | None = None,
  ascales: torch.Tensor | None = None,
  wscales: torch.Tensor | None = None,
  oscales: torch.Tensor | None = None,
  poolout: torch.Tensor | None = None,
  lora_act_in: torch.Tensor | None = None,
  lora_up: torch.Tensor | None = None,
  lora_down: torch.Tensor | None = None,
  lora_act_out: torch.Tensor | None = None,
  norm_q: torch.Tensor | None = None,
  norm_k: torch.Tensor | None = None,
  rotary_emb: torch.Tensor | None = None,
  bias: torch.Tensor | None = None,
  smooth_factor: torch.Tensor | None = None,
  out_vk: torch.Tensor | None = None,
  out_linearattn: torch.Tensor | None = None,
  act_unsigned: bool = False,
  lora_scales: list[float] | None = None,
  fuse_silu: bool = False,
  fp4: bool = False,
  alpha: float | None = 1.0,
  wcscales: torch.Tensor | None = None,
  out_q: torch.Tensor | None = None,
  out_k: torch.Tensor | None = None,
  out_v: torch.Tensor | None = None,
  attn_tokens: int = 0,
) -> torch.Tensor:
  """Full SVDQ W4A4 CUDA GEMM wrapper with optional fusion outputs.

  :param act: Packed activation tensor `[M, K / 2]`.
  :param wgt: Packed quantized weight tensor `[N, K / 2]`.
  :param out: Optional dense output buffer `[M, N]`. Allocated if `None`.
  :param qout: Optional packed quantized output buffer `[M, N / 2]` for the next layer.
  :param ascales: Activation scales `[K / G, M]`, where `G` is 64 for INT4 and 16 for FP4.
  :param wscales: Weight scales `[K / G, N]`, where `G` is 64 for INT4 and 16 for FP4.
  :param oscales: Optional output scales `[N / G, M]` for `qout`.
  :param poolout: Optional pooled output buffer used by specialized fused kernels.
  :param lora_act_in: Optional LoRA activation input `[M, R]`.
  :param lora_up: Optional LoRA up-projection weights `[N, R]`.
  :param lora_down: Optional LoRA down-projection weights `[N, R]` for the next fused layer.
  :param lora_act_out: Optional LoRA activation output buffer `[M, R]` for the next fused layer.
  :param norm_q: Optional query RMSNorm tensor `[HEAD_DIM]`.
  :param norm_k: Optional key RMSNorm tensor `[HEAD_DIM]`.
  :param rotary_emb: Optional packed rotary embeddings `[M, HEAD_DIM / 2, 2, 2]`.
  :param bias: Optional dense output bias `[N]`.
  :param smooth_factor: Optional smoothing factors `[N]` written for next-layer quantization.
  :param out_vk: Optional linear-attention VK output buffer.
  :param out_linearattn: Optional linear-attention output buffer.
  :param act_unsigned: Whether INT4 activations are stored as unsigned values.
  :param lora_scales: Optional per-16-rank LoRA scaling factors `[R / 16]`.
    Defaults to `1.0` per group when `lora_up` is provided.
  :param fuse_silu: Whether to fuse SiLU inside supported kernel variants.
  :param fp4: Whether the packed tensors use FP4/NVFP4 instead of INT4.
  :param alpha: Optional per-tensor FP4 scaling factor. `None` defaults to `1.0`.
  :param wcscales: Optional per-channel FP4 scales `[N]`.
  :param out_q: Optional packed attention-Q output buffer `[B, H, M, D]`.
  :param out_k: Optional packed attention-K output buffer `[B, H, M, D]`.
  :param out_v: Optional packed attention-V output buffer `[B, H, M, D]`.
  :param attn_tokens: Number of attention tokens for fused attention-style kernels.

  :returns: Dense output tensor `[M, N]`. This is the same tensor as `out`
    when `out` is provided.
  """
  return _svdq_gemm_w4a4_ext_impl(
    act=act,
    wgt=wgt,
    out=out,
    qout=qout,
    ascales=ascales,
    wscales=wscales,
    oscales=oscales,
    poolout=poolout,
    lora_act_in=lora_act_in,
    lora_up=lora_up,
    lora_down=lora_down,
    lora_act_out=lora_act_out,
    norm_q=norm_q,
    norm_k=norm_k,
    rotary_emb=rotary_emb,
    bias=bias,
    smooth_factor=smooth_factor,
    out_vk=out_vk,
    out_linearattn=out_linearattn,
    act_unsigned=act_unsigned,
    lora_scales=lora_scales,
    fuse_silu=fuse_silu,
    fp4=fp4,
    alpha=alpha,
    wcscales=wcscales,
    out_q=out_q,
    out_k=out_k,
    out_v=out_v,
    attn_tokens=attn_tokens,
    backend_fn=_CUDA_BE_FN,
  )


def svdq_quantize_w4a4_act_fuse_lora(
  input: torch.Tensor,
  lora_down: torch.Tensor | None = None,
  smooth: torch.Tensor | None = None,
  fuse_glu: bool = False,
  fp4: bool = False,
  pad_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Quantize activations and optionally compute fused LoRA activations.

  :param input: Dense activation tensor `[M, K]`, typically `torch.float16` or `torch.bfloat16`.
  :param lora_down: Optional LoRA down-projection weights `[K, R]`.
  :param smooth: Optional smoothing factors used during activation quantization.
  :param fuse_glu: Whether to fuse GLU inside supported kernel variants.
  :param fp4: Whether to use FP4/NVFP4 quantization. INT4 is used otherwise.
  :param pad_size: Pad the batch dimension to a multiple of this value before
    launching the kernel.
  :returns: A tuple `(output, oscales, lora_act_out)` where `output` is the
    packed quantized activation tensor `[M_pad, K / 2]` with dtype
    `torch.uint8`, `oscales` is the scale tensor `[K / G, M_pad]` with dtype
    `torch.float8_e4m3fn` for FP4 or `input.dtype` for INT4, and
    `lora_act_out` is the LoRA activation output `[M_pad, R]` with dtype
    `torch.float32`.
  """

  return _svdq_quantize_w4a4_act_fuse_lora_impl(
    input=input,
    lora_down=lora_down,
    smooth=smooth,
    fuse_glu=fuse_glu,
    fp4=fp4,
    pad_size=pad_size,
    backend_fn=_CUDA_BE_FN,
  )


def svdq_quantize_w4a4_wgt(
  input: torch.Tensor,
  output: torch.Tensor | None = None,
  oscales: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Quantize dense weights to the packed W4A4 INT4 layout.

  :param input: Dense weight tensor `[N, K]`. `K` must be divisible by 64.
  :param output: Optional destination tensor for the packed weights
    `[N, K / 2]`. When provided, the computed packed weights are copied into
    this tensor and the same tensor is returned.
  :param oscales: Optional destination tensor for the weight scales
    `[K / 64, N]`. When provided, the computed scales are copied into this
    tensor and the same tensor is returned.
  :returns: A tuple `(output, oscales)` where `output` is the packed weight
    tensor `[N, K / 2]` with dtype `torch.int8` and `oscales` is the scale
    tensor `[K / 64, N]` with dtype `input.dtype`.
  """
  return _svdq_quantize_w4a4_wgt_impl(
    input=input,
    output=output,
    oscales=oscales,
    backend_fn=_CUDA_BE_FN,
  )


def svdq_set_faster_i2f_mode(mode: str) -> None:
  """Configure the integer-to-float conversion mode used by the SVDQ kernels.

  :param mode: Backend-specific mode string forwarded to the CUDA extension.
  """

  _svdq_set_faster_i2f_mode_impl(mode=mode, backend_fn=_CUDA_BE_FN)


def svdq_set_log_level(level: str) -> None:
  """Set the log level used by the SVDQ CUDA extension.

  :param level: Log level string understood by the extension runtime.
  """

  _svdq_set_log_level_impl(level=level, backend_fn=_CUDA_BE_FN)


def svdq_extension_is_available() -> bool:
  """Return whether the optional SVDQ CUDA extension is available.

  :returns: `True` when the compiled extension can be loaded successfully.
  """

  return _svdq_extension_is_available_impl()


def svdq_get_load_error() -> Exception | None:
  """Return the cached extension-load error, if one was recorded.

  :returns: The most recent extension import/load error, or `None` when the
    extension loaded successfully.
  """

  return _svdq_get_load_error_impl()


__all__ = [
  # FP8 related ops
  "fp8_comm_per_token_quant",
  "fp8_comm_per_token_dequant",
  "fp8_comm_qkv_permute_quant",
  "fp8_comm_qkv_permute_dequant",
  # Attention related ops
  "fused_merge_attn_states",
  # SVDQuant related ops
  "svdq_get_load_error",
  "svdq_extension_is_available",
  "svdq_gemm_w4a4",
  "svdq_gemm_w4a4_ext",
  "svdq_quantize_w4a4_act_fuse_lora",
  "svdq_quantize_w4a4_wgt",
  "svdq_set_faster_i2f_mode",
  "svdq_set_log_level",
]
