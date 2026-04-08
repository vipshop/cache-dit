import torch

from ._svdquant import _call_svdq_gemm_w4a4
from ._svdquant import _call_svdq_quantize_w4a4_act_fuse_lora
from ._svdquant import _call_svdq_quantize_w4a4_wgt
from ._svdquant import _decode_svdq_output_dtype
from ._svdquant import _encode_svdq_output_dtype
from ._svdquant import _get_required_utils_module
from ._svdquant import _infer_svdq_output_dtype
from ._svdquant import _normalize_svdq_lora_scales
from ._svdquant import svdq_extension_is_available
from ._svdquant import svdq_get_load_error

torch.library.define(
  "cache_dit_cuda_ops::svdq_quantize_w4a4_act_fuse_lora",
  "(Tensor input, Tensor? lora_down, Tensor? smooth, bool fuse_glu, bool fp4, int pad_size) -> (Tensor output, Tensor oscales, Tensor lora_act_out)",
)


@torch.library.impl("cache_dit_cuda_ops::svdq_quantize_w4a4_act_fuse_lora", "CUDA")
def _svdq_quantize_w4a4_act_fuse_lora(
  input: torch.Tensor,
  lora_down: torch.Tensor | None,
  smooth: torch.Tensor | None,
  fuse_glu: bool,
  fp4: bool,
  pad_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  batch_size, channels = input.shape
  batch_size_pad = ((batch_size + pad_size - 1) // pad_size) * pad_size
  rank = 0 if lora_down is None else lora_down.shape[1]

  output = torch.empty(batch_size_pad, channels // 2, dtype=torch.uint8, device=input.device)
  if fp4:
    if channels % 16 != 0:
      raise ValueError(f"Expected channels divisible by 16 for FP4 quantization, got {channels}.")
    oscales = torch.empty(channels // 16,
                          batch_size_pad,
                          dtype=torch.float8_e4m3fn,
                          device=input.device)
  else:
    if channels % 64 != 0:
      raise ValueError(f"Expected channels divisible by 64 for INT4 quantization, got {channels}.")
    oscales = torch.empty(channels // 64, batch_size_pad, dtype=input.dtype, device=input.device)
  lora_act_out = torch.empty(batch_size_pad, rank, dtype=torch.float32, device=input.device)

  _call_svdq_quantize_w4a4_act_fuse_lora(
    input,
    output,
    oscales,
    lora_down,
    lora_act_out,
    smooth,
    fuse_glu,
    fp4,
  )
  return output, oscales, lora_act_out


@torch.library.register_fake("cache_dit_cuda_ops::svdq_quantize_w4a4_act_fuse_lora")
def _fake_svdq_quantize_w4a4_act_fuse_lora(
  input: torch.Tensor,
  lora_down: torch.Tensor | None,
  smooth: torch.Tensor | None,
  fuse_glu: bool,
  fp4: bool,
  pad_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  del smooth, fuse_glu
  batch_size, channels = input.shape
  batch_size_pad = ((batch_size + pad_size - 1) // pad_size) * pad_size
  rank = 0 if lora_down is None else lora_down.shape[1]
  output = input.new_empty((batch_size_pad, channels // 2), dtype=torch.uint8)
  oscales = input.new_empty(
    (channels // (16 if fp4 else 64), batch_size_pad),
    dtype=torch.float8_e4m3fn if fp4 else input.dtype,
  )
  lora_act_out = input.new_empty((batch_size_pad, rank), dtype=torch.float32)
  return output, oscales, lora_act_out


torch.library.define(
  "cache_dit_cuda_ops::svdq_gemm_w4a4",
  "(Tensor act, Tensor wgt, Tensor ascales, Tensor wscales, Tensor? lora_act_in, Tensor? lora_up, Tensor? bias, bool fp4, float alpha, Tensor? wcscales, bool act_unsigned, int out_dtype_id) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::svdq_gemm_w4a4", "CUDA")
def _svdq_gemm_w4a4(
  act: torch.Tensor,
  wgt: torch.Tensor,
  ascales: torch.Tensor,
  wscales: torch.Tensor,
  lora_act_in: torch.Tensor | None,
  lora_up: torch.Tensor | None,
  bias: torch.Tensor | None,
  fp4: bool,
  alpha: float,
  wcscales: torch.Tensor | None,
  act_unsigned: bool,
  out_dtype_id: int,
) -> torch.Tensor:
  output_dtype = _decode_svdq_output_dtype(out_dtype_id)
  output = torch.empty((act.shape[0], wgt.shape[0]), dtype=output_dtype, device=act.device)
  alpha = float(alpha)

  # Keep positional placeholders aligned with csrc/kernels/svdq/ops.h::gemm_w4a4.
  # This compile-friendly wrapper only uses the plain linear path.
  _call_svdq_gemm_w4a4(
    act,
    wgt,
    output,
    None,  # qout: no re-quantized output buffer
    ascales,
    wscales,
    None,  # oscales: no output activation scales requested
    None,  # poolout: no pooled output buffer
    lora_act_in,
    lora_up,
    None,  # lora_down: no fused LoRA-down branch here
    None,  # lora_act_out: no intermediate LoRA activation output
    None,  # norm_q: not running RMSNorm/Q attention fusion
    None,  # norm_k: not running RMSNorm/K attention fusion
    None,  # rotary_emb: not running RoPE fusion
    bias,
    None,  # smooth_factor: not producing next-layer smoothing factors
    None,  # out_vk: not writing linear-attention VK output
    None,  # out_linearattn: not writing linear-attention output
    act_unsigned,
    None,  # lora_scales: not running any LoRA fusion paths that require scale normalization
    False,  # fuse_silu: not running SiLU fusion
    fp4,
    alpha,
    wcscales,
    None,  # out_q: not writing attention Q output
    None,  # out_k: not writing attention K output
    None,  # out_v: not writing attention V output
    0,
  )
  return output


torch.library.define(
  "cache_dit_cuda_ops::svdq_gemm_w4a4_ext",
  "(Tensor act, Tensor wgt, Tensor out, Tensor? qout, Tensor? ascales, Tensor? wscales, Tensor? oscales, Tensor? poolout, Tensor? lora_act_in, Tensor? lora_up, Tensor? lora_down, Tensor? lora_act_out, Tensor? norm_q, Tensor? norm_k, Tensor? rotary_emb, Tensor? bias, Tensor? smooth_factor, Tensor? out_vk, Tensor? out_linearattn, bool act_unsigned, float[] lora_scales, bool fuse_silu, bool fp4, float alpha, Tensor? wcscales, Tensor? out_q, Tensor? out_k, Tensor? out_v, int attn_tokens) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::svdq_gemm_w4a4_ext", "CUDA")
def _svdq_gemm_w4a4_ext(
  act: torch.Tensor,
  wgt: torch.Tensor,
  out: torch.Tensor,
  qout: torch.Tensor | None,
  ascales: torch.Tensor | None,
  wscales: torch.Tensor | None,
  oscales: torch.Tensor | None,
  poolout: torch.Tensor | None,
  lora_act_in: torch.Tensor | None,
  lora_up: torch.Tensor | None,
  lora_down: torch.Tensor | None,
  lora_act_out: torch.Tensor | None,
  norm_q: torch.Tensor | None,
  norm_k: torch.Tensor | None,
  rotary_emb: torch.Tensor | None,
  bias: torch.Tensor | None,
  smooth_factor: torch.Tensor | None,
  out_vk: torch.Tensor | None,
  out_linearattn: torch.Tensor | None,
  act_unsigned: bool,
  lora_scales: list[float],
  fuse_silu: bool,
  fp4: bool,
  alpha: float,
  wcscales: torch.Tensor | None,
  out_q: torch.Tensor | None,
  out_k: torch.Tensor | None,
  out_v: torch.Tensor | None,
  attn_tokens: int,
) -> torch.Tensor:
  _call_svdq_gemm_w4a4(
    act,
    wgt,
    out,
    qout,
    ascales,
    wscales,
    oscales,
    poolout,
    lora_act_in,
    lora_up,
    lora_down,
    lora_act_out,
    norm_q,
    norm_k,
    rotary_emb,
    bias,
    smooth_factor,
    out_vk,
    out_linearattn,
    act_unsigned,
    lora_scales,
    fuse_silu,
    fp4,
    alpha,
    wcscales,
    out_q,
    out_k,
    out_v,
    attn_tokens,
  )
  return out


torch.library.define(
  "cache_dit_cuda_ops::svdq_quantize_w4a4_wgt",
  "(Tensor input) -> (Tensor output, Tensor oscales)",
)


@torch.library.impl("cache_dit_cuda_ops::svdq_quantize_w4a4_wgt", "CUDA")
def _svdq_quantize_w4a4_wgt(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  if input.ndim != 2:
    raise ValueError(f"Expected a 2D weight tensor, got shape {tuple(input.shape)}.")
  if input.shape[1] % 64 != 0:
    raise ValueError(f"Expected input features divisible by 64, got {input.shape[1]}.")

  output = torch.empty((input.shape[0], input.shape[1] // 2), dtype=torch.int8, device=input.device)
  oscales = torch.empty((input.shape[1] // 64, input.shape[0]),
                        dtype=input.dtype,
                        device=input.device)
  _call_svdq_quantize_w4a4_wgt(input, output, oscales)
  return output, oscales


@torch.library.register_fake("cache_dit_cuda_ops::svdq_quantize_w4a4_wgt")
def _fake_svdq_quantize_w4a4_wgt(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  if input.ndim != 2:
    raise ValueError(f"Expected a 2D weight tensor, got shape {tuple(input.shape)}.")
  if input.shape[1] % 64 != 0:
    raise ValueError(f"Expected input features divisible by 64, got {input.shape[1]}.")
  return (
    input.new_empty((input.shape[0], input.shape[1] // 2), dtype=torch.int8),
    input.new_empty((input.shape[1] // 64, input.shape[0]), dtype=input.dtype),
  )


@torch.library.register_fake("cache_dit_cuda_ops::svdq_gemm_w4a4_ext")
def _fake_svdq_gemm_w4a4_ext(
  act: torch.Tensor,
  wgt: torch.Tensor,
  out: torch.Tensor,
  qout: torch.Tensor | None,
  ascales: torch.Tensor | None,
  wscales: torch.Tensor | None,
  oscales: torch.Tensor | None,
  poolout: torch.Tensor | None,
  lora_act_in: torch.Tensor | None,
  lora_up: torch.Tensor | None,
  lora_down: torch.Tensor | None,
  lora_act_out: torch.Tensor | None,
  norm_q: torch.Tensor | None,
  norm_k: torch.Tensor | None,
  rotary_emb: torch.Tensor | None,
  bias: torch.Tensor | None,
  smooth_factor: torch.Tensor | None,
  out_vk: torch.Tensor | None,
  out_linearattn: torch.Tensor | None,
  act_unsigned: bool,
  lora_scales: list[float],
  fuse_silu: bool,
  fp4: bool,
  alpha: float,
  wcscales: torch.Tensor | None,
  out_q: torch.Tensor | None,
  out_k: torch.Tensor | None,
  out_v: torch.Tensor | None,
  attn_tokens: int,
) -> torch.Tensor:
  del act, wgt, qout, ascales, wscales, oscales, poolout, lora_act_in, lora_up, lora_down
  del lora_act_out, norm_q, norm_k, rotary_emb, bias, smooth_factor, out_vk, out_linearattn
  del act_unsigned, lora_scales, fuse_silu, fp4, alpha, wcscales, out_q, out_k, out_v, attn_tokens
  return out


@torch.library.register_fake("cache_dit_cuda_ops::svdq_gemm_w4a4")
def _fake_svdq_gemm_w4a4(
  act: torch.Tensor,
  wgt: torch.Tensor,
  ascales: torch.Tensor,
  wscales: torch.Tensor,
  lora_act_in: torch.Tensor | None,
  lora_up: torch.Tensor | None,
  bias: torch.Tensor | None,
  fp4: bool,
  alpha: float,
  wcscales: torch.Tensor | None,
  act_unsigned: bool,
  out_dtype_id: int,
) -> torch.Tensor:
  del ascales, wscales, lora_act_in, lora_up, bias, fp4, alpha, wcscales, act_unsigned
  output_dtype = _decode_svdq_output_dtype(out_dtype_id)
  return act.new_empty((act.shape[0], wgt.shape[0]), dtype=output_dtype)


def svdq_quantize_w4a4_act_fuse_lora(
  input: torch.Tensor,
  lora_down: torch.Tensor | None,
  smooth: torch.Tensor | None,
  fuse_glu: bool,
  fp4: bool,
  pad_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  return torch.ops.cache_dit_cuda_ops.svdq_quantize_w4a4_act_fuse_lora(
    input,
    lora_down,
    smooth,
    fuse_glu,
    fp4,
    pad_size,
  )


def svdq_gemm_w4a4(
  act: torch.Tensor,
  wgt: torch.Tensor,
  ascales: torch.Tensor,
  wscales: torch.Tensor,
  lora_act_in: torch.Tensor | None,
  lora_up: torch.Tensor | None,
  bias: torch.Tensor | None,
  fp4: bool,
  alpha: float | None,
  wcscales: torch.Tensor | None,
  act_unsigned: bool,
  output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
  if alpha is None:
    alpha = 1.0
  if output_dtype is None:
    output_dtype = _infer_svdq_output_dtype(None, lora_up, bias, wscales)
    if output_dtype is None:
      raise ValueError("Unable to infer the output dtype for svdq_gemm_w4a4.")
  return torch.ops.cache_dit_cuda_ops.svdq_gemm_w4a4(
    act,
    wgt,
    ascales,
    wscales,
    lora_act_in,
    lora_up,
    bias,
    fp4,
    alpha,
    wcscales,
    act_unsigned,
    _encode_svdq_output_dtype(output_dtype),
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
  if alpha is None:
    alpha = 1.0
  if out is None:
    output_dtype = _infer_svdq_output_dtype(None, lora_up, bias, wscales)
    if output_dtype is None:
      raise ValueError("Unable to infer the output dtype for svdq_gemm_w4a4_ext.")
    out = torch.empty((act.shape[0], wgt.shape[0]), dtype=output_dtype, device=act.device)
  return torch.ops.cache_dit_cuda_ops.svdq_gemm_w4a4_ext(
    act,
    wgt,
    out,
    qout,
    ascales,
    wscales,
    oscales,
    poolout,
    lora_act_in,
    lora_up,
    lora_down,
    lora_act_out,
    norm_q,
    norm_k,
    rotary_emb,
    bias,
    smooth_factor,
    out_vk,
    out_linearattn,
    act_unsigned,
    _normalize_svdq_lora_scales(lora_scales, lora_up),
    fuse_silu,
    fp4,
    float(alpha),
    wcscales,
    out_q,
    out_k,
    out_v,
    attn_tokens,
  )


def svdq_quantize_w4a4_wgt(
  input: torch.Tensor,
  output: torch.Tensor | None = None,
  oscales: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  custom_output, custom_oscales = torch.ops.cache_dit_cuda_ops.svdq_quantize_w4a4_wgt(input)
  if output is not None:
    output.copy_(custom_output)
    custom_output = output
  if oscales is not None:
    oscales.copy_(custom_oscales)
    custom_oscales = oscales
  return custom_output, custom_oscales


def svdq_set_log_level(level: str) -> None:
  _get_required_utils_module().set_log_level(level)


def svdq_set_faster_i2f_mode(mode: str) -> None:
  _get_required_utils_module().set_faster_i2f_mode(mode)


__all__ = [
  "svdq_get_load_error",
  "svdq_extension_is_available",
  "svdq_gemm_w4a4",
  "svdq_gemm_w4a4_ext",
  "svdq_quantize_w4a4_wgt",
  "svdq_quantize_w4a4_act_fuse_lora",
  "svdq_set_faster_i2f_mode",
  "svdq_set_log_level",
]
