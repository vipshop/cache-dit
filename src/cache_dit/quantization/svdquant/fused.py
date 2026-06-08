"""Fused quantized MLP runtime for SVDQ W4A4 GEMM.

Provides ``fused_gelu_mlp`` (full fc1+GELU+fc2 fusion via qout path) and
``fused_gelu_proj`` (fc1+GELU-only fusion returning fp16 output for use
in single-stream blocks where fc2 input is concatenated).
"""

import torch
from torch import nn

from ...kernels import svdq_gemm_w4a4_ext


@torch.compiler.disable
def fused_gelu_mlp(
  x: torch.Tensor,
  fc1: nn.Module,
  fc2: nn.Module,
  pad_size: int = 256,
) -> torch.Tensor:
  """Fused quantized MLP with GELU activation.

  Combines the first quantized linear layer, GELU activation, and the
  second quantized linear layer into a single CUDA kernel chain.  The
  intermediate fp16 activation is never written back to HBM — the first
  GEMM directly produces 4-bit quantized output consumed by the second
  GEMM.

  :param x: Input tensor with shape ``[..., in_features]`` and dtype
      ``float16`` or ``bfloat16``.
  :param fc1: First quantized linear layer (``SVDQW4A4Linear``, input
      → hidden).
  :param fc2: Second quantized linear layer (``SVDQW4A4Linear``, hidden
      → output).
  :param pad_size: Batch padding size for CUDA kernel efficiency.
      Default is 256.

  :returns: Output tensor with shape ``[..., fc2.out_features]`` and the
      same dtype as ``x``.
  """
  if x.ndim < 2:
    raise ValueError(f"input must have shape [..., in_features], got {tuple(x.shape)}.")

  *leading_shape, channels = x.shape
  token_count = 1
  for extent in leading_shape:
    token_count *= extent
  x = x.reshape(token_count, channels)

  quantized_x, ascales, lora_act = fc1.quantize(x, pad_size=pad_size)

  batch_size_pad = quantized_x.shape[0]

  if fc2.precision == "nvfp4":
    qout_act = torch.empty(
      batch_size_pad,
      fc1.out_features // 2,
      dtype=torch.uint8,
      device=x.device,
    )
    qout_ascales = torch.empty(
      fc1.out_features // 16,
      batch_size_pad,
      dtype=torch.float8_e4m3fn,
      device=x.device,
    )
  else:
    qout_act = torch.empty(
      batch_size_pad,
      fc1.out_features // 2,
      dtype=torch.uint8,
      device=x.device,
    )
    qout_ascales = torch.empty(
      fc1.out_features // 64,
      batch_size_pad,
      dtype=x.dtype,
      device=x.device,
    )
  qout_lora_act = torch.empty(
    batch_size_pad,
    fc2.proj_down.shape[1],
    dtype=torch.float32,
    device=x.device,
  )

  svdq_gemm_w4a4_ext(
    act=quantized_x,
    wgt=fc1.qweight,
    qout=qout_act,
    ascales=ascales,
    wscales=fc1.wscales,
    oscales=qout_ascales,
    lora_act_in=lora_act,
    lora_up=fc1.proj_up,
    lora_down=fc2.proj_down,
    lora_act_out=qout_lora_act,
    bias=fc1.bias,
    smooth_factor=fc2.smooth_factor,
    fp4=fc1.precision == "nvfp4",
    alpha=fc1.wtscale,
    wcscales=fc1.wcscales,
  )

  output = fc2.forward_quant(qout_act, qout_ascales, qout_lora_act)
  output = output[:token_count]
  return output.reshape(*leading_shape, fc2.out_features)


@torch.compiler.disable
def fused_gelu_proj(
  x: torch.Tensor,
  fc1: nn.Module,
  pad_size: int = 256,
) -> torch.Tensor:
  """Fused fc1 GEMM + GELU in a single kernel.

  Uses ``svdq_gemm_w4a4_ext`` with the GELU epilogue to combine the
  first quantized linear layer and GELU activation without a separate
  GELU kernel launch.

  Unlike ``fused_gelu_mlp``, this does **not** fuse the second linear
  layer — the caller is responsible for feeding the returned fp16
  tensor to ``fc2`` (e.g. after concatenation with attention output in
  single-stream blocks).

  :param x: Input tensor with shape ``[..., fc1.in_features]`` and
      dtype ``float16`` or ``bfloat16``.
  :param fc1: First quantized linear layer (``SVDQW4A4Linear``,
      input → hidden).
  :param pad_size: Batch padding size for CUDA kernel efficiency.
      Default is 256.

  :returns: GELU-activated fp16 output with shape
      ``[..., fc1.out_features]``.
  """
  if x.ndim < 2:
    raise ValueError(f"input must have shape [..., in_features], got {tuple(x.shape)}.")

  *leading_shape, channels = x.shape
  token_count = 1
  for extent in leading_shape:
    token_count *= extent
  x = x.reshape(token_count, channels)

  quantized_x, ascales, lora_act = fc1.quantize(x, pad_size=pad_size)

  batch_size_pad = quantized_x.shape[0]
  out = torch.empty(
    batch_size_pad,
    fc1.out_features,
    dtype=x.dtype,
    device=x.device,
  )

  # qout / oscales are required by the ext kernel schema but the
  # caller does not use them — the kernel still populates them.
  if fc1.precision == "nvfp4":
    qout_act = torch.empty(
      batch_size_pad,
      fc1.out_features // 2,
      dtype=torch.uint8,
      device=x.device,
    )
    qout_ascales = torch.empty(
      fc1.out_features // 16,
      batch_size_pad,
      dtype=torch.float8_e4m3fn,
      device=x.device,
    )
  else:
    qout_act = torch.empty(
      batch_size_pad,
      fc1.out_features // 2,
      dtype=torch.uint8,
      device=x.device,
    )
    qout_ascales = torch.empty(
      fc1.out_features // 64,
      batch_size_pad,
      dtype=x.dtype,
      device=x.device,
    )

  # smooth_factor must match fc1.out_features (output dim N),
  # not fc1.in_features (fc1.smooth_factor is for input quant).
  smooth_ones = torch.ones(
    fc1.out_features,
    dtype=x.dtype,
    device=x.device,
  )

  svdq_gemm_w4a4_ext(
    act=quantized_x,
    wgt=fc1.qweight,
    out=out,
    qout=qout_act,
    ascales=ascales,
    wscales=fc1.wscales,
    oscales=qout_ascales,
    lora_act_in=lora_act,
    lora_up=fc1.proj_up,
    bias=fc1.bias,
    smooth_factor=smooth_ones,
    fp4=fc1.precision == "nvfp4",
    alpha=fc1.wtscale,
    wcscales=fc1.wcscales,
  )

  out = out[:token_count]
  return out.reshape(*leading_shape, fc1.out_features)


__all__ = ["fused_gelu_mlp", "fused_gelu_proj"]
