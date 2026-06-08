"""Fused quantized MLP runtime for SVDQ W4A4 GEMM.

Provides ``fused_gelu_mlp``, a drop-in fused path that combines the first
quantized linear layer, GELU activation, and the second quantized linear
layer using a single re-quantization via ``svdq_gemm_w4a4_ext`` with the
qout path.  The intermediate fp16 activation is never written to HBM.
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


__all__ = ["fused_gelu_mlp"]
