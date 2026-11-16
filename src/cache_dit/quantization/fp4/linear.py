"""Online NVFP4 W4A4 linear backed by cache-dit's native operators."""

import torch
import torch.nn.functional as F
from torch import nn

from ...kernels import fp4_gemm
from ...kernels import fp4_quantize

_FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
               -6.0)


def _scale_shape(rows: int, cols: int) -> tuple[int, int]:
  """Return the canonical blocked scale shape.

  :param rows: Logical row count.
  :param cols: Logical reduction width.
  :returns: Blocked scale shape.
  """

  return ((rows + 127) // 128 * 32, (cols + 63) // 64 * 16)


def _unswizzle_scales(scales: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
  """Restore canonical blocked scales to ``[rows, cols // 16]``.

  :param scales: Blocked UE4M3 scales.
  :param rows: Logical row count.
  :param cols: Logical reduction width.
  :returns: Row-major scales before 128-row padding.
  """

  groups = cols // 16
  row_blocks = (rows + 127) // 128
  col_blocks = groups // 4
  rearranged = scales.reshape(row_blocks * col_blocks, 32, 16)
  temp = rearranged.reshape(row_blocks * col_blocks, 32, 4, 4).transpose(1, 2)
  blocks = temp.reshape(row_blocks, col_blocks, 128, 4)
  padded = blocks.permute(0, 2, 1, 3).reshape(row_blocks * 128, col_blocks * 4)
  return padded[:rows, :groups]


def _dequantize_fp4(qdata: torch.Tensor, scales: torch.Tensor,
                    global_scales: torch.Tensor) -> torch.Tensor:
  """Decode packed FP4 data without a TorchAO runtime dependency.

  :param qdata: Packed FP4 tensor ``[R, K / 2]``.
  :param scales: Canonical blocked scales.
  :param global_scales: Per-row FP32 global scales.
  :returns: Dequantized FP32 tensor ``[R, K]``.
  """

  rows, packed_cols = qdata.shape
  packed = qdata.view(torch.uint8)
  table = qdata.new_tensor(_FP4_VALUES, dtype=torch.float32)
  values = table[(packed & 0x0f).long()]
  high = table[(packed >> 4).long()]
  values = torch.stack((values, high), dim=-1).reshape(rows, packed_cols * 2)
  scales = _unswizzle_scales(scales, rows, packed_cols * 2).float()
  return values * scales.repeat_interleave(16, dim=1) * global_scales[:, None]


class NVFP4Linear(nn.Module):
  """Online two-level NVFP4 W4A4 replacement for ``torch.nn.Linear``.

  ``from_linear`` quantizes the BF16 weight once and stores three derived,
  non-persistent buffers: packed E2M1 ``qweight[N,K/2]``, canonical UE4M3
  ``wscales``, and per-output-channel FP32 ``wglobal_scales[N]``. Bias remains
  BF16. No serialized FP4 checkpoint, calibration data, TP/DTensor weight, or
  persistent FP4 state is supported.

  Every forward flattens arbitrary leading dimensions to ``[M,K]``, dynamically
  quantizes the activation into ``qx/sx/gx``, and evaluates::

    y[m,n] = BF16(wglobal_scales[n] *
                  (gx[m] * block_scaled_accumulator[m,n]) + bias[n])

  The output shape is then restored to ``[*leading_shape,N]``. Both activation
  and weight use a per-row FP32 global scale plus a UE4M3 scale per 16 values.
  The two global scales are folded in the GEMM epilogue, not in a separate
  pointwise pass. Eager and ``torch.compile(fullgraph=True)`` use the same
  ``torch.library`` operator and fake metadata.

  On SM120, weights and bias are zero-padded to ``K%256==0`` and ``N%128==0``
  during ``from_linear``. Activations are padded only when needed, and outputs
  are sliced back to the original N. Thus arbitrary Linear shapes preserve the
  original mathematical result while aligned shapes avoid all padding copies.
  """

  def __init__(self,
               in_features: int,
               out_features: int,
               bias: bool = True,
               device: torch.device | str | None = None) -> None:
    """Allocate derived NVFP4 weight buffers.

    :param in_features: Logical input width.
    :param out_features: Logical output width.
    :param bias: Whether to allocate a BF16 bias.
    :param device: Buffer device.
    """

    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.quantized_in_features = (in_features + 255) // 256 * 256
    self.quantized_out_features = (out_features + 127) // 128 * 128
    self.register_buffer(
      "qweight",
      torch.empty((self.quantized_out_features, self.quantized_in_features // 2),
                  dtype=torch.float4_e2m1fn_x2,
                  device=device),
      persistent=False,
    )
    self.register_buffer(
      "wscales",
      torch.empty(_scale_shape(self.quantized_out_features, self.quantized_in_features),
                  dtype=torch.float8_e4m3fn,
                  device=device),
      persistent=False,
    )
    self.register_buffer(
      "wglobal_scales",
      torch.empty((self.quantized_out_features, ), dtype=torch.float32, device=device),
      persistent=False,
    )
    self.bias = nn.Parameter(torch.empty(
      self.quantized_out_features, dtype=torch.bfloat16, device=device),
                             requires_grad=False) if bias else None

  @classmethod
  def from_linear(
    cls,
    linear: nn.Linear,
    *,
    quantize_device: torch.device | str | None = None,
  ) -> "NVFP4Linear":
    """Quantize a BF16 linear weight once.

    :param linear: Source linear layer.
    :param quantize_device: Device used for one-time weight quantization.
    :returns: Native NVFP4 linear on the source weight device.
    """

    if not isinstance(linear, nn.Linear):
      raise TypeError(f"NVFP4Linear.from_linear expects nn.Linear, got {type(linear)}.")
    weight = linear.weight.detach()
    if weight.dim() != 2:
      raise ValueError("NVFP4 quantization requires a 2D linear weight.")
    if weight.dtype != torch.bfloat16:
      raise ValueError(f"cache-dit NVFP4 requires bfloat16 weights, got {weight.dtype}.")
    out_features, in_features = weight.shape
    if quantize_device is None:
      quantize_device = weight.device if weight.is_cuda else torch.device("cuda")

    module = cls(in_features, out_features, linear.bias is not None, device=weight.device)
    quant_weight = weight.to(device=quantize_device, dtype=torch.bfloat16).contiguous()
    quant_weight = F.pad(
      quant_weight,
      (0, module.quantized_in_features - in_features, 0,
       module.quantized_out_features - out_features),
    )
    qweight, wscales, wglobal_scales = fp4_quantize(quant_weight)
    module.qweight.copy_(qweight.to(weight.device))
    module.wscales.copy_(wscales.to(weight.device))
    module.wglobal_scales.copy_(wglobal_scales.to(weight.device))
    if linear.bias is not None:
      module.bias.data.zero_()
      module.bias.data[:out_features].copy_(linear.bias.detach().to(device=weight.device,
                                                                    dtype=torch.bfloat16))
    module._is_inner_quantized = True
    return module

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Dynamically quantize activations and run the two-level NVFP4 GEMM.

    Leading dimensions are flattened only for quantization/GEMM and restored
    afterward. Activation quantization allocates new qdata/scale outputs per
    call; weight buffers are read-only and reused across calls.

    :param x: BF16 activation ``[..., in_features]`` on CUDA.
    :returns: BF16 output ``[..., out_features]``.
    """

    if x.dtype != torch.bfloat16:
      raise ValueError(f"NVFP4Linear expects bfloat16 activations, got {x.dtype}.")
    if (self.qweight.dtype != torch.float4_e2m1fn_x2 or self.wscales.dtype != torch.float8_e4m3fn
        or self.wglobal_scales.dtype != torch.float32):
      raise RuntimeError("NVFP4Linear buffers were cast after quantization; do not call "
                         "`.to(dtype)` on a quantized module.")
    leading_shape = x.shape[:-1]
    flat = x.reshape(-1, self.in_features)
    if self.quantized_in_features != self.in_features:
      flat = F.pad(flat, (0, self.quantized_in_features - self.in_features))
    qx, sx, gx = fp4_quantize(flat)
    out = fp4_gemm(qx, self.qweight, sx, self.wscales, gx, self.wglobal_scales, bias=self.bias)
    out = out[:, :self.out_features]
    return out.reshape(*leading_shape, self.out_features)

  def dequantize_weight(self) -> torch.Tensor:
    """Reconstruct the dense weight for diagnostics.

    :returns: Dequantized BF16 weight ``[out_features, in_features]``.
    """

    dense = _dequantize_fp4(self.qweight, self.wscales, self.wglobal_scales)
    return dense[:self.out_features, :self.in_features].to(torch.bfloat16)

  def extra_repr(self) -> str:
    """:returns: Compact module description."""

    return (f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}")
