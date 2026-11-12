"""Online FP8 linear layer backed by cache-dit's native CUDA operators."""

import torch
from torch import nn

from ...kernels import fp8_gemm
from ...kernels import fp8_quantize
from ...logger import init_logger

logger = init_logger(__name__)


def fp8_weight_scale_shape(out_features: int, in_features: int, mode: str) -> tuple[int, ...]:
  """Return the weight-scale shape for one FP8 granularity.

  :param out_features: Linear output width.
  :param in_features: Linear input width.
  :param mode: FP8 granularity.
  :returns: Weight-scale shape.
  """

  if mode == "per_row":
    return (out_features, )
  if mode == "per_tensor":
    return (1, )
  if mode == "per_block":
    return ((out_features + 127) // 128, (in_features + 127) // 128)
  raise ValueError(f"Unsupported cache-dit FP8 granularity: {mode!r}.")


def _expand_block_scale(scale: torch.Tensor, rows: int, cols: int, is_weight: bool) -> torch.Tensor:
  """Expand a per-block scale table to a dense 2D tensor.

  :param scale: Block scale table.
  :param rows: Dense row count.
  :param cols: Dense column count.
  :param is_weight: Whether the scale uses the ``128 x 128`` weight layout.
  :returns: Dense FP32 scale tensor.
  """

  if is_weight:
    expanded = scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return expanded[:rows, :cols]
  expanded = scale.repeat_interleave(128, dim=1)
  return expanded[:, :cols]


class CacheDitFP8Linear(nn.Module):
  """Linear layer running cache-dit's native FP8 quantize and GEMM operators.

  The FP8 weight and its scale are derived buffers: they are not persistent, so
  a float checkpoint cannot accidentally round-trip an unlabelled quantized
  state. Bias stays a BF16 parameter. Callers must not cast the quantized
  module to another floating dtype; device moves are supported.
  """

  def __init__(
    self,
    in_features: int,
    out_features: int,
    mode: str = "per_row",
    bias: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
  ) -> None:
    """Initialize an empty native FP8 linear.

    :param in_features: Logical input width.
    :param out_features: Logical output width.
    :param mode: ``per_row``, ``per_tensor`` or ``per_block``.
    :param bias: Whether to allocate a BF16 bias.
    :param device: Buffer device.
    :param dtype: Bias dtype; only bfloat16 is supported by the kernels.
    """

    super().__init__()
    if dtype != torch.bfloat16:
      raise ValueError(f"CacheDitFP8Linear requires bfloat16, got {dtype}.")
    self.in_features = in_features
    self.out_features = out_features
    self.mode = mode
    self.register_buffer(
      "qweight",
      torch.empty((out_features, in_features), dtype=torch.float8_e4m3fn, device=device),
      persistent=False,
    )
    self.register_buffer(
      "wscales",
      torch.empty(
        fp8_weight_scale_shape(out_features, in_features, mode),
        dtype=torch.float32,
        device=device,
      ),
      persistent=False,
    )
    self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device),
                             requires_grad=False) if bias else None

  @classmethod
  def from_linear(
    cls,
    linear: nn.Linear,
    mode: str,
    *,
    quantize_device: torch.device | str | None = None,
  ) -> "CacheDitFP8Linear":
    """Quantize a float ``nn.Linear`` weight once and build the runtime module.

    :param linear: Source linear layer. Its weight must be a BF16 2D tensor.
    :param mode: ``per_row``, ``per_tensor`` or ``per_block``.
    :param quantize_device: Device used for the one-time weight quantization;
      defaults to the current CUDA device when the source is on CPU.
    :returns: A native FP8 linear on the source device.
    """

    if not isinstance(linear, nn.Linear):
      raise TypeError(f"CacheDitFP8Linear.from_linear expects nn.Linear, got {type(linear)}.")
    weight = linear.weight.detach()
    if weight.dim() != 2:
      raise ValueError("FP8 quantization requires a 2D linear weight.")
    if weight.dtype != torch.bfloat16:
      raise ValueError(f"cache-dit FP8 requires bfloat16 weights, got {weight.dtype}.")
    out_features, in_features = weight.shape
    if in_features % 16 != 0 or out_features % 8 != 0:
      raise ValueError("cache-dit FP8 requires in_features % 16 == 0 and "
                       f"out_features % 8 == 0, got ({out_features}, {in_features}).")
    if mode == "per_block" and (in_features % 128 != 0 or out_features % 128 != 0):
      raise ValueError("cache-dit FP8 per-block requires both linear dimensions to be "
                       f"multiples of 128, got ({out_features}, {in_features}).")
    if quantize_device is None:
      quantize_device = weight.device if weight.is_cuda else torch.device("cuda")

    quant_weight = weight.to(device=quantize_device, dtype=torch.bfloat16).contiguous()
    qweight, wscales = fp8_quantize(quant_weight, mode=mode, is_weight=True)
    module = cls(
      in_features=in_features,
      out_features=out_features,
      mode=mode,
      bias=linear.bias is not None,
      device=weight.device,
      dtype=torch.bfloat16,
    )
    module.qweight.copy_(qweight.to(weight.device))
    module.wscales.copy_(wscales.to(weight.device))
    if linear.bias is not None:
      module.bias.data.copy_(linear.bias.detach().to(device=weight.device, dtype=torch.bfloat16))
    module._is_inner_quantized = True
    return module

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Quantize activations dynamically and run the FP8 GEMM.

    :param x: Input activation ``[..., in_features]`` on CUDA.
    :returns: BF16 output ``[..., out_features]``.
    """

    if x.dtype != torch.bfloat16:
      raise ValueError(f"CacheDitFP8Linear expects bfloat16 activations, got {x.dtype}.")
    if self.qweight.dtype != torch.float8_e4m3fn or self.wscales.dtype != torch.float32:
      raise RuntimeError("CacheDitFP8Linear buffers were cast after quantization; do not call "
                         "`.to(dtype)` on a quantized module.")
    leading_shape = x.shape[:-1]
    flat = x.reshape(-1, self.in_features)
    qx, sx = fp8_quantize(flat, mode=self.mode)
    out = fp8_gemm(qx, self.qweight, sx, self.wscales, bias=self.bias, mode=self.mode)
    return out.reshape(*leading_shape, self.out_features)

  def dequantize_weight(self) -> torch.Tensor:
    """Reconstruct the dense FP32-equivalent weight for diagnostics.

    :returns: Dequantized BF16 weight ``[out_features, in_features]``.
    """

    weight = self.qweight.float()
    if self.mode == "per_row":
      weight = weight * self.wscales[:, None]
    elif self.mode == "per_tensor":
      weight = weight * self.wscales[0]
    else:
      weight = weight * _expand_block_scale(self.wscales, self.out_features, self.in_features, True)
    return weight.to(torch.bfloat16)

  def extra_repr(self) -> str:
    """:returns: Compact module description."""

    return (f"in_features={self.in_features}, out_features={self.out_features}, "
            f"mode={self.mode}, bias={self.bias is not None}")
