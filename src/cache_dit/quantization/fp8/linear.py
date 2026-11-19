"""Online quantize + W8A8 linear layer backed by cache-dit's native operators."""

import torch
import torch.nn.functional as F
from torch import nn

from ...kernels import fp8_gemm
from ...kernels import fp8_quantize
from ...kernels.fp8 import fp8_quantize_torch
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


class QW8A8Linear(nn.Module):
  """Online W8A8 E4M3 replacement for ``torch.nn.Linear``.

  Both weights and activations use the same quantization convention: each scale
  is ``amax / 448``, qdata is ``satfinite_e4m3(x / scale)``, and dequantization
  is ``qdata * scale``. Unlike NVFP4, there is only one FP32 scale level. The
  three modes share the math but differ in scale granularity: per-token/output-
  channel, per-tensor, or 1x128 activation and 128x128 weight blocks.

  Lifecycle and shape contract
  ----------------------------
  :meth:`from_linear` quantizes the weight once. Before quantization it zero-pads
  BF16 weight and bias to the mode's required dimensions, so arbitrary input and
  output widths are supported without a per-forward weight repack:

  * per-row/per-tensor: quantized K/N align to 16/8 for the CUTLASS collective;
  * per-block: quantized K/N align to 128/128 for blockwise scale layouts.

  :meth:`forward` flattens leading dimensions to ``[M,K]``, pads activation K if
  needed, dynamically quantizes it, calls :func:`cache_dit.kernels.fp8_gemm`, and
  slices output N back to ``out_features``. Padding qdata, padding scales and
  padding bias are zero, so this does not change the original output region.
  Eager uses the custom CUDA quantizer; compile uses the equivalent pure-PyTorch
  quantizer to keep reduction/pointwise operations visible to Inductor. GEMM uses
  the CUTLASS collective by default; ``fp8_gemm(use_native=False)`` is the
  per-row/per-tensor ``torch._scaled_mm`` reference path.

  ``qweight`` and ``wscales`` are non-persistent derived buffers: they are
  rebuilt from a BF16 checkpoint instead of being serialized as an unlabelled
  quantized state. ``bias`` remains BF16 and requires no gradient. Moving devices
  is supported, but calling ``.to(dtype=...)`` on the quantized module is not:
  casting qdata or FP32 scales would invalidate the layout and numerical recipe.
  :meth:`dequantize_weight` is only a diagnostic approximation and also depads
  the stored weight to the original shape.
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
      raise ValueError(f"QW8A8Linear requires bfloat16, got {dtype}.")
    self.in_features = in_features
    self.out_features = out_features
    self.mode = mode
    k_multiple, n_multiple = (128, 128) if mode == "per_block" else (16, 8)
    self.quantized_in_features = (in_features + k_multiple - 1) // k_multiple * k_multiple
    self.quantized_out_features = (out_features + n_multiple - 1) // n_multiple * n_multiple
    self.register_buffer(
      "qweight",
      torch.empty((self.quantized_out_features, self.quantized_in_features),
                  dtype=torch.float8_e4m3fn,
                  device=device),
      persistent=False,
    )
    self.register_buffer(
      "wscales",
      torch.empty(
        fp8_weight_scale_shape(self.quantized_out_features, self.quantized_in_features, mode),
        dtype=torch.float32,
        device=device,
      ),
      persistent=False,
    )
    self.bias = nn.Parameter(torch.empty(self.quantized_out_features, dtype=dtype, device=device),
                             requires_grad=False) if bias else None

  @classmethod
  def from_linear(
    cls,
    linear: nn.Linear,
    mode: str,
    *,
    quantize_device: torch.device | str | None = None,
  ) -> "QW8A8Linear":
    """Pad and quantize a BF16 Linear weight once, then build the runtime module.

    Padding is deliberately done in BF16 before ``fp8_quantize``. A padded row,
    column, or 128x128 block therefore has amax zero, scale zero and qdata zero;
    its bias is also explicitly initialized to zero. This is stronger and simpler
    than padding only at GEMM time: both weight quantization and every later
    activation/GEMM call see one consistent padded layout. The public logical
    ``in_features/out_features`` remain unchanged and are used to depad output
    and diagnostic dequantization.

    :param linear: Source layer with a 2D BF16 weight. Any positive K/N is
      accepted; alignment is handled by ``quantized_in/out_features``.
    :param mode: ``per_row``, ``per_tensor`` or ``per_block``. Per-row weight
      scales are per output channel; per-block uses 128x128 weight blocks.
    :param quantize_device: Device for the one-time reduction and E4M3 cast;
      defaults to the weight device, or current CUDA for a CPU source. Buffers
      are copied back to the original weight device afterward.
    :returns: An online W8A8 module whose qweight/scales are derived,
      non-persistent buffers. No calibrated or serialized FP8 checkpoint is
      produced.
    """

    if not isinstance(linear, nn.Linear):
      raise TypeError(f"QW8A8Linear.from_linear expects nn.Linear, got {type(linear)}.")
    weight = linear.weight.detach()
    if weight.dim() != 2:
      raise ValueError("FP8 quantization requires a 2D linear weight.")
    if weight.dtype != torch.bfloat16:
      raise ValueError(f"cache-dit FP8 requires bfloat16 weights, got {weight.dtype}.")
    out_features, in_features = weight.shape
    if quantize_device is None:
      quantize_device = weight.device if weight.is_cuda else torch.device("cuda")

    module = cls(
      in_features=in_features,
      out_features=out_features,
      mode=mode,
      bias=linear.bias is not None,
      device=weight.device,
      dtype=torch.bfloat16,
    )
    quant_weight = weight.to(device=quantize_device, dtype=torch.bfloat16).contiguous()
    quant_weight = F.pad(
      quant_weight,
      (0, module.quantized_in_features - in_features, 0,
       module.quantized_out_features - out_features),
    )
    qweight, wscales = fp8_quantize(quant_weight, mode=mode, is_weight=True)
    module.qweight.copy_(qweight.to(weight.device))
    module.wscales.copy_(wscales.to(weight.device))
    if linear.bias is not None:
      module.bias.data.zero_()
      module.bias.data[:out_features].copy_(linear.bias.detach().to(device=weight.device,
                                                                    dtype=torch.bfloat16))
    module._is_inner_quantized = True
    return module

  def forward_quant(
    self,
    qx: torch.Tensor,
    sx: torch.Tensor,
    leading_shape: tuple[int, ...],
    activation: str | None = None,
  ) -> torch.Tensor:
    """Evaluate a padded W8A8 GEMM from an already quantized activation.

    :param qx: E4M3 activation ``[M, quantized_in_features]``.
    :param sx: FP32 scales matching this layer's granularity.
    :param leading_shape: Original activation dimensions before flattening.
    :param activation: Optional exact/tanh GELU epilogue.
    :returns: BF16 output with its original leading dimensions and logical width.
    """

    out = fp8_gemm(qx,
                   self.qweight,
                   sx,
                   self.wscales,
                   bias=self.bias,
                   mode=self.mode,
                   activation=activation)
    out = out[:, :self.out_features]
    return out.reshape(*leading_shape, self.out_features)

  def forward(self, x: torch.Tensor, activation: str | None = None) -> torch.Tensor:
    """Dynamically quantize activation and evaluate padded W8A8 GEMM.

    The steps are ``[... ,K] -> flatten [M,K] -> zero-pad K -> FP8 quantize ->
    CUTLASS GEMM -> depad N -> restore leading dimensions``. Weight qdata and
    scales are reused from ``from_linear``; only activation amax/qdata are
    produced here. The scale granularity follows ``self.mode``, so per-block
    activation uses 1x128 scales while the cached weight uses 128x128 scales.

    :param x: BF16 activation ``[..., in_features]`` on CUDA. Arbitrary leading
      dimensions and unaligned K are accepted.
    :param activation: Optional ``gelu`` or ``gelu_tanh`` GEMM epilogue.
    :returns: BF16 output ``[..., out_features]``. Padding columns never leak
      into the public shape.
    """

    if x.dtype != torch.bfloat16:
      raise ValueError(f"QW8A8Linear expects bfloat16 activations, got {x.dtype}.")
    if self.qweight.dtype != torch.float8_e4m3fn or self.wscales.dtype != torch.float32:
      raise RuntimeError("QW8A8Linear buffers were cast after quantization; do not call "
                         "`.to(dtype)` on a quantized module.")
    leading_shape = x.shape[:-1]
    flat = x.reshape(-1, self.in_features)
    if self.quantized_in_features != self.in_features:
      flat = F.pad(flat, (0, self.quantized_in_features - self.in_features))
    # Both branches implement the same amax/448 recipe. Eager uses one custom-op
    # boundary; compile uses ordinary torch ops so Inductor can fuse quantization
    # with neighboring pointwise/reduction work. fp8_gemm below still selects
    # the CUTLASS collective by default in either execution mode.
    if torch.compiler.is_compiling():
      qx, sx = fp8_quantize_torch(flat, mode=self.mode)
    else:
      qx, sx = fp8_quantize(flat, mode=self.mode)
    return self.forward_quant(qx, sx, leading_shape, activation=activation)

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
      weight = weight * _expand_block_scale(self.wscales, self.quantized_out_features,
                                            self.quantized_in_features, True)
    return weight[:self.out_features, :self.in_features].to(torch.bfloat16)

  def extra_repr(self) -> str:
    """:returns: Compact module description."""

    return (f"in_features={self.in_features}, out_features={self.out_features}, "
            f"mode={self.mode}, bias={self.bias is not None}")
