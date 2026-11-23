"""SVDQ-W4A8: E4M3 activations, packed E2M1 residual weights and a BF16 low-rank branch.

The residual uses one FP32 scale per output channel and is packed with even-k in the low nibble and
odd-k in the high nibble. The activation producer is exactly the W8A8 producer, including its fused
LoRA-down projection; only the residual-weight format and GEMM differ.
"""

import torch
from torch import nn

from ...kernels import svdq_gemm_w4a8
from ...kernels import svdq_quantize_w4a8
from .lowrank import decompose_lowrank_residual
from .packing import fp_quantize
from .quantizer import _compute_weight_only_smooth_scale
from .quantizer import _normalize_dtype
from .quantizer import compute_smooth_scale

FP4_EPS = 1e-12
SVDQ_W4A8_RANKS = (0, 32, 64, 128)
E2M1_CODEBOOK = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
                 -6.0)


def _pad_to(value: int, multiple: int) -> int:
  return (value + multiple - 1) // multiple * multiple


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
  """Pack E2M1 code indices, with even-k in the low and odd-k in the high nibble.

  :param codes: Integer code indices whose last dimension is even.
  :returns: A contiguous uint8 tensor with half the last-dimension length.
  """
  if codes.shape[-1] % 2:
    raise ValueError("E2M1 packing requires an even K extent.")
  return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8).contiguous()


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
  """Unpack uint8 nibbles into one E2M1 code index per logical element."""
  return torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2).contiguous()


def dequantize_e2m1(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
  """Expand packed E2M1 codes to FP32 and apply the per-row dequantization scale."""
  codebook = torch.tensor(E2M1_CODEBOOK, dtype=torch.float32, device=packed.device)
  codes = unpack_e2m1(packed).long()
  return codebook[codes] * scale.float()[:, None]


def quantize_residual_fp4_per_row(residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Quantize and nibble-pack a residual using one FP32 scale per output channel.

  :param residual: Floating-point residual ``[N,K]`` after the low-rank split.
  :returns: ``(qweight[N,K/2] uint8, wscales[N] FP32)``. Rounding uses the shared SVDQ LUT
    nearest-neighbor rule, including its tie-to-lower-index behavior.
  """
  residual = residual.float()
  rows, channels = residual.shape
  if channels % 2:
    raise ValueError("E2M1 residual quantization requires an even K extent.")
  qweight = torch.empty((rows, channels // 2), dtype=torch.uint8, device=residual.device)
  wscales = torch.empty(rows, dtype=torch.float32, device=residual.device)
  # The LUT computes [rows,K,16] distances; cap that temporary to about 192 MiB at K=3072 instead
  # of allocating several GiB for a large FLUX projection.
  for begin in range(0, rows, 1024):
    chunk = residual[begin:begin + 1024]
    scale = chunk.abs().amax(dim=1, keepdim=True).clamp_min(FP4_EPS) / 6.0
    qweight[begin:begin + 1024] = pack_e2m1(fp_quantize(chunk / scale))
    wscales[begin:begin + 1024] = scale.squeeze(1)
  return qweight, wscales


class SVDQW4A8Linear(nn.Module):
  """Runtime module for SVDQ-W4A8 with packed E2M1 residual weights.

  Logical N/K are preserved while storage is padded to the kernel's 128-wide geometry. `qweight`
  has shape `[padded_N,padded_K/2]`; the other floating-point tensors retain their W8A8 layout.
  """

  def __init__(
    self,
    in_features: int,
    out_features: int,
    rank: int = 32,
    bias: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
  ) -> None:
    """Initialize an empty packed W4A8 module.

    :param in_features: Logical input width of the source linear.
    :param out_features: Logical output width of the source linear.
    :param rank: Low-rank correction rank; one of 0, 32, 64 or 128.
    :param bias: Whether to keep a BF16 bias.
    :param torch_dtype: Dtype of bias and low-rank factors. Only BF16 is implemented.
    :param device: Allocation device; CPU when omitted.
    """
    super().__init__()
    if rank not in SVDQ_W4A8_RANKS:
      raise ValueError(f"SVDQ-W4A8 supports rank 0/32/64/128, got {rank}.")
    if torch_dtype != torch.bfloat16:
      raise ValueError("SVDQW4A8Linear currently requires torch_dtype=torch.bfloat16.")
    if device is None:
      device = torch.device("cpu")
    self.in_features = in_features
    self.out_features = out_features
    self.rank = rank
    self.quantized_in_features = _pad_to(in_features, 128)
    self.quantized_out_features = _pad_to(out_features, 128)
    kp, np_ = self.quantized_in_features, self.quantized_out_features
    self.register_buffer("qweight", torch.zeros((np_, kp // 2), dtype=torch.uint8, device=device))
    self.register_buffer("wscales", torch.zeros(np_, dtype=torch.float32, device=device))
    self.register_buffer("smooth_factor", torch.ones(kp, dtype=torch_dtype, device=device))
    self.register_buffer("proj_down", torch.zeros((rank, kp), dtype=torch_dtype, device=device))
    self.register_buffer("proj_up", torch.zeros((np_, rank), dtype=torch_dtype, device=device))
    self.bias = nn.Parameter(torch.zeros(np_, dtype=torch_dtype, device=device),
                             requires_grad=False) if bias else None

  @classmethod
  def from_linear(
    cls,
    linear: nn.Linear,
    *,
    rank: int = 32,
    smooth_factor: torch.Tensor | None = None,
    alpha: float = 0.5,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
    calibrate_precision: str = "low",
  ) -> "SVDQW4A8Linear":
    """Quantize a float linear layer into the packed W4A8 format.

    :param linear: Source floating-point linear layer.
    :param rank: Low-rank correction rank.
    :param smooth_factor: Optional per-input-channel smoothing vector; `None` uses weight-only
      SmoothQuant with exponent `alpha`.
    :param alpha: SmoothQuant interpolation factor for the weight-only fallback.
    :param torch_dtype: Dtype for bias and low-rank factors.
    :param device: Device on which to materialize the packed module.
    :param calibrate_precision: Precision policy for the low-rank SVD.
    :returns: The packed module.
    """
    device = torch.device(device or linear.weight.device)
    module = cls(linear.in_features,
                 linear.out_features,
                 rank=rank,
                 bias=linear.bias is not None,
                 torch_dtype=torch_dtype,
                 device=device)
    weight = linear.weight.detach().to(device=device, dtype=torch.float32)
    n, k = weight.shape
    if smooth_factor is None:
      weight_span = weight.abs().amax(dim=0)
      smooth = _compute_weight_only_smooth_scale(weight_span,
                                                 alpha=alpha,
                                                 math_dtype=torch.float32,
                                                 output_dtype=torch.float32)
    else:
      smooth = smooth_factor.to(device=device, dtype=torch.float32)
    if smooth.numel() != k:
      raise ValueError(f"smooth_factor must have {k} entries, got {smooth.numel()}.")
    smoothed = weight * smooth.unsqueeze(0)
    down, up, residual = decompose_lowrank_residual(smoothed,
                                                    rank,
                                                    output_dtype=torch.float32,
                                                    svd_precision=calibrate_precision)
    qweight, wscales = quantize_residual_fp4_per_row(residual)
    proj_down = (down.to(torch.float64) / smooth.to(torch.float64).unsqueeze(0)).to(torch_dtype)

    module.qweight.zero_()
    module.wscales.zero_()
    module.proj_up.zero_()
    module.proj_down.zero_()
    module.smooth_factor.fill_(1.0)
    module.qweight[:n, :k // 2] = qweight
    module.wscales[:n] = wscales
    module.smooth_factor[:k] = smooth.to(torch_dtype)
    if rank:
      module.proj_down[:, :k] = proj_down
      module.proj_up[:n, :] = up.to(torch_dtype)
    if module.bias is not None:
      module.bias.zero_()
      module.bias[:n] = linear.bias.detach().to(device=device, dtype=torch_dtype)
    return module

  def dequantize_weight(self) -> torch.Tensor:
    """Return the effective FP32 weight sliced to the logical shape, for diagnostics only."""
    n, k = self.out_features, self.in_features
    residual = dequantize_e2m1(self.qweight[:n, :(k + 1) // 2], self.wscales[:n])[:, :k]
    smooth = self.smooth_factor[:k].float()
    weight = residual / smooth.unsqueeze(0)
    if self.rank:
      weight = weight + self.proj_up[:n].float() @ self.proj_down[:, :k].float()
    return weight

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Run the shared FP8 producer followed by the packed-W4A8 fused GEMM.

    :param x: Activation whose last dimension equals `in_features`.
    :returns: BF16 output with `out_features` channels and the same leading dimensions as `x`.
    """
    shape = x.shape
    rows = x.numel() // shape[-1]
    if rows == 0:
      return x.new_empty((*shape[:-1], self.out_features))
    flat = x.reshape(rows, shape[-1])
    if flat.shape[1] != self.quantized_in_features:
      flat = torch.nn.functional.pad(flat, (0, self.quantized_in_features - flat.shape[1]))
    flat = flat.contiguous()
    qact, ascales, lora_act = svdq_quantize_w4a8(flat,
                                                 self.smooth_factor,
                                                 self.proj_down if self.rank else None,
                                                 pad_rows=True)
    out = svdq_gemm_w4a8(qact,
                         self.qweight,
                         ascales,
                         self.wscales,
                         lora_act_in=lora_act if self.rank else None,
                         lora_up=self.proj_up if self.rank else None,
                         bias=self.bias)
    return out[:rows, :self.out_features].reshape(*shape[:-1], self.out_features)


def quantize_linear_svdq_w4a8(
  linear: nn.Linear,
  representative_activations=None,
  *,
  rank: int = 32,
  alpha: float = 0.5,
  smooth_strategy: str = "identity",
  torch_dtype: torch.dtype | None = None,
  device: str | torch.device | None = None,
  calibrate_precision: str = "low",
) -> SVDQW4A8Linear:
  """Quantize one BF16 linear into W4A8 using DQ or activation-calibrated PTQ.

  :param linear: Source BF16 linear layer.
  :param representative_activations: Activation tensor(s) required by `activation` smoothing.
  :param rank: Low-rank correction rank (0, 32, 64 or 128).
  :param alpha: SmoothQuant interpolation factor.
  :param smooth_strategy: `activation`, `identity`, `weight` or `weight_inv`.
  :param torch_dtype: Packed module floating-point dtype; only BF16 is supported.
  :param device: Target device for the packed module.
  :param calibrate_precision: Precision policy for the low-rank SVD.
  :returns: The packed `SVDQW4A8Linear`.
  """
  if not isinstance(linear, nn.Linear):
    raise TypeError(f"Expected nn.Linear, got {type(linear)}.")
  if type(linear.weight).__name__ == "DTensor":
    raise NotImplementedError("SVDQ-W4A8 quantization does not support DTensor/TP weights.")
  if linear.weight.dtype != torch.bfloat16:
    raise ValueError(f"SVDQ-W4A8 requires BF16 source weights, got {linear.weight.dtype}.")
  if linear.in_features % 2:
    raise ValueError("SVDQ-W4A8 requires an even in_features extent for E2M1 packing.")
  device = torch.device(device or linear.weight.device)
  torch_dtype = _normalize_dtype(torch_dtype, device)
  k = linear.in_features
  if smooth_strategy == "activation":
    if representative_activations is None:
      raise ValueError("smooth_strategy='activation' requires representative activations.")
    acts = representative_activations
    if isinstance(acts, torch.Tensor):
      acts = [acts]
    span = None
    for tensor in acts:
      flat = tensor.detach().reshape(-1, tensor.shape[-1]).to(device=device, dtype=torch.float32)
      current = flat.abs().amax(dim=0)
      span = current if span is None else torch.maximum(span, current)
    weight_span = linear.weight.detach().to(device=device, dtype=torch.float32).abs().amax(dim=0)
    smooth = compute_smooth_scale(span,
                                  weight_span,
                                  alpha=alpha,
                                  math_dtype=torch.float32,
                                  output_dtype=torch.float32)
  elif smooth_strategy == "identity":
    smooth = torch.ones(k, dtype=torch.float32, device=device)
  elif smooth_strategy in ("weight", "weight_inv"):
    weight_span = linear.weight.detach().to(device=device, dtype=torch.float32).abs().amax(dim=0)
    smooth = _compute_weight_only_smooth_scale(weight_span,
                                               alpha=alpha,
                                               inverse=smooth_strategy == "weight_inv",
                                               math_dtype=torch.float32,
                                               output_dtype=torch.float32)
  else:
    raise ValueError(f"Unsupported SVDQ-W4A8 smooth strategy: {smooth_strategy!r}.")
  return SVDQW4A8Linear.from_linear(linear,
                                    rank=rank,
                                    smooth_factor=smooth,
                                    alpha=alpha,
                                    torch_dtype=torch_dtype,
                                    device=device,
                                    calibrate_precision=calibrate_precision)


def quantize_svdq_w4a8(module: nn.Module, quantize_config) -> nn.Module:
  """Quantize a module through the shared SVDQ-FP8-family PTQ/DQ and checkpoint flow."""
  from .fp8 import _quantize_svdq_fp8_family

  return _quantize_svdq_fp8_family(module, quantize_config, quantize_linear_svdq_w4a8, "w4a8")
