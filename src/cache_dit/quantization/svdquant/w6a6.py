"""SVDQ-W6A6 quantization: packed E2M3 runtime module and offline quantizer.

Both operands use the OCP MXFP6 encoding, so the residual weight and the activation share one code
layout and one block-scaled MMA::

    W * s = proj_up @ (proj_down * s) + residual
    wsf[n,g]     = ue8m0(2^ceil(log2(amax_{k in g} |residual[n,:]| / 7.5)))     g = k / 32
    qweight      = satfinite_e2m3(residual / 2^(wsf-127)),  four codes per three bytes
    proj_down    = L2 / s

The forward pass is fully fused: `svdq_quantize_w6a6` produces the packed activation, its UE8M0 row
and the BF16 LoRA-down term in one kernel, and `svdq_gemm_w6a6` feeds both scale rows into the MX
block-scaled MMA while adding the LoRA-up term into the same FP32 accumulator. There is no standalone
BF16 GEMM, no blockwise promotion in the mainloop, and no second pass over the dense output.
"""

import functools

import torch
from torch import nn

from ...kernels import svdq_gemm_w6a6
from ...kernels import svdq_quantize_w6a6
from .lowrank import decompose_lowrank_residual
from .quantizer import _compute_weight_only_smooth_scale
from .quantizer import _normalize_dtype
from .quantizer import compute_smooth_scale

FP6_EPS = 1e-12
E2M3_MAX = 7.5
E2M3_SF_VEC = 32
SVDQ_W6A6_RANKS = (0, 32, 64, 128)


def _pad_to(value: int, multiple: int) -> int:
  return (value + multiple - 1) // multiple * multiple


def _e2m3_magnitudes(device=None) -> torch.Tensor:
  """Return the 32 non-negative E2M3 magnitudes ordered by code ``0x00`` ..

  ``0x1f``.
  """
  exponents = torch.arange(4, dtype=torch.float64, device=device)
  mantissas = torch.arange(8, dtype=torch.float64, device=device) / 8.0
  # Exponent field 0 is the denormal range (m/8); fields 1..3 scale 1 + m/8 by 2^(e-1).
  denormal = mantissas
  normal = (2.0 ** (exponents[1:] - 1.0)).unsqueeze(1) * (1.0 + mantissas).unsqueeze(0)
  return torch.cat((denormal, normal.reshape(-1))).to(torch.float32)


def _e2m3_thresholds(device=None) -> torch.Tensor:
  magnitudes = _e2m3_magnitudes(device).to(torch.float64)
  return ((magnitudes[:-1] + magnitudes[1:]) * 0.5).to(torch.float32)


def quantize_e2m3(values: torch.Tensor) -> torch.Tensor:
  """Round FP32 values already normalized to ``[-7.5, 7.5]`` onto the E2M3 grid.

  :param values: Tensor of normalized residual or activation values.
  :returns: ``uint8`` code tensor with the sign in bit 5 and ties resolved to the even code, which is
    what the device conversion (`cvt.rn.satfinite.e2m3x2`) does. `quantize_e2m3` and the kernel agree
    bit for bit on exact midpoints, so an offline weight and a runtime activation follow one rule.
  """
  magnitude = values.abs().clamp_max(E2M3_MAX)
  thresholds = _e2m3_thresholds(values.device)
  # right=False returns the lower code on an exact midpoint; half-to-even then moves odd codes up.
  codes = torch.bucketize(magnitude.float(), thresholds, right=False)
  midpoint = magnitude == thresholds[codes.clamp_max(thresholds.numel() - 1)]
  codes = torch.where(midpoint & (codes % 2 == 1), codes + 1, codes)
  codes = torch.where(values < 0, codes | 0x20, codes)
  return codes.to(torch.uint8)


def pack_e2m3(codes: torch.Tensor) -> torch.Tensor:
  """Pack E2M3 codes four per three bytes, little-endian within each 24-bit group.

  :param codes: ``uint8`` codes whose last dimension is a multiple of four.
  :returns: ``uint8`` tensor whose last dimension is ``3/4`` of the input width.
  """
  if codes.shape[-1] % 4:
    raise ValueError("E2M3 packing requires a K extent divisible by 4.")
  groups = codes.to(torch.int32).reshape(*codes.shape[:-1], codes.shape[-1] // 4, 4)
  byte0 = groups[..., 0] | (groups[..., 1] << 6)
  byte1 = (groups[..., 1] >> 2) | (groups[..., 2] << 4)
  byte2 = (groups[..., 2] >> 4) | (groups[..., 3] << 2)
  packed = torch.stack((byte0, byte1, byte2), dim=-1).to(torch.uint8)
  return packed.flatten(-2).contiguous()


def unpack_e2m3(packed: torch.Tensor) -> torch.Tensor:
  """Expand packed three-byte groups back into one E2M3 code per logical element."""
  if packed.shape[-1] % 3:
    raise ValueError("E2M3 unpacking requires a packed width divisible by 3.")
  groups = packed.to(torch.int32).reshape(*packed.shape[:-1], packed.shape[-1] // 3, 3)
  code0 = groups[..., 0] & 0x3F
  code1 = ((groups[..., 0] >> 6) | (groups[..., 1] << 2)) & 0x3F
  code2 = ((groups[..., 1] >> 4) | (groups[..., 2] << 4)) & 0x3F
  code3 = (groups[..., 2] >> 2) & 0x3F
  codes = torch.stack((code0, code1, code2, code3), dim=-1).to(torch.uint8)
  return codes.flatten(-2).contiguous()


def ue8m0_codes(block_amax: torch.Tensor) -> torch.Tensor:
  """Map block amax values onto the smallest UE8M0 scale that keeps codes inside ``[-7.5, 7.5]``.

  :param block_amax: Non-negative per-block maxima, any shape.
  :returns: ``uint8`` exponent bytes with bias 127; an all-zero block maps to 127 (scale 1.0).
  """
  target = (block_amax.float() / E2M3_MAX).clamp_min(FP6_EPS)
  mantissa, exponent = torch.frexp(target)
  # frexp returns target = mantissa * 2^exponent with mantissa in [0.5, 1), so ceil(log2(target)) is
  # `exponent` unless the target is an exact power of two (mantissa == 0.5).
  shift = torch.where(mantissa == 0.5, exponent - 1, exponent)
  codes = (shift + 127).clamp(1, 254)
  return torch.where(block_amax > 0, codes, torch.full_like(codes, 127)).to(torch.uint8)


def ue8m0_scales(codes: torch.Tensor) -> torch.Tensor:
  """Return the FP32 multiplier of each UE8M0 code, i.e. ``2**(code-127)``."""
  return torch.ldexp(torch.ones_like(codes, dtype=torch.float32), codes.to(torch.int32) - 127)


def dequantize_e2m3(packed: torch.Tensor, wsf: torch.Tensor) -> torch.Tensor:
  """Expand packed E2M3 codes to FP32 and apply the per-32 UE8M0 micro-scales."""
  codes = unpack_e2m3(packed).long()
  table = _e2m3_magnitudes(packed.device)
  values = table[codes & 0x1F] * torch.where(codes >= 0x20, -1.0, 1.0)
  scale = ue8m0_scales(wsf).repeat_interleave(E2M3_SF_VEC, dim=-1)
  return values.float() * scale[..., :values.shape[-1]]


def quantize_residual_fp6(residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Quantize and pack a residual weight matrix with 32-wide UE8M0 block scales.

  :param residual: FP32 residual ``[N,K]`` after the low-rank split; K must be a multiple of 128.
  :returns: ``(packed[N,K*3/4] uint8, wsf[N,K/32] uint8)``.
  """
  residual = residual.float()
  rows, channels = residual.shape
  if channels % (E2M3_SF_VEC * 4):
    raise ValueError("E2M3 residual quantization requires a K extent divisible by 128.")
  qweight = torch.empty((rows, channels * 3 // 4), dtype=torch.uint8, device=residual.device)
  wsf = torch.empty((rows, channels // E2M3_SF_VEC), dtype=torch.uint8, device=residual.device)
  # Cap the [rows,K,32] temporary instead of materializing several GiB for a large projection.
  for begin in range(0, rows, 1024):
    chunk = residual[begin:begin + 1024]
    blocks = chunk.reshape(chunk.shape[0], channels // E2M3_SF_VEC, E2M3_SF_VEC)
    codes = ue8m0_codes(blocks.abs().amax(dim=-1))
    scale = ue8m0_scales(codes).unsqueeze(-1)
    qweight[begin:begin + 1024] = pack_e2m3(quantize_e2m3(blocks / scale).flatten(1))
    wsf[begin:begin + 1024] = codes
  return qweight, wsf


class SVDQW6A6Linear(nn.Module):
  """Runtime module for packed SVDQ-W6A6 linear layers.

  Buffers are stored padded to the kernel's 128-wide geometry; the original feature sizes are kept so
  `forward` can slice the result back. `proj_down` already contains the inverse smoothing factor,
  which is what allows the fused producer to run the low-rank branch on the raw activation.
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
    """Initialize an empty packed W6A6 module.

    :param in_features: Logical input width of the source linear.
    :param out_features: Logical output width of the source linear.
    :param rank: Low-rank correction rank; one of 0, 32, 64 or 128.
    :param bias: Whether to keep a BF16 bias.
    :param torch_dtype: Dtype of the bias and low-rank factors. Only bfloat16 is implemented.
    :param device: Allocation device; CPU when omitted.
    """
    super().__init__()
    if rank not in SVDQ_W6A6_RANKS:
      raise ValueError(f"SVDQ-W6A6 supports rank 0/32/64/128, got {rank}.")
    if torch_dtype != torch.bfloat16:
      raise ValueError("SVDQW6A6Linear currently requires torch_dtype=torch.bfloat16.")
    if device is None:
      device = torch.device("cpu")
    self.in_features = in_features
    self.out_features = out_features
    self.rank = rank
    self.quantized_in_features = _pad_to(in_features, 128)
    self.quantized_out_features = _pad_to(out_features, 128)
    kp, np_ = self.quantized_in_features, self.quantized_out_features
    self.register_buffer("qweight", torch.zeros((np_, kp * 3 // 4),
                                                dtype=torch.uint8,
                                                device=device))
    self.register_buffer(
      "wsf", torch.full((np_, kp // E2M3_SF_VEC), 127, dtype=torch.uint8, device=device))
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
  ) -> "SVDQW6A6Linear":
    """Quantize a float linear layer into the packed W6A6 format.

    :param linear: Source floating-point linear layer.
    :param rank: Low-rank correction rank.
    :param smooth_factor: Optional calibration-derived per-input-channel smoothing vector. When
      omitted, a weight-only SmoothQuant factor with exponent `alpha` is used (the DQ path).
    :param alpha: SmoothQuant interpolation factor for the weight-only fallback.
    :param torch_dtype: Dtype for the bias and low-rank factors.
    :param device: Device on which to materialize the packed module.
    :param calibrate_precision: Precision policy for the low-rank SVD (`low`/`medium`/`high`).
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
    kp, np_ = module.quantized_in_features, module.quantized_out_features
    padded = torch.zeros((np_, kp), dtype=torch.float32, device=residual.device)
    padded[:n, :k] = residual
    qweight, wsf = quantize_residual_fp6(padded)
    # The inverse smoothing is folded in float64 exactly like the other exporters: dropping this
    # division would leave the whole low-rank branch off by one factor of s.
    proj_down = (down.to(torch.float64) / smooth.to(torch.float64).unsqueeze(0)).to(torch_dtype)

    module.qweight.zero_()
    module.wsf.fill_(127)
    module.proj_up.zero_()
    module.proj_down.zero_()
    module.smooth_factor.fill_(1.0)
    module.qweight.copy_(qweight)
    module.wsf.copy_(wsf)
    module.smooth_factor[:k] = smooth.to(torch_dtype)
    if rank:
      module.proj_down[:, :k] = proj_down
      module.proj_up[:n, :] = up.to(torch_dtype)
    if module.bias is not None:
      module.bias.zero_()
      module.bias[:n] = linear.bias.detach().to(device=device, dtype=torch_dtype)
    return module

  def dequantize_weight(self) -> torch.Tensor:
    """Return the effective full-precision weight, sliced back to the logical shape.

    :returns: Diagnostic reconstruction ``[out_features, in_features]``; not used by `forward`.
    """
    n, k = self.out_features, self.in_features
    residual = dequantize_e2m3(self.qweight, self.wsf)[:n, :k]
    smooth = self.smooth_factor[:k].float()
    weight = residual / smooth.unsqueeze(0)
    if self.rank:
      weight = weight + self.proj_up[:n].float() @ self.proj_down[:, :k].float()
    return weight

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Run the fused producer + block-scaled GEMM chain.

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
    qact, asf, lora_act = svdq_quantize_w6a6(flat,
                                             self.smooth_factor,
                                             self.proj_down if self.rank else None,
                                             pad_rows=True)
    out = svdq_gemm_w6a6(qact,
                         self.qweight,
                         asf,
                         self.wsf,
                         lora_act_in=lora_act if self.rank else None,
                         lora_up=self.proj_up if self.rank else None,
                         bias=self.bias)
    out = out[:rows, :self.out_features]
    return out.reshape(*shape[:-1], self.out_features)


def _quantize_svdq_w6a6(
  linear: nn.Linear,
  representative_activations=None,
  *,
  rank: int = 32,
  alpha: float = 0.5,
  smooth_strategy: str = "identity",
  torch_dtype: torch.dtype | None = None,
  device: str | torch.device | None = None,
  calibrate_precision: str = "low",
) -> SVDQW6A6Linear:
  """Quantize a linear layer into SVDQ-W6A6, with or without calibration data.

  :param linear: Source floating-point linear layer.
  :param representative_activations: Optional tensor or iterable of tensors used to derive the
    SmoothQuant factor when `smooth_strategy` is `activation`.
  :param rank: Low-rank correction rank (0, 32, 64 or 128).
  :param alpha: SmoothQuant interpolation factor.
  :param smooth_strategy: `activation`, `identity`, `weight` or `weight_inv`; `identity` disables
    smoothing, the two weight strategies derive it from the weight alone (DQ).
  :param torch_dtype: Dtype for the packed module's floating-point tensors.
  :param device: Target device for the packed module.
  :param calibrate_precision: Precision policy for the low-rank SVD.
  :returns: The packed `SVDQW6A6Linear`.
  """
  if not isinstance(linear, nn.Linear):
    raise TypeError(f"Expected nn.Linear, got {type(linear)}.")
  if type(linear.weight).__name__ == "DTensor":
    raise NotImplementedError("SVDQ-W6A6 quantization does not support DTensor/TP weights.")
  if linear.weight.dtype != torch.bfloat16:
    raise ValueError(f"SVDQ-W6A6 requires BF16 source weights, got {linear.weight.dtype}.")
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
    raise ValueError(f"Unsupported SVDQ-W6A6 smooth strategy: {smooth_strategy!r}.")
  return SVDQW6A6Linear.from_linear(linear,
                                    rank=rank,
                                    smooth_factor=smooth,
                                    alpha=alpha,
                                    torch_dtype=torch_dtype,
                                    device=device,
                                    calibrate_precision=calibrate_precision)


def _get_svdq_w6a6_quantizer(precision: str):
  """Return the module-level quantizer for an MXFP6 manifest precision or alias."""
  from .wNaN import _quantize_svdq_wNaN

  quantizers = {
    # FP6 is the encoding alias; W6A6 is the weight/activation semantic name.
    "fp6":
    functools.partial(_quantize_svdq_wNaN, quantize_linear=_quantize_svdq_w6a6, precision="fp6"),
    "w6a6":
    functools.partial(_quantize_svdq_wNaN, quantize_linear=_quantize_svdq_w6a6, precision="w6a6"),
  }
  try:
    return quantizers[precision]
  except KeyError as exc:
    raise ValueError(f"Unsupported SVDQ FP6 precision: {precision!r}.") from exc
