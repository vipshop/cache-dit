from __future__ import annotations

import dataclasses
import typing as tp

import torch
from torch import nn

from .linear import SVDQW4A4Linear
from .lowrank import decompose_lowrank_residual
from .packing import adapt_svdq_module_state_dict
from .packing import export_raw_svdq_w4a4_state_dict

CalibrationInputs = torch.Tensor | tp.Iterable[torch.Tensor]

__all__ = [
  "CalibrationInputs",
  "compute_smooth_scale",
  "quantize_linear_svdq_w4a4",
  "standardize_calibration_activations",
  "validate_svdq_linear_geometry",
]

_CALIBRATE_PRECISIONS = ("low", "medium", "high")
_SVDQ_SMOOTH_STRATEGIES = ("activation", "identity", "weight", "weight_inv")
_WEIGHT_ONLY_SMOOTH_CLAMP_RANGE = (0.25, 4.0)


def _resolve_svdq_quant_mode(quant_type: str | None) -> str:
  if quant_type is None:
    return "ptq"
  if not isinstance(quant_type, str):
    raise TypeError(f"quant_type must be a str or None, got {type(quant_type)}.")
  return "dq" if quant_type.lower().endswith("_dq") else "ptq"


def validate_svdq_linear_geometry(
  in_features: int,
  out_features: int,
  *,
  rank: int,
  precision: str = "int4",
) -> None:
  """Validate that a linear layer fits the migrated SVDQ W4A4 runtime.

  :param in_features: Logical input width of the candidate linear layer.
  :param out_features: Logical output width of the candidate linear layer.
  :param rank: Requested low-rank residual rank.
  :param precision: Weight format requested by the quantizer.

  :raises NotImplementedError: If `precision` is not supported by the minimal SVDQ
    quantizer.
  :raises ValueError: If the geometry or low-rank rank is incompatible with the W4A4
    packer/runtime contract.
  """

  if precision != "int4":
    raise NotImplementedError("The minimal SVDQuant quantizer currently supports INT4 only.")
  if in_features % 64 != 0:
    raise ValueError(f"INT4 SVDQuant requires in_features divisible by 64, got {in_features}.")
  if in_features % 128 != 0:
    raise ValueError(
      f"The migrated W4A4 packer/runtime requires in_features divisible by 128, got {in_features}.")
  if out_features % 128 != 0:
    raise ValueError(
      f"The migrated W4A4 packer/runtime requires out_features divisible by 128, got {out_features}."
    )
  if rank < 0:
    raise ValueError(f"rank must be non-negative, got {rank}.")
  if rank != 0 and rank % 16 != 0:
    raise ValueError(f"The migrated W4A4 runtime requires rank 0 or a multiple of 16, got {rank}.")


def _normalize_dtype(torch_dtype: torch.dtype | None, device: torch.device | str) -> torch.dtype:
  if torch_dtype is not None:
    if torch_dtype not in (torch.float16, torch.bfloat16):
      raise ValueError(f"torch_dtype must be torch.float16 or torch.bfloat16, got {torch_dtype}.")
    return torch_dtype
  if isinstance(device, str):
    device = torch.device(device)
  if device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    return torch.bfloat16
  return torch.float16


@torch.inference_mode()
def _iter_standardized_calibration_activations(
  representative_activations: CalibrationInputs,
  *,
  in_features: int,
) -> tp.Iterator[torch.Tensor]:
  tensors = ((representative_activations, ) if isinstance(representative_activations, torch.Tensor)
             else representative_activations)
  found_tensor = False
  for tensor in tensors:
    found_tensor = True
    if not isinstance(tensor, torch.Tensor):
      raise TypeError("representative_activations must be a tensor or a sequence of tensors.")
    if tensor.shape[-1] != in_features:
      raise ValueError(
        f"Expected representative activations with last dim {in_features}, got {tensor.shape[-1]}.")
    yield tensor.reshape(-1, in_features)
  if not found_tensor:
    raise ValueError("At least one representative activation tensor is required.")


@torch.inference_mode()
def standardize_calibration_activations(
  representative_activations: CalibrationInputs,
  *,
  in_features: int,
) -> list[torch.Tensor]:
  """Normalize representative activations into `[tokens, in_features]` tensors.

  :param representative_activations: A tensor or iterable of tensors whose last
    dimension matches `in_features`.
  :param in_features: Expected activation channel dimension.

  :returns: A list of 2D tensors reshaped to `[tokens, in_features]` for calibration.
  """

  return list(
    _iter_standardized_calibration_activations(
      representative_activations,
      in_features=in_features,
    ))


def _repair_invalid_scale(scale: torch.Tensor) -> torch.Tensor:
  scale = scale.clone()
  scale[scale == 0] = 1
  scale[~torch.isfinite(scale)] = 1
  return scale


def _normalize_calibrate_precision(calibrate_precision: str) -> str:
  if not isinstance(calibrate_precision, str):
    raise TypeError(f"calibrate_precision must be a str, got {type(calibrate_precision)}.")
  normalized = calibrate_precision.lower()
  if normalized not in _CALIBRATE_PRECISIONS:
    raise ValueError(
      f"calibrate_precision must be one of {_CALIBRATE_PRECISIONS}, got {calibrate_precision!r}.")
  return normalized


def _resolve_math_dtype(torch_dtype: torch.dtype, calibrate_precision: str) -> torch.dtype:
  calibrate_precision = _normalize_calibrate_precision(calibrate_precision)
  return torch_dtype if calibrate_precision == "low" else torch.float32


def _normalize_svdq_smooth_strategy(smooth_strategy: str) -> str:
  if not isinstance(smooth_strategy, str):
    raise TypeError(f"smooth_strategy must be a str, got {type(smooth_strategy)}.")
  normalized = smooth_strategy.lower()
  if normalized not in _SVDQ_SMOOTH_STRATEGIES:
    raise ValueError(
      f"smooth_strategy must be one of {_SVDQ_SMOOTH_STRATEGIES}, got {smooth_strategy!r}.")
  return normalized


@torch.inference_mode()
def compute_smooth_scale(
  activation_span: torch.Tensor,
  weight_span: torch.Tensor,
  *,
  alpha: float = 0.5,
  math_dtype: torch.dtype = torch.float32,
  output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
  """Compute the per-input-channel smoothing factor used by SVDQ.

  :param activation_span: Per-channel activation maxima for the representative data.
  :param weight_span: Per-channel weight maxima for the source linear layer.
  :param alpha: SmoothQuant interpolation factor in `[0, 1]`.
  :param math_dtype: Intermediate dtype used for the scale computation.
  :param output_dtype: Optional dtype for the returned scale tensor.
  :returns: A 1D tensor of per-input-channel smoothing factors.
  """

  if not 0.0 <= alpha <= 1.0:
    raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
  beta = 1.0 - alpha
  smooth_scale = activation_span.to(dtype=math_dtype).pow(alpha)
  if beta > 0:
    smooth_scale = smooth_scale.div_(weight_span.to(dtype=math_dtype).pow(beta))
  smooth_scale = _repair_invalid_scale(smooth_scale)
  if output_dtype is not None:
    smooth_scale = smooth_scale.to(dtype=output_dtype)
  return smooth_scale


@torch.inference_mode()
def _compute_weight_only_smooth_scale(
  weight_span: torch.Tensor,
  *,
  alpha: float = 0.5,
  inverse: bool = False,
  math_dtype: torch.dtype = torch.float32,
  output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
  """Compute a purely weight-derived smooth factor for experimental DQ.

  This heuristic treats the activation proxy as a constant one-vector, which reduces SVDQ smoothing
  to a weight-only power law. The default direction (`inverse=False`) equalizes weight spans via
  inverse scaling; the inverse direction (`inverse=True`) intentionally pushes more quantization
  difficulty toward the weight tensor so activation channels may become easier to quantize. The
  resulting scale is then normalized to unit geometric mean and clamped to a conservative range so
  it remains numerically stable for the existing W4A4 runtime.

  :param weight_span: Per-channel weight maxima for the source linear layer.
  :param alpha: SmoothQuant interpolation factor in `[0, 1]`.
  :param inverse: Whether to flip the exponent sign and bias the heuristic toward easier activation
    quantization instead of easier weight quantization.
  :param math_dtype: Intermediate dtype used for the scale computation.
  :param output_dtype: Optional dtype for the returned scale tensor.
  :returns: A 1D tensor of per-input-channel smoothing factors.
  """

  if not 0.0 <= alpha <= 1.0:
    raise ValueError(f"alpha must be in [0, 1], got {alpha}.")

  beta = 1.0 - alpha
  if beta == 0.0:
    smooth_scale = torch.ones_like(weight_span, dtype=math_dtype)
  else:
    exponent = beta if inverse else -beta
    smooth_scale = weight_span.to(dtype=math_dtype).pow(exponent)
  smooth_scale = _repair_invalid_scale(smooth_scale)

  log_scale = smooth_scale.log()
  geometric_mean = log_scale.mean().exp()
  if torch.isfinite(geometric_mean) and geometric_mean > 0:
    smooth_scale = smooth_scale.div_(geometric_mean)

  clamp_min, clamp_max = _WEIGHT_ONLY_SMOOTH_CLAMP_RANGE
  smooth_scale = smooth_scale.clamp_(min=clamp_min, max=clamp_max)
  smooth_scale = _repair_invalid_scale(smooth_scale)
  if output_dtype is not None:
    smooth_scale = smooth_scale.to(dtype=output_dtype)
  return smooth_scale


@torch.inference_mode()
def _compute_group_scales(
  weight: torch.Tensor,
  group_size: int = 64,
  *,
  math_dtype: torch.dtype = torch.float32,
  output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
  """Compute per-group INT4 scales from the residual weight matrix.

  :param weight: Residual weight matrix with shape `[out_features, in_features]`.
  :param group_size: Number of input channels covered by each scale value.
  :param math_dtype: Intermediate dtype used during the reduction.
  :param output_dtype: Optional dtype for the returned scale tensor.
  :returns: A scale tensor with shape `[out_features, 1, num_groups, 1]` compatible with the
    packing/export path.
  """

  out_features, in_features = weight.shape
  if in_features % group_size != 0:
    raise ValueError(f"Expected in_features divisible by {group_size}, got {in_features}.")
  weight = weight.to(dtype=math_dtype).view(out_features, 1, in_features // group_size, group_size)
  scales = weight.abs().amax(dim=-1, keepdim=True).div_(7.0)
  scales = _repair_invalid_scale(scales)
  if output_dtype is not None:
    scales = scales.to(dtype=output_dtype)
  return scales


@torch.inference_mode()
def _compute_batch_activation_span(
  tensor: torch.Tensor,
  *,
  device: torch.device,
  torch_dtype: torch.dtype,
  math_dtype: torch.dtype,
) -> torch.Tensor:
  device_tensor = tensor.to(device=device, dtype=torch_dtype)
  batch_span = device_tensor.to(dtype=math_dtype).abs().amax(dim=0).to(device="cpu")
  del device_tensor
  return batch_span


@dataclasses.dataclass
class _ActivationSpanAccumulator:
  """Streaming accumulator for per-channel activation spans.

  The accumulator keeps calibration memory bounded by reducing each observed batch to a single
  `[in_features]` span vector and periodically merging buffered spans on CPU.
  """

  device: torch.device
  torch_dtype: torch.dtype
  math_dtype: torch.dtype
  flush_sample_count: int | None = 1
  flush_cpu_bytes: int | None = None
  buffered_spans: list[torch.Tensor] = dataclasses.field(default_factory=list)
  activation_span: torch.Tensor | None = None
  buffered_cpu_bytes: int = 0
  observed_samples: int = 0

  def add_tensor(self, tensor: torch.Tensor) -> None:
    self.add_span(
      _compute_batch_activation_span(
        tensor,
        device=self.device,
        torch_dtype=self.torch_dtype,
        math_dtype=self.math_dtype,
      ))

  def add_span(self, span: torch.Tensor) -> None:
    cpu_span = span.to(device="cpu", dtype=self.math_dtype)
    self.buffered_spans.append(cpu_span)
    self.buffered_cpu_bytes += cpu_span.numel() * cpu_span.element_size()
    self.observed_samples += 1
    if self._should_flush():
      self.flush()

  def _should_flush(self) -> bool:
    sample_limit_reached = (self.flush_sample_count is not None
                            and len(self.buffered_spans) >= self.flush_sample_count)
    cpu_bytes_limit_reached = (self.flush_cpu_bytes is not None
                               and self.buffered_cpu_bytes >= self.flush_cpu_bytes)
    return sample_limit_reached or cpu_bytes_limit_reached

  def flush(self) -> None:
    if not self.buffered_spans:
      return

    if len(self.buffered_spans) == 1:
      chunk_span = self.buffered_spans[0]
    else:
      chunk_span = torch.stack(self.buffered_spans, dim=0).amax(dim=0)

    if self.activation_span is None:
      self.activation_span = chunk_span
    else:
      self.activation_span = torch.maximum(self.activation_span, chunk_span)

    self.buffered_spans.clear()
    self.buffered_cpu_bytes = 0

  def finalize(self) -> torch.Tensor:
    self.flush()
    if self.activation_span is None:
      raise ValueError("At least one representative activation tensor is required.")
    return self.activation_span

  @property
  def has_observations(self) -> bool:
    return self.observed_samples > 0


@torch.inference_mode()
def _compute_activation_span(
  standardized_acts: list[torch.Tensor],
  *,
  device: torch.device,
  torch_dtype: torch.dtype,
  math_dtype: torch.dtype,
  streaming: bool,
) -> torch.Tensor:
  if streaming:
    activation_span: torch.Tensor | None = None
    for tensor in standardized_acts:
      device_tensor = tensor.to(device=device, dtype=torch_dtype)
      tensor_span = device_tensor.to(dtype=math_dtype).abs().amax(dim=0)
      if activation_span is None:
        activation_span = tensor_span
      else:
        activation_span = torch.maximum(activation_span, tensor_span)
      del device_tensor, tensor_span
    if activation_span is None:
      raise ValueError("At least one representative activation tensor is required.")
    return activation_span

  eager_acts = [tensor.to(device=device, dtype=torch_dtype) for tensor in standardized_acts]
  activation_span = torch.stack(
    [tensor.to(dtype=math_dtype).abs().amax(dim=0) for tensor in eager_acts],
    dim=0,
  ).amax(dim=0)
  del eager_acts
  return activation_span


@torch.inference_mode()
def _resolve_svdq_smooth_scale(
  weight: torch.Tensor,
  activation_span: torch.Tensor | None,
  *,
  quant_mode: str,
  smooth_strategy: str,
  in_features: int,
  alpha: float,
  device: torch.device,
  torch_dtype: torch.dtype,
  math_dtype: torch.dtype,
) -> torch.Tensor:
  """Resolve the per-input-channel smoothing factor for PTQ or DQ.

  DQ defaults to an identity-smooth workflow, but can also opt into experimental weight-only
  heuristics. PTQ continues to derive a data-dependent smooth factor from activation and weight
  spans.

  :param weight: Floating-point weight matrix on the target device.
  :param activation_span: Optional per-input-channel activation maxima.
  :param quant_mode: Either `"ptq"` or `"dq"`.
  :param smooth_strategy: Smoothing strategy used to derive the returned scale.
  :param in_features: Logical input width of the source linear layer.
  :param alpha: SmoothQuant interpolation factor.
  :param device: Target device for the returned smooth factor.
  :param torch_dtype: Output dtype for the returned smooth factor.
  :param math_dtype: Intermediate dtype for span reductions and PTQ smoothing.
  :returns: A per-input-channel smoothing factor with shape `[in_features]`.
  """

  smooth_strategy = _normalize_svdq_smooth_strategy(smooth_strategy)
  if quant_mode == "dq" and smooth_strategy not in {"identity", "weight", "weight_inv"}:
    raise ValueError(
      "SVDQ DQ currently only supports smooth_strategy in {'identity', 'weight', 'weight_inv'}.")

  if smooth_strategy == "identity":
    return torch.ones(in_features, device=device, dtype=torch_dtype)

  if smooth_strategy == "weight":
    if quant_mode != "dq":
      raise ValueError("Weight-only SVDQ smoothing is currently only supported for DQ workflows.")
    weight_span = weight.to(dtype=math_dtype).abs().amax(dim=0)
    return _compute_weight_only_smooth_scale(
      weight_span,
      alpha=alpha,
      math_dtype=math_dtype,
      output_dtype=torch_dtype,
    )

  if smooth_strategy == "weight_inv":
    if quant_mode != "dq":
      raise ValueError(
        "Weight-inverse SVDQ smoothing is currently only supported for DQ workflows.")
    weight_span = weight.to(dtype=math_dtype).abs().amax(dim=0)
    return _compute_weight_only_smooth_scale(
      weight_span,
      alpha=alpha,
      inverse=True,
      math_dtype=math_dtype,
      output_dtype=torch_dtype,
    )

  if quant_mode != "ptq":
    raise ValueError(
      "Activation-derived SVDQ smoothing is only valid for PTQ calibration workflows.")

  if activation_span is None:
    raise ValueError("activation_span must be provided for SVDQ PTQ.")
  if activation_span.ndim != 1 or activation_span.shape[0] != in_features:
    raise ValueError("activation_span must be a 1D tensor with length equal to linear.in_features, "
                     f"got shape {tuple(activation_span.shape)} for in_features={in_features}.")

  activation_span = activation_span.to(device=device, dtype=math_dtype)
  weight_span = weight.to(dtype=math_dtype).abs().amax(dim=0)
  return compute_smooth_scale(
    activation_span,
    weight_span,
    alpha=alpha,
    math_dtype=math_dtype,
    output_dtype=torch_dtype,
  )


@torch.inference_mode()
def _quantize_linear_svdq_w4a4_from_activation_span(
  linear: nn.Linear,
  activation_span: torch.Tensor | None,
  *,
  quant_type: str | None = None,
  rank: int = 32,
  alpha: float = 0.5,
  smooth_strategy: str = "activation",
  precision: str = "int4",
  act_unsigned: bool = False,
  torch_dtype: torch.dtype | None = None,
  device: torch.device | str | None = None,
  return_state_dict: bool = False,
  calibrate_precision: str = "low",
  runtime_kernel: str = "v1",
) -> SVDQW4A4Linear | dict[str, torch.Tensor]:
  """Quantize a linear layer from either PTQ activation spans or DQ mode.

  This is the lower-level worker behind `quantize_linear_svdq_w4a4`. It
  supports two internal modes controlled by `quant_type`:

  - PTQ: calibration has already been reduced to a single per-channel
    activation span, then smoothing, low-rank decomposition, residual scale
    computation, packing, and optional module instantiation are applied.
  - DQ: smoothing defaults to `smooth_scale = 1`, but may opt into
    experimental `weight` or `weight_inv` heuristics while still avoiding
    activation calibration.

  :param linear: Source floating-point linear layer.
  :param activation_span: Optional per-input-channel calibration maxima with
    length `linear.in_features`. Required for PTQ and ignored for DQ.
  :param quant_type: Optional SVDQ quant type. When it ends with `_dq`, the
    worker uses the dynamic quantization path without activation calibration.
  :param rank: Rank of the low-rank residual correction.
  :param alpha: SmoothQuant interpolation factor.
  :param smooth_strategy: Smoothing strategy used for this quantization pass.
  :param precision: Target weight format.
  :param act_unsigned: Whether the runtime activation path should use unsigned INT4.
  :param torch_dtype: Floating-point dtype for scales and residual tensors.
  :param device: Device where quantization intermediates and outputs are placed.
  :param return_state_dict: Whether to return the module `state_dict` instead
    of an instantiated `SVDQW4A4Linear`.
  :param calibrate_precision: Shared precision policy for calibration math and
    low-rank decomposition. For activation and weightsmoothing, `"low"` keeps
    calibration math in `torch_dtype`, while `"medium"` and `"high"` use float32
    calibration math. For the SVD low-rank decomposition, `"low"` uses `torch.svd_lowrank`
    in `torch_dtype`, `"medium"` uses the default full SVD route in float32, and `"high"`
    uses float64 SVD with CUDA `gesvd` for maximum precision.
  :param runtime_kernel: Packed runtime GEMM implementation to bind into the
    instantiated `SVDQW4A4Linear` module.
  :returns: Either a ready-to-load SVDQ module state dict or an instantiated
    `SVDQW4A4Linear`.
  """

  if not isinstance(linear, nn.Linear):
    raise TypeError(f"Expected nn.Linear, got {type(linear)}.")

  device = torch.device(device or linear.weight.device)
  torch_dtype = _normalize_dtype(torch_dtype, device)
  quant_mode = _resolve_svdq_quant_mode(quant_type)
  calibrate_precision = _normalize_calibrate_precision(calibrate_precision)
  math_dtype = _resolve_math_dtype(torch_dtype, calibrate_precision)
  validate_svdq_linear_geometry(linear.in_features,
                                linear.out_features,
                                rank=rank,
                                precision=precision)

  weight = linear.weight.detach().to(device=device, dtype=torch_dtype)
  smooth_scale = _resolve_svdq_smooth_scale(
    weight,
    activation_span,
    quant_mode=quant_mode,
    smooth_strategy=smooth_strategy,
    in_features=linear.in_features,
    alpha=alpha,
    device=device,
    torch_dtype=torch_dtype,
    math_dtype=math_dtype,
  )

  smoothed_weight = weight.to(dtype=math_dtype) * smooth_scale.to(dtype=math_dtype).unsqueeze(0)
  lowrank_down, lowrank_up, residual = decompose_lowrank_residual(
    smoothed_weight,
    rank,
    output_dtype=torch_dtype,
    svd_precision=calibrate_precision,
  )
  scales = _compute_group_scales(
    residual,
    group_size=64,
    math_dtype=math_dtype,
    output_dtype=torch_dtype,
  )

  bias = (None
          if linear.bias is None else linear.bias.detach().to(device=device, dtype=torch_dtype))
  raw_state_dict = export_raw_svdq_w4a4_state_dict(
    residual,
    scale=scales,
    bias=bias,
    smooth=smooth_scale,
    lora=None if rank == 0 else (lowrank_down, lowrank_up),
    float_point=False,
  )
  module_state_dict = adapt_svdq_module_state_dict(
    raw_state_dict,
    in_features=linear.in_features,
    out_features=linear.out_features,
    rank=rank,
    torch_dtype=torch_dtype,
    device=device,
    has_bias=linear.bias is not None,
  )

  if return_state_dict:
    return module_state_dict

  quantized = SVDQW4A4Linear.from_linear(
    linear,
    rank=rank,
    precision=precision,
    act_unsigned=act_unsigned,
    runtime_kernel=runtime_kernel,
    torch_dtype=torch_dtype,
    device=device,
  )
  incompatible = quantized.load_state_dict(module_state_dict, strict=True)
  if incompatible.missing_keys or incompatible.unexpected_keys:
    raise RuntimeError(
      f"Unexpected SVDQuant state_dict mismatch: missing={incompatible.missing_keys}, "
      f"unexpected={incompatible.unexpected_keys}.")
  return quantized


@torch.inference_mode()
def quantize_linear_svdq_w4a4(
  linear: nn.Linear,
  representative_activations: CalibrationInputs,
  *,
  rank: int = 32,
  alpha: float = 0.5,
  precision: str = "int4",
  act_unsigned: bool = False,
  torch_dtype: torch.dtype | None = None,
  device: torch.device | str | None = None,
  return_state_dict: bool = False,
  calibrate_precision: str = "low",
  streaming: bool = True,
  activation_buffer_flush_sample_count: int | None = 1,
  activation_buffer_flush_cpu_bytes: int | None = None,
) -> SVDQW4A4Linear | dict[str, torch.Tensor]:
  """Quantize a float `nn.Linear` into the cache-dit SVDQ W4A4 format.

  :param linear: Source floating-point linear layer.
  :param representative_activations: Tensor or iterable of tensors whose last
    dimension matches `linear.in_features`.
  :param rank: Rank of the low-rank residual correction.
  :param alpha: SmoothQuant interpolation factor used to derive `smooth_factor`.
  :param precision: Weight format requested by the quantizer. The current minimal
    implementation supports `int4` only.
  :param act_unsigned: Whether the runtime activation quantizer should emit unsigned
    4-bit activations.
  :param torch_dtype: Floating-point dtype used for the packed runtime tensors.
  :param device: Device on which quantization intermediates and the returned module
    should be materialized.
  :param return_state_dict: When `True`, return the packed module `state_dict`
    instead of instantiating `SVDQW4A4Linear`.
  :param calibrate_precision: Shared precision policy for calibration math and
    low-rank decomposition.
  :param streaming: Whether to accumulate activation spans incrementally instead of
    materializing all standardized activations at once.
  :param activation_buffer_flush_sample_count: Number of buffered span samples to keep
    before merging them when `streaming=True`.
  :param activation_buffer_flush_cpu_bytes: CPU buffer limit that also triggers a
    merge when `streaming=True`.

  :returns: Either a quantized `SVDQW4A4Linear` module or the corresponding module
  `state_dict`, depending on `return_state_dict`.
  """

  if not isinstance(linear, nn.Linear):
    raise TypeError(f"Expected nn.Linear, got {type(linear)}.")

  device = torch.device(device or linear.weight.device)
  torch_dtype = _normalize_dtype(torch_dtype, device)
  calibrate_precision = _normalize_calibrate_precision(calibrate_precision)
  math_dtype = _resolve_math_dtype(torch_dtype, calibrate_precision)
  validate_svdq_linear_geometry(linear.in_features,
                                linear.out_features,
                                rank=rank,
                                precision=precision)

  if streaming:
    activation_span_accumulator = _ActivationSpanAccumulator(
      device=device,
      torch_dtype=torch_dtype,
      math_dtype=math_dtype,
      flush_sample_count=activation_buffer_flush_sample_count,
      flush_cpu_bytes=activation_buffer_flush_cpu_bytes,
    )
    for tensor in _iter_standardized_calibration_activations(
        representative_activations,
        in_features=linear.in_features,
    ):
      activation_span_accumulator.add_tensor(tensor)
    activation_span = activation_span_accumulator.finalize()
  else:
    standardized_acts = standardize_calibration_activations(
      representative_activations,
      in_features=linear.in_features,
    )
    activation_span = _compute_activation_span(
      standardized_acts,
      device=device,
      torch_dtype=torch_dtype,
      math_dtype=math_dtype,
      streaming=False,
    )
  return _quantize_linear_svdq_w4a4_from_activation_span(
    linear,
    activation_span,
    rank=rank,
    alpha=alpha,
    precision=precision,
    act_unsigned=act_unsigned,
    torch_dtype=torch_dtype,
    device=device,
    return_state_dict=return_state_dict,
    calibrate_precision=calibrate_precision,
  )
