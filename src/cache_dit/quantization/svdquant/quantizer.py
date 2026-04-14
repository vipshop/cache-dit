from __future__ import annotations

import dataclasses
import math
import typing as tp
import warnings

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
_SVDQ_SMOOTH_STRATEGIES = ("activation", "identity", "weight", "weight_inv", "few_shot")
_WEIGHT_ONLY_SMOOTH_CLAMP_RANGE = (0.25, 4.0)
_FEW_SHOT_RELAX_STRATEGIES = ("fixed", "top", "auto", "stable_auto", "power", "log", "rank")
_FEW_SHOT_LOG_CURVE_STRENGTH = 9.0
_FEW_SHOT_POWER_GAMMA = 2.0
_FEW_SHOT_STABLE_AUTO_BUCKETS = 8
_FEW_SHOT_RELAX_WARN_THRESHOLD = 3.0


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


def _validate_few_shot_relax_factor(relax_factor: float) -> float:
  if isinstance(relax_factor, bool) or not isinstance(relax_factor, (int, float)):
    raise TypeError(f"relax_factor must be a float, got {type(relax_factor)}.")
  resolved = float(relax_factor)
  if resolved < 1.0:
    raise ValueError(f"relax_factor must be >= 1.0, got {relax_factor}.")
  return resolved


def _validate_few_shot_relax_top_ratio(relax_top_ratio: float) -> float:
  if isinstance(relax_top_ratio, bool) or not isinstance(relax_top_ratio, (int, float)):
    raise TypeError(f"relax_top_ratio must be a float, got {type(relax_top_ratio)}.")
  resolved = float(relax_top_ratio)
  if not 0.0 < resolved <= 1.0:
    raise ValueError(f"relax_top_ratio must be in the range (0, 1], got {relax_top_ratio}.")
  return resolved


def _normalize_few_shot_relax_strategy(relax_strategy: str) -> str:
  if not isinstance(relax_strategy, str):
    raise TypeError(f"relax_strategy must be a str, got {type(relax_strategy)}.")
  normalized = relax_strategy.lower()
  if normalized == "top_q4":
    normalized = "top"
  if normalized not in _FEW_SHOT_RELAX_STRATEGIES:
    raise ValueError(
      f"relax_strategy must be one of {_FEW_SHOT_RELAX_STRATEGIES} or the alias 'top_q4', got "
      f"{relax_strategy!r}.")
  return normalized


def _warn_about_aggressive_few_shot_relax_factor(relax_factor: float) -> None:
  if relax_factor <= _FEW_SHOT_RELAX_WARN_THRESHOLD:
    return
  warnings.warn(
    f"relax_factor={relax_factor} is aggressive for SVDQ few-shot relaxation and may oversmooth "
    "or blur outputs. Prefer values around 1.5-2.5 first, then tune relax_strategy or "
    "relax_top_ratio if more headroom is needed.",
    RuntimeWarning,
    stacklevel=3,
  )


@torch.inference_mode()
def _map_few_shot_response_to_multipliers(
  response: torch.Tensor,
  *,
  relax_factor: float,
) -> torch.Tensor:
  """Map a normalized relax response to the per-channel activation-span multiplier.

  The input `response` is interpreted as a dimensionless score in `[0, 1]`:

  - `response = 0` means "keep the original activation span unchanged"
  - `response = 1` means "apply the configured maximum relax factor to activation span"

  We intentionally clamp the response to `[0, 1]` before the affine map

  `multiplier = 1 + response * (relax_factor - 1)`

  so the final multiplier is guaranteed to stay inside `[1, relax_factor]`. This bound matters for
  two reasons:

  - it prevents numerical overshoot caused by floating-point noise or future response builders that
    might otherwise emit values outside the normalized range;
  - combined with the invariant `relax_factor >= 1`, it guarantees `multiplier >= 1`, so the final
    relaxed activation span `a_relaxed = a_orig * multiplier` never shrinks the original observed
    activation maxima.

  The important detail is that we do not multiply the whole activation-span vector by a single global
  `relax_factor`. A uniform rescale would preserve every channel-to-channel ratio exactly, so it
  would only shift the whole vector upward without changing its shape. That does not mimic the
  effect of observing more calibration samples, because longer observation windows usually increase
  the maxima of "outlier-prone" channels more than stable channels. The response vector therefore
  acts as a per-channel gate that lets high-risk channels expand more aggressively than low-risk
  ones.

  :param response: Per-channel normalized relax response.
  :param relax_factor: Maximum multiplicative activation-span relax factor. Must be >= 1.
  :returns: Per-channel activation-span multipliers in `[1, relax_factor]`.
  """

  if response.ndim != 1:
    raise ValueError(f"response must be a 1D tensor, got shape {tuple(response.shape)}.")
  response = response.clamp_(0.0, 1.0)
  # After clamping, the affine map sends 0 -> 1 and 1 -> relax_factor. Because relax_factor is
  # constrained to be >= 1, every channel either keeps its original activation span or is
  # amplified. No channel can be attenuated by this step.
  return response.mul_(relax_factor - 1.0).add_(1.0)


@torch.inference_mode()
def _build_few_shot_relax_response(
  activation_span: torch.Tensor,
  *,
  relax_top_ratio: float,
  relax_strategy: str,
) -> torch.Tensor:
  """Build a normalized `[0, 1]` relax response from a per-channel activation-span vector.

  The output of this function is not the final multiplier yet. Instead, it builds a monotonic
  response variable `r_i` that is later converted to the final multiplier by
  `_map_few_shot_response_to_multipliers()`.

  The modeling intuition is: few-shot mode only sees a small number of runtime samples, so the
  observed activation span is a lower-confidence estimate of what the layer would look like after a
  longer run. If we had seen more samples, channels that already exhibit larger activation maxima are
  also the channels most likely to encounter even larger future activation maxima. In other words,
  a large current activation span is treated as a proxy for higher outlier probability under a longer
  observation horizon.

  Because of that, the relaxation must be channel-wise and monotonic rather than a single uniform
  multiplication. Directly applying `activation_span *= relax_factor` would enlarge every channel by
  the same proportion, preserve the original vector shape, and therefore fail to create any extra
  smoothing preference between stable channels and outlier-prone channels. The role of this
  function is precisely to reshape that vector: channels with stronger outlier evidence receive a
  larger response, while conservative channels stay closer to 0.

  Let `a_i` be the original activation span for channel `i`, `tau` the quantile threshold determined
  by `few_shot_relax_top_ratio`, and `a_min` the minimum activation span in the current layer. The
  strategy-specific response definitions are:

  - `fixed`:
      `r_i = 0`
    This disables the relax step entirely. After the final affine map every multiplier stays at 1,
    so the runtime activation span is exactly the original few-shot observed activation span. It is
    useful as a control/baseline when we want few-shot activation collection without any extra
    channel-wise amplification before recomputing smooth scale.

  - `top`:
      `r_i = 1[a_i >= tau]`
    Only channels at or above the threshold receive the maximum relax factor; all other channels are
    left unchanged after the affine map. The corresponding smooth scale is then recomputed from the
    relaxed activation span and the unchanged weight span.

  - `auto`:
      `r_i = clip((a_i - a_min) / (tau - a_min), 0, 1)`
    This is a linear ramp. Small activation spans stay near 0, larger activation spans move toward
    1, and all channels at or above the threshold saturate to the same maximum relax factor.

  - `stable_auto`:
      `r_i = round(B * auto_i) / B`, with a small fixed bucket count `B`
    This keeps the same magnitude-aware linear ramp as `auto`, but snaps nearby channels to a
    shared response bucket before the final affine map. The goal is not to make the whole runtime
    path bitwise deterministic; rather, it makes the relax policy less sensitive to small
    first-forward activation-span fluctuations, so repeated few-shot runs are more likely to land
    on the same per-channel relax multipliers.

  - `power`:
      `r_i = auto_i ** gamma`, with `gamma > 1`
    This convex transform suppresses the low/mid channels relative to `auto`, so the amplification
    budget is concentrated more aggressively on the largest activation spans.

  - `log`:
      `r_i = log(1 + k * auto_i) / log(1 + k)`
    This concave transform raises the low/mid channels earlier than `auto`, while still saturating
    to 1 at the threshold.

  - `rank`:
      `r_i = clip(rank_percentile_i / threshold_q, 0, 1)`
    This uses the channel order statistics instead of raw activation-span gaps. It is useful when the
    layer contains extreme outliers and we want the relax policy to depend more on relative rank than
    on the exact numeric spacing between channels.

  :param activation_span: Original per-channel activation-span vector.
  :param relax_top_ratio: Fraction of top channels used to determine the saturation threshold.
  :param relax_strategy: Strategy name.
  :returns: A normalized per-channel response vector in `[0, 1]`.
  """

  relax_strategy = _normalize_few_shot_relax_strategy(relax_strategy)
  activation_float = activation_span.float()
  if relax_strategy == "fixed":
    # `fixed` is the explicit no-op baseline: every response stays at 0, so the later affine map
    # produces multiplier = 1 for every channel and the original activation span is preserved.
    return torch.zeros_like(activation_float)
  threshold_q = max(0.0, 1.0 - relax_top_ratio)
  threshold = torch.quantile(activation_float, threshold_q)

  if relax_strategy == "top":
    # Hard indicator response: channels above the threshold map to 1, others to 0. After the final
    # affine map this becomes multiplier = relax_factor for the top set and multiplier = 1 for the
    # rest, i.e. non-top channels keep their original activation span.
    #
    # Interpretation: if we only trust the most extreme channels to be genuinely outlier-prone,
    # we relax only that top subset and leave the rest untouched. This is the most conservative way
    # to approximate "seeing more samples" without globally inflating the whole layer.
    response = torch.zeros_like(activation_float)
    response[activation_float >= threshold] = 1.0
    return response

  lower = activation_float.amin()
  if not torch.isfinite(lower) or not torch.isfinite(threshold):
    return torch.zeros_like(activation_float)
  if threshold <= lower:
    # Degenerate case: all channels collapse to the same activation span (or the threshold is numerically
    # indistinguishable from the minimum). In that situation there is no meaningful ordering, so all
    # channels receive the saturated response.
    return torch.ones_like(activation_float)

  # Normalize the observed activation spans into a response coordinate where 0 corresponds to the
  # smallest/most stable channel in the layer and 1 corresponds to the saturation threshold for the
  # selected top ratio. This is the key step that turns absolute activation magnitude into a per-channel
  # "future outlier likelihood" score. We use this normalized score instead of a single scalar so
  # that different channels can grow by different amounts.
  normalized = activation_float.sub(lower).div_(threshold - lower).clamp_(0.0, 1.0)
  if relax_strategy == "auto":
    # Linear ramp from the layer minimum to the threshold. Larger activation spans imply larger
    # normalized response, so after the affine map the final relax multiplier increases linearly
    # with the observed few-shot activation span until it saturates at the threshold.
    #
    # Interpretation: this is the default "more samples -> larger current outliers grow more"
    # heuristic. Every channel can expand, but the expansion is proportional to how extreme its
    # current activation span already looks.
    return normalized
  if relax_strategy == "stable_auto":
    # Bucketized linear ramp: preserve the same magnitude-aware ordering as `auto`, but snap the
    # response to a small number of evenly spaced levels. This damps run-to-run jitter when the
    # first observed activation spans move slightly yet should still imply the same coarse relax
    # decision.
    return normalized.mul(_FEW_SHOT_STABLE_AUTO_BUCKETS).add_(0.5).floor_().div_(
      _FEW_SHOT_STABLE_AUTO_BUCKETS)
  if relax_strategy == "power":
    # Convex curve: relative to `auto`, this keeps low/mid channels closer to 0 and allocates more
    # of the amplification budget to the largest channels near the threshold.
    #
    # Interpretation: use this when you believe future extra samples will mostly strengthen the
    # already-extreme channels, not the moderate ones. It is a sharper outlier-prior than `auto`.
    return normalized.pow(_FEW_SHOT_POWER_GAMMA)
  if relax_strategy == "log":
    # Concave curve: relative to `auto`, this lifts low/mid channels earlier, but still saturates at
    # 1 near the threshold. It is useful when we want a broader, softer amplification profile.
    #
    # Interpretation: use this when you expect additional samples to reveal somewhat broader tail
    # behavior, so mid-ranked channels should also receive noticeable relaxation instead of focusing
    # almost entirely on the very largest channels.
    return torch.log1p(_FEW_SHOT_LOG_CURVE_STRENGTH * normalized).div_(
      math.log1p(_FEW_SHOT_LOG_CURVE_STRENGTH))
  if relax_strategy == "rank":
    if activation_float.numel() <= 1:
      return torch.ones_like(activation_float)
    # Rank-based response: ignore the exact magnitude gaps and convert the sorted channel order to a
    # percentile in [0, 1]. This makes the relax policy depend on relative ordering rather than raw
    # scale distance, which is often more robust when a few channels are extreme outliers.
    #
    # Interpretation: this says "I trust the ordering more than the absolute spacing". Even if one
    # channel is numerically much larger than the others, we do not let that single gap dominate the
    # entire response curve; we only use the fact that it ranks near the top.
    order = torch.argsort(activation_float, stable=True)
    rank_fraction = torch.empty_like(activation_float)
    rank_fraction[order] = torch.linspace(
      0.0,
      1.0,
      steps=activation_float.numel(),
      dtype=activation_float.dtype,
      device=activation_float.device,
    )
    if threshold_q <= 0.0:
      return torch.ones_like(rank_fraction)
    return rank_fraction.div_(threshold_q).clamp_(0.0, 1.0)
  raise AssertionError(f"Unhandled few-shot relax strategy: {relax_strategy}.")


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
def _apply_few_shot_relaxation(
  activation_span: torch.Tensor,
  weight_span: torch.Tensor,
  *,
  alpha: float = 0.5,
  relax_factor: float = 1.5,
  relax_top_ratio: float = 0.25,
  relax_strategy: str = "auto",
  math_dtype: torch.dtype = torch.float32,
  output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Relax few-shot activation spans, then recompute smooth scales.

  The few-shot policy is applied to `activation_span`, because the uncertainty being compensated is
  the limited number of observed runtime activations, not the already-mixed smooth scale. After the
  activation span is relaxed, the function recomputes both the original and relaxed smooth scales
  with the same `weight_span` and SmoothQuant exponent `alpha`.

  :param activation_span: The base few-shot activation-span vector.
  :param weight_span: The per-channel weight-span vector used to derive smooth scale.
  :param alpha: SmoothQuant interpolation factor used during the smooth-scale recomputation.
  :param relax_factor: Maximum multiplicative factor applied to activation-span relaxation. Because
    it is constrained to be >= 1, the relaxed activation span is guaranteed not to be smaller than
    the original one. The `fixed` strategy ignores this value because it performs no relaxation.
  :param relax_top_ratio: Fraction of channels used to define the relax threshold. The `fixed`
    strategy ignores this value because it performs no relaxation.
  :param relax_strategy: Relax strategy. `fixed` preserves the original few-shot activation span,
    `top` keeps channels below the threshold unchanged, and the other strategies build a monotonic
    response from activation span and then map that response into the multiplicative interval
    `[1, relax_factor]`.
  :param math_dtype: Intermediate dtype used for activation-span relaxation and smooth-scale
    recomputation.
  :param output_dtype: Optional dtype for the returned smooth-scale tensors.
  :returns: A tuple `(relaxed_smooth_scale, original_smooth_scale)`.
  """

  if activation_span.ndim != 1:
    raise ValueError(
      f"activation_span must be a 1D tensor, got shape {tuple(activation_span.shape)}.")
  if weight_span.ndim != 1:
    raise ValueError(f"weight_span must be a 1D tensor, got shape {tuple(weight_span.shape)}.")
  if activation_span.shape != weight_span.shape:
    raise ValueError("activation_span and weight_span must have the same shape, got "
                     f"{tuple(activation_span.shape)} and {tuple(weight_span.shape)}.")
  if not 0.0 <= alpha <= 1.0:
    raise ValueError(f"alpha must be in [0, 1], got {alpha}.")

  relax_strategy = _normalize_few_shot_relax_strategy(relax_strategy)
  activation_span = activation_span.to(device=weight_span.device, dtype=math_dtype)
  weight_span = weight_span.to(device=weight_span.device, dtype=math_dtype)
  original = compute_smooth_scale(
    activation_span,
    weight_span,
    alpha=alpha,
    math_dtype=math_dtype,
    output_dtype=output_dtype,
  )
  if relax_strategy == "fixed":
    preserved = original if output_dtype is None else original.to(dtype=output_dtype)
    return preserved.clone(), preserved

  relax_factor = _validate_few_shot_relax_factor(relax_factor)
  _warn_about_aggressive_few_shot_relax_factor(relax_factor)
  relax_top_ratio = _validate_few_shot_relax_top_ratio(relax_top_ratio)
  relax_response = _build_few_shot_relax_response(
    activation_span,
    relax_top_ratio=relax_top_ratio,
    relax_strategy=relax_strategy,
  )
  multipliers = _map_few_shot_response_to_multipliers(
    relax_response,
    relax_factor=relax_factor,
  )
  # Relax the activation statistics first, then recompute the mixed smooth scale with the same
  # weight span. Because the activation multiplier is >= 1 and alpha >= 0, the recomputed smooth
  # scale is also guaranteed not to shrink.
  relaxed_activation_span = activation_span.mul(multipliers)
  relaxed = compute_smooth_scale(
    relaxed_activation_span,
    weight_span,
    alpha=alpha,
    math_dtype=math_dtype,
    output_dtype=output_dtype,
  )
  return relaxed, original


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
  if quant_mode == "dq" and smooth_strategy not in {"identity", "weight", "weight_inv", "few_shot"}:
    raise ValueError(
      "SVDQ DQ currently only supports smooth_strategy in {'identity', 'weight', 'weight_inv', 'few_shot'}."
    )

  if smooth_strategy == "few_shot":
    raise ValueError("The 'few_shot' smooth strategy requires runtime activation collection.")

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
def _resolve_activation_smooth_scale(
  weight: torch.Tensor,
  activation_span: torch.Tensor,
  *,
  in_features: int,
  alpha: float,
  math_dtype: torch.dtype,
  torch_dtype: torch.dtype,
  device: torch.device,
) -> torch.Tensor:
  """Resolve activation-derived SVDQ smooth scales from a finalized span vector."""

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
def _quantize_linear_svdq_w4a4_from_smooth_scale(
  linear: nn.Linear,
  smooth_scale: torch.Tensor,
  *,
  smooth_scale_orig: torch.Tensor | None = None,
  quant_type: str | None = None,
  rank: int = 32,
  precision: str = "int4",
  act_unsigned: bool = False,
  torch_dtype: torch.dtype | None = None,
  device: torch.device | str | None = None,
  return_state_dict: bool = False,
  calibrate_precision: str = "low",
  runtime_kernel: str = "v1",
) -> SVDQW4A4Linear | dict[str, torch.Tensor]:
  """Quantize a linear layer from an explicitly resolved runtime smooth vector."""

  if not isinstance(linear, nn.Linear):
    raise TypeError(f"Expected nn.Linear, got {type(linear)}.")

  device = torch.device(device or linear.weight.device)
  torch_dtype = _normalize_dtype(torch_dtype, device)
  calibrate_precision = _normalize_calibrate_precision(calibrate_precision)
  math_dtype = _resolve_math_dtype(torch_dtype, calibrate_precision)
  validate_svdq_linear_geometry(
    linear.in_features,
    linear.out_features,
    rank=rank,
    precision=precision,
  )

  if smooth_scale.ndim != 1 or smooth_scale.shape[0] != linear.in_features:
    raise ValueError("smooth_scale must be a 1D tensor with length equal to linear.in_features, "
                     f"got shape {tuple(smooth_scale.shape)} for in_features={linear.in_features}.")

  if smooth_scale_orig is None:
    smooth_scale_orig = smooth_scale
  if smooth_scale_orig.ndim != 1 or smooth_scale_orig.shape[0] != linear.in_features:
    raise ValueError(
      "smooth_scale_orig must be a 1D tensor with length equal to linear.in_features, "
      f"got shape {tuple(smooth_scale_orig.shape)} for in_features={linear.in_features}.")

  weight = linear.weight.detach().to(device=device, dtype=torch_dtype)
  smooth_scale = _repair_invalid_scale(smooth_scale.to(device=device, dtype=torch_dtype))
  smooth_scale_orig = _repair_invalid_scale(smooth_scale_orig.to(device=device, dtype=torch_dtype))

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
    smooth_orig=smooth_scale_orig,
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
  validate_svdq_linear_geometry(
    linear.in_features,
    linear.out_features,
    rank=rank,
    precision=precision,
  )

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
  return _quantize_linear_svdq_w4a4_from_smooth_scale(
    linear,
    smooth_scale,
    smooth_scale_orig=smooth_scale,
    quant_type=quant_type,
    rank=rank,
    precision=precision,
    act_unsigned=act_unsigned,
    torch_dtype=torch_dtype,
    device=device,
    return_state_dict=return_state_dict,
    calibrate_precision=calibrate_precision,
    runtime_kernel=runtime_kernel,
  )


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
  validate_svdq_linear_geometry(
    linear.in_features,
    linear.out_features,
    rank=rank,
    precision=precision,
  )

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
