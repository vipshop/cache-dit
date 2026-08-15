"""Cd cache-dit pytest tests/quantization/test_svdquant_quantizer.py -v -s."""

import math
import os
from pathlib import Path
import time
import warnings

import pytest
import torch
from torch import nn

import cache_dit.quantization.svdquant.quantizer as svdq_quantizer
from cache_dit.kernels import svdq_extension_is_available
from cache_dit.quantization.svdquant.lowrank import decompose_lowrank_residual
from cache_dit.quantization.svdquant.packing import fp_quantize
from cache_dit.quantization.svdquant import SVDQW4A4Linear
from cache_dit.quantization.svdquant import quantize_linear_svdq_w4a4
from tests.quantization._svdq_test_utils import EVALUATED_RANKS
from tests.quantization._svdq_test_utils import RANKS_WITH_BASELINE
from tests.quantization._svdq_test_utils import assert_rank_metric_trend
from tests.quantization._svdq_test_utils import build_empty_quantized_toy_model
from tests.quantization._svdq_test_utils import compute_accuracy_metrics
from tests.quantization._svdq_test_utils import format_markdown_table
from tests.quantization._svdq_test_utils import format_rank_report
from tests.quantization._svdq_test_utils import make_rank_sensitive_linear
from tests.quantization._svdq_test_utils import make_spectral_decay_weight
from tests.quantization._svdq_test_utils import make_token_batch
from tests.quantization._svdq_test_utils import make_token_samples
from tests.quantization._svdq_test_utils import make_toy_model
from tests.quantization._svdq_test_utils import quantize_toy_model
from tests.quantization._svdq_test_utils import runtime_dtype

_CALIBRATE_PRECISION = os.getenv("CACHE_DIT_SVDQ_TEST_CALIBRATE_PRECISION", "low").lower()
_ENABLE_STREAMING_MEMORY_BENCH = os.getenv("CACHE_DIT_SVDQ_TEST_LARGE_MEMORY", "0").lower() == "1"
_ENABLE_LARGE_HEAD_NUMBER = os.getenv("CACHE_DIT_SVDQ_TEST_LARGE_HEAD_NUM", "0").lower() == "1"
_LARGE_MEMORY_TOTAL_GIB = float(os.getenv("CACHE_DIT_SVDQ_TEST_LARGE_MEMORY_GIB", "10"))
_LARGE_MEMORY_CHUNK_MIB = int(os.getenv("CACHE_DIT_SVDQ_TEST_LARGE_MEMORY_CHUNK_MIB", "256"))
_STREAMING_MEMORY_THRESHOLD_PCT = float(
  os.getenv("CACHE_DIT_SVDQ_TEST_STREAMING_MEMORY_THRESHOLD_PCT", "25"))
_LARGE_MEMORY_MIN_DEVICE_GIB = float(os.getenv("CACHE_DIT_SVDQ_TEST_LARGE_MEMORY_MIN_GIB", "12"))

if _CALIBRATE_PRECISION not in {"low", "medium", "high"}:
  raise ValueError("CACHE_DIT_SVDQ_TEST_CALIBRATE_PRECISION must be one of low, medium, high.")


def _quantizer_kwargs(**overrides: object) -> dict[str, object]:
  kwargs: dict[str, object] = {
    "calibrate_precision": _CALIBRATE_PRECISION,
    "streaming": True,
    "activation_buffer_flush_sample_count": 1,
    "activation_buffer_flush_cpu_bytes": None,
  }
  kwargs.update(overrides)
  return kwargs


def _current_tolerance() -> tuple[float, float]:
  if _CALIBRATE_PRECISION == "high":
    return 4e-2, 1e-2
  if _CALIBRATE_PRECISION == "medium":
    return 6e-2, 2e-2
  return 1e-1, 1e-1


def _make_large_cpu_calibration_list(
  *,
  in_features: int,
  total_gib: float,
  chunk_mib: int,
  dtype: torch.dtype,
) -> list[torch.Tensor]:
  bytes_per_elem = torch.empty((), dtype=dtype).element_size()
  chunk_bytes = chunk_mib * 1024 * 1024
  rows_per_chunk = max(1, chunk_bytes // (in_features * bytes_per_elem))
  total_bytes = int(total_gib * (1024 ** 3))

  calibration: list[torch.Tensor] = []
  allocated = 0
  while allocated < total_bytes:
    remaining_rows = max(1, (total_bytes - allocated) // (in_features * bytes_per_elem))
    rows = min(rows_per_chunk, remaining_rows)
    tensor = torch.zeros((rows, in_features), dtype=dtype, device="cpu")
    calibration.append(tensor)
    allocated += tensor.numel() * tensor.element_size()
  return calibration


def _measure_quantizer_peak_memory(
  linear: nn.Linear,
  representative: list[torch.Tensor],
  *,
  rank: int = 32,
  dtype: torch.dtype,
  streaming: bool,
) -> int:
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()
  torch.cuda.synchronize()
  _ = quantize_linear_svdq_w4a4(
    linear,
    representative,
    rank=rank,
    device=linear.weight.device,
    torch_dtype=dtype,
    return_state_dict=True,
    **_quantizer_kwargs(streaming=streaming),
  )
  torch.cuda.synchronize()
  peak = torch.cuda.max_memory_allocated()
  torch.cuda.empty_cache()
  return peak


def _make_cpu_linear(in_features: int, out_features: int, *, bias: bool = True) -> nn.Linear:
  torch.manual_seed(0)
  linear = nn.Linear(in_features, out_features, bias=bias, device="cpu", dtype=torch.bfloat16)
  return linear


def test_svdquant_quantizer_returns_module_state_dict() -> None:
  linear = _make_cpu_linear(128, 256)
  representative = torch.randn(3, 5, 128, dtype=torch.float32)

  state_dict: dict[str, torch.Tensor] = quantize_linear_svdq_w4a4(
    linear,
    representative,
    rank=16,
    device="cpu",
    torch_dtype=torch.bfloat16,
    return_state_dict=True,
    **_quantizer_kwargs(),
  )

  assert set(state_dict) == {
    "bias",
    "proj_down",
    "proj_up",
    "qweight",
    "smooth_factor",
    "smooth_factor_orig",
    "wscales",
  }
  assert state_dict["qweight"].shape == (256, 64)
  assert state_dict["wscales"].shape == (2, 256)
  assert state_dict["bias"].shape == (256, )
  assert state_dict["smooth_factor"].shape == (128, )
  assert state_dict["smooth_factor_orig"].shape == (128, )
  assert state_dict["proj_down"].shape == (128, 16)
  assert state_dict["proj_up"].shape == (256, 16)


# `torch_dtype` values a real PTQ run passes. float32 keeps the residual bookkeeping exact, while
# bfloat16 rounds the stored residual and is what exposes precision slips in the refinement loop.
_REFINE_TORCH_DTYPES = [torch.float32, torch.bfloat16]


def _refine_split_pieces(
  weight: torch.Tensor,
  *,
  precision: str,
  refine_iters: int,
  torch_dtype: torch.dtype,
  calibrate_precision: str | None = None,
  rank: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Run the production low-rank split and return the pieces the runtime combines, in float32.

  :returns: A tuple `(lowrank, residual, residual_q)` where `lowrank` is `up @ down`, `residual` is
    the matrix handed to the residual branch, and `residual_q` is its quantize/dequantize
    round-trip. The W4A4 kernel evaluates `lowrank + residual_q`.
  """

  calibrate_precision = calibrate_precision or _CALIBRATE_PRECISION
  math_dtype = svdq_quantizer._resolve_math_dtype(torch_dtype, calibrate_precision)
  down, up, residual = svdq_quantizer._refine_lowrank_split(
    weight.to(math_dtype),
    rank,
    refine_iters=refine_iters,
    precision=precision,
    calibrate_precision=calibrate_precision,
    math_dtype=math_dtype,
    torch_dtype=torch_dtype,
  )
  residual_q = svdq_quantizer._fake_quantize_dequantize_residual(
    residual,
    precision=precision,
    math_dtype=math_dtype,
    torch_dtype=torch_dtype,
  )
  return (up.to(torch.float32) @ down.to(torch.float32), residual.to(torch.float32),
          residual_q.to(torch.float32))


def test_svdquant_refine_iters_codebook_matches_packing() -> None:
  # The refinement simulation carries its own copy of the FP4 codebook so it can index values back
  # out of `fp_quantize`'s indices. Guard against the two copies drifting apart.
  codebook = torch.tensor(svdq_quantizer._NVFP4_CODEBOOK, dtype=torch.float32)
  probe = codebook.clone()
  assert torch.equal(fp_quantize(probe, codebook=codebook), fp_quantize(probe))
  assert torch.equal(codebook[fp_quantize(probe)], probe)


@pytest.mark.parametrize("torch_dtype", _REFINE_TORCH_DTYPES, ids=["fp32", "bf16"])
@pytest.mark.parametrize("precision", ["int4", "nvfp4"])
def test_svdquant_refine_iters_fake_quantize_matches_packing(
  precision: str,
  torch_dtype: torch.dtype,
) -> None:
  # Refinement is only as good as its model of the residual quantizer, so pin the simulation
  # against an independent replication of `pack_svdq_w4a4_linear_tensors`. The NVFP4 path is
  # deliberately asymmetric: the packer normalizes by the `torch_dtype` group scales but
  # `SVDQWeightPacker.pack_micro_scale` stores them as FP8 E4M3, so the kernel dequantizes with a
  # different value than it normalized with. INT4 group scales are not micro-scales and stay at
  # `torch_dtype` on both sides.
  residual = make_spectral_decay_weight(256, 128, seed=3, device="cpu", dtype=torch.float32)
  out_features, in_features = residual.shape
  math_dtype = torch.float32
  group_size = 16 if precision == "nvfp4" else 64
  grouped = residual.to(math_dtype).view(out_features, 1, in_features // group_size, group_size)

  if precision == "nvfp4":
    channel_scales, group_scales = svdq_quantizer._compute_nvfp4_channel_and_group_scales(
      residual,
      math_dtype=math_dtype,
      output_dtype=torch_dtype,
    )
    channel_scales = channel_scales.to(math_dtype)
    normalize_scales = group_scales.to(math_dtype)
    stored_scales = group_scales.to(torch.float8_e4m3fn).to(math_dtype)
    codebook = torch.tensor(svdq_quantizer._NVFP4_CODEBOOK, dtype=math_dtype)
    codes = fp_quantize(grouped / channel_scales / normalize_scales, codebook=codebook)
    expected = (codebook[codes] * stored_scales * channel_scales).view(out_features, in_features)
  else:
    group_scales = svdq_quantizer._compute_group_scales(
      residual,
      group_size=group_size,
      math_dtype=math_dtype,
      output_dtype=torch_dtype,
    ).to(math_dtype)
    codes = (grouped / group_scales).round().clamp(-8, 7)
    expected = (codes * group_scales).view(out_features, in_features)

  actual = svdq_quantizer._fake_quantize_dequantize_residual(
    residual,
    precision=precision,
    math_dtype=math_dtype,
    torch_dtype=torch_dtype,
  )
  assert torch.equal(actual, expected.to(residual.dtype))


@pytest.mark.parametrize("torch_dtype", _REFINE_TORCH_DTYPES, ids=["fp32", "bf16"])
@pytest.mark.parametrize("precision", ["int4", "nvfp4"])
def test_svdquant_refine_iters_keeps_residual_anchored_on_weight(
  precision: str,
  torch_dtype: torch.dtype,
) -> None:
  # The residual is quantized and packed independently of the factors, so the runtime evaluates
  # `up @ down + quantize(residual)`. That only reconstructs the weight when the returned residual
  # is `weight - up @ down`: a residual left relative to an intermediate refit target subtracts the
  # previous round's correction twice, and reconstructing from already-downcast factors instead of
  # the SVD's working precision inflates the drift several-fold at bfloat16. Refinement is held to
  # the drift the one-shot split already has at this dtype, which is pure storage rounding.
  weight = make_spectral_decay_weight(256, 128, seed=0, device="cpu", dtype=torch.float32)

  def _drift(refine_iters: int) -> float:
    lowrank, residual, _ = _refine_split_pieces(
      weight,
      precision=precision,
      refine_iters=refine_iters,
      torch_dtype=torch_dtype,
    )
    return compute_accuracy_metrics(weight - lowrank, residual).rel_l2

  baseline_drift = _drift(0)
  assert baseline_drift < 1e-2, "one-shot split should not drift beyond storage rounding"
  for refine_iters in (1, 2, 5):
    drift = _drift(refine_iters)
    assert drift <= baseline_drift * 1.1 + 1e-6, (
      f"residual drifted off the weight at refine_iters={refine_iters}: "
      f"{drift:.3e} vs one-shot {baseline_drift:.3e}")


@pytest.mark.parametrize("calibrate_precision", ["low", "medium", "high"])
@pytest.mark.parametrize("precision", ["int4", "nvfp4"])
def test_svdquant_refine_iters_anchoring_survives_calibrate_precision(
  precision: str,
  calibrate_precision: str,
) -> None:
  # `_resolve_math_dtype` returns the raw `torch_dtype` for the "low" route and float32 otherwise,
  # while the "high" route runs the SVD in float64. The re-anchoring has to follow that internal
  # working precision, so cover all three routes at a bfloat16 `torch_dtype`.
  weight = make_spectral_decay_weight(256, 128, seed=0, device="cpu", dtype=torch.float32)

  def _drift(refine_iters: int) -> float:
    lowrank, residual, _ = _refine_split_pieces(
      weight,
      precision=precision,
      refine_iters=refine_iters,
      torch_dtype=torch.bfloat16,
      calibrate_precision=calibrate_precision,
    )
    return compute_accuracy_metrics(weight - lowrank, residual).rel_l2

  baseline_drift = _drift(0)
  for refine_iters in (1, 5):
    assert _drift(refine_iters) <= baseline_drift * 1.1 + 1e-6, (
      f"residual drifted at calibrate_precision={calibrate_precision!r}, "
      f"refine_iters={refine_iters}")


@pytest.mark.parametrize("torch_dtype", _REFINE_TORCH_DTYPES, ids=["fp32", "bf16"])
@pytest.mark.parametrize("precision", ["int4", "nvfp4"])
def test_svdquant_refine_iters_improves_weight_error(
  precision: str,
  torch_dtype: torch.dtype,
) -> None:
  weight = make_spectral_decay_weight(256, 128, seed=0, device="cpu", dtype=torch.float32)

  def _rel_l2(refine_iters: int) -> float:
    lowrank, _, residual_q = _refine_split_pieces(
      weight,
      precision=precision,
      refine_iters=refine_iters,
      torch_dtype=torch_dtype,
    )
    return compute_accuracy_metrics(weight, lowrank + residual_q).rel_l2

  baseline_rel_l2 = _rel_l2(0)
  # Refinement must strictly beat the one-shot split, and never regress as rounds are added.
  previous_rel_l2 = baseline_rel_l2
  for refine_iters in (1, 2, 5):
    current_rel_l2 = _rel_l2(refine_iters)
    assert current_rel_l2 < baseline_rel_l2
    assert current_rel_l2 <= previous_rel_l2
    previous_rel_l2 = current_rel_l2


@pytest.mark.parametrize("precision", ["int4", "nvfp4"])
def test_svdquant_refine_iters_is_noop_at_rank_zero(precision: str) -> None:
  # At rank 0 the factors are empty, so there is nothing to refit and the split must be untouched.
  weight = make_spectral_decay_weight(256, 128, seed=0, device="cpu", dtype=torch.float32)
  baseline = _refine_split_pieces(weight,
                                  precision=precision,
                                  refine_iters=0,
                                  torch_dtype=torch.bfloat16,
                                  rank=0)
  refined = _refine_split_pieces(weight,
                                 precision=precision,
                                 refine_iters=5,
                                 torch_dtype=torch.bfloat16,
                                 rank=0)
  for expected, actual in zip(baseline, refined, strict=True):
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("precision", ["int4", "nvfp4"])
def test_svdquant_quantizer_accepts_refine_iters(precision: str) -> None:
  linear = make_rank_sensitive_linear(
    in_features=128,
    out_features=256,
    seed=0,
    device="cpu",
    dtype=torch.bfloat16,
  )
  representative = make_token_batch(
    batch_size=4,
    seq_len=8,
    width=128,
    seed=1,
    device="cpu",
    dtype=torch.bfloat16,
  )

  state_dict = quantize_linear_svdq_w4a4(
    linear,
    representative,
    rank=32,
    device="cpu",
    torch_dtype=torch.bfloat16,
    precision=precision,
    return_state_dict=True,
    **_quantizer_kwargs(svd_refine_iters=5),
  )
  assert state_dict["proj_down"].shape == (128, 32)
  assert state_dict["proj_up"].shape == (256, 32)
  for name in ("proj_down", "proj_up"):
    assert torch.isfinite(state_dict[name].to(torch.float32)).all()


def test_svdquant_quantizer_returns_nvfp4_module_state_dict() -> None:
  linear = _make_cpu_linear(128, 256)
  representative = torch.randn(3, 5, 128, dtype=torch.float32)

  state_dict: dict[str, torch.Tensor] = quantize_linear_svdq_w4a4(
    linear,
    representative,
    rank=16,
    precision="nvfp4",
    device="cpu",
    torch_dtype=torch.bfloat16,
    return_state_dict=True,
    **_quantizer_kwargs(),
  )

  assert set(state_dict) == {
    "bias",
    "proj_down",
    "proj_up",
    "qweight",
    "smooth_factor",
    "smooth_factor_orig",
    "wcscales",
    "wscales",
  }
  assert state_dict["qweight"].shape == (256, 64)
  assert state_dict["wscales"].shape == (8, 256)
  assert state_dict["wcscales"].shape == (256, )
  assert state_dict["bias"].shape == (256, )


def test_svdquant_quantizer_rejects_nvfp4_v2_runtime_kernel() -> None:
  linear = _make_cpu_linear(128, 128)

  with pytest.raises(ValueError, match="NVFP4 currently only supports"):
    svdq_quantizer._quantize_from_activation_span(
      linear,
      torch.ones(128, dtype=torch.float32),
      rank=16,
      precision="nvfp4",
      device="cpu",
      torch_dtype=torch.bfloat16,
      runtime_kernel="v2",
      return_state_dict=True,
      calibrate_precision=_CALIBRATE_PRECISION,
    )


def test_svdquant_quantizer_repairs_invalid_smooth_scales() -> None:
  linear = _make_cpu_linear(128, 128, bias=False)
  with torch.no_grad():
    linear.weight.zero_()

  state_dict: dict[str, torch.Tensor] = quantize_linear_svdq_w4a4(
    linear,
    torch.zeros(4, 128, dtype=torch.float32),
    rank=0,
    device="cpu",
    torch_dtype=torch.bfloat16,
    return_state_dict=True,
    **_quantizer_kwargs(),
  )

  assert torch.equal(state_dict["smooth_factor"], torch.ones_like(state_dict["smooth_factor"]))
  assert torch.equal(state_dict["smooth_factor_orig"],
                     torch.ones_like(state_dict["smooth_factor_orig"]))
  assert state_dict["proj_down"].shape == (128, 0)
  assert state_dict["proj_up"].shape == (128, 0)


def test_svdquant_quantizer_rejects_unsupported_geometry() -> None:
  linear = _make_cpu_linear(128, 96)

  with pytest.raises(ValueError, match="out_features"):
    quantize_linear_svdq_w4a4(
      linear,
      torch.randn(2, 128, dtype=torch.float32),
      rank=16,
      device="cpu",
      torch_dtype=torch.bfloat16,
      return_state_dict=True,
      **_quantizer_kwargs(),
    )


def test_svdquant_quantizer_applies_configurable_few_shot_relaxation() -> None:
  activation_span = torch.linspace(1.0, 8.0, steps=8, dtype=torch.float32)
  weight_span = torch.ones_like(activation_span)

  relaxed, original = svdq_quantizer._apply_few_shot_relaxation(
    activation_span,
    weight_span,
    alpha=0.5,
    relax_factor=3.0,
    relax_top_ratio=0.25,
    relax_strategy="top",
  )

  expected = svdq_quantizer.compute_smooth_scale(
    activation_span,
    weight_span,
    alpha=0.5,
    output_dtype=torch.float32,
  )
  torch.testing.assert_close(original, expected, rtol=0.0, atol=0.0)
  expected = expected.clone()
  expected[-2:] = expected[-2:] * math.sqrt(3.0)
  torch.testing.assert_close(relaxed, expected, rtol=1e-6, atol=1e-6)


def test_svdquant_quantizer_uses_new_default_few_shot_relax_factor() -> None:
  activation_span = torch.linspace(1.0, 8.0, steps=8, dtype=torch.float32)
  weight_span = torch.ones_like(activation_span)

  relaxed, original = svdq_quantizer._apply_few_shot_relaxation(activation_span, weight_span)

  expected = svdq_quantizer.compute_smooth_scale(
    activation_span,
    weight_span,
    alpha=0.5,
    output_dtype=torch.float32,
  )
  torch.testing.assert_close(original, expected, rtol=0.0, atol=0.0)
  threshold = torch.quantile(activation_span, 0.75)
  normalized = activation_span.sub(activation_span.amin()).div(threshold - activation_span.amin())
  normalized = normalized.clamp(0.0, 1.0)
  expected_multiplier = normalized.mul(0.5).add(1.0).sqrt()
  torch.testing.assert_close(relaxed, expected * expected_multiplier, rtol=1e-6, atol=1e-6)


def test_svdquant_quantizer_fixed_few_shot_relaxation_keeps_original_scale() -> None:
  activation_span = torch.linspace(1.0, 8.0, steps=8, dtype=torch.float32)
  weight_span = torch.linspace(0.5, 1.5, steps=8, dtype=torch.float32)

  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    relaxed, original = svdq_quantizer._apply_few_shot_relaxation(
      activation_span,
      weight_span,
      relax_factor=4.0,
      relax_top_ratio=0.1,
      relax_strategy="fixed",
    )

  expected = svdq_quantizer.compute_smooth_scale(
    activation_span,
    weight_span,
    alpha=0.5,
    output_dtype=torch.float32,
  )
  torch.testing.assert_close(original, expected, rtol=0.0, atol=0.0)
  torch.testing.assert_close(relaxed, expected, rtol=0.0, atol=0.0)
  assert caught == []


def test_svdquant_quantizer_rejects_relax_factor_smaller_than_one() -> None:
  activation_span = torch.linspace(1.0, 8.0, steps=8, dtype=torch.float32)
  weight_span = torch.ones_like(activation_span)

  with pytest.raises(ValueError, match=r">= 1.0"):
    svdq_quantizer._apply_few_shot_relaxation(
      activation_span,
      weight_span,
      relax_factor=0.5,
      relax_top_ratio=0.25,
      relax_strategy="auto",
    )


def test_svdquant_quantizer_warns_for_large_relax_factor() -> None:
  activation_span = torch.linspace(1.0, 8.0, steps=8, dtype=torch.float32)
  weight_span = torch.ones_like(activation_span)

  with pytest.warns(RuntimeWarning, match="oversmooth or blur outputs"):
    svdq_quantizer._apply_few_shot_relaxation(
      activation_span,
      weight_span,
      relax_factor=4.0,
      relax_top_ratio=0.25,
      relax_strategy="top",
    )


def test_svdquant_quantizer_auto_few_shot_relaxation_is_monotonic_and_bounded() -> None:
  activation_span = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
  weight_span = torch.ones_like(activation_span)

  relaxed, original = svdq_quantizer._apply_few_shot_relaxation(
    activation_span,
    weight_span,
    relax_factor=2.0,
    relax_top_ratio=0.25,
    relax_strategy="auto",
  )

  multipliers = relaxed / original
  assert torch.isclose(multipliers[0], torch.tensor(1.0))
  assert torch.isclose(multipliers[-1], torch.tensor(math.sqrt(2.0)))
  assert torch.all(multipliers[1:] >= multipliers[:-1])
  assert torch.all(multipliers >= 1.0)
  assert torch.all(multipliers <= math.sqrt(2.0))


def test_svdquant_quantizer_stable_auto_bucketizes_relaxation_response() -> None:
  activation_span = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
  weight_span = torch.ones_like(activation_span)

  relaxed, original = svdq_quantizer._apply_few_shot_relaxation(
    activation_span,
    weight_span,
    relax_factor=2.5,
    relax_top_ratio=0.25,
    relax_strategy="stable_auto",
  )

  threshold = torch.quantile(activation_span, 0.75)
  auto_response = activation_span.sub(activation_span.amin()).div(threshold -
                                                                  activation_span.amin())
  auto_response = auto_response.clamp(0.0, 1.0)
  bucket_count = svdq_quantizer._FEW_SHOT_STABLE_AUTO_BUCKETS
  stable_response = auto_response.mul(bucket_count).add(0.5).floor().div(bucket_count)
  expected_multiplier = stable_response.mul(1.5).add(1.0).sqrt()
  bucket_indices = stable_response.mul(bucket_count)

  torch.testing.assert_close(relaxed, original * expected_multiplier, rtol=1e-6, atol=1e-6)
  torch.testing.assert_close(bucket_indices, bucket_indices.round(), rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("strategy", ["stable_auto", "power", "log", "rank"])
def test_svdquant_quantizer_extra_few_shot_relax_strategies_are_monotonic_and_bounded(
  strategy: str, ) -> None:
  activation_span = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
  weight_span = torch.ones_like(activation_span)

  relaxed, original = svdq_quantizer._apply_few_shot_relaxation(
    activation_span,
    weight_span,
    relax_factor=2.5,
    relax_top_ratio=0.25,
    relax_strategy=strategy,
  )

  multipliers = relaxed / original
  assert torch.isclose(multipliers[0], torch.tensor(1.0))
  assert torch.isclose(multipliers[-1], torch.tensor(math.sqrt(2.5)))
  assert torch.all(multipliers[1:] >= multipliers[:-1])
  assert torch.all(multipliers >= 1.0)
  assert torch.all(multipliers <= math.sqrt(2.5))


def test_svdquant_quantizer_from_smooth_scale_preserves_runtime_and_original_vectors() -> None:
  linear = _make_cpu_linear(128, 128)
  smooth_orig = torch.linspace(0.5, 2.0, steps=128, dtype=torch.float32)
  smooth_runtime = smooth_orig.clone()
  smooth_runtime[-32:] = smooth_runtime[-32:] * 2.0

  state_dict: dict[str, torch.Tensor] = svdq_quantizer._quantize_from_smooth_scale(
    linear,
    smooth_runtime,
    smooth_scale_orig=smooth_orig,
    rank=16,
    device="cpu",
    torch_dtype=torch.bfloat16,
    return_state_dict=True,
    calibrate_precision=_CALIBRATE_PRECISION,
  )

  assert state_dict["smooth_factor"].shape == (128, )
  assert state_dict["smooth_factor_orig"].shape == (128, )
  assert not torch.allclose(
    state_dict["smooth_factor"].float(),
    state_dict["smooth_factor_orig"].float(),
    rtol=0.0,
    atol=1e-4,
  )


def test_svdquant_quantizer_state_dict_loads_into_module() -> None:
  linear = _make_cpu_linear(128, 128)
  representative = [
    torch.randn(4, 128, dtype=torch.float32),
    torch.randn(2, 3, 128, dtype=torch.float32),
  ]
  state_dict: dict[str, torch.Tensor] = quantize_linear_svdq_w4a4(
    linear,
    representative,
    rank=16,
    device="cpu",
    torch_dtype=torch.bfloat16,
    return_state_dict=True,
    **_quantizer_kwargs(),
  )

  module = SVDQW4A4Linear.from_linear(
    linear,
    rank=16,
    precision="int4",
    torch_dtype=torch.bfloat16,
    device="cpu",
  )
  incompatible = module.load_state_dict(state_dict, strict=True)
  assert incompatible.missing_keys == []
  assert incompatible.unexpected_keys == []


def test_svdquant_quantizer_nvfp4_state_dict_loads_into_module() -> None:
  linear = _make_cpu_linear(128, 128)
  representative = [
    torch.randn(4, 128, dtype=torch.float32),
    torch.randn(2, 3, 128, dtype=torch.float32),
  ]
  state_dict: dict[str, torch.Tensor] = quantize_linear_svdq_w4a4(
    linear,
    representative,
    rank=16,
    precision="nvfp4",
    device="cpu",
    torch_dtype=torch.bfloat16,
    return_state_dict=True,
    **_quantizer_kwargs(),
  )

  module = SVDQW4A4Linear.from_linear(
    linear,
    rank=16,
    precision="nvfp4",
    torch_dtype=torch.bfloat16,
    device="cpu",
  )
  incompatible = module.load_state_dict(state_dict, strict=True)
  assert incompatible.missing_keys == []
  assert incompatible.unexpected_keys == []


def test_svdquant_quantizer_streaming_matches_eager_state_dict() -> None:
  linear = _make_cpu_linear(128, 128)
  representative = [
    torch.randn(4, 128, dtype=torch.bfloat16),
    torch.randn(2, 3, 128, dtype=torch.bfloat16),
  ]

  streamed = quantize_linear_svdq_w4a4(
    linear,
    representative,
    rank=16,
    device="cpu",
    torch_dtype=torch.bfloat16,
    return_state_dict=True,
    calibrate_precision=_CALIBRATE_PRECISION,
    streaming=True,
  )
  eager = quantize_linear_svdq_w4a4(
    linear,
    representative,
    rank=16,
    device="cpu",
    torch_dtype=torch.bfloat16,
    return_state_dict=True,
    calibrate_precision=_CALIBRATE_PRECISION,
    streaming=False,
  )

  assert set(streamed) == set(eager)
  for key in streamed:
    torch.testing.assert_close(streamed[key], eager[key], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
  "buffer_kwargs",
  [
    {
      "activation_buffer_flush_sample_count": 2
    },
    {
      "activation_buffer_flush_cpu_bytes": 256
    },
    {
      "activation_buffer_flush_sample_count": 3,
      "activation_buffer_flush_cpu_bytes": 256,
    },
  ],
)
def test_svdquant_quantizer_streaming_flush_thresholds_match_eager_state_dict(
  buffer_kwargs: dict[str, int], ) -> None:
  linear = _make_cpu_linear(128, 128)
  representative = [
    torch.randn(4, 128, dtype=torch.bfloat16),
    torch.randn(2, 3, 128, dtype=torch.bfloat16),
    torch.randn(1, 7, 128, dtype=torch.bfloat16),
    torch.randn(6, 128, dtype=torch.bfloat16),
  ]

  buffered = quantize_linear_svdq_w4a4(
    linear,
    (tensor for tensor in representative),
    rank=16,
    device="cpu",
    torch_dtype=torch.bfloat16,
    return_state_dict=True,
    calibrate_precision=_CALIBRATE_PRECISION,
    streaming=True,
    **buffer_kwargs,
  )
  eager = quantize_linear_svdq_w4a4(
    linear,
    representative,
    rank=16,
    device="cpu",
    torch_dtype=torch.bfloat16,
    return_state_dict=True,
    calibrate_precision=_CALIBRATE_PRECISION,
    streaming=False,
  )

  assert set(buffered) == set(eager)
  for key in buffered:
    torch.testing.assert_close(buffered[key], eager[key], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("svd_precision", ["low", "medium", "high"])
def test_decompose_lowrank_residual_modes_return_finite_tensors(svd_precision: str) -> None:
  weight = torch.randn(128, 128, dtype=torch.bfloat16)

  down, up, residual = decompose_lowrank_residual(
    weight,
    rank=16,
    output_dtype=torch.bfloat16,
    svd_precision=svd_precision,
  )

  assert down.shape == (16, 128)
  assert up.shape == (128, 16)
  assert residual.shape == (128, 128)
  assert torch.isfinite(down).all()
  assert torch.isfinite(up).all()
  assert torch.isfinite(residual).all()


def test_decompose_lowrank_residual_low_uses_svd_lowrank_and_float32_retry(
  monkeypatch: pytest.MonkeyPatch, ) -> None:
  original = torch.svd_lowrank
  calls: list[tuple[torch.dtype, int, int]] = []

  def wrapped(matrix: torch.Tensor, *args, **kwargs):
    calls.append((matrix.dtype, kwargs["q"], kwargs["niter"]))
    if matrix.dtype == torch.bfloat16:
      raise RuntimeError("simulated low-precision svd_lowrank failure")
    return original(matrix, *args, **kwargs)

  monkeypatch.setattr(torch, "svd_lowrank", wrapped)

  down, up, residual = decompose_lowrank_residual(
    torch.randn(64, 64, dtype=torch.bfloat16),
    rank=16,
    output_dtype=torch.bfloat16,
    svd_precision="low",
  )

  assert calls == [(torch.bfloat16, 26, 4), (torch.float32, 26, 4)]
  assert torch.isfinite(down).all()
  assert torch.isfinite(up).all()
  assert torch.isfinite(residual).all()


def test_decompose_lowrank_residual_low_is_deterministic() -> None:
  weight = torch.randn(64, 64, dtype=torch.bfloat16)

  first = decompose_lowrank_residual(
    weight,
    rank=16,
    output_dtype=torch.bfloat16,
    svd_precision="low",
  )
  second = decompose_lowrank_residual(
    weight,
    rank=16,
    output_dtype=torch.bfloat16,
    svd_precision="low",
  )

  for lhs, rhs in zip(first, second):
    torch.testing.assert_close(lhs, rhs, rtol=0.0, atol=0.0)


def test_decompose_lowrank_residual_medium_retries_in_float32(
  monkeypatch: pytest.MonkeyPatch, ) -> None:
  original = torch.linalg.svd
  calls: list[tuple[torch.dtype, str | None]] = []

  def wrapped(matrix: torch.Tensor, *args, **kwargs):
    calls.append((matrix.dtype, kwargs.get("driver")))
    if matrix.dtype == torch.bfloat16:
      raise RuntimeError("simulated low-precision full SVD failure")
    return original(matrix, *args, **kwargs)

  monkeypatch.setattr(torch.linalg, "svd", wrapped)

  down, up, residual = decompose_lowrank_residual(
    torch.randn(64, 64, dtype=torch.bfloat16),
    rank=16,
    output_dtype=torch.bfloat16,
    svd_precision="medium",
  )

  assert calls == [(torch.bfloat16, None), (torch.float32, None)]
  assert torch.isfinite(down).all()
  assert torch.isfinite(up).all()
  assert torch.isfinite(residual).all()


def test_decompose_lowrank_residual_high_uses_expected_driver(
  monkeypatch: pytest.MonkeyPatch, ) -> None:
  original = torch.linalg.svd
  calls: list[tuple[torch.dtype, str | None]] = []

  def wrapped(matrix: torch.Tensor, *args, **kwargs):
    calls.append((matrix.dtype, kwargs.get("driver")))
    return original(matrix, *args, **kwargs)

  monkeypatch.setattr(torch.linalg, "svd", wrapped)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dtype = (torch.bfloat16
           if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32)
  weight = torch.randn(64, 64, device=device, dtype=dtype)

  decompose_lowrank_residual(
    weight,
    rank=16,
    output_dtype=dtype,
    svd_precision="high",
  )

  expected_driver = "gesvd" if device.type == "cuda" else None
  assert calls == [(torch.float64, expected_driver)]


def test_svdquant_quantizer_runtime_rank32_beats_rank0() -> None:
  if not torch.cuda.is_available() or not svdq_extension_is_available():
    pytest.skip("CUDA runtime validation requires the optional SVDQuant extension.")

  device = "cuda"
  dtype = runtime_dtype()
  in_features = 128
  out_features = 128

  linear = make_rank_sensitive_linear(
    in_features=in_features,
    out_features=out_features,
    seed=17,
    device=device,
    dtype=dtype,
  )
  calibration = make_token_samples(
    num_samples=4,
    batch_size=1,
    seq_len=16,
    width=in_features,
    seed=29,
    device="cpu",
    dtype=dtype,
  )
  rank0_module: SVDQW4A4Linear = quantize_linear_svdq_w4a4(linear,
                                                           calibration,
                                                           rank=0,
                                                           device=device,
                                                           torch_dtype=dtype,
                                                           **_quantizer_kwargs())
  rank16_module: SVDQW4A4Linear = quantize_linear_svdq_w4a4(linear,
                                                            calibration,
                                                            rank=16,
                                                            device=device,
                                                            torch_dtype=dtype,
                                                            **_quantizer_kwargs())
  rank32_module: SVDQW4A4Linear = quantize_linear_svdq_w4a4(linear,
                                                            calibration,
                                                            rank=32,
                                                            device=device,
                                                            torch_dtype=dtype,
                                                            **_quantizer_kwargs())
  rank128_module: SVDQW4A4Linear = quantize_linear_svdq_w4a4(linear,
                                                             calibration,
                                                             rank=128,
                                                             device=device,
                                                             torch_dtype=dtype,
                                                             **_quantizer_kwargs())

  x = make_token_batch(
    batch_size=2,
    seq_len=16,
    width=in_features,
    seed=41,
    device=device,
    dtype=dtype,
  )
  with torch.inference_mode():
    reference = linear(x)
    rank0_output = rank0_module(x)
    rank16_output = rank16_module(x)
    rank32_output = rank32_module(x)
    rank128_output = rank128_module(x)
    torch.cuda.synchronize()

  metrics_by_rank = {
    0: compute_accuracy_metrics(reference, rank0_output),
    16: compute_accuracy_metrics(reference, rank16_output),
    32: compute_accuracy_metrics(reference, rank32_output),
    128: compute_accuracy_metrics(reference, rank128_output),
  }
  print(format_rank_report("SVDQ linear module accuracy report\n", metrics_by_rank))

  rank0_error = metrics_by_rank[0].mae
  rank16_error = metrics_by_rank[16].mae
  rank32_error = metrics_by_rank[32].mae
  rank128_error = metrics_by_rank[128].mae
  assert rank16_error < rank0_error
  assert rank32_error < rank16_error
  assert rank128_error < rank32_error


def test_svdquant_toymodel_rank_accuracy_roundtrip_report(tmp_path: Path) -> None:
  if not torch.cuda.is_available() or not svdq_extension_is_available():
    pytest.skip("CUDA runtime validation requires the optional SVDQuant extension.")

  device = "cuda"
  dtype = runtime_dtype()  # torch.bfloat16
  num_heads = 16 if not _ENABLE_LARGE_HEAD_NUMBER else 32
  embed_dim = 128 * num_heads

  model = make_toy_model(
    embed_dim=embed_dim,
    num_heads=num_heads,
    seed=0,
    device=device,
    dtype=dtype,
  )
  # case 0: large head number with shorter sequence length to reduce quantization time.
  # case 1: small head number with longer sequence length to better simulate the quantization.
  calibration_samples = make_token_samples(
    num_samples=8,
    batch_size=1,
    seq_len=8192 if not _ENABLE_LARGE_HEAD_NUMBER else 1024,
    width=embed_dim,
    seed=0,
    device=device,
    dtype=dtype,
  )
  # For simplicity, we use the same calibration samples as evaluation inputs. The main
  # goal of this test is to validate the quantizer's offline-to-runtime accuracy trend
  # and state dict integrity, rather than to benchmark on a separate evaluation set.
  eval_inputs = torch.cat(calibration_samples, dim=0)
  H, D, B, S = num_heads, embed_dim, eval_inputs.shape[0], eval_inputs.shape[1]

  metrics_by_rank = {}
  quantization_latency_rows: list[tuple[object, ...]] = []
  # Warmup
  with torch.inference_mode():
    reference = model(eval_inputs)
    torch.cuda.synchronize()
  # Profile reference latency, repeats=10
  with torch.inference_mode():
    start_time = time.perf_counter()
    for _ in range(10):
      _ = model(eval_inputs)
    torch.cuda.synchronize()
    reference_latency = (time.perf_counter() - start_time) / 10
    metrics_by_rank[-1] = compute_accuracy_metrics(
      reference,
      reference,
      latency_ms=reference_latency * 1000,  # reference latency in milliseconds
    )

  for rank in RANKS_WITH_BASELINE:
    quantize_start_time = time.perf_counter()
    quantized_model = quantize_toy_model(
      model,
      calibration_samples,
      rank=rank,
      device=device,
      dtype=dtype,
      calibrate_precision=_CALIBRATE_PRECISION,
    )
    torch.cuda.synchronize()
    quantize_latency = time.perf_counter() - quantize_start_time
    quantization_latency_rows.append((rank, f"{quantize_latency:.6f}"))

    checkpoint_path = tmp_path / f"svdq_toy_rank{rank}.pt"
    torch.save(
      {
        "model_config": {
          "embed_dim": embed_dim,
          "num_heads": num_heads
        },
        "rank": rank,
        "state_dict": quantized_model.state_dict(),
      },
      checkpoint_path,
    )

    payload = torch.load(checkpoint_path, map_location=device)
    model_config = payload["model_config"]
    reloaded_model = build_empty_quantized_toy_model(
      embed_dim=model_config["embed_dim"],
      num_heads=model_config["num_heads"],
      rank=payload["rank"],
      device=device,
      dtype=dtype,
    )
    incompatible = reloaded_model.load_state_dict(payload["state_dict"], strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    # Warmup
    with torch.inference_mode():
      quantized_output = quantized_model(eval_inputs)
      reloaded_output = reloaded_model(eval_inputs)
      torch.cuda.synchronize()

    # Profile and validate outputs, repeats=10
    with torch.inference_mode():
      start_time = time.perf_counter()
      for _ in range(10):
        _ = reloaded_model(eval_inputs)
      torch.cuda.synchronize()
      reloaded_latency = (time.perf_counter() - start_time) / 10
    # May not bitwise-deterministic due to non-determinism in CUDA.
    # BFloat16 atol can be ranged in [4e-3, 8e-3].
    atol, rtol = _current_tolerance()
    torch.testing.assert_close(reloaded_output, quantized_output, rtol=rtol, atol=atol)
    metrics_by_rank[rank] = compute_accuracy_metrics(
      reference,
      reloaded_output,
      reloaded_latency * 1000,  # reloaded latency in milliseconds
    )

  print(
    format_markdown_table(
      "SVDQ ToyModel profiling config\n",
      ("num_heads", "embed_dim", "batch", "seq_len", "calibrate_precision"),
      [(H, D, B, S, _CALIBRATE_PRECISION)],
    ))
  print(
    format_markdown_table(
      "SVDQ ToyModel quantization latency\n",
      ("rank", "quantization_s"),
      quantization_latency_rows,
    ))
  print(format_rank_report("SVDQ ToyModel accuracy report\n", metrics_by_rank))
  assert_rank_metric_trend(metrics_by_rank, "mae", ranks=RANKS_WITH_BASELINE)
  assert_rank_metric_trend(metrics_by_rank, "rel_l2", ranks=RANKS_WITH_BASELINE)
  for rank in EVALUATED_RANKS:
    assert metrics_by_rank[rank].mae < metrics_by_rank[0].mae


@pytest.mark.skipif(
  not torch.cuda.is_available() or not _ENABLE_STREAMING_MEMORY_BENCH,
  reason="Streaming memory benchmark requires CUDA and CACHE_DIT_SVDQ_TEST_LARGE_MEMORY=1.",
)
def test_svdquant_streaming_memory_peak_is_lower() -> None:
  device_props = torch.cuda.get_device_properties(0)
  total_gib = device_props.total_memory / (1024 ** 3)
  if total_gib < _LARGE_MEMORY_MIN_DEVICE_GIB:
    pytest.skip(
      f"Streaming memory benchmark requires at least {_LARGE_MEMORY_MIN_DEVICE_GIB:.1f} GiB, got {total_gib:.1f} GiB."
    )

  device = torch.device("cuda")
  dtype = runtime_dtype()
  linear = nn.Linear(128, 128, bias=False, device=device, dtype=dtype).eval()
  representative = _make_large_cpu_calibration_list(
    in_features=128,
    total_gib=_LARGE_MEMORY_TOTAL_GIB,
    chunk_mib=_LARGE_MEMORY_CHUNK_MIB,
    dtype=dtype,
  )

  try:
    streaming_peak = _measure_quantizer_peak_memory(
      linear,
      representative,
      dtype=dtype,
      streaming=True,
    )
    eager_peak = _measure_quantizer_peak_memory(
      linear,
      representative,
      dtype=dtype,
      streaming=False,
    )
  except torch.cuda.OutOfMemoryError as exc:
    pytest.skip(f"Not enough free GPU memory for the eager streaming benchmark: {exc}")

  assert eager_peak > streaming_peak
  savings_pct = 100.0 * (eager_peak - streaming_peak) / eager_peak
  print(
    format_markdown_table(
      "SVDQ streaming memory benchmark\n",
      (
        "rank",
        "cpu_calibration_gib",
        "streaming_peak_gib",
        "eager_peak_gib",
        "savings_pct",
      ),
      [(
        32,
        f"{_LARGE_MEMORY_TOTAL_GIB:.2f}",
        f"{streaming_peak / 2**30:.4f}",
        f"{eager_peak / 2**30:.4f}",
        f"{savings_pct:.2f}",
      )],
    ))
  assert savings_pct >= _STREAMING_MEMORY_THRESHOLD_PCT
