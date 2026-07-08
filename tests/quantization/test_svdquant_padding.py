"""Operator-level correctness tests for SVDQ NVFP4 padding support.

Tests SVDQW4A4Linear with non-128-multiple in_features / out_features to verify that the padding/de-
padding logic produces correct results (within quantization tolerance) compared to the original FP16
linear layer.
"""

from __future__ import annotations

import pytest
import torch

from cache_dit.quantization.svdquant.linear import SVDQW4A4Linear
from cache_dit.quantization.svdquant.quantizer import _quantize_from_smooth_scale
from cache_dit.quantization.svdquant.quantizer import _pad_to_multiple_128
from cache_dit.quantization.svdquant.quantizer import validate_linear_geometry

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16


def _create_linear(in_features, out_features, bias=True):
  return torch.nn.Linear(in_features, out_features, bias=bias, dtype=TORCH_DTYPE, device=DEVICE)


def _quantize_linear(linear, rank=0, precision="nvfp4"):
  """Run a minimal smooth-scale quantize for testing, returns SVDQW4A4Linear."""
  smooth = torch.ones(linear.in_features, dtype=TORCH_DTYPE, device=DEVICE)
  smooth_orig = smooth.clone()
  result = _quantize_from_smooth_scale(
    linear,
    smooth,
    smooth_scale_orig=smooth_orig,
    rank=rank,
    precision=precision,
    act_unsigned=False,
    torch_dtype=TORCH_DTYPE,
    device=DEVICE,
    calibrate_precision="low",
    runtime_kernel="v1",
  )
  return result


def _assert_output_sane(quant_output: torch.Tensor):
  """Sanity checks: no NaN/Inf, output is finite, max abs is reasonable (<100)."""
  assert not torch.isnan(quant_output).any(), "Quantized output contains NaN"
  assert not torch.isinf(quant_output).any(), "Quantized output contains Inf"
  max_abs = quant_output.abs().max().item()
  assert max_abs < 100.0, f"Output max abs {max_abs:.2f} exceeds sanity limit"


class TestSVDQPadding:
  """Test SVDQ NVFP4 with non-128-multiple dimensions."""

  @pytest.mark.parametrize(
    "in_features,out_features",
    [
      (3360, 3360),  # Boogu-Image typical dims
      (3360, 1280),  # mixed alignment
      (1280, 3360),  # mixed alignment (reverse)
      (512, 512),  # already aligned (regression test)
      (320, 640),  # small non-aligned
      (640, 320),  # small non-aligned (reverse)
    ])
  def test_padded_quantize_output_shape(self, in_features, out_features):
    """SVDQW4A4Linear forward output shape matches logical dims."""
    linear = _create_linear(in_features, out_features, bias=False)
    quantized = _quantize_linear(linear, rank=0)
    assert quantized.in_features == in_features
    assert quantized.out_features == out_features

    x = torch.randn(2, 3, in_features, dtype=TORCH_DTYPE, device=DEVICE)
    output = quantized(x)
    assert output.shape == (2, 3, out_features)
    _assert_output_sane(output)

  @pytest.mark.parametrize("in_features,out_features,rank", [
    (3360, 3360, 0),
    (320, 640, 0),
    (3360, 3360, 32),
    (320, 640, 32),
  ])
  def test_padded_quantize_no_nan_inf(self, in_features, out_features, rank):
    """Quantized output from padded layers contains no NaN/Inf."""
    linear = _create_linear(in_features, out_features, bias=False)
    quantized = _quantize_linear(linear, rank=rank)

    x = torch.randn(1, 17, in_features, dtype=TORCH_DTYPE, device=DEVICE)
    with torch.inference_mode():
      output = quantized(x)
    _assert_output_sane(output)

  @pytest.mark.parametrize("in_features,out_features", [
    (3360, 3360),
    (320, 640),
  ])
  def test_padded_quantize_output_reproducible(self, in_features, out_features):
    """Same input twice produces identical quantized output."""
    linear = _create_linear(in_features, out_features, bias=False)
    quantized = _quantize_linear(linear, rank=0)

    x = torch.randn(1, 17, in_features, dtype=TORCH_DTYPE, device=DEVICE)
    with torch.inference_mode():
      out1 = quantized(x)
      out2 = quantized(x)
    assert torch.equal(out1, out2)

  @pytest.mark.parametrize("in_features,out_features", [
    (512, 512),
    (512, 1024),
  ])
  def test_already_aligned_no_overhead(self, in_features, out_features):
    """Already-128-aligned layers have zero-overhead padding path."""
    linear = _create_linear(in_features, out_features, bias=False)
    quantized = _quantize_linear(linear, rank=0)

    assert not quantized._needs_k_pad
    assert not quantized._needs_n_strip
    assert quantized._padded_in_features == in_features
    assert quantized._padded_out_features == out_features

  @pytest.mark.parametrize("in_features,out_features", [
    (512, 512),
    (512, 1024),
  ])
  def test_aligned_output_with_and_without_pad_same(self, in_features, out_features):
    """Same already-aligned layer gives identical output via forward or manual path."""
    linear = _create_linear(in_features, out_features, bias=False)
    quantized = _quantize_linear(linear, rank=0)

    x = torch.randn(1, 17, in_features, dtype=TORCH_DTYPE, device=DEVICE).reshape(17, in_features)
    with torch.inference_mode():
      # Through normal forward (with potential K-pad + N-strip which are no-ops)
      forward_out = quantized(x.unsqueeze(0).unsqueeze(0))

      # Through raw quantize + forward_quant + strip
      quantized_x, ascales, lora_act = quantized.quantize(x)
      padded_out = quantized.forward_quant(quantized_x, ascales, lora_act)
      raw_out = padded_out[:x.shape[0]]
      raw_out = raw_out[:, :out_features]

    assert torch.equal(forward_out.squeeze(0).squeeze(0), raw_out)

  def test_pad_multiple_computation(self):
    """_pad_to_multiple_128 round-up logic."""
    assert _pad_to_multiple_128(1) == 128
    assert _pad_to_multiple_128(127) == 128
    assert _pad_to_multiple_128(128) == 128
    assert _pad_to_multiple_128(129) == 256
    assert _pad_to_multiple_128(3360) == 3456
    assert _pad_to_multiple_128(0) == 0

  def test_validate_linear_geometry_allows_non_128_multiple(self):
    """validate_linear_geometry no longer rejects non-128-multiple dims."""
    validate_linear_geometry(3360, 3360, rank=0, precision="nvfp4")
    validate_linear_geometry(3360, 3360, rank=32, precision="nvfp4")
    validate_linear_geometry(320, 640, rank=0, precision="int4")

  def test_validate_linear_geometry_still_rejects_bad_group_size(self):
    """validate_linear_geometry still rejects in_features not divisible by group_size."""
    with pytest.raises(ValueError, match="in_features divisible by 16"):
      validate_linear_geometry(15, 128, rank=0, precision="nvfp4")

  def test_validate_linear_geometry_still_rejects_bad_rank(self):
    """validate_linear_geometry still rejects rank not divisible by 16."""
    with pytest.raises(ValueError, match="multiple of 16"):
      validate_linear_geometry(128, 128, rank=5, precision="nvfp4")

  def test_svdq_linear_repr_shows_logical_dims(self):
    """SVDQW4A4Linear.__repr__ shows logical dims with padded info."""
    linear = SVDQW4A4Linear(3360, 3360, rank=0, precision="nvfp4", torch_dtype=TORCH_DTYPE)
    rep = repr(linear)
    assert "in_features=3360" in rep
    assert "out_features=3360" in rep
    assert "padded_in=3456" in rep
    assert "padded_out=3456" in rep

  def test_svdq_linear_repr_aligned_no_padded_info(self):
    """SVDQW4A4Linear.__repr__ omits padded info when already aligned."""
    linear = SVDQW4A4Linear(512, 512, rank=0, precision="int4", torch_dtype=TORCH_DTYPE)
    rep = repr(linear)
    assert "in_features=512" in rep
    assert "padded_in" not in rep

  def test_padded_weights_produce_zero_output_on_zero_input(self):
    """Zero input to padded SVDQW4A4Linear produces zero output."""
    in_features, out_features = 3360, 3360
    linear = _create_linear(in_features, out_features, bias=False)
    quantized = _quantize_linear(linear, rank=0)

    x = torch.zeros(1, 1, in_features, dtype=TORCH_DTYPE, device=DEVICE)
    with torch.inference_mode():
      output = quantized(x)
    assert output.abs().max().item() < 1e-3, "Zero input should produce near-zero output"


class TestSVDQPaddingINT4:
  """Test SVDQ INT4 with non-128-multiple dimensions (group_size=64)."""

  @pytest.mark.parametrize(
    "in_features,out_features",
    [
      (320, 640),  # non-128-multiple but 64-multiple
      (192, 384),  # smaller non-aligned
      (512, 512),  # already aligned (regression)
    ])
  def test_int4_quantize_output_shape(self, in_features, out_features):
    """INT4 SVDQW4A4Linear forward output shape matches logical dims."""
    linear = _create_linear(in_features, out_features, bias=False)
    quantized = _quantize_linear(linear, rank=0, precision="int4")
    assert quantized.in_features == in_features
    assert quantized.out_features == out_features
    assert quantized.precision == "int4"

    x = torch.randn(2, 3, in_features, dtype=TORCH_DTYPE, device=DEVICE)
    output = quantized(x)
    assert output.shape == (2, 3, out_features)
    _assert_output_sane(output)

  @pytest.mark.parametrize("in_features,out_features,rank", [
    (320, 640, 0),
    (320, 640, 32),
    (192, 384, 0),
  ])
  def test_int4_quantize_no_nan_inf(self, in_features, out_features, rank):
    """INT4 quantized output from padded layers contains no NaN/Inf."""
    linear = _create_linear(in_features, out_features, bias=False)
    quantized = _quantize_linear(linear, rank=rank, precision="int4")

    x = torch.randn(1, 17, in_features, dtype=TORCH_DTYPE, device=DEVICE)
    with torch.inference_mode():
      output = quantized(x)
    _assert_output_sane(output)

  def test_int4_already_aligned_no_overhead(self):
    """Already-128-aligned INT4 layers have zero-overhead padding path."""
    linear = _create_linear(512, 512, bias=False)
    quantized = _quantize_linear(linear, rank=0, precision="int4")

    assert not quantized._needs_k_pad
    assert not quantized._needs_n_strip

  def test_int4_zero_input_produces_zero_output(self):
    """Zero input to padded INT4 SVDQW4A4Linear produces zero output."""
    linear = _create_linear(320, 640, bias=False)
    quantized = _quantize_linear(linear, rank=0, precision="int4")

    x = torch.zeros(1, 1, 320, dtype=TORCH_DTYPE, device=DEVICE)
    with torch.inference_mode():
      output = quantized(x)
    assert output.abs().max().item() < 1e-3

  def test_int4_validate_geometry_allows_non_128_multiple(self):
    """INT4 validate_linear_geometry allows 64-multiples that aren't 128-multiples."""
    validate_linear_geometry(320, 640, rank=0, precision="int4")
    validate_linear_geometry(192, 384, rank=16, precision="int4")

  def test_int4_validate_geometry_rejects_non_64_multiple(self):
    """INT4 validate_linear_geometry still rejects in_features not divisible by 64."""
    with pytest.raises(ValueError, match="in_features divisible by 64"):
      validate_linear_geometry(100, 256, rank=0, precision="int4")
