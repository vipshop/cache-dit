"""Cd cache-dit pytest tests/quantization/test_svdquant_imports.py -v -s."""

import pytest
import torch

from cache_dit.kernels import svdq_extension_is_available
from cache_dit.kernels import svdq_get_load_error
from cache_dit.quantization.svdquant import SVDQW4A4Linear


def test_svdquant_import_safe_without_extension() -> None:
  available = svdq_extension_is_available()
  error = svdq_get_load_error()

  if available:
    assert error is None
  else:
    assert error is None or isinstance(error, Exception)


def test_svdq_w4a4_linear_int4_parameter_shapes() -> None:
  layer = SVDQW4A4Linear(
    in_features=128,
    out_features=64,
    rank=32,
    precision="int4",
    torch_dtype=torch.bfloat16,
  )

  assert layer.group_size == 64
  assert layer.qweight.shape == (64, 64)
  assert layer.wscales.shape == (2, 64)
  assert layer.smooth_factor.shape == (128, )
  assert layer.smooth_factor_orig.shape == (128, )
  assert layer.proj_down.shape == (128, 32)
  assert layer.proj_up.shape == (64, 32)
  assert layer.wtscale is None
  assert layer.wcscales is None
  assert layer.runtime_kernel == "v1"


def test_svdq_w4a4_linear_rejects_unsupported_geometry() -> None:
  try:
    SVDQW4A4Linear(in_features=96, out_features=64, rank=32, precision="int4")
  except ValueError as exc:
    assert "group_size" in str(exc)
  else:
    raise AssertionError("Expected unsupported INT4 geometry to raise ValueError.")


def test_svdq_w4a4_linear_rejects_unknown_runtime_kernel() -> None:
  try:
    SVDQW4A4Linear(
      in_features=128,
      out_features=64,
      rank=32,
      precision="int4",
      runtime_kernel="v4",
    )
  except ValueError as exc:
    assert "runtime_kernel" in str(exc)
  else:
    raise AssertionError("Expected unsupported runtime_kernel to raise ValueError.")


def test_svdq_w4a4_linear_forward_accepts_2d_input(monkeypatch: pytest.MonkeyPatch, ) -> None:
  layer = SVDQW4A4Linear(
    in_features=128,
    out_features=64,
    rank=32,
    precision="int4",
    torch_dtype=torch.bfloat16,
  )

  def fake_quantize(x: torch.Tensor, pad_size: int = 256):
    del pad_size
    token_count = x.shape[0]
    quantized_x = torch.zeros(token_count + 2, 64, dtype=torch.int8)
    ascales = torch.ones(token_count + 2, 1, dtype=torch.bfloat16)
    lora_act = torch.zeros(token_count + 2, layer.rank, dtype=torch.bfloat16)
    return quantized_x, ascales, lora_act

  def fake_forward_quant(
    quantized_x: torch.Tensor,
    ascales: torch.Tensor,
    lora_act: torch.Tensor,
    output: torch.Tensor | None = None,
  ) -> torch.Tensor:
    del ascales, lora_act
    result = torch.arange(
      quantized_x.shape[0] * layer.out_features,
      dtype=torch.bfloat16,
    ).reshape(quantized_x.shape[0], layer.out_features)
    if output is not None:
      output.copy_(result)
      return output
    return result

  monkeypatch.setattr(layer, "quantize", fake_quantize)
  monkeypatch.setattr(layer, "forward_quant", fake_forward_quant)

  x = torch.randn(3, 128, dtype=torch.bfloat16)
  output = layer(x)

  assert output.shape == (3, 64)
  torch.testing.assert_close(
    output,
    torch.arange(3 * 64, dtype=torch.bfloat16).reshape(3, 64),
    rtol=0.0,
    atol=0.0,
  )
