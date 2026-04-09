"""Cd cache-dit pytest tests/quantization/test_svdquant_compile.py -v -s."""

import pytest
import torch
from torch import nn

from cache_dit.kernels import svdq_extension_is_available
from cache_dit.quantization.svdquant import quantize_linear_svdq_w4a4


def _require_svdquant_compile_runtime() -> None:
  if not torch.cuda.is_available():
    pytest.skip("CUDA is required for the SVDQuant torch.compile smoke test.")
  if not svdq_extension_is_available():
    pytest.skip("The optional Cache-DiT SVDQuant CUDA extension is not built in this environment.")
  if not hasattr(torch, "compile"):
    pytest.skip("torch.compile is not available in this torch build.")


def _runtime_dtype() -> torch.dtype:
  return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def test_svdquant_w4a4_module_torch_compile_fullgraph_smoke() -> None:
  _require_svdquant_compile_runtime()

  torch.compiler.reset()
  torch.manual_seed(0)

  device = torch.device("cuda")
  dtype = _runtime_dtype()

  linear = nn.Linear(128, 128, bias=True, device=device, dtype=dtype).eval()
  calibration = torch.randn(64, 128, device="cpu", dtype=dtype)
  quantized = quantize_linear_svdq_w4a4(
    linear,
    calibration,
    rank=16,
    device=device,
    torch_dtype=dtype,
    high_precision=False,
    fp32_fallback=True,
    streaming=True,
  ).eval()

  x = torch.randn(2, 16, 128, device=device, dtype=dtype)

  with torch.inference_mode():
    eager = quantized(x)

    compiled_module = torch.compile(quantized, fullgraph=True)
    compiled_module(x)
    compiled = compiled_module(x)
    torch.cuda.synchronize()

  assert eager.shape == compiled.shape == (2, 16, 128)
  assert eager.dtype == compiled.dtype == dtype
  assert torch.isfinite(compiled).all()
  torch.testing.assert_close(compiled, eager, rtol=0.0, atol=0.0)


def test_svdquant_w4a4_module_torch_compile_fullgraph_v2_smoke() -> None:
  _require_svdquant_compile_runtime()

  torch.compiler.reset()
  torch.manual_seed(0)

  device = torch.device("cuda")
  dtype = _runtime_dtype()

  linear = nn.Linear(128, 128, bias=True, device=device, dtype=dtype).eval()
  calibration = torch.randn(64, 128, device="cpu", dtype=dtype)
  quantized = quantize_linear_svdq_w4a4(
    linear,
    calibration,
    rank=16,
    device=device,
    torch_dtype=dtype,
    high_precision=False,
    fp32_fallback=True,
    streaming=True,
  ).eval()
  quantized.runtime_kernel = "v2"

  x = torch.randn(2, 16, 128, device=device, dtype=dtype)

  with torch.inference_mode():
    eager = quantized(x)

    compiled_module = torch.compile(quantized, fullgraph=True)
    compiled_module(x)
    compiled = compiled_module(x)
    torch.cuda.synchronize()

  assert eager.shape == compiled.shape == (2, 16, 128)
  assert eager.dtype == compiled.dtype == dtype
  assert torch.isfinite(compiled).all()
  torch.testing.assert_close(compiled, eager, rtol=0.0, atol=0.0)
