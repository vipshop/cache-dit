"""FlashInfer SVDQuant runtime tests (require FlashInfer and the CUDA extension)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cache_dit.kernels import cuda_ext_is_available
from cache_dit.kernels.flashinfer import CUTE_DSL_RUNTIME
from cache_dit.kernels.flashinfer import CUTLASS_SM120_RUNTIME
from cache_dit.kernels.flashinfer import flashinfer_available
from cache_dit.quantization.svdquant.linear import SVDQW4A4Linear
from cache_dit.quantization.svdquant.quantizer import quantize_linear_svdq_w4a4

pytestmark = pytest.mark.skipif(
  not torch.cuda.is_available() or not cuda_ext_is_available() or not flashinfer_available(),
  reason="The FlashInfer SVDQuant runtime requires CUDA, the SVDQuant extension, and FlashInfer.",
)


def _require_sm120() -> None:
  if torch.cuda.get_device_capability(0)[0] < 12:
    pytest.skip("SM120 is required")


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
  return ((actual.float() - expected.float()).norm() /
          expected.float().norm().clamp_min(1e-12)).item()


def _make_pair(m: int, n: int, k: int, rank: int = 32, runtime: str = CUTE_DSL_RUNTIME):
  torch.manual_seed(0)
  linear = nn.Linear(k, n, bias=True).to(device="cuda", dtype=torch.bfloat16)
  x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / (k ** 0.25)
  native = quantize_linear_svdq_w4a4(linear, [x], rank=rank, precision="nvfp4").to("cuda")
  other = SVDQW4A4Linear(k,
                         n,
                         rank=rank,
                         precision="nvfp4",
                         runtime_kernel=runtime,
                         torch_dtype=torch.bfloat16,
                         device="cuda")
  other.load_state_dict(native.state_dict())
  other = other.to("cuda")
  return linear, x, native, other


def test_flashinfer_layer_tracks_the_native_runtime() -> None:
  _require_sm120()
  linear, x, native, flash = _make_pair(512, 320, 320)
  with torch.no_grad():
    y_native = native(x)
    y_flash = flash(x)
  assert y_flash.shape == (512, 320) and y_flash.dtype == torch.bfloat16
  assert torch.isfinite(y_flash).all()

  reference = x.float() @ linear.weight.detach().float().t() + linear.bias.detach().float()
  # Both arms quantize the same checkpoint; the gap is the E4M3 re-encode of the weight scale
  # plus FlashInfer's per-tensor activation global scale, not an equivalence claim.
  assert _relative_l2(y_flash, y_native) < 0.15
  assert _relative_l2(y_flash, reference) < _relative_l2(y_native, reference) * 1.25


@pytest.mark.parametrize("m", [512, 300, 2])
def test_flashinfer_layer_handles_leading_dims_and_output_buffers(m: int) -> None:
  _require_sm120()
  _, x, _, flash = _make_pair(m, 320, 320)
  rows = 2 if m % 2 == 0 else 1
  x3 = x.reshape(rows, m // rows, -1)
  with torch.no_grad():
    expected = flash(x)
    actual = flash(x3)
    assert actual.shape == (rows, m // rows, 320)
    assert _relative_l2(actual.reshape(m, 320), expected) < 1e-3

    out = torch.empty_like(actual)
    returned = flash(x3, out)
    assert returned.data_ptr() == out.data_ptr()
    assert _relative_l2(returned.reshape(m, 320), expected) < 1e-6


def test_flashinfer_runtime_rejects_the_split_quantize_api() -> None:
  _, x, _, flash = _make_pair(256, 256, 256)
  with pytest.raises(RuntimeError, match="flashinfer"):
    flash.quantize(x)
  with pytest.raises(RuntimeError, match="flashinfer"):
    flash.forward_quant(None, None, None)


def test_flashinfer_cutlass_runtime_matches_the_cute_chain() -> None:
  _require_sm120()
  if torch.cuda.get_device_capability(0) != (12, 0):
    pytest.skip("FlashInfer's cutlass-sm120 chain requires a 12.0 device")
  _, x, _, cute = _make_pair(512, 320, 320)
  _, _, _, cutlass = _make_pair(512, 320, 320, runtime=CUTLASS_SM120_RUNTIME)
  with torch.no_grad():
    expected = cute(x).float()
    actual = cutlass(x).float()
  assert torch.isfinite(actual).all()
  # Both runtimes share the derived pack and the alpha/l1 contract; only FlashInfer's chain
  # differs, so this is a parity check between two FlashInfer routes.
  assert _relative_l2(actual, expected) < 5e-3
