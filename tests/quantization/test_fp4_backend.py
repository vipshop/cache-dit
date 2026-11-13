"""Tests for cache-dit's native online NVFP4 backend and routing."""

import pytest
import torch
from torch import nn

from cache_dit.kernels import cuda_ext_is_available
from cache_dit.kernels import fp4_quantize
from cache_dit.quantization import QuantizeConfig
from cache_dit.quantization import load
from cache_dit.quantization import quantize
from cache_dit.quantization.backend import QuantizeBackend
from cache_dit.quantization.fp4.linear import NVFP4Linear

requires_native_fp4 = pytest.mark.skipif(
  not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12
  or not cuda_ext_is_available(),
  reason="cache-dit native FP4 requires sm120 and the compiled extension",
)


class _Block(nn.Module):
  """Tiny repeated block used by selection tests."""

  def __init__(self) -> None:
    super().__init__()
    self.proj = nn.Linear(128, 128, bias=True)
    self.bad_shape = nn.Linear(48, 24)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Run the supported projection.

    :param x: Input activation.
    :returns: Projected activation.
    """

    return self.proj(x)


class _Transformer(nn.Module):
  """Tiny transformer-shaped module with an excluded modulation branch."""

  def __init__(self) -> None:
    super().__init__()
    self._repeated_blocks = ["_Block"]
    self.blocks = nn.ModuleList([_Block(), _Block()])
    self.modulation = nn.Linear(128, 128)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Run every branch.

    :param x: Input activation.
    :returns: Output activation.
    """

    for block in self.blocks:
      x = block(x)
    return self.modulation(x)


def _plain_state(module: nn.Module) -> dict:
  """Clone float weights before quantization.

  :param module: Float module.
  :returns: State dict copy.
  """

  return {key: value.detach().clone() for key, value in module.state_dict().items()}


@requires_native_fp4
def test_native_fp4_quantizes_only_supported_repeated_blocks() -> None:
  torch.manual_seed(0)
  model = _Transformer().cuda().to(torch.bfloat16).eval()
  out = quantize(
    model,
    QuantizeConfig(
      backend=QuantizeBackend.CACHE_DIT,
      quant_type="nvfp4",
      exclude_layers=["modulation"],
    ),
  )
  assert all(type(block.proj) is NVFP4Linear for block in out.blocks)
  assert all(isinstance(block.bad_shape, nn.Linear) for block in out.blocks)
  assert isinstance(out.modulation, nn.Linear)
  assert out._quantize_type == "nvfp4"
  assert len(out._fp4_quantized_layers) == 2
  assert len(out._fp4_skipped_layers) == 2
  assert out._quantize_config.backend == QuantizeBackend.CACHE_DIT


@requires_native_fp4
def test_native_fp4_forward_tracks_bf16_reference() -> None:
  torch.manual_seed(1)
  model = _Transformer().cuda().to(torch.bfloat16).eval()
  reference = _Transformer().cuda().to(torch.bfloat16).eval()
  reference.load_state_dict(_plain_state(model))
  x = torch.randn(2, 128, dtype=torch.bfloat16, device="cuda")
  with torch.no_grad():
    expected = reference(x)
    out = quantize(
      model,
      QuantizeConfig(
        backend="cache_dit",
        quant_type="nvfp4",
        exclude_layers=["modulation"],
      ),
    )(x)
  diff = (out.float() - expected.float()).norm() / expected.float().norm()
  assert float(diff.detach()) < 0.35


@requires_native_fp4
def test_fp4_weight_dequantize_matches_quantized_linear() -> None:
  torch.manual_seed(2)
  holder = nn.Module()
  holder.proj = nn.Linear(128, 128).cuda().to(torch.bfloat16)
  quantize(holder, QuantizeConfig(backend="cache_dit", quant_type="nvfp4", regional_quantize=False))
  packed = holder.proj
  dense = packed.dequantize_weight()
  assert dense.shape == (128, 128)
  assert dense.dtype == torch.bfloat16
  x = torch.randn(3, 128, dtype=torch.bfloat16, device="cuda")
  qx, sx, gx = fp4_quantize(x)
  from cache_dit.quantization.fp4.linear import _dequantize_fp4

  activation = _dequantize_fp4(qx, sx, gx)
  expected = torch.nn.functional.linear(activation, dense.float(), packed.bias.float())
  diff = (packed(x) - expected).float().norm() / expected.float().norm()
  assert float(diff.detach()) < 5e-3


@requires_native_fp4
def test_native_fp4_compile_matches_eager() -> None:
  torch.manual_seed(3)
  holder = nn.Module()
  holder.proj = nn.Linear(128, 128).cuda().to(torch.bfloat16)
  quantize(holder, QuantizeConfig(backend="cache_dit", quant_type="nvfp4", regional_quantize=False))
  x = torch.randn(2, 4, 128, dtype=torch.bfloat16, device="cuda")
  expected = holder.proj(x)
  actual = torch.compile(holder.proj, fullgraph=True)(x)
  diff = (actual.float() - expected.float()).norm() / expected.float().norm()
  assert float(diff.detach()) < 5e-3


def test_backend_routing_and_validation() -> None:
  auto = QuantizeConfig(quant_type="nvfp4")
  assert auto.backend == QuantizeBackend.TORCHAO
  assert auto.strify() == "nvfp4"
  explicit = QuantizeConfig(backend="cache_dit", quant_type="nvfp4")
  assert explicit.strify() == "nvfp4_cache_dit"
  mixed = QuantizeConfig(
    quant_type="nvfp4",
    components_to_quantize={
      "transformer": {
        "quant_type": "nvfp4",
        "backend": "cache_dit",
      },
      "text_encoder": {
        "quant_type": "nvfp4",
        "backend": "torchao",
      },
    },
  )
  assert "<transformer:nvfp4_cache_dit>" in mixed.strify()
  assert "<text_encoder:nvfp4>" in mixed.strify()

  with pytest.raises(ValueError, match="three FP8 types and NVFP4"):
    QuantizeConfig(backend="cache_dit", quant_type="int8_per_row")
  with pytest.raises(ValueError, match="online-only"):
    QuantizeConfig(backend="cache_dit",
                   quant_type="nvfp4",
                   calibrate_fn=lambda **_: None,
                   serialize_to=".tmp/native_nvfp4/test-checkpoint")
  with pytest.raises(ValueError, match="three FP8 types and NVFP4"):
    QuantizeConfig(backend="cache_dit", quant_type="nvfp4_weight_only")


@requires_native_fp4
def test_native_fp4_rejects_a_selection_with_no_supported_linears() -> None:
  holder = nn.Module()
  holder.bad = nn.Linear(48, 24).cuda().to(torch.bfloat16)
  with pytest.raises(ValueError, match="did not quantize any linear"):
    quantize(
      holder,
      QuantizeConfig(
        backend="cache_dit",
        quant_type="nvfp4",
        regional_quantize=False,
      ),
    )


@requires_native_fp4
def test_native_fp4_load_is_rejected() -> None:
  module = nn.Linear(128, 128).cuda().to(torch.bfloat16)
  config = QuantizeConfig(backend="cache_dit", quant_type="nvfp4")
  with pytest.raises(ValueError, match="online-only"):
    load(module, config)
