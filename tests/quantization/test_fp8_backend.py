"""Tests for cache-dit's native online FP8 backend and backend routing."""

import pytest
import torch
from torch import nn

from cache_dit.quantization import QuantizeConfig
from cache_dit.quantization import load
from cache_dit.kernels import cuda_ext_is_available
from cache_dit.quantization import quantize
from cache_dit.quantization.backend import QuantizeBackend

requires_native_fp8 = pytest.mark.skipif(
  not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0)
  or not cuda_ext_is_available(),
  reason="cache-dit native FP8 requires sm100 or newer and the compiled extension",
)


class _Block(nn.Module):
  """Tiny repeated block used by the selection tests."""

  def __init__(self, width: int = 128) -> None:
    super().__init__()
    self.proj = nn.Linear(width, width, bias=True)
    self.bad_shape = nn.Linear(48, 24)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Run the block.

    :param x: Input activation.
    :returns: Output activation.
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


@requires_native_fp8
@pytest.mark.parametrize("mode", ["per_row", "per_tensor", "per_block"])
def test_native_fp8_quantizes_only_supported_repeated_blocks(mode: str) -> None:
  torch.manual_seed(0)
  model = _Transformer().cuda().to(torch.bfloat16).eval()
  reference = _Transformer().cuda().to(torch.bfloat16).eval()
  reference.load_state_dict(_plain_state(model))

  out = quantize(
    model,
    QuantizeConfig(
      backend=QuantizeBackend.CACHE_DIT,
      quant_type=f"float8_{mode}",
      exclude_layers=["modulation"],
    ),
  )
  assert all(type(block.proj).__name__ == "QW8A8Linear" for block in out.blocks)
  assert isinstance(out.modulation, nn.Linear)
  if mode == "per_block":
    assert all(isinstance(block.bad_shape, nn.Linear) for block in out.blocks)
    assert len(out._fp8_skipped_layers) == 2
  else:
    assert all(type(block.bad_shape).__name__ == "QW8A8Linear" for block in out.blocks)
  assert out._quantize_type == f"float8_{mode}"
  assert len(out._fp8_quantized_layers) == (2 if mode == "per_block" else 4)
  assert out._quantize_config.backend == QuantizeBackend.CACHE_DIT


@requires_native_fp8
def test_native_fp8_forward_tracks_bf16_reference() -> None:
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
        quant_type="float8_per_row",
        exclude_layers=["modulation"],
      ),
    )(x)
  diff = (out.float() - expected.float()).norm() / expected.float().norm()
  assert float(diff.detach()) < 0.08


@requires_native_fp8
def test_fp8_weight_dequantize_matches_quantized_linear() -> None:
  torch.manual_seed(2)
  linear = nn.Linear(128, 128).cuda().to(torch.bfloat16)
  from cache_dit.quantization.fp8.linear import QW8A8Linear

  packed = QW8A8Linear.from_linear(linear, "per_row")
  dense = packed.dequantize_weight()
  assert dense.shape == linear.weight.shape
  assert dense.dtype == torch.bfloat16
  x = torch.randn(3, 128, dtype=torch.bfloat16, device="cuda")
  from cache_dit.kernels import fp8_quantize

  qx, sx = fp8_quantize(x, "per_row")
  expected = torch.nn.functional.linear(qx.float() * sx[:, None], dense.float(),
                                        packed.bias.float())
  diff = (packed(x) - expected).float().norm() / expected.float().norm()
  assert float(diff.detach()) < 3e-3


@requires_native_fp8
def test_backend_routing_and_validation() -> None:
  auto_svdq = QuantizeConfig(quant_type="svdq_int4_r32_dq")
  assert auto_svdq.backend == QuantizeBackend.CACHE_DIT
  auto_fp8 = QuantizeConfig(quant_type="float8_per_row")
  assert auto_fp8.backend == QuantizeBackend.TORCHAO
  assert auto_fp8.strify() == "float8_per_row"
  explicit = QuantizeConfig(backend="cache_dit", quant_type="float8_per_row")
  assert explicit.strify() == "float8_per_row_cache_dit"
  mixed = QuantizeConfig(
    quant_type="float8_per_row",
    components_to_quantize={
      "transformer": {
        "quant_type": "float8_per_row",
        "backend": "cache_dit",
      },
      "text_encoder": {
        "quant_type": "float8_per_row",
        "backend": "torchao",
      },
    },
  )
  assert "<transformer:float8_per_row_cache_dit>" in mixed.strify()
  assert "<text_encoder:float8_per_row>" in mixed.strify()
  assert QuantizeConfig(quant_type="float8_per_row",
                        backend=None).backend == QuantizeBackend.TORCHAO

  with pytest.raises(ValueError, match="backend='cache_dit'"):
    QuantizeConfig(backend="torchao", quant_type="svdq_int4_r32_dq")
  with pytest.raises(ValueError, match="three FP8 types"):
    QuantizeConfig(backend="cache_dit", quant_type="int8_per_row")
  with pytest.raises(ValueError, match="online-only"):
    QuantizeConfig(backend="cache_dit",
                   quant_type="float8_per_row",
                   calibrate_fn=lambda **_: None,
                   serialize_to=".tmp/fp8_backend/test-checkpoint")


@requires_native_fp8
def test_native_fp8_rejects_a_selection_with_no_supported_linears() -> None:
  model = nn.Linear(48, 24).cuda().to(torch.bfloat16)
  with pytest.raises(ValueError, match="did not quantize any linear"):
    quantize(
      model,
      QuantizeConfig(
        backend="cache_dit",
        quant_type="float8_per_block",
        regional_quantize=False,
      ),
    )


@requires_native_fp8
def test_native_fp8_load_is_rejected() -> None:
  module = nn.Linear(128, 128).cuda().to(torch.bfloat16)
  config = QuantizeConfig(backend="cache_dit", quant_type="float8_per_row")
  with pytest.raises(ValueError, match="online-only"):
    load(module, config)
