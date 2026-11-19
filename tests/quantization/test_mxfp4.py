import pytest
import torch

from cache_dit.cli.utils import get_args, maybe_postprocess_args
from cache_dit.quantization import MXFP4Linear, QuantizeConfig, quantize
from cache_dit.quantization.mxfp4 import mxfp4_is_available
from cache_dit.quantization.mxfp4.tensor import dequantize_mxfp4, quantize_mxfp4_weight


def test_mxfp4_config_and_cli() -> None:
  config = QuantizeConfig(quant_type="mxfp4")
  assert config.strify() == "mxfp4_cache_dit"
  assert config.backend.value == "CACHE_DIT"

  weight_only_config = QuantizeConfig(quant_type="mxfp4_weight_only")
  assert weight_only_config.backend.value == "CACHE_DIT"

  with pytest.raises(ValueError, match="mxfp4_block_size"):
    QuantizeConfig(quant_type="mxfp4", mxfp4_block_size=64)

  parser = get_args(parse=False)
  args = maybe_postprocess_args(parser.parse_args(["--mxfp4"]))
  assert args.quantize is True
  assert args.quantize_type == "mxfp4"

  args = maybe_postprocess_args(parser.parse_args(["--mxfp4-weight-only"]))
  assert args.quantize is True
  assert args.quantize_type == "mxfp4_weight_only"

  args = maybe_postprocess_args(parser.parse_args(["--quantize-type", "mxfp4"]))
  assert args.quantize is True
  assert args.quantize_type == "mxfp4"

  with pytest.raises(ValueError, match="mutually exclusive"):
    maybe_postprocess_args(parser.parse_args(["--mxfp4", "--nvfp4"]))


def _require_mxfp4_device() -> None:
  if not mxfp4_is_available():
    pytest.skip("MXFP4 requires torch.float4_e2m1fn_x2 + torch.float8_e8m0fnu.")
  if not torch.cuda.is_available():
    pytest.skip("MXFP4 tests require CUDA.")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_mxfp4_quant_dequant_roundtrip(dtype: torch.dtype) -> None:
  _require_mxfp4_device()
  torch.manual_seed(0)
  device = torch.device("cuda")
  weight = torch.randn(128, 256, device=device, dtype=dtype)

  packed, scales = quantize_mxfp4_weight(weight)
  assert packed.shape == (128, 128)  # 2 fp4 codes per byte along K
  assert scales.shape == (128, 8)  # one E8M0 scale per group of 32
  assert packed.dtype == torch.float4_e2m1fn_x2
  assert scales.dtype == torch.float8_e8m0fnu

  dequant = dequantize_mxfp4(packed, scales, 128, 256, dtype=torch.float32)
  rel_err = ((dequant - weight.float()).abs().mean() / weight.float().abs().mean()).item()
  # MXFP4 group quantization noise for N(0,1) weights is ~10-12% MAE-relative;
  # the bound mainly guards against packing/scale-order regressions.
  assert rel_err < 0.15, f"unexpected MXFP4 roundtrip error: {rel_err}"
  # Storage: 4-bit payload + 1/32-byte E8M0 scale -> ~0.28 bytes per element,
  # i.e. ~3.6x smaller than bf16 and ~7x smaller than fp32 weights.
  stored_bytes = packed.numel() + scales.numel()
  assert stored_bytes < weight.numel() * weight.element_size() / 2.5


def test_mxfp4_quantize_linear_via_dispatch() -> None:
  _require_mxfp4_device()
  torch.manual_seed(0)

  class ToyBlock(torch.nn.Module):

    def __init__(self) -> None:
      super().__init__()
      self.proj = torch.nn.Linear(256, 256, dtype=torch.bfloat16)
      self.excluded = torch.nn.Linear(256, 64, dtype=torch.bfloat16)
      self.odd = torch.nn.Linear(48, 32, dtype=torch.bfloat16)  # not packable (48 % 64 != 0)

  class ToyModel(torch.nn.Module):

    def __init__(self) -> None:
      super().__init__()
      self.blocks = torch.nn.ModuleList([ToyBlock() for _ in range(2)])

    def forward(self, x):
      for block in self.blocks:
        x = block.proj(x)
      return x

  model = ToyModel().cuda()
  x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
  with torch.no_grad():
    ref = model(x)

  quantized = quantize(model, QuantizeConfig(quant_type="mxfp4", exclude_layers=["excluded"]))
  assert isinstance(quantized.blocks[0].proj, MXFP4Linear)
  assert not isinstance(quantized.blocks[0].excluded, MXFP4Linear)
  assert not isinstance(quantized.blocks[0].odd, MXFP4Linear)

  with torch.no_grad():
    out = quantized(x)
  rel_err = ((out.float() - ref.float()).abs().mean() / ref.float().abs().mean()).item()
  # Two chained MXFP4 linears compound per-layer ~12% noise; guard against
  # packing/scale-order bugs rather than quantization noise itself.
  assert rel_err < 0.30, f"unexpected MXFP4 end-to-end error: {rel_err}"
