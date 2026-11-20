import pytest
import torch

from cache_dit.cli.utils import get_args, maybe_postprocess_args
from cache_dit.quantization import MXFP4Linear, QuantizeConfig, quantize
from cache_dit.quantization.mxfp4 import mxfp4_is_available, resolve_mxfp4_quantize_device
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


def _require_mxfp4_dtypes() -> None:
  if not mxfp4_is_available():
    pytest.skip("MXFP4 requires torch.float4_e2m1fn_x2 + torch.float8_e8m0fnu.")


def test_mxfp4_quantize_device_resolution() -> None:
  weight_cpu = torch.randn(64, 64, dtype=torch.bfloat16)

  assert resolve_mxfp4_quantize_device(weight_cpu, "cpu") == torch.device("cpu")

  auto = resolve_mxfp4_quantize_device(weight_cpu, "auto")
  assert auto == resolve_mxfp4_quantize_device(weight_cpu, None)
  expected = (torch.device("cuda", torch.cuda.current_device())
              if torch.cuda.is_available() else torch.device("cpu"))
  assert auto == expected

  with pytest.raises(ValueError, match="quantize_device"):
    resolve_mxfp4_quantize_device(weight_cpu, "xpu")

  if torch.cuda.is_available():
    # A weight already on the accelerator stays there under "auto".
    weight_cuda = weight_cpu.cuda()
    assert resolve_mxfp4_quantize_device(weight_cuda, "auto") == weight_cuda.device
  else:
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
      resolve_mxfp4_quantize_device(weight_cpu, "cuda")


def test_mxfp4_config_rejects_bad_quantize_device() -> None:
  with pytest.raises(ValueError, match="mxfp4_quantize_device"):
    QuantizeConfig(quant_type="mxfp4", mxfp4_quantize_device="xpu")
  # "auto"/"cpu"/"cuda" are all accepted; "cuda" is validated at packing time.
  QuantizeConfig(quant_type="mxfp4", mxfp4_quantize_device="cpu")
  QuantizeConfig(quant_type="mxfp4", mxfp4_quantize_device="auto")


def test_mxfp4_component_config_keeps_mxfp4_fields() -> None:
  config = QuantizeConfig(quant_type="mxfp4",
                          components_to_quantize={
                            "transformer": {
                              "mxfp4_quantize_device": "cpu",
                              "mxfp4_block_size": 32
                            }
                          })
  expanded = QuantizeConfig.expand_configs(config)
  assert len(expanded) == 1
  assert expanded[0].mxfp4_quantize_device == "cpu"
  assert expanded[0].mxfp4_block_size == 32
  assert expanded[0].components_to_quantize == ["transformer"]


@pytest.mark.parametrize("requested", ["auto", "cuda"])
def test_mxfp4_accelerator_pack_is_bit_identical(requested: str) -> None:
  """Packing on the accelerator must not change a single payload or scale bit."""
  _require_mxfp4_dtypes()
  if not torch.cuda.is_available():
    pytest.skip("cross-device packing parity needs an accelerator")
  torch.manual_seed(0)
  weight = torch.randn(128, 512, dtype=torch.bfloat16)

  device = resolve_mxfp4_quantize_device(weight, requested)
  assert device.type == "cuda"
  packed, scales = quantize_mxfp4_weight(weight, device=device)
  ref_packed, ref_scales = quantize_mxfp4_weight(weight, device=torch.device("cpu"))

  # Buffers always come back to the weight's device, not the packing device.
  assert packed.device == weight.device
  assert scales.device == weight.device
  assert torch.equal(packed.view(torch.uint8), ref_packed.view(torch.uint8))
  assert torch.equal(scales.view(torch.uint8), ref_scales.view(torch.uint8))


def test_mxfp4_quantize_device_reaches_packing(monkeypatch) -> None:
  """The config field must reach the packing math, not just be stored."""
  _require_mxfp4_dtypes()
  from cache_dit.quantization.mxfp4 import tensor as tensor_mod

  seen = []
  orig = tensor_mod.quantize_mxfp4_weight

  def spy(weight, block_size=tensor_mod.MXFP4_BLOCK_SIZE, device=None):
    seen.append(device)
    return orig(weight, block_size=block_size, device=device)

  monkeypatch.setattr(tensor_mod, "quantize_mxfp4_weight", spy)

  class ToyBlock(torch.nn.Module):

    def __init__(self) -> None:
      super().__init__()
      self.proj = torch.nn.Linear(256, 32, dtype=torch.bfloat16)

  quantize(ToyBlock(), QuantizeConfig(quant_type="mxfp4", mxfp4_quantize_device="cpu"))
  assert seen == [torch.device("cpu")]

  seen.clear()
  quantize(ToyBlock(), QuantizeConfig(quant_type="mxfp4"))
  expected = (torch.device("cuda", torch.cuda.current_device())
              if torch.cuda.is_available() else torch.device("cpu"))
  assert seen == [expected]


@pytest.mark.parametrize("requested", ["auto", "cpu", "cuda"])
def test_mxfp4_quantizes_cpu_module_without_moving_it(requested: str) -> None:
  """Quantization runs before the pipeline is moved to the GPU: the module must
  stay on CPU (only the packing math is offloaded) and still run afterwards."""
  _require_mxfp4_dtypes()
  if requested == "cuda" and not torch.cuda.is_available():
    pytest.skip("mxfp4_quantize_device='cuda' needs an accelerator")
  torch.manual_seed(0)

  class ToyBlock(torch.nn.Module):

    def __init__(self) -> None:
      super().__init__()
      self.proj = torch.nn.Linear(256, 32, dtype=torch.bfloat16)

  model = ToyBlock()
  config = QuantizeConfig(quant_type="mxfp4", mxfp4_quantize_device=requested)
  quantized = quantize(model, config)

  assert isinstance(quantized.proj, MXFP4Linear)
  assert quantized.proj.weight_packed.device.type == "cpu"
  assert quantized.proj.weight_scales.device.type == "cpu"

  x = torch.randn(2, 256, dtype=torch.bfloat16)
  with torch.no_grad():
    out = quantized.proj(x)
  assert out.shape == (2, 32)
  assert torch.isfinite(out).all()


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
