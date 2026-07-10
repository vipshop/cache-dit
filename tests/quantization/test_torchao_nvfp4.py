import importlib.util

import pytest
import torch

from cache_dit._utils.utils import get_args, maybe_postprocess_args
from cache_dit.quantization import QuantizeConfig, quantize
from cache_dit.quantization.torchao.quantize_ao import _get_torchao_config

torchao = pytest.importorskip("torchao")
from torchao.quantization.quantize_.common import KernelPreference

nvfp4_workflow = pytest.importorskip("torchao.prototype.mx_formats.inference_workflow")
NVFP4Tensor = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor").NVFP4Tensor


def test_nvfp4_config_and_cli() -> None:
  float8_config = _get_torchao_config("float8_per_row")
  assert float8_config.kernel_preference is KernelPreference.TORCH

  config = QuantizeConfig(quant_type="nvfp4")
  assert config.strify() == "nvfp4"
  assert config.backend.value == "TORCHAO"

  torchao_config = _get_torchao_config("nvfp4")
  assert isinstance(torchao_config, nvfp4_workflow.NVFP4DynamicActivationNVFP4WeightConfig)
  assert torchao_config.use_triton_kernel is True
  assert torchao_config.use_dynamic_per_tensor_scale is True

  weight_only_config = QuantizeConfig(quant_type="nvfp4_weight_only")
  assert weight_only_config.strify() == "nvfp4_weight_only"
  weight_only_torchao_config = _get_torchao_config("nvfp4_weight_only")
  assert isinstance(weight_only_torchao_config, nvfp4_workflow.NVFP4WeightOnlyConfig)
  assert weight_only_torchao_config.use_dynamic_per_tensor_scale is True

  parser = get_args(parse=False)
  args = maybe_postprocess_args(parser.parse_args(["--nvfp4"]))
  assert args.quantize is True
  assert args.quantize_type == "nvfp4"

  args = maybe_postprocess_args(parser.parse_args(["--quantize-type", "nvfp4"]))
  assert args.quantize is True
  assert args.quantize_type == "nvfp4"

  args = maybe_postprocess_args(parser.parse_args(["--nvfp4-weight-only"]))
  assert args.quantize is True
  assert args.quantize_type == "nvfp4_weight_only"

  with pytest.raises(ValueError, match="mutually exclusive"):
    maybe_postprocess_args(parser.parse_args(["--nvfp4", "--svdq-nvfp4-r128-dq"]))


def _require_nvfp4_device() -> None:
  if not torch.cuda.is_available():
    pytest.skip("TorchAO NVFP4 tests require CUDA.")
  if torch.cuda.get_device_capability() < (10, 0):
    pytest.skip("TorchAO dynamic NVFP4 requires SM100 or newer.")
  if importlib.util.find_spec("mslk") is None:
    pytest.skip("TorchAO dynamic NVFP4 tests require MSLK.")


def test_nvfp4_quantizes_supported_linears_and_skips_unsupported_linears() -> None:
  _require_nvfp4_device()

  class ToyModel(torch.nn.Module):

    def __init__(self) -> None:
      super().__init__()
      self.large = torch.nn.Linear(2048, 2048, bias=False, dtype=torch.bfloat16)
      self.small = torch.nn.Linear(1024, 64, bias=False, dtype=torch.bfloat16)
      self.unaligned = torch.nn.Linear(1025, 2048, bias=False, dtype=torch.bfloat16)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
      return self.large(input)

  model = ToyModel().cuda()
  quantize(
    model,
    QuantizeConfig(quant_type="nvfp4", regional_quantize=False),
  )

  assert isinstance(model.large.weight, NVFP4Tensor)
  assert not isinstance(model.small.weight, NVFP4Tensor)
  assert not isinstance(model.unaligned.weight, NVFP4Tensor)
  assert model.large.weight.act_quant_kwargs.use_triton_kernel is True
  assert model.large.weight.act_quant_kwargs.use_dynamic_per_tensor_scale is True

  output = model(torch.randn(128, 2048, device="cuda", dtype=torch.bfloat16))
  assert output.shape == (128, 2048)
  assert torch.isfinite(output).all()


def test_nvfp4_compiles_and_runs() -> None:
  _require_nvfp4_device()

  model = torch.nn.Linear(2048, 2048, bias=False, dtype=torch.bfloat16, device="cuda")
  quantize(
    model,
    QuantizeConfig(quant_type="nvfp4", regional_quantize=False),
  )
  compiled_model = torch.compile(model, fullgraph=True)
  output = compiled_model(torch.randn(128, 2048, device="cuda", dtype=torch.bfloat16))
  assert output.shape == (128, 2048)
  assert torch.isfinite(output).all()


def test_nvfp4_weight_only_quantizes_and_compiles() -> None:
  _require_nvfp4_device()

  class ToyModel(torch.nn.Module):

    def __init__(self) -> None:
      super().__init__()
      self.large = torch.nn.Linear(2048, 2048, bias=False, dtype=torch.bfloat16)
      self.unaligned = torch.nn.Linear(1025, 2048, bias=False, dtype=torch.bfloat16)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
      return self.large(input)

  model = ToyModel().cuda()
  quantize(
    model,
    QuantizeConfig(quant_type="nvfp4_weight_only", regional_quantize=False),
  )

  assert isinstance(model.large.weight, NVFP4Tensor)
  assert model.large.weight.act_quant_kwargs is None
  assert not isinstance(model.unaligned.weight, NVFP4Tensor)

  compiled_model = torch.compile(model, fullgraph=True)
  output = compiled_model(torch.randn(128, 2048, device="cuda", dtype=torch.bfloat16))
  assert output.shape == (128, 2048)
  assert torch.isfinite(output).all()
