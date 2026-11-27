"""CLI coverage for the cache-dit/TorchAO quantization backend selector."""

from types import SimpleNamespace

import pytest
from torch import nn

from cache_dit.cli import quantize as quantize_mod
from cache_dit.cli import utils as cli_utils
from cache_dit.kernels import cuda_ext_is_available
from cache_dit.quantization.backend import QuantizeBackend

requires_native_fp8 = pytest.mark.skipif(
  not cuda_ext_is_available(),
  reason="explicit cache-dit QuantizeConfig requires the compiled extension",
)


def _parser():
  """Return the CLI parser without parsing the process arguments.

  :returns: Configured argparse parser.
  """

  return cli_utils.get_args(parse=False)


def test_quantize_backend_defaults_to_auto(monkeypatch) -> None:
  monkeypatch.setattr("sys.argv", ["cache-dit-generate"])
  args = cli_utils.maybe_postprocess_args(_parser().parse_args([]))
  assert args.quantize_fields == {}
  assert cli_utils._resolve_cli_quantize_backend(args) == QuantizeBackend.AUTO


@pytest.mark.parametrize(
  "value,expected",
  [
    ("torchao", QuantizeBackend.TORCHAO),
    ("cache_dit", QuantizeBackend.CACHE_DIT),
  ],
)
def test_quantize_backend_param(value: str, expected: QuantizeBackend) -> None:
  args = cli_utils.maybe_postprocess_args(_parser().parse_args(
    ["--quantize-params", f"backend={value}"]))
  assert args.quantize is True
  assert args.quantize_type == "float8_per_row"
  assert cli_utils._resolve_cli_quantize_backend(args) == expected


def test_invalid_quantize_backend_is_rejected() -> None:
  with pytest.raises(ValueError, match="Invalid --quantize-params 'backend=nope'"):
    cli_utils.maybe_postprocess_args(_parser().parse_args(["--quantize-params", "backend=nope"]))


@requires_native_fp8
@pytest.mark.parametrize("quant_type", ["float8_per_row", "nvfp4"])
def test_transformer_cli_passes_backend_to_config(monkeypatch, quant_type: str) -> None:
  captured = []

  def fake_quantize(module, quantize_config):
    captured.append(quantize_config)
    return module

  class _Toy(nn.Module):
    """Minimal transformer accepted by the CLI helper."""

    def __init__(self) -> None:
      super().__init__()
      self._repeated_blocks = ["_Block"]
      self.proj = nn.Linear(2, 2)

  monkeypatch.setattr(quantize_mod, "quantize", fake_quantize)
  args = cli_utils.maybe_postprocess_args(_parser().parse_args([
    "--quantize-params",
    f"quant_type={quant_type},backend=cache_dit",
    "--steps",
    "1",
  ]))
  transformer = _Toy()
  pipe = SimpleNamespace(transformer=transformer)
  cli_utils.maybe_quantize_transformer(args, pipe)
  assert len(captured) == 1
  assert captured[0].backend == QuantizeBackend.CACHE_DIT
  assert captured[0].quant_type == quant_type


@pytest.mark.skipif(not cuda_ext_is_available(), reason="TorchAO backend availability")
def test_torchao_backend_is_passed_explicitly(monkeypatch) -> None:
  captured = []
  monkeypatch.setattr(quantize_mod, "quantize",
                      lambda module, quantize_config: captured.append(quantize_config) or module)

  class _Toy(nn.Module):
    """Minimal transformer accepted by the CLI helper."""

    def __init__(self) -> None:
      super().__init__()
      self._repeated_blocks = ["_Block"]
      self.proj = nn.Linear(2, 2)

  args = cli_utils.maybe_postprocess_args(_parser().parse_args([
    "--quantize-params",
    "backend=torchao",
  ]))
  cli_utils.maybe_quantize_transformer(args, SimpleNamespace(transformer=_Toy()))
  assert captured[0].backend == QuantizeBackend.TORCHAO


@requires_native_fp8
def test_fp8_fused_mlp_cli_is_passed_to_transformer(monkeypatch) -> None:
  captured = []
  monkeypatch.setattr(quantize_mod, "quantize",
                      lambda module, quantize_config: captured.append(quantize_config) or module)

  class _Toy(nn.Module):

    def __init__(self) -> None:
      super().__init__()
      self._repeated_blocks = ["_Block"]
      self.proj = nn.Linear(2, 2)

  args = cli_utils.maybe_postprocess_args(_parser().parse_args([
    "--quantize-params",
    "backend=cache_dit,fp8_fused_mlp=true",
    "--steps",
    "1",
  ]))
  cli_utils.maybe_quantize_transformer(args, SimpleNamespace(transformer=_Toy()))
  assert len(captured) == 1
  assert captured[0].backend == QuantizeBackend.CACHE_DIT
  assert captured[0].quant_type == "float8_per_row"
  assert captured[0].fp8_fused_mlp is True
