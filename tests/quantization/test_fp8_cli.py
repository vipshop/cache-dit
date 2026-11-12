"""CLI coverage for the cache-dit/TorchAO quantization backend selector."""

from types import SimpleNamespace

import pytest
from torch import nn

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
  args = _parser().parse_args([])
  assert args.quantize_backend is None
  assert cli_utils._resolve_cli_quantize_backend(args) == QuantizeBackend.AUTO


@pytest.mark.parametrize(
  "value,expected",
  [
    ("torchao", QuantizeBackend.TORCHAO),
    ("cache_dit", QuantizeBackend.CACHE_DIT),
  ],
)
def test_quantize_backend_cli_flag(value: str, expected: QuantizeBackend) -> None:
  args = cli_utils.maybe_postprocess_args(_parser().parse_args([
    "--quantize-type",
    "float8_per_row",
    "--quantize-backend",
    value,
  ]))
  assert args.quantize is True
  assert args.quantize_type == "float8_per_row"
  assert cli_utils._resolve_cli_quantize_backend(args) == expected


def test_quantize_backend_alone_defaults_the_type() -> None:
  args = cli_utils.maybe_postprocess_args(_parser().parse_args(["--quantize-backend", "cache_dit"]))
  assert args.quantize is True
  assert args.quantize_type == "float8_per_row"


def test_invalid_quantize_backend_is_rejected() -> None:
  with pytest.raises(SystemExit):
    _parser().parse_args(["--quantize-backend", "nope"])


@requires_native_fp8
def test_transformer_cli_passes_backend_to_config(monkeypatch) -> None:
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

  monkeypatch.setattr(cli_utils, "quantize", fake_quantize)
  args = cli_utils.maybe_postprocess_args(_parser().parse_args([
    "--quantize-type",
    "float8_per_row",
    "--quantize-backend",
    "cache_dit",
    "--steps",
    "1",
  ]))
  transformer = _Toy()
  pipe = SimpleNamespace(transformer=transformer)
  cli_utils.maybe_quantize_transformer(args, pipe)
  assert len(captured) == 1
  assert captured[0].backend == QuantizeBackend.CACHE_DIT
  assert captured[0].quant_type == "float8_per_row"


@pytest.mark.skipif(not cuda_ext_is_available(), reason="TorchAO backend availability")
def test_torchao_backend_is_passed_explicitly(monkeypatch) -> None:
  captured = []
  monkeypatch.setattr(cli_utils, "quantize",
                      lambda module, quantize_config: captured.append(quantize_config) or module)

  class _Toy(nn.Module):
    """Minimal transformer accepted by the CLI helper."""

    def __init__(self) -> None:
      super().__init__()
      self._repeated_blocks = ["_Block"]
      self.proj = nn.Linear(2, 2)

  args = cli_utils.maybe_postprocess_args(_parser().parse_args([
    "--quantize-type",
    "float8_per_row",
    "--quantize-backend",
    "torchao",
  ]))
  cli_utils.maybe_quantize_transformer(args, SimpleNamespace(transformer=_Toy()))
  assert captured[0].backend == QuantizeBackend.TORCHAO
