"""CLI coverage for native FP4 fused MLP."""

from types import SimpleNamespace

import pytest
from torch import nn

from cache_dit.cli import utils as cli_utils
from cache_dit.kernels import cuda_ext_is_available
from cache_dit.quantization.backend import QuantizeBackend

requires_native_fp4 = pytest.mark.skipif(not cuda_ext_is_available(),
                                         reason="native FP4 QuantizeConfig requires the extension")


def test_fp4_fused_mlp_cli_defaults_off() -> None:
  args = cli_utils.maybe_postprocess_args(cli_utils.get_args(parse=False).parse_args([]))
  assert args.quantize_fields == {}


@requires_native_fp4
def test_fp4_fused_mlp_cli_is_passed_to_transformer(monkeypatch) -> None:
  captured = []
  monkeypatch.setattr(cli_utils, "quantize",
                      lambda module, quantize_config: captured.append(quantize_config) or module)

  class _Toy(nn.Module):

    def __init__(self) -> None:
      super().__init__()
      self._repeated_blocks = ["_Block"]
      self.proj = nn.Linear(2, 2)

  args = cli_utils.maybe_postprocess_args(
    cli_utils.get_args(parse=False).parse_args([
      "--quantize",
      "--quantize-params",
      "quant_type=nvfp4,backend=cache_dit,fp4_fused_mlp=true",
      "--steps",
      "1",
    ]))
  cli_utils.maybe_quantize_transformer(args, SimpleNamespace(transformer=_Toy()))
  assert len(captured) == 1
  assert captured[0].backend == QuantizeBackend.CACHE_DIT
  assert captured[0].quant_type == "nvfp4"
  assert captured[0].fp4_fused_mlp is True
