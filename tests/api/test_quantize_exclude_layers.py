"""Regression tests for the quantization exclusion table.

`--quantize-exclude-layers` used to replace `QuantizeConfig.exclude_layers`, which let a CLI run
drop the default `embed` / `modulation` guards. It also forwarded `None` verbatim whenever the flag
was absent, and every consumer iterates the list, so a plain `--quantize` run crashed in torchao.
"""
from types import SimpleNamespace

from cache_dit.quantization.config import DEFAULT_EXCLUDE_LAYERS, QuantizeConfig


def test_explicit_none_falls_back_to_the_default_table():
  config = QuantizeConfig(quant_type="float8_per_row", exclude_layers=None)
  assert config.exclude_layers == list(DEFAULT_EXCLUDE_LAYERS)


def test_default_table_protects_embeddings():
  config = QuantizeConfig(quant_type="float8_per_row")
  assert "embed" in config.exclude_layers


def test_cli_flag_extends_the_default_table():
  from cache_dit._utils.utils import _resolve_cli_quantize_exclude_layers

  args = SimpleNamespace(quantize_exclude_layers=["adaln_proj", "norm_out"])
  merged = _resolve_cli_quantize_exclude_layers(args)
  assert "adaln_proj" in merged and "norm_out" in merged
  for name in DEFAULT_EXCLUDE_LAYERS:
    assert name in merged, f"'{name}' must survive the CLI override"


def test_absent_flag_reports_no_override():
  from cache_dit._utils.utils import _resolve_cli_quantize_exclude_layers

  assert _resolve_cli_quantize_exclude_layers(SimpleNamespace(quantize_exclude_layers=None)) is None
