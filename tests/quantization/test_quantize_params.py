"""CLI coverage for `--quantize-params` and `--svdq-params`."""

import argparse
import dataclasses

import pytest

from cache_dit.cli import utils as cli_utils
from cache_dit.cli.params import QUANTIZE_PARAM_SPECS, SVDQ_PARAM_SPECS, parse_param_pairs
from cache_dit.quantization.backend import QuantizeBackend
from cache_dit.quantization import QuantizeConfig
from cache_dit.quantization.config import (
  _SVDQ_KWARGS_DEFAULTS,
  _parse_svdq_quant_type,
  ONLINE_QUANT_TYPES,
  SVDQ_DQ_QUANT_TYPES,
)

parser = cli_utils.get_args(parse=False)

SVDQ = ("--quantize", "--quantize-params", "quant_type=svdq_int4_r32_dq", "--svdq-params")


def resolve(argv):
  return cli_utils.maybe_postprocess_args(parser.parse_args(argv))


def test_param_specs_match_quantize_config() -> None:
  assert set(SVDQ_PARAM_SPECS) == set(_SVDQ_KWARGS_DEFAULTS)
  assert set(QUANTIZE_PARAM_SPECS) - {"text_encoder"} <= {
    field.name
    for field in dataclasses.fields(QuantizeConfig)
  }


def test_svdq_dq_quant_types_parse() -> None:
  for quant_type in SVDQ_DQ_QUANT_TYPES:
    assert _parse_svdq_quant_type(quant_type)[2] is True


@pytest.mark.parametrize("quant_type", [*ONLINE_QUANT_TYPES, *SVDQ_DQ_QUANT_TYPES])
def test_quantize_type_param_accepts_every_listed_type(quant_type) -> None:
  args = resolve(["--quantize", "--quantize-params", f"quant_type={quant_type.replace('_', '-')}"])
  assert args.quantize_type == quant_type


def test_parse_param_pairs_syntax() -> None:
  assert parse_param_pairs(None, "--f", "'a=1'") is None
  assert parse_param_pairs(" A = 1 , b=x,,", "--f", "'a=1'") == {"a": "1", "b": "x"}
  for bad in ("a", "=1", "a=1,b"):
    with pytest.raises(argparse.ArgumentTypeError, match="--f entry"):
      parse_param_pairs(bad, "--f", "'a=1'")


def test_quantize_is_a_bare_switch_with_a_default_type() -> None:
  args = resolve(["--quantize"])
  assert args.quantize is True
  assert args.quantize_type == "float8_per_row"
  assert args.quantize_fields == {}
  assert args.svdq_kwargs is None


def test_quantize_type_param() -> None:
  for text in ("svdq-nvfp4-r64-dq", "svdq_nvfp4_r64_dq"):
    args = resolve(["--quantize", "--quantize-params", f"quant_type={text}"])
    assert args.quantize_type == "svdq_nvfp4_r64_dq"
  with pytest.raises(ValueError, match="Invalid --quantize-params 'quant_type=nope'"):
    resolve(["--quantize", "--quantize-params", "quant_type=nope"])


@pytest.mark.parametrize(
  "text, attr, expected",
  [
    ("backend=cache_dit", "quantize_backend", "cache_dit"),
    ("regional_quantize=false", "disable_regional_quantize", True),
    ("per_tensor_fallback=off", "disable_per_tensor_fallback", True),
    ("exclude_layers=adaln_proj+norm_out", "quantize_exclude_layers", ["adaln_proj", "norm_out"]),
    ("verbose=true", "quantize_verbose", True),
    ("fp8_fused_mlp=1", "fp8_fused_mlp", True),
    ("mxfp4_quantize_device=cpu", "mxfp4_quantize_device", "cpu"),
  ],
)
def test_quantize_params_are_resolved(text, attr, expected) -> None:
  args = resolve(["--quantize", "--quantize-params", text])
  assert getattr(args, attr) == expected


def test_quantize_params_reject_unknown_keys_and_bad_values() -> None:
  with pytest.raises(ValueError, match="Unknown --quantize-params keys \\['nope'\\]"):
    resolve(["--quantize", "--quantize-params", "nope=1"])
  with pytest.raises(ValueError, match="Invalid --quantize-params 'regional_quantize=maybe'"):
    resolve(["--quantize", "--quantize-params", "regional_quantize=maybe"])
  with pytest.raises(ValueError, match="Invalid --quantize-params 'backend=nonsense'"):
    resolve(["--quantize", "--quantize-params", "backend=nonsense"])


def test_quantize_params_require_quantize_except_text_encoder() -> None:
  with pytest.raises(ValueError, match="require --quantize"):
    resolve(["--quantize-params", "quant_type=float8_per_row"])
  args = resolve(["--quantize-params", "text_encoder=bitsandbytes_4bit"])
  assert args.quantize is False
  assert args.quantize_text_encoder is True
  assert args.quantize_text_type == "bitsandbytes_4bit"


def test_text_encoder_param() -> None:
  args = resolve(["--quantize", "--quantize-params", "text_encoder=true"])
  assert args.quantize_text_type == "float8_per_row"
  args = resolve(["--quantize", "--quantize-params", "text_encoder=int8_per_row"])
  assert args.quantize_text_type == "int8_per_row"
  with pytest.raises(ValueError, match="'text_encoder=true' requires --quantize"):
    resolve(["--quantize-params", "text_encoder=true"])
  with pytest.raises(ValueError, match="Invalid --quantize-params 'text_encoder=nvfp4'"):
    resolve(["--quantize", "--quantize-params", "text_encoder=nvfp4"])


def test_svdq_params_require_an_svdq_type_and_known_keys() -> None:
  with pytest.raises(ValueError, match="--svdq-params requires an svdq_\\* quant_type"):
    resolve(["--quantize", "--svdq-params", "smooth_strategy=few_shot"])
  with pytest.raises(ValueError, match="Unknown --svdq-params keys \\['nope'\\]"):
    resolve([*SVDQ, "nope=1"])


def test_svdq_params_are_resolved() -> None:
  kwargs = resolve([
    *SVDQ,
    "runtime_kernel=auto,lora_layout=none,max_inflight_prefetch_bytes=8GiB,few_shot_steps=none,"
    "streaming=false,svd_refine_iters=2",
  ]).svdq_kwargs
  assert kwargs["runtime_kernel"] is None
  assert kwargs["lora_layout"] is None
  assert kwargs["max_inflight_prefetch_bytes"] == 8 * 1024 ** 3
  assert kwargs["few_shot_steps"] is None
  assert kwargs["streaming"] is False
  assert kwargs["svd_refine_iters"] == 2


def test_svdq_params_bad_values_fail_at_postprocess() -> None:
  with pytest.raises(ValueError, match="Invalid --svdq-params 'few_shot_steps=abc'"):
    resolve([*SVDQ, "few_shot_steps=abc"])
  with pytest.raises(ValueError, match="few_shot_relax_strategy"):
    resolve([*SVDQ, "few_shot_relax_strategy=nope"])


def test_svdq_layerwise_keys_inherit_and_override_layerwise_flags() -> None:
  kwargs = resolve([
    "--quantize", "--quantize-params", "quant_type=svdq_int4_r32_dq",
    "--layerwise-transfer-buckets", "4"
  ]).svdq_kwargs
  assert kwargs["transfer_buckets"] == 4
  kwargs = resolve([
    "--quantize", "--quantize-params", "quant_type=svdq_int4_r32_dq",
    "--layerwise-transfer-buckets", "4", "--svdq-params", "transfer_buckets=2"
  ]).svdq_kwargs
  assert kwargs["transfer_buckets"] == 2


def test_svdq_cli_derivations_and_explicit_keys_win() -> None:

  def kwargs(extra):
    return resolve([*SVDQ, extra, "--compile"]).svdq_kwargs

  assert kwargs("smooth_strategy=few_shot")["few_shot_auto_compile"] is True
  assert kwargs(
    "smooth_strategy=few_shot,few_shot_auto_compile=false")["few_shot_auto_compile"] is False
  few_shot_offload = kwargs("smooth_strategy=few_shot,layerwise_offload=true")
  assert few_shot_offload["offload_quantized_layers_to_cpu"] is True
  assert few_shot_offload["defer_move_to_execution_device"] is True
  assert kwargs(
    "smooth_strategy=few_shot,layerwise_offload=true,defer_move_to_execution_device=false"
  )["defer_move_to_execution_device"] is False


def test_build_quantize_config_uses_the_resolved_options() -> None:
  args = resolve([
    "--quantize",
    "--quantize-params",
    "quant_type=svdq_int4_r32_dq,exclude_layers=adaln_proj,regional_quantize=false,verbose=true",
    "--svdq-params",
    "smooth_strategy=few_shot,few_shot_relax_factor=2.0",
  ])
  config = cli_utils.build_quantize_config(args)
  assert config.quant_type == "svdq_int4_r32_dq"
  assert config.backend == QuantizeBackend.CACHE_DIT
  assert config.regional_quantize is False
  assert config.verbose is True
  assert config.exclude_layers[-1] == "adaln_proj"
  assert config.get_svdq_kwargs()["smooth_strategy"] == "few_shot"
  assert config.get_svdq_kwargs()["few_shot_relax_factor"] == 2.0


def test_deprecated_flags_still_work_and_conflict_with_params() -> None:
  old = resolve(
    ["--svdq-nvfp4-r32-dq", "--svdq-runtime", "cute", "--quantize", "--quantize-verbose"])
  new = resolve([
    "--quantize",
    "--quantize-params",
    "quant_type=svdq_nvfp4_r32_dq,verbose=true",
    "--svdq-params",
    "runtime_kernel=cute",
  ])
  assert (old.quantize_type, old.svdq_kwargs,
          old.quantize_verbose) == (new.quantize_type, new.svdq_kwargs, new.quantize_verbose)
  with pytest.raises(ValueError, match="mutually exclusive"):
    resolve(["--svdq-int4-r32-dq", "--quantize-params", "quant_type=float8_per_row"])
