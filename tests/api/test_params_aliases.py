"""Coverage for the `--xxx-config` aliases and the bare first-entry type shorthand."""

import pytest

from cache_dit.cli.entrypoint import get_args
from cache_dit.cli.utils import maybe_postprocess_args

parser = get_args(parse=False)


def _resolve(argv):
  return maybe_postprocess_args(parser.parse_args(argv))


@pytest.mark.parametrize("params_flag, config_flag, value, attribute, expected", [
  ("--cache-params", "--cache-config", "DBCache,rdt=0.2", "residual_diff_threshold", 0.2),
  ("--parallel-params", "--parallel-config", "ulysses=2", "parallel_type", "ulysses"),
  ("--attn-params", "--attn-config", "native", "attn", "native"),
  ("--quantize-params", "--quantize-config", "float8_per_row", "quantize_type", "float8_per_row"),
  ("--svdq-params", "--svdq-config", "smooth_strategy=few_shot", "svdq_params", {
    "smooth_strategy": "few_shot"
  }),
  ("--quant-communicate-params", "--quant-communicate-config", "bits=8", "quant_communicate_params",
   {
     "bits": "8"
   }),
  ("--quant-communicate-params", "--quant-comm-params", "bits=8", "quant_communicate_params", {
    "bits": "8"
  }),
  ("--quant-communicate-params", "--quant-comm-config", "bits=8", "quant_communicate_params", {
    "bits": "8"
  }),
])
def test_config_is_an_alias_of_params(params_flag, config_flag, value, attribute, expected):
  argv = ["--quantize-params", "svdq_int4_r32_dq"] if params_flag == "--svdq-params" else []
  args = _resolve([*argv, params_flag, value])
  assert getattr(args, attribute) == expected
  alias = _resolve([*argv, config_flag, value])
  assert getattr(alias, attribute) == expected


def test_calibrator_config_alias_and_bare_type():
  args = _resolve(["--cache-params", "DBCache", "--calibrator-config", "svd,tau=0.8"])
  assert args.calibrator == "svd"
  assert args.calibrator_fields == {"svd_tau": 0.8}


@pytest.mark.parametrize("flag, value, attribute, expected", [
  ("--cache-config", "SeaCache,seacache_mode=vp", "seacache_mode", "vp"),
  ("--quantize-config", "float8_per_row,backend=cache_dit", "quantize_type", "float8_per_row"),
  ("--attn-config", "ffpa_fp8,hybrid=true", "attn", "ffpa_fp8"),
])
def test_bare_first_entry_sets_the_type(flag, value, attribute, expected):
  assert getattr(_resolve([flag, value]), attribute) == expected


def test_explicit_type_still_wins_on_the_first_entry():
  args = _resolve(["--cache-config", "type=SoftCache,softcache_tau=0.4"])
  assert args.cache_type == "softcache"
  assert args.softcache_tau == 0.4


@pytest.mark.parametrize("argv", [
  ["--cache-config", "Nope"],
  ["--attn-config", "nope"],
])
def test_unknown_bare_type_is_rejected(argv):
  with pytest.raises((SystemExit, ValueError)):
    _resolve(argv)


def test_bare_type_still_allows_options():
  args = _resolve(["--cache-config", "DBCache,Fn=2,Bn=1,scm=fast"])
  assert (args.Fn_compute_blocks, args.Bn_compute_blocks, args.mask_policy) == (2, 1, "fast")
