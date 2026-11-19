import os

import pytest

from cache_dit.caching.cache_contexts import (
  DBCacheConfig,
  DMDCalibratorConfig,
  FoCaCalibratorConfig,
  SVDCalibratorConfig,
  TaylorSeerCalibratorConfig,
)
from cache_dit.caching.cache_contexts.calibrators import (
  Calibrator,
  SVDCalibrator,
  normalize_calibrator_type,
)
from cache_dit.caching.load_configs import load_cache_config, load_configs
from cache_dit.cli.utils import build_calibrator_config, get_args, maybe_postprocess_args

PARSER = get_args(parse=False)

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "configs")


def _cli(argv):
  return maybe_postprocess_args(PARSER.parse_args(argv))


class TestSVDCalibratorConfig:

  def test_defaults(self):
    """SVD-Cache defaults mirror the paper's tau/beta and keep an uncapped rank."""
    config = SVDCalibratorConfig()
    assert config.enable_calibrator is True
    assert config.enable_encoder_calibrator is True
    assert config.calibrator_type == "svd"
    assert config.calibrator_cache_type == "residual"
    assert config.svd_tau == 0.9
    assert config.svd_beta == 0.9
    assert config.svd_max_rank == 0
    assert config.svd_predictor == "ema"
    assert config.svd_order == 1

  def test_to_kwargs(self):
    """to_kwargs() translates the config fields into SVDCalibrator constructor kwargs."""
    kwargs = SVDCalibratorConfig(svd_tau=0.85, svd_beta=0.5, svd_max_rank=8).to_kwargs()
    assert kwargs == {
      "tau": 0.85,
      "beta": 0.5,
      "max_rank": 8,
      "predictor": "ema",
      "n_derivatives": 1,
    }

  def test_strify(self):
    """Short tags stay filename-safe and gain the rank only when it is capped."""
    assert SVDCalibratorConfig().strify() == "SVDT9B9"
    assert SVDCalibratorConfig(svd_max_rank=8).strify() == "SVDT9B9R8"
    assert SVDCalibratorConfig().strify(details=True) == "SVD_T(0.9, ema, B=0.9, R=0)"

  def test_taylor_predictor_config(self):
    """The SVD+Taylor mode gets its own tag and forwards the ladder order."""
    config = SVDCalibratorConfig(svd_predictor="taylor", svd_order=2)
    assert config.strify() == "SVDT9T2"
    assert config.strify(details=True) == "SVD_T(0.9, taylor2, R=0)"
    kwargs = config.to_kwargs()
    assert kwargs["predictor"] == "taylor"
    assert kwargs["n_derivatives"] == 2

  def test_factory_accepts_canonical_and_long_names(self):
    """The factory resolves the canonical name and the long aliases."""
    for name in ("svd", "svd-calibrator", "svd-cache", "SVD"):
      calibrator = Calibrator(SVDCalibratorConfig(calibrator_type=name))
      assert isinstance(calibrator, SVDCalibrator)
      assert calibrator.tau == 0.9
    assert normalize_calibrator_type("DMD-Calibrator") == "dmd"


class TestSVDConfigLoading:

  def test_load_from_dict_basic(self):
    """Only enable_svd set: encoder follows it and the paper defaults apply."""
    result = load_configs({"cache_config": {"enable_svd": True}}, return_dict=True)
    calibrator = result["calibrator_config"]
    assert isinstance(calibrator, SVDCalibratorConfig)
    assert calibrator.enable_calibrator is True
    assert calibrator.enable_encoder_calibrator is True
    assert calibrator.svd_tau == 0.9
    assert calibrator.svd_beta == 0.9
    assert calibrator.svd_max_rank == 0

  def test_load_from_dict_full(self):
    """All SVD-Cache fields load verbatim."""
    result = load_configs(
      {
        "cache_config": {
          "enable_svd": True,
          "enable_encoder_svd": False,
          "calibrator_cache_type": "hidden_states",
          "svd_tau": 0.85,
          "svd_beta": 0.5,
          "svd_max_rank": 12,
        },
      },
      return_dict=True,
    )
    calibrator = result["calibrator_config"]
    assert isinstance(calibrator, SVDCalibratorConfig)
    assert calibrator.enable_encoder_calibrator is False
    assert calibrator.calibrator_cache_type == "hidden_states"
    assert calibrator.svd_tau == 0.85
    assert calibrator.svd_beta == 0.5
    assert calibrator.svd_max_rank == 12

  def test_load_from_yaml(self):
    """The shipped example config loads into an SVDCalibratorConfig."""
    result = load_configs(os.path.join(_CONFIG_DIR, "cache_svd.yaml"), return_dict=True)
    calibrator = result["calibrator_config"]
    assert isinstance(calibrator, SVDCalibratorConfig)
    assert isinstance(result["cache_config"], DBCacheConfig)
    assert calibrator.enable_calibrator is True

  def test_load_cache_config_tuple(self):
    """load_cache_config returns the DBCache/SVD pair."""
    cache_config, calibrator_config = load_cache_config(
      {"cache_config": {
        "enable_svd": True,
        "svd_tau": 0.8
      }}, )
    assert isinstance(cache_config, DBCacheConfig)
    assert isinstance(calibrator_config, SVDCalibratorConfig)
    assert calibrator_config.svd_tau == 0.8

  @pytest.mark.parametrize("other", ["enable_taylorseer", "enable_dmd", "enable_foca"])
  def test_mutual_exclusion(self, other):
    """enable_svd conflicts with every other calibrator switch."""
    with pytest.raises(ValueError, match="mutually exclusive"):
      load_configs(
        {
          "cache_config": {
            "enable_svd": True,
            other: True,
          },
        },
        return_dict=True,
      )

  def test_existing_calibrators_still_load(self):
    """Regression: the three existing calibrator switches are unaffected."""
    cases = {
      "enable_taylorseer": TaylorSeerCalibratorConfig,
      "enable_dmd": DMDCalibratorConfig,
      "enable_foca": FoCaCalibratorConfig,
    }
    for key, expected in cases.items():
      result = load_configs({"cache_config": {key: True}}, return_dict=True)
      assert isinstance(result["calibrator_config"], expected)


class TestCalibratorCLI:

  def test_unified_entry_builds_svd_config(self):
    """--calibrator svd plus --calibrator-params fills the SVD fields."""
    args = _cli(
      ["--cache", "--calibrator", "svd", "--calibrator-params", "tau=0.85,beta=0.5,max_rank=8"])
    config = build_calibrator_config(args)
    assert isinstance(config, SVDCalibratorConfig)
    assert (config.svd_tau, config.svd_beta, config.svd_max_rank) == (0.85, 0.5, 8)

  def test_long_name_is_accepted(self):
    """The long 'svd-calibrator' spelling resolves to the same config."""
    args = _cli(["--cache", "--calibrator", "svd-calibrator"])
    assert isinstance(build_calibrator_config(args), SVDCalibratorConfig)

  def test_common_params(self):
    """Common keys work for any calibrator."""
    args = _cli([
      "--cache", "--calibrator", "svd", "--calibrator-params",
      "calibrator_cache_type=hidden_states,encoder=false"
    ])
    config = build_calibrator_config(args)
    assert config.calibrator_cache_type == "hidden_states"
    assert config.enable_encoder_calibrator is False

  def test_taylor_predictor_params(self):
    """Predictor/order reach the config through the same entry point."""
    args = _cli(
      ["--cache", "--calibrator", "svd", "--calibrator-params", "predictor=taylor,order=2"])
    config = build_calibrator_config(args)
    assert config.svd_predictor == "taylor"
    assert config.svd_order == 2

  def test_invalid_predictor_is_rejected(self):
    """An unknown predictor value is validated instead of silently accepted."""
    with pytest.raises(ValueError, match="expects one of"):
      _cli(["--cache", "--calibrator", "svd", "--calibrator-params", "predictor=cubic"])

  def test_dmd_params(self):
    """DMD options go through the same entry point."""
    args = _cli(
      ["--cache", "--calibrator", "dmd", "--calibrator-params", "history=5,svd_precision=low"])
    config = build_calibrator_config(args)
    assert isinstance(config, DMDCalibratorConfig)
    assert config.dmd_history == 5
    assert config.dmd_svd_precision == "low"
    defaults = build_calibrator_config(_cli(["--cache", "--calibrator", "dmd"]))
    assert (defaults.dmd_history, defaults.dmd_svd_precision) == (6, "medium")

  def test_taylorseer_params(self):
    """TaylorSeer order uses --calibrator-params and defaults to one."""
    args = _cli(["--cache", "--calibrator", "taylorseer", "--calibrator-params", "order=2"])
    config = build_calibrator_config(args)
    assert isinstance(config, TaylorSeerCalibratorConfig)
    assert config.taylorseer_order == 2
    defaults = build_calibrator_config(_cli(["--cache", "--calibrator", "taylorseer"]))
    assert defaults.taylorseer_order == 1

  def test_no_calibrator_builds_nothing(self):
    """Without --calibrator the cache runs without a forecast layer."""
    assert build_calibrator_config(_cli(["--cache"])) is None

  def test_unknown_param_key_raises(self):
    """Unknown keys fail loudly instead of being silently ignored."""
    args = _cli(["--cache", "--calibrator", "svd"])
    args.calibrator_params = {"bogus": "1"}
    with pytest.raises(ValueError, match="Unknown --calibrator-params key"):
      maybe_postprocess_args(args)

  def test_unknown_calibrator_is_rejected(self):
    """Argparse rejects an unknown calibrator name."""
    with pytest.raises(SystemExit):
      PARSER.parse_args(["--cache", "--calibrator", "nope"])

  @pytest.mark.parametrize("legacy_flag", ["--taylorseer", "--dmd", "--foca", "--svd", "--hicache"])
  def test_legacy_selector_flags_are_rejected(self, legacy_flag):
    """Calibrator selection has a single entry point: --calibrator."""
    with pytest.raises(SystemExit):
      PARSER.parse_args(["--cache", legacy_flag])

  @pytest.mark.parametrize("legacy_argv", [
    ["--taylorseer-order", "1"],
    ["--order", "1"],
    ["-order", "1"],
    ["--dmd-history", "4"],
    ["--dmd-svd-precision", "low"],
    ["--dmd-svd", "low"],
  ])
  def test_legacy_value_flags_are_rejected(self, legacy_argv):
    """Algorithm options are configured only through --calibrator-params."""
    with pytest.raises(SystemExit):
      PARSER.parse_args(["--cache", *legacy_argv])

  def test_calibrator_requires_cache(self):
    """--calibrator without --cache would be a silent no-op, so it fails."""
    with pytest.raises(ValueError, match="requires --cache"):
      _cli(["--calibrator", "svd"])

  def test_params_require_calibrator(self):
    """--calibrator-params alone has no target and fails."""
    with pytest.raises(ValueError, match="requires --calibrator"):
      _cli(["--cache", "--calibrator-params", "tau=0.9"])

  def test_config_path_conflict(self):
    """A config file already owns the calibrator settings."""
    with pytest.raises(ValueError, match="cannot be combined with --config-path"):
      _cli(["--cache", "--config-path", "some.yaml", "--calibrator", "svd"])

  def test_postprocess_is_idempotent(self):
    """Resolving the params must not consume them, so a second pass is a no-op."""
    once = _cli(["--cache", "--calibrator", "svd", "--calibrator-params", "tau=0.85,beta=0.5"])
    twice = maybe_postprocess_args(once)
    assert once.calibrator_fields == {"svd_tau": 0.85, "svd_beta": 0.5}
    assert twice.calibrator_fields == once.calibrator_fields
    assert build_calibrator_config(twice).svd_tau == 0.85

  def test_out_of_range_param_is_rejected(self):
    """A numeric value outside the algorithm's domain fails before any inference."""
    args = _cli(["--cache", "--calibrator", "svd", "--calibrator-params", "tau=1.5"])
    with pytest.raises(ValueError, match="tau must be in"):
      build_calibrator_config(args)
