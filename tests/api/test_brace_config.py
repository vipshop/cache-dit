import os

import pytest

from cache_dit.caching.cache_contexts import (
  BRACECalibratorConfig,
  DBCacheConfig,
  DMDCalibratorConfig,
  FoCaCalibratorConfig,
  HiCacheCalibratorConfig,
  SVDCalibratorConfig,
  TaylorSeerCalibratorConfig,
)
from cache_dit.caching.cache_contexts.calibrators import (
  BRACECalibrator,
  Calibrator,
  normalize_calibrator_type,
)
from cache_dit.caching.load_configs import load_cache_config, load_configs
from cache_dit.cli.utils import build_calibrator_config, get_args, maybe_postprocess_args

PARSER = get_args(parse=False)
_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "configs")


def _cli(argv):
  return maybe_postprocess_args(PARSER.parse_args(argv))


class TestBRACECalibratorConfig:

  def test_defaults(self):
    """BRACE defaults use the paper's C=3 window and gamma=0.7."""
    config = BRACECalibratorConfig()
    assert config.enable_calibrator is True
    assert config.enable_encoder_calibrator is True
    assert config.calibrator_type == "brace"
    assert config.calibrator_cache_type == "residual"
    assert config.brace_capacity == 3
    assert config.brace_gamma == 0.7

  def test_to_kwargs(self):
    """Config fields translate into BRACECalibrator constructor kwargs."""
    assert BRACECalibratorConfig(brace_capacity=2, brace_gamma=0.6).to_kwargs() == {
      "capacity": 2,
      "gamma": 0.6,
    }

  def test_to_kwargs_survives_reset(self):
    """The refresh path leaves calibrator_kwargs=None without losing BRACE fields."""
    config = BRACECalibratorConfig().reset(
      enable_calibrator=True,
      enable_encoder_calibrator=True,
      calibrator_cache_type="residual",
      brace_capacity=2,
      brace_gamma=0.6,
    )
    assert config.to_kwargs() == {"capacity": 2, "gamma": 0.6}

  def test_strify(self):
    """Summary tags contain both parameters and remain safe in filenames."""
    config = BRACECalibratorConfig()
    assert config.strify() == "BRACEC3G7"
    assert config.strify(details=True) == "BRACE_C(3, gamma=0.7)"
    assert "/" not in config.strify()
    assert " " not in config.strify()

  @pytest.mark.parametrize("capacity,gamma", [(0, 0.7), (4, 0.7), (True, 0.7), (3, 0.0), (3, 0.4),
                                              (3, True), (3, -1.0), (3, float("inf"))])
  def test_invalid_params_are_rejected(self, capacity, gamma):
    """Invalid BRACE values fail while the config is constructed."""
    with pytest.raises(ValueError):
      BRACECalibratorConfig(brace_capacity=capacity, brace_gamma=gamma)

  def test_factory_accepts_canonical_and_long_names(self):
    """The factory resolves canonical and long BRACE spellings."""
    for name in ("brace", "brace-calibrator", "BRACE"):
      calibrator = Calibrator(BRACECalibratorConfig(calibrator_type=name))
      assert isinstance(calibrator, BRACECalibrator)
      assert (calibrator.capacity, calibrator.gamma) == (3, 0.7)
    assert normalize_calibrator_type("BRACE-Calibrator") == "brace"


class TestBRACEConfigLoading:

  def test_load_from_dict_basic(self):
    """Only enable_brace uses defaults and enables the encoder stream too."""
    result = load_configs({"cache_config": {"enable_brace": True}}, return_dict=True)
    calibrator = result["calibrator_config"]
    assert isinstance(calibrator, BRACECalibratorConfig)
    assert calibrator.enable_calibrator is True
    assert calibrator.enable_encoder_calibrator is True
    assert calibrator.brace_capacity == 3
    assert calibrator.brace_gamma == 0.7

  def test_load_from_dict_full(self):
    """All BRACE fields and the explicit encoder switch load verbatim."""
    result = load_configs(
      {
        "cache_config": {
          "enable_brace": True,
          "enable_encoder_brace": False,
          "calibrator_cache_type": "hidden_states",
          "brace_capacity": 2,
          "brace_gamma": 0.6,
        },
      },
      return_dict=True,
    )
    calibrator = result["calibrator_config"]
    assert calibrator.enable_encoder_calibrator is False
    assert calibrator.calibrator_cache_type == "hidden_states"
    assert calibrator.brace_capacity == 2
    assert calibrator.brace_gamma == 0.6

  def test_load_from_yaml(self):
    """The shipped BRACE example loads into the expected config types."""
    result = load_configs(os.path.join(_CONFIG_DIR, "cache_brace.yaml"), return_dict=True)
    assert isinstance(result["cache_config"], DBCacheConfig)
    assert isinstance(result["calibrator_config"], BRACECalibratorConfig)
    assert result["calibrator_config"].brace_capacity == 3

  def test_reset_path_keeps_every_field(self):
    """refresh_context reset reconstructs all BRACE fields and builds the backend."""
    _, calibrator = load_cache_config(
      {"cache_config": {
        "enable_brace": True,
        "brace_capacity": 2,
        "brace_gamma": 0.6,
      }},
      reset=True,
    )
    assert calibrator.brace_capacity == 2
    assert calibrator.brace_gamma == 0.6
    assert calibrator.enable_encoder_calibrator is True
    backend = Calibrator(calibrator)
    assert isinstance(backend, BRACECalibrator)
    assert (backend.capacity, backend.gamma) == (2, 0.6)

  @pytest.mark.parametrize(
    "other", ["enable_taylorseer", "enable_dmd", "enable_foca", "enable_svd", "enable_hicache"])
  def test_mutual_exclusion(self, other):
    """BRACE conflicts with every other calibrator switch."""
    with pytest.raises(ValueError, match="mutually exclusive"):
      load_configs(
        {"cache_config": {
          "enable_brace": True,
          other: True,
        }},
        return_dict=True,
      )

  def test_existing_calibrators_still_load(self):
    """Adding BRACE does not change existing calibrator dispatch."""
    cases = {
      "enable_taylorseer": TaylorSeerCalibratorConfig,
      "enable_dmd": DMDCalibratorConfig,
      "enable_foca": FoCaCalibratorConfig,
      "enable_svd": SVDCalibratorConfig,
      "enable_hicache": HiCacheCalibratorConfig,
    }
    for key, expected in cases.items():
      result = load_configs({"cache_config": {key: True}}, return_dict=True)
      assert isinstance(result["calibrator_config"], expected)


class TestBRACECLI:

  def test_unified_entry_builds_brace_config(self):
    """--calibrator brace and short params populate the BRACE config."""
    args = _cli([
      "--cache", "--calibrator", "brace", "--calibrator-params",
      "capacity=2,gamma=0.6,encoder=false"
    ])
    config = build_calibrator_config(args)
    assert isinstance(config, BRACECalibratorConfig)
    assert (config.brace_capacity, config.brace_gamma) == (2, 0.6)
    assert config.enable_encoder_calibrator is False

  def test_defaults_without_params(self):
    """No params selects C=3 and gamma=0.7."""
    config = build_calibrator_config(_cli(["--cache", "--calibrator", "brace"]))
    assert (config.brace_capacity, config.brace_gamma) == (3, 0.7)

  def test_long_name_is_accepted(self):
    """The long brace-calibrator spelling resolves to BRACE."""
    args = _cli(["--cache", "--calibrator", "brace-calibrator"])
    assert isinstance(build_calibrator_config(args), BRACECalibratorConfig)

  @pytest.mark.parametrize("params",
                           ["capacity=4", "capacity=0", "gamma=0", "gamma=0.4", "gamma=-1"])
  def test_out_of_range_param_is_rejected(self, params):
    """Invalid CLI values fail before model inference starts."""
    with pytest.raises(ValueError):
      args = _cli(["--cache", "--calibrator", "brace", "--calibrator-params", params])
      build_calibrator_config(args)

  def test_requires_cache(self):
    """BRACE without --cache is rejected instead of becoming a silent no-op."""
    with pytest.raises(ValueError, match="requires --cache"):
      _cli(["--calibrator", "brace"])

  def test_config_path_conflict(self):
    """A config file and unified calibrator flags cannot both own the settings."""
    with pytest.raises(ValueError, match="cannot be combined with --config-path"):
      _cli(["--cache", "--config-path", "some.yaml", "--calibrator", "brace"])

  def test_postprocess_is_idempotent(self):
    """Resolving BRACE params twice preserves the same config fields."""
    once = _cli(["--cache", "--calibrator", "brace", "--calibrator-params", "capacity=2,gamma=0.6"])
    twice = maybe_postprocess_args(once)
    assert once.calibrator_fields == {"brace_capacity": 2, "brace_gamma": 0.6}
    assert twice.calibrator_fields == once.calibrator_fields
    assert build_calibrator_config(twice).brace_gamma == 0.6
