import dataclasses
import os

import pytest

from cache_dit.caching.cache_contexts import (
  BRACECalibratorConfig,
  BeziCastCalibratorConfig,
  DBCacheConfig,
  DMDCalibratorConfig,
  FoCaCalibratorConfig,
  HiCacheCalibratorConfig,
  SVDCalibratorConfig,
  TaylorSeerCalibratorConfig,
)
from cache_dit.caching.cache_contexts.calibrators import (
  BeziCastCalibrator,
  Calibrator,
  normalize_calibrator_type,
)
from cache_dit.caching.load_configs import load_cache_config, load_configs
from cache_dit.cli.utils import build_calibrator_config, get_args, maybe_postprocess_args

PARSER = get_args(parse=False)
_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "configs")


def _cli(argv):
  return maybe_postprocess_args(PARSER.parse_args(argv))


class TestBeziCastCalibratorConfig:

  def test_defaults(self):
    """Pure-predictor defaults use the paper's K=6, n=4, ridge=0.5."""
    config = BeziCastCalibratorConfig()
    assert config.enable_calibrator is True
    assert config.enable_encoder_calibrator is True
    assert config.calibrator_type == "bezicast"
    assert config.bezicast_history == 6
    assert config.bezicast_control_points == 4
    assert config.bezicast_ridge == 0.5
    assert config.bezicast_guardrail is False
    assert config.bezicast_u_max == 1.5
    assert config.bezicast_beta_max == 1.0
    assert config.bezicast_rho is None
    assert config.bezicast_gamma == 1.0

  def test_to_kwargs(self):
    """Every public config field reaches the BeziCast constructor."""
    config = BeziCastCalibratorConfig(
      bezicast_history=4,
      bezicast_control_points=3,
      bezicast_ridge=0.25,
      bezicast_guardrail=True,
      bezicast_u_max=2.0,
      bezicast_beta_max=0.8,
      bezicast_rho=0.2,
      bezicast_gamma=0.5,
    )
    assert config.to_kwargs() == {
      "history": 4,
      "control_points": 3,
      "ridge": 0.25,
      "guardrail": True,
      "u_max": 2.0,
      "beta_max": 0.8,
      "rho": 0.2,
      "gamma": 0.5,
    }

  def test_to_kwargs_survives_reset(self):
    """Refresh reset retains explicit fitting and automatic guardrail fields."""
    config = BeziCastCalibratorConfig().reset(
      enable_calibrator=True,
      enable_encoder_calibrator=True,
      calibrator_cache_type="residual",
      bezicast_history=4,
      bezicast_control_points=3,
      bezicast_ridge=0.25,
      bezicast_guardrail=True,
      bezicast_u_max=2.0,
      bezicast_beta_max=0.8,
      bezicast_rho=None,
      bezicast_gamma=0.5,
    )
    kwargs = config.to_kwargs()
    assert kwargs["history"] == 4
    assert kwargs["control_points"] == 3
    assert kwargs["guardrail"] is True
    assert kwargs["rho"] is None
    assert kwargs["gamma"] == 0.5

  def test_strify(self):
    """Summary tags expose the effective fitting and guardrail configuration."""
    config = BeziCastCalibratorConfig()
    assert config.strify() == "BZCK6N4R5U1.5G0"
    assert "guardrail=False" in config.strify(details=True)
    assert "rho=auto" in config.strify(details=True)
    assert "/" not in config.strify()
    assert " " not in config.strify()

  def test_guardrail_tags_do_not_collide(self):
    """Automatic and explicit guardrail thresholds produce distinct artifact names."""
    automatic = BeziCastCalibratorConfig(bezicast_guardrail=True)
    explicit = BeziCastCalibratorConfig(
      bezicast_guardrail=True,
      bezicast_rho=0.2,
      bezicast_beta_max=0.3,
      bezicast_gamma=0.1,
    )
    assert automatic.strify() == "BZCK6N4R5U1.5G1RHOautoB1GM1"
    assert explicit.strify() == "BZCK6N4R5U1.5G1RHO0.2B0.3GM0.1"
    assert automatic.strify() != explicit.strify()

  @pytest.mark.parametrize("field,value", [
    ("bezicast_history", 1),
    ("bezicast_history", 7),
    ("bezicast_control_points", 1),
    ("bezicast_control_points", 7),
    ("bezicast_ridge", -0.1),
    ("bezicast_guardrail", 1),
    ("bezicast_u_max", 0.5),
    ("bezicast_beta_max", 1.1),
    ("bezicast_rho", -0.1),
    ("bezicast_gamma", 0.0),
  ])
  def test_invalid_params_are_rejected(self, field, value):
    """Invalid values fail while the config is constructed."""
    with pytest.raises(ValueError):
      dataclasses.replace(BeziCastCalibratorConfig(), **{field: value})

  def test_history_and_control_points_cross_validation(self):
    """The control-point count cannot exceed a smaller history window."""
    with pytest.raises(ValueError, match="control_points"):
      BeziCastCalibratorConfig(bezicast_history=3, bezicast_control_points=4)

  def test_factory_accepts_canonical_and_aliases(self):
    """Canonical and long BeziCast names construct the intended backend."""
    for name in ("bezicast", "bezicast-calibrator", "bezier-cast", "BeziCast"):
      calibrator = Calibrator(BeziCastCalibratorConfig(calibrator_type=name))
      assert isinstance(calibrator, BeziCastCalibrator)
      assert (calibrator.history, calibrator.control_points) == (6, 4)
    assert normalize_calibrator_type("BeziCast-Calibrator") == "bezicast"


class TestBeziCastConfigLoading:

  def test_load_from_dict_basic(self):
    """Only enable_bezicast selects all pure-predictor defaults."""
    result = load_configs({"cache_config": {"enable_bezicast": True}}, return_dict=True)
    config = result["calibrator_config"]
    assert isinstance(config, BeziCastCalibratorConfig)
    assert config.enable_encoder_calibrator is True
    assert config.bezicast_guardrail is False
    assert config.bezicast_rho is None

  def test_load_from_dict_full(self):
    """All fitting, guardrail, cache-type, and encoder fields load verbatim."""
    result = load_configs(
      {
        "cache_config": {
          "enable_bezicast": True,
          "enable_encoder_bezicast": False,
          "calibrator_cache_type": "hidden_states",
          "bezicast_history": 4,
          "bezicast_control_points": 3,
          "bezicast_ridge": 0.25,
          "bezicast_guardrail": True,
          "bezicast_u_max": 2.0,
          "bezicast_beta_max": 0.8,
          "bezicast_rho": 0.2,
          "bezicast_gamma": 0.5,
        },
      },
      return_dict=True,
    )
    config = result["calibrator_config"]
    assert config.enable_encoder_calibrator is False
    assert config.calibrator_cache_type == "hidden_states"
    assert config.bezicast_history == 4
    assert config.bezicast_control_points == 3
    assert config.bezicast_guardrail is True
    assert config.bezicast_rho == 0.2

  def test_load_from_yaml(self):
    """The shipped YAML loads automatic rho as None and guardrail as false."""
    result = load_configs(os.path.join(_CONFIG_DIR, "cache_bezicast.yaml"), return_dict=True)
    assert isinstance(result["cache_config"], DBCacheConfig)
    config = result["calibrator_config"]
    assert isinstance(config, BeziCastCalibratorConfig)
    assert config.bezicast_rho is None
    assert config.bezicast_guardrail is False

  def test_reset_path_keeps_every_field(self):
    """refresh_context reset reconstructs all fields and the actual backend."""
    _, config = load_cache_config(
      {
        "cache_config": {
          "enable_bezicast": True,
          "bezicast_history": 4,
          "bezicast_control_points": 3,
          "bezicast_ridge": 0.25,
          "bezicast_guardrail": True,
          "bezicast_u_max": 2.0,
          "bezicast_beta_max": 0.8,
          "bezicast_rho": None,
          "bezicast_gamma": 0.5,
        },
      },
      reset=True,
    )
    backend = Calibrator(config)
    assert isinstance(backend, BeziCastCalibrator)
    assert backend.history == 4
    assert backend.control_points == 3
    assert backend.ridge == 0.25
    assert backend.guardrail is True
    assert backend.u_max == 2.0
    assert backend.beta_max == 0.8
    assert backend.rho is None
    assert backend.gamma == 0.5

  @pytest.mark.parametrize("other", [
    "enable_taylorseer",
    "enable_dmd",
    "enable_foca",
    "enable_svd",
    "enable_hicache",
    "enable_brace",
  ])
  def test_mutual_exclusion(self, other):
    """BeziCast conflicts with every other calibrator switch."""
    with pytest.raises(ValueError, match="mutually exclusive"):
      load_configs(
        {"cache_config": {
          "enable_bezicast": True,
          other: True,
        }},
        return_dict=True,
      )

  def test_existing_calibrators_still_load(self):
    """Adding BeziCast does not change existing calibrator dispatch."""
    cases = {
      "enable_taylorseer": TaylorSeerCalibratorConfig,
      "enable_dmd": DMDCalibratorConfig,
      "enable_foca": FoCaCalibratorConfig,
      "enable_svd": SVDCalibratorConfig,
      "enable_hicache": HiCacheCalibratorConfig,
      "enable_brace": BRACECalibratorConfig,
    }
    for key, expected in cases.items():
      result = load_configs({"cache_config": {key: True}}, return_dict=True)
      assert isinstance(result["calibrator_config"], expected)


class TestBeziCastCLI:

  def test_unified_entry_builds_full_config(self):
    """Short CLI keys populate fitting, guardrail, and encoder fields."""
    args = _cli([
      "--cache-params", "type=DBCache", "--calibrator-params",
      "type=bezicast,history=4,control_points=3,ridge=0.25,guardrail=true,u_max=2.0,"
      "beta_max=0.8,rho=auto,gamma=0.5,encoder=false"
    ])
    config = build_calibrator_config(args)
    assert isinstance(config, BeziCastCalibratorConfig)
    assert config.bezicast_history == 4
    assert config.bezicast_control_points == 3
    assert config.bezicast_ridge == 0.25
    assert config.bezicast_guardrail is True
    assert config.bezicast_u_max == 2.0
    assert config.bezicast_beta_max == 0.8
    assert config.bezicast_rho is None
    assert config.bezicast_gamma == 0.5
    assert config.enable_encoder_calibrator is False

  def test_defaults_without_params(self):
    """No params selects pure Bezier fitting with guardrail disabled."""
    config = build_calibrator_config(
      _cli(["--cache-params", "type=DBCache", "--calibrator-params", "type=bezicast"]))
    assert config.bezicast_guardrail is False
    assert (config.bezicast_history, config.bezicast_control_points) == (6, 4)

  def test_numeric_rho_and_long_name(self):
    """The long calibrator name and numeric guardrail threshold are accepted."""
    args = _cli([
      "--cache-params", "type=DBCache", "--calibrator-params",
      "type=bezicast-calibrator,guardrail=true,rho=0.28"
    ])
    config = build_calibrator_config(args)
    assert isinstance(config, BeziCastCalibratorConfig)
    assert config.bezicast_rho == 0.28

  @pytest.mark.parametrize("params", [
    "history=7",
    "control_points=1",
    "history=3,control_points=4",
    "ridge=-1",
    "u_max=0.5",
    "beta_max=2",
    "rho=-1",
    "gamma=0",
  ])
  def test_out_of_range_param_is_rejected(self, params):
    """Invalid CLI values fail before model inference starts."""
    with pytest.raises(ValueError):
      args = _cli(
        ["--cache-params", "type=DBCache", "--calibrator-params", f"type=bezicast,{params}"])
      build_calibrator_config(args)

  def test_requires_cache(self):
    """BeziCast without --cache is rejected instead of becoming a silent no-op."""
    with pytest.raises(ValueError, match="requires --cache-params"):
      _cli(["--calibrator-params", "type=bezicast"])

  def test_config_path_conflict(self):
    """A config file and unified calibrator params cannot both own the settings."""
    with pytest.raises(ValueError, match="cannot be combined with --config-path"):
      _cli([
        "--cache-params", "type=DBCache", "--config-path", "some.yaml", "--calibrator-params",
        "type=bezicast"
      ])

  def test_postprocess_is_idempotent_with_auto_rho(self):
    """Resolving rho=auto twice preserves None and all other fields."""
    once = _cli([
      "--cache-params", "type=DBCache", "--calibrator-params",
      "type=bezicast,control_points=3,ridge=0.25,guardrail=true,rho=auto"
    ])
    twice = maybe_postprocess_args(once)
    assert twice.calibrator_fields == once.calibrator_fields
    config = build_calibrator_config(twice)
    assert config.bezicast_rho is None
    assert config.bezicast_guardrail is True
    assert config.bezicast_control_points == 3
