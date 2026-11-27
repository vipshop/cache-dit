import os

import pytest

from cache_dit.caching.cache_contexts import (
  DBCacheConfig,
  DMDCalibratorConfig,
  FoCaCalibratorConfig,
  HiCacheCalibratorConfig,
  SVDCalibratorConfig,
  TaylorSeerCalibratorConfig,
)
from cache_dit.caching.cache_contexts.calibrators import (
  Calibrator,
  HiCacheCalibrator,
  normalize_calibrator_type,
)
from cache_dit.caching.load_configs import load_cache_config, load_configs
from cache_dit.cli.utils import build_calibrator_config, get_args, maybe_postprocess_args

PARSER = get_args(parse=False)

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "configs")


def _cli(argv):
  return maybe_postprocess_args(PARSER.parse_args(argv))


class TestHiCacheCalibratorConfig:

  def test_defaults(self):
    """HiCache defaults follow the paper's FLUX setting (order 2) and a mid damping factor."""
    config = HiCacheCalibratorConfig()
    assert config.enable_calibrator is True
    assert config.enable_encoder_calibrator is True
    assert config.calibrator_type == "hicache"
    assert config.calibrator_cache_type == "residual"
    assert config.hicache_order == 2
    assert config.hicache_sigma == 0.5

  def test_to_kwargs(self):
    """to_kwargs() translates the config fields into HiCacheCalibrator constructor kwargs."""
    assert HiCacheCalibratorConfig(hicache_order=3, hicache_sigma=0.4).to_kwargs() == {
      "n_derivatives": 3,
      "sigma": 0.4,
    }

  def test_to_kwargs_survives_reset(self):
    """The reset path leaves calibrator_kwargs as None; to_kwargs() must still work."""
    config = HiCacheCalibratorConfig().reset(
      enable_calibrator=True,
      enable_encoder_calibrator=True,
      calibrator_cache_type="residual",
      hicache_order=2,
      hicache_sigma=0.4,
    )
    assert config.to_kwargs() == {"n_derivatives": 2, "sigma": 0.4}

  def test_strify(self):
    """Short tags stay filename-safe and keep both the order and the damping factor."""
    assert HiCacheCalibratorConfig().strify() == "HC2S0.5"
    assert HiCacheCalibratorConfig(hicache_order=1, hicache_sigma=1.0).strify() == "HC1S1"
    assert HiCacheCalibratorConfig().strify(details=True) == "HiCache_O(2, sigma=0.5)"

  @pytest.mark.parametrize("sigma,order", [(0.0, 1), (1.2, 1), (0.5, -1), (0.5, 5)])
  def test_invalid_params_are_rejected(self, sigma, order):
    """Out-of-domain hyper-parameters fail while the config is built."""
    with pytest.raises(ValueError):
      HiCacheCalibratorConfig(hicache_order=order, hicache_sigma=sigma)

  def test_factory_accepts_canonical_and_long_names(self):
    """The factory resolves the canonical name and the long aliases."""
    for name in ("hicache", "hicache-calibrator", "hi-cache", "HiCache"):
      calibrator = Calibrator(HiCacheCalibratorConfig(calibrator_type=name))
      assert isinstance(calibrator, HiCacheCalibrator)
      assert calibrator.sigma == 0.5
    assert normalize_calibrator_type("HiCache-Calibrator") == "hicache"


class TestHiCacheConfigLoading:

  def test_load_from_dict_basic(self):
    """Only enable_hicache set: the encoder switch follows it and the defaults apply."""
    result = load_configs({"cache_config": {"enable_hicache": True}}, return_dict=True)
    calibrator = result["calibrator_config"]
    assert isinstance(calibrator, HiCacheCalibratorConfig)
    assert calibrator.enable_calibrator is True
    assert calibrator.enable_encoder_calibrator is True
    assert calibrator.calibrator_cache_type == "residual"
    assert calibrator.hicache_order == 2
    assert calibrator.hicache_sigma == 0.5

  def test_load_from_dict_full(self):
    """All HiCache fields load verbatim."""
    result = load_configs(
      {
        "cache_config": {
          "enable_hicache": True,
          "enable_encoder_hicache": False,
          "calibrator_cache_type": "hidden_states",
          "hicache_order": 1,
          "hicache_sigma": 0.4,
        },
      },
      return_dict=True,
    )
    calibrator = result["calibrator_config"]
    assert isinstance(calibrator, HiCacheCalibratorConfig)
    assert calibrator.enable_encoder_calibrator is False
    assert calibrator.calibrator_cache_type == "hidden_states"
    assert calibrator.hicache_order == 1
    assert calibrator.hicache_sigma == 0.4

  def test_load_from_yaml(self):
    """The shipped example config loads into a HiCacheCalibratorConfig."""
    result = load_configs(os.path.join(_CONFIG_DIR, "cache_hicache.yaml"), return_dict=True)
    calibrator = result["calibrator_config"]
    assert isinstance(calibrator, HiCacheCalibratorConfig)
    assert isinstance(result["cache_config"], DBCacheConfig)
    assert calibrator.enable_calibrator is True
    assert calibrator.hicache_order == 2

  def test_load_cache_config_tuple(self):
    """load_cache_config returns the DBCache/HiCache pair."""
    cache_config, calibrator_config = load_cache_config(
      {"cache_config": {
        "enable_hicache": True,
        "hicache_sigma": 0.4
      }}, )
    assert isinstance(cache_config, DBCacheConfig)
    assert isinstance(calibrator_config, HiCacheCalibratorConfig)
    assert calibrator_config.hicache_sigma == 0.4

  def test_reset_path_keeps_every_field(self):
    """The refresh_context(reset=True) path must not lose the new fields or the shared defaults."""
    _, calibrator = load_cache_config(
      {"cache_config": {
        "enable_hicache": True,
        "hicache_order": 1,
        "hicache_sigma": 0.4
      }},
      reset=True,
    )
    assert calibrator.hicache_order == 1
    assert calibrator.hicache_sigma == 0.4
    assert calibrator.enable_encoder_calibrator is True
    assert calibrator.calibrator_cache_type == "residual"
    assert isinstance(Calibrator(calibrator), HiCacheCalibrator)

  @pytest.mark.parametrize("other",
                           ["enable_taylorseer", "enable_dmd", "enable_foca", "enable_svd"])
  def test_mutual_exclusion(self, other):
    """enable_hicache conflicts with every other calibrator switch."""
    with pytest.raises(ValueError, match="mutually exclusive"):
      load_configs(
        {
          "cache_config": {
            "enable_hicache": True,
            other: True,
          },
        },
        return_dict=True,
      )

  def test_existing_calibrators_still_load(self):
    """Regression: the four existing calibrator switches are unaffected."""
    cases = {
      "enable_taylorseer": TaylorSeerCalibratorConfig,
      "enable_dmd": DMDCalibratorConfig,
      "enable_foca": FoCaCalibratorConfig,
      "enable_svd": SVDCalibratorConfig,
    }
    for key, expected in cases.items():
      result = load_configs({"cache_config": {key: True}}, return_dict=True)
      assert isinstance(result["calibrator_config"], expected)


class TestHiCacheCLI:

  def test_unified_entry_builds_hicache_config(self):
    """`type=hicache` plus option keys fills the HiCache fields."""
    args = _cli(["--cache", "--calibrator-params", "type=hicache,order=1,sigma=0.4"])
    config = build_calibrator_config(args)
    assert isinstance(config, HiCacheCalibratorConfig)
    assert (config.hicache_order, config.hicache_sigma) == (1, 0.4)

  def test_defaults_without_params(self):
    """Without option keys the paper setting (order 2, sigma 0.5) is used."""
    config = build_calibrator_config(_cli(["--cache", "--calibrator-params", "type=hicache"]))
    assert (config.hicache_order, config.hicache_sigma) == (2, 0.5)

  def test_long_name_is_accepted(self):
    """The long 'hicache-calibrator' spelling resolves to the same config."""
    args = _cli(["--cache", "--calibrator-params", "type=hicache-calibrator"])
    assert isinstance(build_calibrator_config(args), HiCacheCalibratorConfig)

  @pytest.mark.parametrize("params", ["sigma=1.5", "order=5"])
  def test_out_of_range_param_is_rejected(self, params):
    """A value outside the algorithm's domain fails before any inference."""
    args = _cli(["--cache", "--calibrator-params", f"type=hicache,{params}"])
    with pytest.raises(ValueError):
      build_calibrator_config(args)

  def test_requires_cache(self):
    """A calibrator without --cache would be a silent no-op, so it fails."""
    with pytest.raises(ValueError, match="requires --cache"):
      _cli(["--calibrator-params", "type=hicache"])

  def test_config_path_conflict(self):
    """A config file already owns the calibrator settings."""
    with pytest.raises(ValueError, match="cannot be combined with --config-path"):
      _cli(["--cache", "--config-path", "some.yaml", "--calibrator-params", "type=hicache"])

  def test_postprocess_is_idempotent(self):
    """Resolving the params must not consume them, so a second pass is a no-op."""
    once = _cli(["--cache", "--calibrator-params", "type=hicache,order=1,sigma=0.4"])
    twice = maybe_postprocess_args(once)
    assert once.calibrator_fields == {"hicache_order": 1, "hicache_sigma": 0.4}
    assert twice.calibrator_fields == once.calibrator_fields
    assert build_calibrator_config(twice).hicache_sigma == 0.4
