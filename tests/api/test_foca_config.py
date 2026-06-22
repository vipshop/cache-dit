import pytest
import os

from cache_dit.caching.load_configs import load_configs, load_cache_config
from cache_dit.caching.cache_contexts import (
  FoCaCalibratorConfig,
  TaylorSeerCalibratorConfig,
  DMDCalibratorConfig,
  DBCacheConfig,
)


class TestFoCaConfigLoading:

  def test_load_foca_from_dict_basic(self):
    """Load FoCa config from a dict with only enable_foca set, verify defaults."""
    result = load_configs(
      {"cache_config": {
        "enable_foca": True
      }},
      return_dict=True,
    )
    calibrator = result["calibrator_config"]
    assert isinstance(calibrator, FoCaCalibratorConfig), \
        f"Expected FoCaCalibratorConfig, got {type(calibrator).__name__}"
    assert calibrator.enable_calibrator is True
    assert calibrator.enable_encoder_calibrator is False
    assert calibrator.calibrator_type == "foca"

  def test_load_foca_from_dict_full(self):
    """Load FoCa config from a dict with all fields explicitly set."""
    result = load_configs(
      {
        "cache_config": {
          "enable_foca": True,
          "enable_encoder_foca": True,
          "calibrator_cache_type": "hidden_states",
        },
      },
      return_dict=True,
    )
    calibrator = result["calibrator_config"]
    assert isinstance(calibrator, FoCaCalibratorConfig)
    assert calibrator.enable_calibrator is True
    assert calibrator.enable_encoder_calibrator is True
    assert calibrator.calibrator_cache_type == "hidden_states"

  def test_load_foca_from_yaml(self):
    """Load FoCa config from examples/configs/cache_foca.yaml."""
    yaml_path = os.path.join(
      os.path.dirname(__file__),
      "..",
      "..",
      "examples",
      "configs",
      "cache_foca.yaml",
    )
    result = load_configs(yaml_path, return_dict=True)
    calibrator = result["calibrator_config"]
    assert isinstance(calibrator, FoCaCalibratorConfig)
    assert calibrator.enable_calibrator is True

  def test_foca_mutual_exclusion_with_taylorseer(self):
    """Setting both enable_foca and enable_taylorseer must raise ValueError."""
    with pytest.raises(ValueError, match="mutually exclusive"):
      load_configs(
        {
          "cache_config": {
            "enable_foca": True,
            "enable_taylorseer": True,
          },
        },
        return_dict=True,
      )

  def test_foca_mutual_exclusion_with_dmd(self):
    """Setting both enable_foca and enable_dmd must raise ValueError."""
    with pytest.raises(ValueError, match="mutually exclusive"):
      load_configs(
        {
          "cache_config": {
            "enable_foca": True,
            "enable_dmd": True,
          },
        },
        return_dict=True,
      )

  def test_foca_all_three_mutual_exclusion(self):
    """Setting all three calibrators must raise ValueError."""
    with pytest.raises(ValueError, match="mutually exclusive"):
      load_configs(
        {
          "cache_config": {
            "enable_foca": True,
            "enable_taylorseer": True,
            "enable_dmd": True,
          },
        },
        return_dict=True,
      )

  def test_taylorseer_and_dmd_still_work(self):
    """Regression: TaylorSeer and DMD config loading still functions."""
    result_ts = load_configs(
      {"cache_config": {
        "enable_taylorseer": True,
        "taylorseer_order": 2
      }},
      return_dict=True,
    )
    assert isinstance(result_ts["calibrator_config"], TaylorSeerCalibratorConfig)
    assert result_ts["calibrator_config"].taylorseer_order == 2

    result_dmd = load_configs(
      {"cache_config": {
        "enable_dmd": True,
        "dmd_history": 5
      }},
      return_dict=True,
    )
    assert isinstance(result_dmd["calibrator_config"], DMDCalibratorConfig)
    assert result_dmd["calibrator_config"].dmd_history == 5

  def test_foca_to_kwargs(self):
    """FoCaCalibratorConfig.to_kwargs() returns an empty dict."""
    result = load_configs(
      {"cache_config": {
        "enable_foca": True
      }},
      return_dict=True,
    )
    kwargs = result["calibrator_config"].to_kwargs()
    assert kwargs == {}

  def test_load_cache_config_foca(self):
    """load_cache_config returns correct tuple for FoCa."""
    cache_config, calibrator_config = load_cache_config({"cache_config": {"enable_foca": True}}, )
    assert isinstance(cache_config, DBCacheConfig)
    assert isinstance(calibrator_config, FoCaCalibratorConfig)
