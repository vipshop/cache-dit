import pytest
import yaml
import tempfile
import os

from cache_dit.caching.load_configs import load_configs, load_cache_config
from cache_dit.caching.cache_contexts import (
  DMDCalibratorConfig,
  TaylorSeerCalibratorConfig,
  DBCacheConfig,
)


class TestDMDConfigLoading:

  def test_load_dmd_from_dict_basic(self):
    """Load DMD config from a dict with only enable_dmd set, verify defaults."""
    result = load_configs(
      {"cache_config": {
        "enable_dmd": True
      }},
      return_dict=True,
    )
    calibrator = result["calibrator_config"]
    assert isinstance(calibrator, DMDCalibratorConfig), \
        f"Expected DMDCalibratorConfig, got {type(calibrator).__name__}"
    assert calibrator.enable_calibrator is True
    assert calibrator.enable_encoder_calibrator is True  # follows enable_dmd
    assert calibrator.dmd_history == 6
    assert calibrator.dmd_rank == 0
    assert calibrator.dmd_ridge == 1e-8
    assert calibrator.dmd_svd_precision == "medium"
    assert calibrator.calibrator_type == "dmd"

  def test_load_dmd_from_dict_full(self):
    """Load DMD config from a dict with all fields explicitly set."""
    result = load_configs(
      {
        "cache_config": {
          "enable_dmd": True,
          "enable_encoder_dmd": False,
          "calibrator_cache_type": "hidden_states",
          "dmd_history": 4,
          "dmd_rank": 2,
          "dmd_ridge": 1e-6,
          "dmd_svd_precision": "high",
        },
      },
      return_dict=True,
    )
    calibrator = result["calibrator_config"]
    assert isinstance(calibrator, DMDCalibratorConfig)
    assert calibrator.enable_calibrator is True
    assert calibrator.enable_encoder_calibrator is False
    assert calibrator.calibrator_cache_type == "hidden_states"
    assert calibrator.dmd_history == 4
    assert calibrator.dmd_rank == 2
    assert calibrator.dmd_ridge == 1e-6
    assert calibrator.dmd_svd_precision == "high"

  def test_load_dmd_from_yaml(self):
    """Load DMD config from examples/configs/cache_dmd.yaml."""
    yaml_path = os.path.join(
      os.path.dirname(__file__),
      "..",
      "..",
      "examples",
      "configs",
      "cache_dmd.yaml",
    )
    result = load_configs(yaml_path, return_dict=True)
    calibrator = result["calibrator_config"]
    assert isinstance(calibrator, DMDCalibratorConfig)
    assert calibrator.enable_calibrator is True
    assert calibrator.dmd_history == 6
    assert calibrator.dmd_svd_precision == "medium"

  def test_load_dmd_from_temp_yaml(self):
    """Load DMD config from a temporary YAML file to avoid fs dependency."""
    yaml_content = {
      "cache_config": {
        "enable_dmd": True,
        "dmd_history": 3,
        "dmd_svd_precision": "low",
        "max_warmup_steps": 4,
      },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
      yaml.dump(yaml_content, f)
      tmp_path = f.name

    try:
      result = load_configs(tmp_path, return_dict=True)
      calibrator = result["calibrator_config"]
      cache = result["cache_config"]
      assert isinstance(calibrator, DMDCalibratorConfig)
      assert calibrator.dmd_history == 3
      assert calibrator.dmd_svd_precision == "low"
      assert isinstance(cache, DBCacheConfig)
      assert cache.max_warmup_steps == 4
    finally:
      os.unlink(tmp_path)

  def test_load_cache_config_dmd(self):
    """load_cache_config returns correct tuple for DMD."""
    cache_config, calibrator_config = load_cache_config(
      {"cache_config": {
        "enable_dmd": True,
        "dmd_history": 5
      }}, )
    assert isinstance(cache_config, DBCacheConfig)
    assert isinstance(calibrator_config, DMDCalibratorConfig)
    assert calibrator_config.dmd_history == 5

  def test_dmd_and_taylorseer_mutual_exclusion(self):
    """Setting both enable_dmd and enable_taylorseer must raise ValueError."""
    with pytest.raises(ValueError, match="mutually exclusive"):
      load_configs(
        {
          "cache_config": {
            "enable_taylorseer": True,
            "enable_dmd": True,
          },
        },
        return_dict=True,
      )

  def test_taylorseer_still_works(self):
    """Regression: TaylorSeer config loading still functions."""
    result = load_configs(
      {
        "cache_config": {
          "enable_taylorseer": True,
          "taylorseer_order": 2,
        },
      },
      return_dict=True,
    )
    calibrator = result["calibrator_config"]
    assert isinstance(calibrator, TaylorSeerCalibratorConfig)
    assert calibrator.taylorseer_order == 2

  def test_dmd_to_kwargs(self):
    """DMDCalibratorConfig.to_kwargs() maps fields to DMDCalibrator kwargs."""
    result = load_configs(
      {"cache_config": {
        "enable_dmd": True,
        "dmd_history": 7
      }},
      return_dict=True,
    )
    kwargs = result["calibrator_config"].to_kwargs()
    assert kwargs["history"] == 7
    assert kwargs["rank"] == 0
    assert kwargs["ridge"] == 1e-8
    assert kwargs["svd_precision"] == "medium"
