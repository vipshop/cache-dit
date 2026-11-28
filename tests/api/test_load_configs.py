from pathlib import Path

import pytest
import cache_dit

from cache_dit.caching.load_configs import load_cache_config

CONFIG = ["api/config.yaml"]


@pytest.mark.parametrize("config", CONFIG)
def test_load_configs(config):
  configs = cache_dit.load_configs(config)
  assert "cache_config" in configs
  assert "calibrator_config" in configs
  assert configs["parallelism_config"].local_weight_shards is True
  assert "attention_backend" in configs
  assert "quantize_config" in configs
  print("Loaded configs:", configs, flush=True)


def test_load_seacache_yaml_keeps_dbcache():
  """The SeaCache YAML selects the optional DBCache probe with global geometry."""

  path = Path(__file__).resolve().parents[2] / "examples" / "configs" / "cache_seacache.yaml"
  configs = cache_dit.load_configs(str(path))
  config = configs["cache_config"]
  assert str(config.cache_type) == str(cache_dit.CacheType.DBCache)
  assert config.enable_seacache is True
  assert config.seacache_mode == "flow"
  assert config.seacache_feature_shape == [64, 64]
  assert config.seacache_power_exp == 2.0


def test_load_softcache_yaml():
  """The SoftCache YAML enables SoftCache on DBCache + SeaCache with Residual Motion."""

  path = Path(__file__).resolve().parents[2] / "examples" / "configs" / "cache_softcache.yaml"
  configs = cache_dit.load_configs(str(path))
  config = configs["cache_config"]
  assert str(config.cache_type) == str(cache_dit.CacheType.DBCache)
  assert config.enable_seacache is True and config.seacache_mode == "flow"
  assert config.enable_softcache is True
  assert config.softcache_tau == 0.5 and config.softcache_jf_max == 8.0
  assert config.max_continuous_cached_steps == -1
  calibrator = configs["calibrator_config"]
  assert calibrator is not None and calibrator.enable_calibrator is True
  assert isinstance(calibrator, cache_dit.ResidualMotionCalibratorConfig)
  assert calibrator.residual_motion_beta == 1.0


def test_seacache_fields_survive_refresh_reset():
  """The refresh/reset loader retains explicit SeaCache fields without a new config branch."""

  config, _ = load_cache_config(
    {
      "cache_config": {
        "enable_seacache": True,
        "seacache_mode": "vp",
        "seacache_feature_shape": [2, 8, 8],
        "seacache_power_exp": 3.0,
        "seacache_ab_schedule": [[0.8, 0.2], [0.7, 0.3]],
      },
    },
    reset=True,
  )
  assert config.enable_seacache is True
  assert config.seacache_mode == "vp"
  assert config.seacache_feature_shape == [2, 8, 8]
  assert config.seacache_power_exp == 3.0
  assert config.seacache_ab_schedule == [[0.8, 0.2], [0.7, 0.3]]
