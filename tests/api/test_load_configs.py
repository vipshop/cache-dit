import pytest
import cache_dit

CONFIG = ["api/config.yaml"]


@pytest.mark.parametrize("config", CONFIG)
def test_load_configs(config):
    configs = cache_dit.load_configs(config)
    assert "cache_config" in configs
    assert "calibrator_config" in configs
    assert "parallelism_config" in configs
    print("Loaded configs:", configs, flush=True)
