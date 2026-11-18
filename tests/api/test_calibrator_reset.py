import gc

import pytest
import torch

import cache_dit
from cache_dit import (
  BlockAdapter,
  DBCacheConfig,
  ForwardPattern,
)
from cache_dit.caching.cache_contexts.calibrators import Calibrator, TaylorSeerCalibrator
from cache_dit.caching.load_configs import load_cache_config
from utils import RandPipeline

# `cache_dit.refresh_context(transformer, **shorthand)` is the only `reset=True` caller. It rebuilds
# the config through `CalibratorConfig.empty()`, which nulls every non-constant field including
# `calibrator_kwargs`, so every `to_kwargs()` has to tolerate `None`.
_CALIBRATORS = {
  "enable_taylorseer": "TaylorSeerCalibrator",
  "enable_dmd": "DMDCalibrator",
  "enable_foca": "FoCaCalibrator",
  "enable_svd": "SVDCalibrator",
  "enable_hicache": "HiCacheCalibrator",
}


@pytest.mark.parametrize("switch,expected", _CALIBRATORS.items())
def test_reset_path_still_builds_the_calibrator(switch, expected):
  """A reset config carries `calibrator_kwargs=None`; the factory must still build."""
  _, config = load_cache_config({"cache_config": {switch: True}}, reset=True)
  assert config.calibrator_kwargs is None
  assert type(Calibrator(config)).__name__ == expected


def test_base_config_survives_reset_too():
  """The base `CalibratorConfig.to_kwargs()` is reachable through the same reset path."""
  from cache_dit.caching.cache_contexts import CalibratorConfig

  config = CalibratorConfig()
  config.reset()
  assert config.calibrator_kwargs is None
  assert config.to_kwargs() == {}


@pytest.mark.parametrize(
  "switch,field,value,attr",
  [
    ("enable_taylorseer", "taylorseer_order", 2, "n_derivatives"),
    ("enable_dmd", "dmd_history", 4, "history"),
    ("enable_svd", "svd_tau", 0.8, "tau"),
    ("enable_hicache", "hicache_sigma", 0.4, "sigma"),
  ],
)
def test_reset_path_preserves_explicit_params(switch, field, value, attr):
  """Re-filling after `empty()` must reach the calibrator, not just the config."""
  _, config = load_cache_config({"cache_config": {switch: True, field: value}}, reset=True)
  calibrator = Calibrator(config)
  assert getattr(calibrator, attr) == value


def test_refresh_context_accepts_calibrator_shorthand():
  """The public `refresh_context` entry rebuilds through the reset path and keeps working.

  `enable_cache` deliberately starts WITHOUT a calibrator: only then does the refresh replace the
  `calibrator_config` object wholesale (the buggy branch), instead of merging into an existing
  config whose `calibrator_kwargs` is still `{}` and hides the crash.
  """
  gc.collect()
  torch.manual_seed(0)
  pipe = RandPipeline(pattern=ForwardPattern.Pattern_0)
  transformer = pipe.transformer
  adapter = cache_dit.enable_cache(
    BlockAdapter(
      transformer=transformer,
      blocks=transformer.transformer_blocks,
      forward_pattern=ForwardPattern.Pattern_0,
    ),
    cache_config=DBCacheConfig(max_warmup_steps=1, residual_diff_threshold=0.99),
  )
  cache_dit.refresh_context(transformer, enable_taylorseer=True, taylorseer_order=2)
  output = pipe(
    torch.randn(1, 64, 16),
    encoder_hidden_states=torch.randn(1, 64, 16),
    num_inference_steps=2,
  )
  sample = output.sample if hasattr(output, "sample") else output
  assert torch.isfinite(sample).all()
  assert "T1O2" in cache_dit.strify(transformer)
  # The context object exists right after the refresh; only `_current_context` is bound lazily,
  # in the forward pass (`pattern_base` sets it), which is why `get_context()` needs one run.
  calibrator, _ = transformer._context_manager.get_context().get_calibrators()
  assert isinstance(calibrator, TaylorSeerCalibrator)
  assert calibrator.n_derivatives == 2
  cache_dit.disable_cache(adapter)
  del pipe, adapter
  gc.collect()
