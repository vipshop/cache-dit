"""CPU tests for the MOVA CLI example.

Pins the registration, the env-path mapping and the cache preset — including the warmup clamp that
keeps a short run from spending every step in the warmup phase and reporting a zero-hit run.
"""

import argparse

from cache_dit._utils.examples import _mova_cache_config, _env_path_mapping
from cache_dit._utils.registers import ExampleRegister


def _args(**overrides) -> argparse.Namespace:
  """Build the subset of CLI arguments the example reads.

  :param overrides: Argument overrides.
  :returns: A namespace with the CLI defaults.
  """

  defaults = dict(
    cache=True,
    max_warmup_steps=8,
    residual_diff_threshold=0.12,
    max_cached_steps=-1,
    max_continuous_cached_steps=-1,
  )
  defaults.update(overrides)
  return argparse.Namespace(**defaults)


def test_example_is_registered():
  """`python -m cache_dit.generate list` finds the model."""

  assert "mova_360p" in ExampleRegister._example_registry


def test_env_path_mapping():
  """`MOVA_DIR` overrides the hub id with the local checkpoint root."""

  assert _env_path_mapping["MOVA_DIR"] == "OpenMOSS-Team/MOVA-360p"


def test_cache_config_is_disabled_without_the_flag():
  """No `--cache` means no cache config, not a default one."""

  assert _mova_cache_config(_args(cache=False), steps=8) is None


def test_cache_config_clamps_the_warmup():
  """`--max-warmup-steps` covering every step must be clamped so a hit is still possible."""

  config = _mova_cache_config(_args(max_warmup_steps=8), steps=8)
  assert config.max_warmup_steps == 7
  assert config.num_inference_steps == 8
  assert config.Bn_compute_blocks == 1
  assert config.Fn_compute_blocks == 1
  assert config.enable_separate_cfg is True


def test_cache_config_keeps_a_short_warmup():
  """A warmup below the step count is used verbatim."""

  config = _mova_cache_config(_args(max_warmup_steps=1), steps=8)
  assert config.max_warmup_steps == 1


def test_cache_config_never_zero_warmup():
  """A single-step run still keeps one warmup step (``-0`` and ``0`` are both invalid)."""

  config = _mova_cache_config(_args(max_warmup_steps=8), steps=1)
  assert config.max_warmup_steps == 1
