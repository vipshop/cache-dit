"""Contract tests for `cache_dit.models.pipelines.env`.

The module is the one place a model pipeline's environment variables are read. Today that is the
``CACHE_DIT_FUSED_OPS`` switch, and it is tri-state: unset is not "on", it means "off until a
cache-dit entry point installs the fused transformer". Both halves fail silently when wrong - a
family that stays off looks exactly like one that is merely slow, and a family that turns itself on
has already changed the numerics before anyone measures.
"""

import importlib
import os
import pathlib
import subprocess
import sys

import pytest

from cache_dit.models.pipelines import env

_FAMILIES = (
  "cache_dit.models.pipelines.flux2_klein.fused_ops",
  "cache_dit.models.pipelines.flux3_klein.fused_ops",
  "cache_dit.models.pipelines.qwen_image21.fused_ops",
  "cache_dit.models.pipelines.boogu_image.fused_ops",
)
# The names this variable replaced: the per-model switches, plus their per-operator allow-lists and
# the two FLUX.2 sub-feature knobs. They must not come back one model at a time.
_REMOVED = (
  "CACHE_DIT_FLUX2_KLEIN_FUSED_OPS",
  "CACHE_DIT_FLUX3_KLEIN_FUSED_OPS",
  "CACHE_DIT_QWEN_IMAGE21_FUSED_OPS",
  "CACHE_DIT_BOOGU_FUSED_OPS",
  "CACHE_DIT_FLUX2_KLEIN_OPS",
  "CACHE_DIT_FLUX3_KLEIN_OPS",
  "CACHE_DIT_QWEN_IMAGE21_OPS",
  "CACHE_DIT_FLUX2_KLEIN_ROPE_CACHE",
  "CACHE_DIT_FLUX2_KLEIN_JOIN",
)


def test_the_switch_is_one_unprefixed_variable():
  assert env.FUSED_OPS_ENV == "CACHE_DIT_FUSED_OPS"


def test_unset_is_neither_on_nor_off(monkeypatch):
  monkeypatch.delenv(env.FUSED_OPS_ENV, raising=False)

  assert env.fused_ops_from_env() is None
  # Unset does not enable the operators at import, and does not forbid the install from doing it.
  assert env.fused_ops_default() is False
  assert env.fused_ops_enabled_on_install() is True


def test_only_explicit_falsey_values_turn_the_switch_off(monkeypatch):
  for value in ("0", "false", "off", "no", " OFF "):
    monkeypatch.setenv(env.FUSED_OPS_ENV, value)
    assert env.fused_ops_from_env() is False
    assert env.fused_ops_default() is False
    assert env.fused_ops_enabled_on_install() is False

  # Anything else is "on": a typo must never turn a kernel off silently.
  for value in ("1", "on", "yes", "", "true"):
    monkeypatch.setenv(env.FUSED_OPS_ENV, value)
    assert env.fused_ops_from_env() is True
    assert env.fused_ops_default() is True
    assert env.fused_ops_enabled_on_install() is True


@pytest.mark.parametrize("name", _FAMILIES)
def test_install_activates_the_family_unless_the_environment_forbids_it(monkeypatch, name):
  module = importlib.import_module(name)
  # The switch is read at call time, so the test can drive the module attribute directly;
  # `monkeypatch` restores it, which matters because the install writes the same global.
  monkeypatch.setattr(module, "FUSED_OPS_ENABLED", False)

  monkeypatch.delenv(env.FUSED_OPS_ENV, raising=False)
  module.activate_fused_ops()
  assert module.FUSED_OPS_ENABLED is True

  monkeypatch.setattr(module, "FUSED_OPS_ENABLED", False)
  monkeypatch.setenv(env.FUSED_OPS_ENV, "0")
  module.activate_fused_ops()
  assert module.FUSED_OPS_ENABLED is False


@pytest.mark.parametrize("name", _FAMILIES)
def test_a_clean_process_imports_the_family_disabled(name):
  """Unset means off at import - the half an in-process assertion cannot see.

  Every kernel test forces the flag on when it is imported, and pytest imports every collected
  module before the first test runs, so asserting the default from inside this process would be
  measuring collection order. A child interpreter with the variable removed is the honest check,
  and it is the assertion that fails if a family goes back to `FUSED_OPS_ENABLED = True`.
  """

  environment = {key: value for key, value in os.environ.items() if key != env.FUSED_OPS_ENV}
  completed = subprocess.run(
    [sys.executable, "-c", f"import {name} as m; print(m.FUSED_OPS_ENABLED)"],
    capture_output=True,
    text=True,
    env=environment,
    check=True)

  assert completed.stdout.strip().splitlines()[-1] == "False"


@pytest.mark.parametrize("name", _FAMILIES)
def test_no_family_keeps_a_switch_of_its_own(name):
  module = importlib.import_module(name)

  # No per-family constant, no per-family parser, and no model-prefixed flag attribute.
  for attribute in ("_ENV_SWITCH", "_BOOGU_FUSED_OPS_ENV", "_BOOGU_FUSED_OPS_ENABLED"):
    assert not hasattr(module, attribute), attribute


def test_the_removed_per_model_names_are_gone_from_the_package():
  root = pathlib.Path(env.__file__).parents[2]
  offenders = [
    str(path.relative_to(root)) for path in root.rglob("*.py")
    if any(name in path.read_text(encoding="utf-8") for name in _REMOVED)
  ]

  assert offenders == [], f"per-model fused-ops switches reintroduced in: {offenders}"
