"""Checks for the CausVid-Wan `cache_dit.generate` registration.

Usage:
  python -m pytest tests/utils/test_causvid_wan_example.py -q
"""

import argparse

import pytest

from cache_dit._utils import causvid_wan_example as public_example
from cache_dit._utils import registers
from cache_dit._utils.examples import (
  _env_path_mapping,
  _path_env_mapping,
  causvid_wan_example,
)
from cache_dit._utils.registers import ExampleRegister, ExampleType


@pytest.fixture(autouse=True)
def _cpu_distributed(monkeypatch):
  """`Example.__init__` asks the distributed helpers whether this run is parallel."""
  monkeypatch.setattr(registers, "maybe_init_distributed", lambda _args: (0, "cpu"))


@pytest.fixture(autouse=True)
def _clear_model_env(monkeypatch):
  """Developer shells may export the model dir; the default assertions need it absent."""
  monkeypatch.delenv("CAUSVID_DIR", raising=False)
  monkeypatch.delenv("WAN_DIR", raising=False)


def _factory_args(**overrides) -> argparse.Namespace:
  values = dict(
    model_path=None,
    height=None,
    width=None,
    num_frames=None,
    num_inference_steps=None,
    parallel_type=None,
    config_path=None,
  )
  values.update(overrides)
  return argparse.Namespace(**values)


def test_example_is_registered():
  assert "causvid_wan" in ExampleRegister.list_examples()
  assert ExampleRegister.get_default("causvid_wan") == "tianweiy/CausVid"


def test_public_export_is_not_a_stub():
  # `_safe_import` degrades to a stub on a broken import chain.
  assert public_example is causvid_wan_example


def test_env_mapping_is_bidirectional():
  assert _env_path_mapping["CAUSVID_DIR"] == "tianweiy/CausVid"
  assert _path_env_mapping["tianweiy/CausVid"] == "CAUSVID_DIR"


def test_defaults_follow_the_released_recipe():
  example = causvid_wan_example(_factory_args())
  data = example.input_data

  assert example.init_config.task_type == ExampleType.T2V
  assert example.init_config.pipeline_class.__name__ == "CausVidPipeline"
  # The base checkpoint supplies the text encoder / tokenizer / VAE; the student comes from
  # CAUSVID_DIR.
  assert example.init_config.model_name_or_path == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
  assert data.height == 480
  assert data.width == 832
  assert data.num_frames == 81
  assert data.num_inference_steps == 3
  assert data.guidance_scale == 1.0
  assert data.extra_input_kwargs == {"output_type": "pil"}


def test_cli_geometry_overrides():
  example = causvid_wan_example(_factory_args(num_frames=21, height=384, width=640))
  data = example.input_data

  assert data.num_frames == 21
  assert data.height == 384
  assert data.width == 640


def test_transformer_factory_asks_for_the_student_checkpoint():
  from cache_dit._utils.examples import _causvid_wan_transformer_factory

  factory = _causvid_wan_transformer_factory("/definitely/missing", "/also/missing")
  with pytest.raises(FileNotFoundError, match="autoregressive_checkpoint"):
    factory()
