"""Checks for the LingBot-Video `cache_dit.generate` registration.

Usage:
  python -m pytest tests/utils/test_lingbot_video_example.py -q
"""

import argparse

import pytest

from cache_dit._utils import lingbot_video_example as public_example
from cache_dit._utils import registers
from cache_dit._utils.examples import (
  _env_path_mapping,
  _path_env_mapping,
  lingbot_video_example,
)
from cache_dit._utils.registers import ExampleRegister, ExampleType


@pytest.fixture(autouse=True)
def _cpu_distributed(monkeypatch):
  """`Example.__init__` asks the distributed helpers whether this run is parallel."""
  monkeypatch.setattr(registers, "maybe_init_distributed", lambda _args: (0, "cpu"))


@pytest.fixture(autouse=True)
def _clear_model_env(monkeypatch):
  """Developer shells may export the model dir; the default assertions need it absent."""
  monkeypatch.delenv("LINGBOT_VIDEO_DIR", raising=False)


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
  examples = ExampleRegister.list_examples()
  assert "lingbot_video" in examples
  assert ExampleRegister.get_default("lingbot_video") == "robbyant/lingbot-video-dense-1.3b"


def test_public_export_is_not_a_stub():
  # `_safe_import` degrades to an `import_error_adapter`-style stub on a broken import chain.
  assert public_example is lingbot_video_example


def test_env_mapping_round_trips():
  assert _env_path_mapping["LINGBOT_VIDEO_DIR"] == "robbyant/lingbot-video-dense-1.3b"
  assert _path_env_mapping["robbyant/lingbot-video-dense-1.3b"] == "LINGBOT_VIDEO_DIR"


def test_defaults_follow_the_official_recipe():
  example = lingbot_video_example(_factory_args())
  assert example.init_config.task_type == ExampleType.T2V
  assert example.input_data.height == 480
  assert example.input_data.width == 832
  assert example.input_data.num_frames == 81
  assert example.input_data.num_inference_steps == 40
  assert example.input_data.guidance_scale == 3.0
  # The writer takes PIL frames; the pipeline default returns `(T, H, W, C)` arrays.
  assert example.input_data.extra_input_kwargs["output_type"] == "pil"
  # LingBot-Video conditions on the structured JSON caption form, not on a plain sentence.
  assert example.input_data.prompt.lstrip().startswith('{"comprehensive_description"')


def test_cli_overrides_are_honoured():
  example = lingbot_video_example(
    _factory_args(height=256, width=256, num_frames=9, num_inference_steps=4))
  assert example.input_data.height == 256
  assert example.input_data.width == 256
  assert example.input_data.num_frames == 9
  assert example.input_data.num_inference_steps == 4
