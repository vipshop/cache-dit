"""Regression tests for the SANA-Video 2.0 `generate` CLI examples.

The checkpoint is native (merged `.pth` + `config.yaml`), so the factory has to resolve to the
vendored pipeline class rather than a `_safe_import` stub, and the defaults have to stay on the
release recipe (720p canvas, `8n+1` frames, CFG 8, flow shift 12). Both the text-to-video and the
first-frame-conditioned entry are covered.

Usage:
  python -m pytest tests/utils/test_sana_video2_example.py -q
"""
import argparse

import pytest

from cache_dit._utils import registers
from cache_dit._utils import sana_video2_example as public_example
from cache_dit._utils.examples import (
  _env_path_mapping,
  _path_env_mapping,
  _SANA_VIDEO2_PROMPT,
  sana_video2_example,
  sana_video2_ti2v_example,
)
from cache_dit._utils.registers import ExampleRegister, ExampleType


@pytest.fixture(autouse=True)
def _cpu_distributed(monkeypatch):
  """`Example.__init__` asks the distributed helpers whether this run is parallel."""
  monkeypatch.setattr(registers, "maybe_init_distributed", lambda _args: (0, "cpu"))


@pytest.fixture(autouse=True)
def _clear_model_env(monkeypatch):
  """Developer shells may export the model dir; the default assertions need it absent."""
  monkeypatch.delenv("SANA_VIDEO2_DIR", raising=False)


def _factory_args(**overrides) -> argparse.Namespace:
  values = dict(
    model_path=None,
    image_path=None,
    height=None,
    width=None,
    num_frames=None,
    num_inference_steps=None,
    parallel_type=None,
    parallel_text_encoder=False,
    parallel_vae=False,
    config_path=None,
  )
  values.update(overrides)
  return argparse.Namespace(**values)


def test_examples_are_registered():
  names = ExampleRegister.list_examples()
  assert "sana_video2" in names
  assert "sana_video2_ti2v" in names
  assert (
    ExampleRegister.get_default("sana_video2") == "Efficient-Large-Model/SANA-Video_2.0_5B_720p")
  assert public_example is sana_video2_example


def test_env_mapping_points_at_the_release():
  assert _env_path_mapping["SANA_VIDEO2_DIR"] == "Efficient-Large-Model/SANA-Video_2.0_5B_720p"
  assert _path_env_mapping["Efficient-Large-Model/SANA-Video_2.0_5B_720p"] == "SANA_VIDEO2_DIR"


def test_t2v_defaults_follow_the_release_recipe():
  example = sana_video2_example(_factory_args())
  assert example.init_config.task_type == ExampleType.T2V
  assert (example.init_config.model_name_or_path == "Efficient-Large-Model/SANA-Video_2.0_5B_720p")
  data = example.input_data
  assert (data.height, data.width) == (736, 1280)
  assert data.num_frames == 97
  assert (data.num_frames - 1) % 8 == 0
  assert data.num_inference_steps == 30
  assert data.guidance_scale == 8.0
  assert data.extra_input_kwargs["flow_shift"] == 12.0
  assert data.extra_input_kwargs["motion_score"] == 10
  assert data.prompt == _SANA_VIDEO2_PROMPT


def test_cli_overrides_replace_the_canvas():
  example = sana_video2_example(_factory_args(height=512, width=832, num_frames=17))
  data = example.input_data
  assert (data.height, data.width) == (512, 832)
  assert data.num_frames == 17


def test_ti2v_example_conditions_on_a_first_frame(tmp_path):
  # Always feed a local frame: the fallback asset is fetched from the hub, which is not available in
  # a unit-test environment.
  from PIL import Image

  frame = tmp_path / "frame.png"
  Image.new("RGB", (64, 32), (255, 0, 0)).save(frame)
  example = sana_video2_ti2v_example(_factory_args(image_path=str(frame)))
  assert example.init_config.task_type == ExampleType.I2V
  data = example.input_data
  assert data.image is not None
  assert data.skip_image_resize is True
  assert data.extra_input_kwargs["flow_shift"] == 12.0
