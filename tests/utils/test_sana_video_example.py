"""Regression tests for the SANA-Video (2B 480p) `generate` CLI examples.

Both entries load the released diffusers checkpoint, so the factories must resolve to diffusers'
`SanaVideoPipeline` / `SanaImageToVideoPipeline` and keep the released recipe (480x832, `frames=81`,
CFG 6, 50 DPM steps). The pipeline names its frame count `frames`, not `num_frames`, which is the
one wiring detail that silently falls back to the pipeline default when it is dropped.

Usage:
  python -m pytest tests/utils/test_sana_video_example.py -q
"""

import argparse

import pytest

from cache_dit._utils import registers
from cache_dit._utils import sana_video_example as public_example
from cache_dit._utils.examples import (
  _env_path_mapping,
  _path_env_mapping,
  _SANA_VIDEO_NEGATIVE_PROMPT,
  _SANA_VIDEO_PROMPT,
  sana_video_example,
  sana_video_i2v_example,
)
from cache_dit._utils.registers import ExampleRegister, ExampleType

_DEFAULT = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"


@pytest.fixture(autouse=True)
def _cpu_distributed(monkeypatch):
  """`Example.__init__` asks the distributed helpers whether this run is parallel."""

  monkeypatch.setattr(registers, "maybe_init_distributed", lambda _args: (0, "cpu"))


@pytest.fixture(autouse=True)
def _clear_model_env(monkeypatch):
  """Developer shells may export the model dir; the default assertions need it absent."""

  monkeypatch.delenv("SANA_VIDEO_DIR", raising=False)


def _data_args(**overrides) -> argparse.Namespace:
  """Every attribute `ExampleInputData.data()` reads, so the forwarding can be asserted."""

  values = dict(
    model_path=None,
    image_path=None,
    mask_image_path=None,
    height=None,
    width=None,
    input_height=None,
    input_width=None,
    num_frames=None,
    num_inference_steps=None,
    prompt=None,
    negative_prompt=None,
    skip_negative_prompt=False,
    extra_input_kwargs=None,
    generator_device=None,
    seed=None,
    parallel_type=None,
    parallel_text_encoder=False,
    parallel_vae=False,
    config_path=None,
  )
  values.update(overrides)
  return argparse.Namespace(**values)


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
  assert "sana_video" in names
  assert "sana_video_i2v" in names
  assert ExampleRegister.get_default("sana_video") == _DEFAULT
  assert ExampleRegister.get_default("sana_video_i2v") == _DEFAULT


def test_public_reexport_points_at_the_same_factory():
  assert public_example is sana_video_example


def test_env_mapping_and_default_path():
  assert _env_path_mapping["SANA_VIDEO_DIR"] == _DEFAULT
  assert _path_env_mapping[_DEFAULT] == "SANA_VIDEO_DIR"

  example = sana_video_example(_factory_args())
  assert example.init_config.model_name_or_path == _DEFAULT

  # SANA-Video 2.0 must keep its own mapping: the two families share a name prefix only.
  assert _env_path_mapping["SANA_VIDEO2_DIR"] != _DEFAULT


def test_t2v_defaults_follow_the_release_recipe():
  from diffusers import SanaVideoPipeline

  example = sana_video_example(_factory_args())
  assert example.init_config.pipeline_class is SanaVideoPipeline
  assert example.init_config.task_type == ExampleType.T2V

  data = example.input_data
  assert data.height == 480
  assert data.width == 832
  assert data.num_inference_steps == 50
  assert data.guidance_scale == 6.0
  assert data.prompt == _SANA_VIDEO_PROMPT
  assert data.negative_prompt == _SANA_VIDEO_NEGATIVE_PROMPT
  # The pipeline takes `frames`; `num_frames` would be swallowed as an unexpected keyword.
  assert data.use_num_frames is False
  assert data.extra_input_kwargs["frames"] == 81


def test_i2v_defaults_match_the_t2v_canvas(monkeypatch):
  from diffusers import SanaImageToVideoPipeline

  monkeypatch.setattr("cache_dit._utils.examples.load_image", lambda *args, **kwargs: "IMAGE")
  example = sana_video_i2v_example(_factory_args(image_path="dummy.png"))

  assert example.init_config.pipeline_class is SanaImageToVideoPipeline
  assert example.init_config.task_type == ExampleType.I2V
  data = example.input_data
  assert data.image == "IMAGE"
  assert (data.height, data.width) == (480, 832)
  assert data.extra_input_kwargs["frames"] == 81


def test_frames_flag_reaches_the_pipeline_kwargs(monkeypatch):
  """`--num-frames` must land as `frames`; a stray `num_frames` would be an unexpected keyword."""

  from PIL import Image

  # The factory and `ExampleInputData.data()` each imported `load_image` into their own module.
  fake_loader = lambda *args, **kwargs: Image.new("RGB", (8, 8))
  monkeypatch.setattr("cache_dit._utils.examples.load_image", fake_loader)
  monkeypatch.setattr(registers, "load_image", fake_loader)

  for requested, expected in ((None, 81), (77, 77)):
    args = _data_args(image_path="dummy.png", num_frames=requested)
    example = sana_video_i2v_example(args)
    kwargs = example.input_data.data(args)
    assert kwargs["frames"] == expected
    assert "num_frames" not in kwargs
