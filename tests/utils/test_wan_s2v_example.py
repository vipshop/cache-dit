"""Checks for the Wan2.2-S2V `cache_dit.generate` registration.

Wan2.2-S2V names its frame count `num_frames_per_chunk`, not `num_frames`, so the example maps
`--num-frames` itself and has to keep the generic forwarding from leaking a `num_frames` kwarg into
a pipeline that does not accept it.

Usage:
  python -m pytest tests/utils/test_wan_s2v_example.py -q
"""

import argparse

import pytest

from cache_dit._utils import registers, wan_s2v_example as public_example
from cache_dit._utils.examples import (
  _env_path_mapping,
  _path_env_mapping,
  wan_s2v_example,
)
from cache_dit._utils.registers import ExampleRegister, ExampleType


@pytest.fixture(autouse=True)
def _cpu_distributed(monkeypatch):
  """`Example.__init__` asks the distributed helpers whether this run is parallel."""
  monkeypatch.setattr(registers, "maybe_init_distributed", lambda _args: (0, "cpu"))


@pytest.fixture(autouse=True)
def _clear_model_env(monkeypatch):
  """Developer shells may export the model dir; the default assertions need it absent."""
  monkeypatch.delenv("WAN_2_2_S2V_DIR", raising=False)
  monkeypatch.delenv("WAN_2_2_S2V_DEMO_DIR", raising=False)


def _factory_args(**overrides) -> argparse.Namespace:
  values = dict(
    model_path=None,
    height=None,
    width=None,
    num_frames=None,
    num_inference_steps=None,
    image_path=None,
    audio_path=None,
  )
  values.update(overrides)
  return argparse.Namespace(**values)


def test_example_is_registered():
  examples = ExampleRegister.list_examples()
  assert "wan2.2_s2v" in examples
  assert ExampleRegister.get_default("wan2.2_s2v") == "Wan-AI/Wan2.2-S2V-14B-Diffusers"


def test_public_export_is_not_a_stub():
  # `_safe_import` degrades to a stub on a broken import chain.
  assert public_example is wan_s2v_example


def test_env_mapping_round_trips():
  assert _env_path_mapping["WAN_2_2_S2V_DIR"] == "Wan-AI/Wan2.2-S2V-14B-Diffusers"
  assert _path_env_mapping["Wan-AI/Wan2.2-S2V-14B-Diffusers"] == "WAN_2_2_S2V_DIR"


def test_defaults_follow_the_official_recipe():
  example = wan_s2v_example(_factory_args())
  assert example.init_config.task_type == ExampleType.S2V
  assert example.input_data.height == 480
  assert example.input_data.width == 832
  assert example.input_data.guidance_scale == 4.5
  assert example.input_data.num_inference_steps == 40
  # 84 chunk frames minus the 3 warm-up frames the first chunk drops.
  assert example.input_data.extra_input_kwargs["num_frames_per_chunk"] == 84
  assert example.input_data.extra_input_kwargs["sampling_fps"] == 16
  # ndarray frames: imageio (the video writer) rejects PIL frames.
  assert example.input_data.extra_input_kwargs["output_type"] == "np"
  # The pipeline center-crops the reference image itself.
  assert example.input_data.skip_image_resize is True


def test_num_frames_maps_to_the_chunk_length():
  """`--num-frames` must land on `num_frames_per_chunk` and never on `num_frames`."""
  example = wan_s2v_example(_factory_args(num_frames=48))
  assert example.input_data.extra_input_kwargs["num_frames_per_chunk"] == 48
  assert example.input_data.num_frames is None


def test_cli_overrides_are_honoured():
  example = wan_s2v_example(_factory_args(height=256, width=256, num_inference_steps=4))
  assert example.input_data.height == 256
  assert example.input_data.width == 256
  assert example.input_data.num_inference_steps == 4


def test_audio_is_a_required_input():
  """The waveform is the generation input, not a soundtrack: it has to reach the pipeline."""
  example = wan_s2v_example(_factory_args())
  audio = example.input_data.extra_input_kwargs["audio"]
  assert audio.ndim == 1
  assert example.input_data.extra_input_kwargs["sampling_rate"] == 16000
  import numpy as np

  assert isinstance(audio, np.ndarray)


def test_synthetic_placeholder_covers_the_requested_chunk():
  """Without any asset the placeholder is sized from the chunk length, not from a fixed duration."""
  from cache_dit._utils.examples import _wan_s2v_synthetic_audio

  audio = _wan_s2v_synthetic_audio(84, 16)
  assert audio.shape[0] == 16000 * 84 // 16
