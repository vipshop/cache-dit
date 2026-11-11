"""CLI registration tests for the MAGI-2 Preview example.

`python -m cache_dit.generate magi2_t2va` must resolve without importing the 114B model code, and the
example must expose the released task type (video **with** audio) plus the `MAGI2_DIR` override.
"""
import argparse

from cache_dit.cli.examples import _env_path_mapping, magi2_i2va_example, magi2_t2va_example
from cache_dit.cli.registers import ExampleRegister, ExampleType


def _args(**overrides) -> argparse.Namespace:
  args = argparse.Namespace(
    height=None,
    width=None,
    num_inference_steps=None,
    image_path=None,
    model_path=None,
    num_frames=None,
    parallel_type=None,
    config_path=None,
  )
  for key, value in overrides.items():
    setattr(args, key, value)
  return args


def test_examples_are_registered():
  """Both entry points must be discoverable by name."""

  names = ExampleRegister.list_examples()
  assert "magi2_t2va" in names
  assert "magi2_i2va" in names


def test_env_path_mapping_uses_the_checkpoint_root_variable():
  """`MAGI2_DIR` is the documented override for the checkpoint root."""

  assert _env_path_mapping["MAGI2_DIR"] == "sand-ai/MAGI-2-preview"


def test_t2va_example_defaults():
  """The task type must carry audio, and the defaults must stay in development geometry."""

  example = magi2_t2va_example(_args())
  assert example.init_config.task_type == ExampleType.T2VA
  assert example.input_data.height % 16 == 0
  assert example.input_data.width % 16 == 0
  assert example.input_data.num_inference_steps == 4
  # MAGI-2 names its geometry `latent_frames` / `audio_frames`; `--num-frames` must not be forwarded.
  assert example.input_data.use_num_frames is False
  assert example.input_data.extra_input_kwargs["latent_frames"] == 4
  assert example.input_data.extra_input_kwargs["audio_frames"] == 32


def test_t2va_example_honours_cli_overrides():
  """`--height/--width/--steps` must reach the example."""

  example = magi2_t2va_example(_args(height=320, width=576, num_inference_steps=8))
  assert example.input_data.height == 320
  assert example.input_data.width == 576
  assert example.input_data.num_inference_steps == 8


def test_i2va_example_requires_a_reference_still():
  """The image variant must fail fast without `--image-path` instead of rendering a t2va clip."""

  import pytest

  with pytest.raises(ValueError, match="--image-path"):
    magi2_i2va_example(_args())
