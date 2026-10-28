"""Regression tests for the HiDream-O1-Image `generate` CLI example.

The pipeline is vendored inside cache-dit (`cache_dit.models.pipelines.hidream_o1_image`), so these
tests pin down that the factory resolves to the built-in class (not a `_safe_import` stub), that the
env var points at a local checkout, that the defaults stay on the release recipe, and that the
unsupported parallel modes (no VAE, no text encoder) are rejected instead of silently ignored.

Usage:
  python -m pytest tests/utils/test_hidream_o1_image_example.py -q
"""

import argparse

import pytest

from cache_dit._utils import hidream_o1_image_example as public_example
from cache_dit._utils import registers
from cache_dit._utils.examples import (
  _env_path_mapping,
  _HIDREAM_O1_IMAGE_PROMPT,
  _path_env_mapping,
  hidream_o1_image_example,
)
from cache_dit._utils.registers import ExampleRegister, ExampleType
from cache_dit.models.pipelines.hidream_o1_image import HiDreamO1ImagePipeline


@pytest.fixture(autouse=True)
def _cpu_distributed(monkeypatch):
  """`Example.__init__` asks the distributed helpers whether this run is parallel."""
  monkeypatch.setattr(registers, "maybe_init_distributed", lambda _args: (0, "cpu"))


@pytest.fixture(autouse=True)
def _clear_model_env(monkeypatch):
  """Developer shells may export the model dir; the default assertions need it absent."""
  monkeypatch.delenv("HIDREAM_O1_IMAGE_DIR", raising=False)


def _factory_args(**overrides) -> argparse.Namespace:
  values = dict(
    model_path=None,
    num_inference_steps=None,
    parallel_type=None,
    parallel_text_encoder=False,
    parallel_vae=False,
    config_path=None,
  )
  values.update(overrides)
  return argparse.Namespace(**values)


def test_example_is_registered_with_its_hf_model_id():
  assert "hidream_o1_image" in ExampleRegister.list_examples()
  assert ExampleRegister.get_default("hidream_o1_image") == "HiDream-ai/HiDream-O1-Image"


def test_public_reexport_resolves_to_the_factory():
  assert public_example is hidream_o1_image_example


def test_env_var_points_the_example_at_a_local_checkout(monkeypatch):
  monkeypatch.setenv("HIDREAM_O1_IMAGE_DIR", "/models/hidream-o1")
  example = hidream_o1_image_example(_factory_args())
  assert example.init_config.model_name_or_path == "/models/hidream-o1"


def test_defaults_are_the_release_recipe():
  example = hidream_o1_image_example(_factory_args())
  assert example.init_config.task_type is ExampleType.T2I
  assert example.init_config.pipeline_class is HiDreamO1ImagePipeline
  assert example.init_config.model_name_or_path == "HiDream-ai/HiDream-O1-Image"

  input_data = example.input_data
  assert input_data.prompt == _HIDREAM_O1_IMAGE_PROMPT
  assert (input_data.height, input_data.width) == (2048, 2048)
  assert input_data.num_inference_steps == 50
  # `guidance_scale=1.0` would leave a single forward per step and break the separate-CFG pairing.
  assert input_data.guidance_scale == 5.0


def test_cli_model_path_overrides_the_env_mapping(monkeypatch):
  monkeypatch.setenv("HIDREAM_O1_IMAGE_DIR", "/models/hidream-o1")
  example = hidream_o1_image_example(_factory_args(model_path="/elsewhere/hidream"))
  assert example.init_config.model_name_or_path == "/elsewhere/hidream"


def test_env_mapping_stays_in_sync_with_the_factory_default():
  assert _env_path_mapping["HIDREAM_O1_IMAGE_DIR"] == "HiDream-ai/HiDream-O1-Image"
  assert _path_env_mapping["HiDream-ai/HiDream-O1-Image"] == "HIDREAM_O1_IMAGE_DIR"


def test_pixel_space_model_rejects_vae_parallelism():
  with pytest.raises(NotImplementedError, match="no VAE"):
    hidream_o1_image_example(_factory_args(parallel_vae=True))


def test_model_without_text_encoder_rejects_text_parallelism():
  with pytest.raises(NotImplementedError, match="no text encoder"):
    hidream_o1_image_example(_factory_args(parallel_text_encoder=True))
