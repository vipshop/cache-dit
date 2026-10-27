"""Regression tests for the MOSS-Image `generate` CLI examples.

MOSS-Image is the first model whose pipeline is vendored inside cache-dit
(`cache_dit.models.pipelines.moss_image`) instead of imported from diffusers, so these tests pin
down that the factory resolves to the built-in class (not a stub from `_safe_import`), that the
environment variables point both variants at local checkouts, and that the CFG skip recipe reaches
the pipeline through `extra_input_kwargs`.

Usage:
  python -m pytest tests/utils/test_moss_image_example.py -q
"""
import argparse

import pytest

from cache_dit._utils import moss_image_base_example as public_base_example
from cache_dit._utils import moss_image_sft_example as public_sft_example
from cache_dit._utils import registers
from cache_dit._utils.examples import (
  _env_path_mapping,
  _MOSS_IMAGE_PROMPT,
  _path_env_mapping,
  moss_image_base_example,
  moss_image_sft_example,
)
from cache_dit._utils.registers import ExampleRegister, ExampleType
from cache_dit.models.pipelines.moss_image import MOSSImagePipeline


@pytest.fixture(autouse=True)
def _cpu_distributed(monkeypatch):
  """`Example.__init__` asks the distributed helpers whether this run is parallel."""
  monkeypatch.setattr(registers, "maybe_init_distributed", lambda _args: (0, "cpu"))


@pytest.fixture(autouse=True)
def _clear_model_env(monkeypatch):
  """Developer shells may export the model dirs; the default assertions need them absent."""
  for name in ("MOSS_IMAGE_SFT_DIR", "MOSS_IMAGE_BASE_DIR"):
    monkeypatch.delenv(name, raising=False)


def _factory_args(**overrides) -> argparse.Namespace:
  values = dict(
    model_path=None,
    num_inference_steps=None,
    parallel_type=None,
    config_path=None,
  )
  values.update(overrides)
  return argparse.Namespace(**values)


def test_both_variants_are_registered_with_their_hf_model_ids():
  examples = ExampleRegister.list_examples()
  assert "moss_image_sft" in examples
  assert "moss_image_base" in examples
  assert ExampleRegister.get_default("moss_image_sft") == "OpenMOSS-Team/MOSS-Image-SFT"
  assert ExampleRegister.get_default("moss_image_base") == "OpenMOSS-Team/MOSS-Image-Base"


def test_public_reexports_resolve_to_the_factories():
  """`_safe_import` degrades to a stub on ImportError, which would hide broken wiring."""
  assert public_sft_example is moss_image_sft_example
  assert public_base_example is moss_image_base_example


def test_env_vars_point_the_examples_at_local_checkouts(monkeypatch):
  monkeypatch.setenv("MOSS_IMAGE_SFT_DIR", "/models/moss-sft")
  monkeypatch.setenv("MOSS_IMAGE_BASE_DIR", "/models/moss-base")
  sft_path = moss_image_sft_example(_factory_args()).init_config.model_name_or_path
  base_path = moss_image_base_example(_factory_args()).init_config.model_name_or_path
  assert sft_path == "/models/moss-sft"
  assert base_path == "/models/moss-base"


def test_defaults_are_the_documented_ones():
  example = moss_image_sft_example(_factory_args())
  assert example.init_config.task_type is ExampleType.T2I
  assert example.init_config.pipeline_class is MOSSImagePipeline
  assert example.init_config.model_name_or_path == "OpenMOSS-Team/MOSS-Image-SFT"

  input_data = example.input_data
  assert input_data.prompt == _MOSS_IMAGE_PROMPT
  assert (input_data.height, input_data.width) == (1024, 1024)
  assert input_data.num_inference_steps == 50
  # `guidance_scale=1.0` would leave a single forward per step and break the separate-CFG pairing.
  assert input_data.guidance_scale == 3.0
  assert input_data.extra_input_kwargs == {"uncond_skip_layers": [22]}


def test_cli_model_path_overrides_the_env_mapping(monkeypatch):
  monkeypatch.setenv("MOSS_IMAGE_SFT_DIR", "/models/moss-sft")
  example = moss_image_sft_example(_factory_args(model_path="/elsewhere/moss"))
  assert example.init_config.model_name_or_path == "/elsewhere/moss"


def test_env_mapping_values_stay_in_sync_with_the_factory_defaults():
  """`_path()` reverse-resolves the env name from the default string, so both must match."""
  assert _env_path_mapping["MOSS_IMAGE_SFT_DIR"] == "OpenMOSS-Team/MOSS-Image-SFT"
  assert _path_env_mapping["OpenMOSS-Team/MOSS-Image-SFT"] == "MOSS_IMAGE_SFT_DIR"
  assert _path_env_mapping["OpenMOSS-Team/MOSS-Image-Base"] == "MOSS_IMAGE_BASE_DIR"


def test_skip_recipe_survives_the_input_data_builder():
  data = moss_image_sft_example(_factory_args()).input_data.data(_data_args())
  assert data["uncond_skip_layers"] == [22]
  assert data["guidance_scale"] == 3.0


def _data_args(**overrides) -> argparse.Namespace:
  """Namespace with the attributes `ExampleInputData.data` reads."""
  values = dict(
    model_path=None,
    prompt=None,
    negative_prompt=None,
    skip_negative_prompt=False,
    height=None,
    width=None,
    num_inference_steps=None,
    num_frames=None,
    image_path=None,
    last_image_path=None,
    mask_image_path=None,
    input_height=None,
    input_width=None,
    generator_device=None,
    seed=None,
    extra_input_kwargs=None,
  )
  values.update(overrides)
  return argparse.Namespace(**values)
