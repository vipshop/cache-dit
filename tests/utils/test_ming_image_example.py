import argparse

import pytest
from PIL import Image

from cache_dit._utils import ming_image_design_example as public_design_example
from cache_dit._utils import ming_image_design_layer_example as public_layer_example
from cache_dit._utils import registers
from cache_dit._utils.examples import (
  _env_path_mapping,
  ming_image_design_example,
  ming_image_design_layer_example,
)
from cache_dit._utils.registers import ExampleRegister, ExampleType
from cache_dit.models.pipelines.ming_image import MingImagePipeline


@pytest.fixture(autouse=True)
def _cpu_distributed(monkeypatch):
  monkeypatch.setattr(registers, "maybe_init_distributed", lambda _args: (0, "cpu"))


@pytest.fixture(autouse=True)
def _clear_model_env(monkeypatch):
  for name in ("MING_IMAGE_DESIGN_DIR", "MING_IMAGE_DESIGN_LAYER_DIR"):
    monkeypatch.delenv(name, raising=False)


def _factory_args(**overrides):
  values = dict(model_path=None, num_inference_steps=None, parallel_type=None, config_path=None)
  values.update(overrides)
  return argparse.Namespace(**values)


def _data_args(**overrides):
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


def test_both_variants_are_registered_and_publicly_reexported():
  assert "ming_image_design" in ExampleRegister.list_examples()
  assert "ming_image_design_layer" in ExampleRegister.list_examples()
  assert public_design_example is ming_image_design_example
  assert public_layer_example is ming_image_design_layer_example


def test_variant_defaults_and_env_mapping(monkeypatch):
  design = ming_image_design_example(_factory_args())
  layered = ming_image_design_layer_example(_factory_args())
  assert design.init_config.task_type is ExampleType.T2I
  assert layered.init_config.task_type is ExampleType.IE2I
  assert design.init_config.pipeline_class is MingImagePipeline
  assert design.input_data.guidance_scale == 1.0
  assert layered.input_data.guidance_scale == 2.0
  assert layered.input_data.extra_input_kwargs == {"num_layers": 4}
  assert layered.init_config.preserve_image_list is True
  assert design.init_config.preserve_image_list is False
  assert design.input_data.num_inference_steps == 12

  monkeypatch.setenv("MING_IMAGE_DESIGN_DIR", "/models/design")
  monkeypatch.setenv("MING_IMAGE_DESIGN_LAYER_DIR", "/models/layer")
  assert ming_image_design_example(
    _factory_args()).init_config.model_name_or_path == "/models/design"
  assert ming_image_design_layer_example(
    _factory_args()).init_config.model_name_or_path == "/models/layer"
  assert _env_path_mapping["MING_IMAGE_DESIGN_DIR"] == "inclusionAI/Ming-Image-0.1-Design"
  assert _env_path_mapping[
    "MING_IMAGE_DESIGN_LAYER_DIR"] == "inclusionAI/Ming-Image-0.1-Design-Layer"


def test_cli_model_path_overrides_environment(monkeypatch):
  monkeypatch.setenv("MING_IMAGE_DESIGN_DIR", "/models/design")
  example = ming_image_design_example(_factory_args(model_path="/elsewhere/ming"))
  assert example.init_config.model_name_or_path == "/elsewhere/ming"


def test_rgba_input_survives_cli_image_builder(tmp_path):
  path = tmp_path / "reference.png"
  Image.new("RGBA", (32, 32), (10, 20, 30, 127)).save(path)
  example = ming_image_design_layer_example(_factory_args())

  data = example.input_data.data(_data_args(image_path=str(path)))

  assert data["image"].mode == "RGBA"
  assert data["image"].getpixel((0, 0))[3] == 127
  assert "keep_alpha" not in data
  assert data["num_layers"] == 4


def test_parallel_text_is_supported_for_custom_mllm():
  from cache_dit.distributed.text_encoders.planners import (
    _activate_text_encoder_tp_planners, )
  from cache_dit.distributed.text_encoders.register import (
    TextEncoderTensorParallelismPlannerRegister, )
  from cache_dit.models.pipelines.ming_image.mllm import MingImageMLLM

  ming_image_design_example(_factory_args(parallel_text_encoder=True))
  assert MingImageMLLM._cache_dit_text_encoder is True
  _activate_text_encoder_tp_planners()
  planner = TextEncoderTensorParallelismPlannerRegister.get_planner("MingImageMLLM")
  assert planner.__name__ == "MingImageMLLMTensorParallelismPlanner"
