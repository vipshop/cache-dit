"""Regression tests for the Ideogram 4 `generate` CLI example.

`Ideogram4Pipeline` is the first T2I example whose guidance is a per-step schedule instead of a
scalar, and `check_inputs` rejects a schedule whose length differs from `num_inference_steps`. The
failure modes pinned down here:

1. A schedule baked into `ExampleInputData` stays 48 entries long no matter what
   `--num-inference-steps` / `--warmup-num-inference-steps` ask for, and the run then dies in
   `check_inputs` after the weights are already on the device.
2. `ExampleInputData.data()` drops `None` but keeps `False`, so `prompt_upsampling=False` has to
   reach the pipeline while `guidance_scale` must not.
3. Setting both `guidance_scale` and `guidance_schedule` raises, so the scalar has to be removed
   when the schedule is filled in.
4. The default checkpoint is the official FP8 release. The example no longer converts it itself: it
   enables `Ideogram4Loader`, which patches `Ideogram4Pipeline.from_pretrained`. The conversion
   itself is covered by `tests/loaders/test_ideogram4_loader.py`; what is pinned down here is that
   the factory really installs the patch.

Usage:
  python -m pytest tests/utils/test_ideogram4_example.py -q
"""
import argparse
import json
import inspect
from types import SimpleNamespace

import pytest

try:
  from diffusers import Ideogram4Pipeline
except ImportError:
  pytest.skip("diffusers without Ideogram4 support", allow_module_level=True)

from cache_dit._utils import ideogram4_example as public_ideogram4_example
from cache_dit._utils import registers
from cache_dit._utils.examples import (
  _IDEOGRAM4_PROMPT,
  _IDEOGRAM4_STEPS,
  _Ideogram4Example,
  _ideogram4_guidance_schedule,
  ideogram4_example,
)
from cache_dit._utils.registers import ExampleRegister, ExampleType
from cache_dit.loaders import Ideogram4Loader
from cache_dit.loaders import base as loaders_base

# kwargs of the latest stand-in pipeline call
_SEEN = {}


@pytest.fixture(autouse=True)
def _cpu_distributed(monkeypatch):
  """`Example.__init__` asks the distributed helpers whether this run is parallel."""
  monkeypatch.setattr(registers, "maybe_init_distributed", lambda _args: (0, "cpu"))


@pytest.fixture(autouse=True)
def _unpatched_pipeline_class():
  """The factory patches `Ideogram4Pipeline` in place, so no test may inherit that patch."""
  yield
  loader = Ideogram4Pipeline.__dict__.get(loaders_base.LOADER_PATCH_ATTR)
  if loader is not None:
    loader.remove()


def _factory_args(**overrides) -> argparse.Namespace:
  """Namespace with the attributes the example factory and `Example.__init__` read."""
  values = dict(
    model_path=None,
    num_inference_steps=None,
    parallel_type=None,
    config_path=None,
  )
  values.update(overrides)
  return argparse.Namespace(**values)


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


def _capture(**kwargs) -> SimpleNamespace:
  """Stand-in for the pipeline: records the kwargs of one call."""
  _SEEN.clear()
  _SEEN.update(kwargs)
  return SimpleNamespace(images=[None])


def test_example_is_registered_with_the_hf_model_id():
  assert "ideogram4" in ExampleRegister.list_examples()
  assert ExampleRegister.get_default("ideogram4") == "ideogram-ai/ideogram-4-fp8"


def test_public_reexport_resolves_to_the_example_factory():
  """`_safe_import` degrades to a stub on an ImportError, which would hide broken wiring."""
  assert public_ideogram4_example is ideogram4_example


def test_env_var_points_the_example_at_a_local_checkout(monkeypatch):
  monkeypatch.setenv("IDEOGRAM4_DIR", "/models/ideogram-v4")
  example = ideogram4_example(_factory_args())
  assert example.init_config.model_name_or_path == "/models/ideogram-v4"


def test_defaults_are_the_documented_ones():
  example = ideogram4_example(_factory_args())
  assert isinstance(example, _Ideogram4Example)
  assert example.init_config.task_type is ExampleType.T2I
  assert example.init_config.pipeline_class is Ideogram4Pipeline
  # The prompt upsampler rewrites the prompt, so the head is dropped instead of loaded.
  assert example.init_config.skip_components == ["prompt_enhancer_head"]
  # Asymmetric CFG: two transformers, one forward each, blended outside of both.
  assert example.init_config.extra_optimize_kwargs == {"enable_separate_cfg": False}

  input_data = example.input_data
  assert input_data.prompt == _IDEOGRAM4_PROMPT
  assert input_data.num_inference_steps == _IDEOGRAM4_STEPS
  assert (input_data.height, input_data.width) == (1024, 1024)
  assert input_data.guidance_scale is None


def test_default_prompt_is_the_official_structured_json_caption():
  prompt = json.loads(_IDEOGRAM4_PROMPT)
  assert list(prompt) == [
    "high_level_description",
    "style_description",
    "compositional_deconstruction",
  ]
  assert list(prompt["style_description"]) == ["aesthetics", "lighting", "photo", "medium"]
  composition = prompt["compositional_deconstruction"]
  assert list(composition) == ["background", "elements"]
  assert list(composition["elements"][0]) == ["type", "bbox", "desc"]
  assert list(composition["elements"][1]) == ["type", "bbox", "text", "desc"]
  assert composition["elements"][1]["text"] == "hello world"
  assert [element["type"] for element in composition["elements"]] == ["obj", "text"]
  assert [element["bbox"] for element in composition["elements"]] == [
    [120, 100, 920, 900],
    [500, 300, 680, 700],
  ]
  assert _IDEOGRAM4_PROMPT == json.dumps(
    prompt,
    separators=(",", ":"),
    ensure_ascii=False,
  )


def test_cli_step_override_reaches_the_example():
  example = ideogram4_example(_factory_args(num_inference_steps=8))
  assert example.input_data.num_inference_steps == 8


@pytest.mark.parametrize("steps", [48, 30, 8, 4, 3, 2, 1])
def test_schedule_length_always_matches_the_step_count(steps):
  assert len(_ideogram4_guidance_schedule(steps)) == steps


def test_schedule_keeps_the_polish_tail_bounded():
  assert _ideogram4_guidance_schedule(48) == [7.0] * 45 + [3.0] * 3
  assert _ideogram4_guidance_schedule(4) == [7.0, 3.0, 3.0, 3.0]
  assert _ideogram4_guidance_schedule(2) == [3.0, 3.0]
  assert _ideogram4_guidance_schedule(1) == [3.0]


def test_call_pipe_rebuilds_the_schedule_for_the_forwarded_steps():
  example = ideogram4_example(_factory_args())
  for steps in (48, 30, 2):
    example.call_pipe(_capture, {"prompt": "a cat", "num_inference_steps": steps})
    assert _SEEN["guidance_schedule"] == _ideogram4_guidance_schedule(steps)


def test_warmup_step_override_gets_its_own_schedule():
  """`run()` overwrites `num_inference_steps` for the warmup pass; the schedule has to follow."""
  example = ideogram4_example(_factory_args())
  example.call_pipe(_capture, {"prompt": "a cat", "num_inference_steps": 2})
  assert _SEEN["num_inference_steps"] == 2
  assert _SEEN["guidance_schedule"] == [3.0, 3.0]


def test_missing_step_count_falls_back_to_the_example_default():
  example = ideogram4_example(_factory_args(num_inference_steps=30))
  example.call_pipe(_capture, {"prompt": "a cat"})
  assert _SEEN["guidance_schedule"] == _ideogram4_guidance_schedule(30)


def test_call_pipe_preserves_explicit_guidance_scale():
  """Setting both the scalar and the schedule raises in `check_inputs`."""
  example = ideogram4_example(_factory_args())
  example.call_pipe(_capture, {"prompt": "a cat", "guidance_scale": 5.0})
  assert _SEEN["guidance_scale"] == 5.0
  assert _SEEN["guidance_schedule"] is None


def test_input_data_forwards_prompt_upsampling_but_no_guidance():
  data = ideogram4_example(_factory_args()).input_data.data(_data_args())
  assert data["prompt_upsampling"] is False
  # None values are filtered, so the mutually exclusive scalar never reaches the pipeline.
  assert "guidance_scale" not in data
  # The schedule is derived per call instead of being baked into the input dict.
  assert "guidance_schedule" not in data


def test_generated_schedules_pass_the_pipeline_input_check():
  input_data = ideogram4_example(_factory_args()).input_data
  # `check_inputs` only touches these two attributes before validating the schedule.
  stub = SimpleNamespace(vae_scale_factor=8, patch_size=2)

  def _check(steps: int, schedule) -> None:
    Ideogram4Pipeline.check_inputs(
      stub,
      prompt=input_data.prompt,
      height=input_data.height,
      width=input_data.width,
      num_inference_steps=steps,
      guidance_scale=None,
      guidance_schedule=schedule,
    )

  for steps in (input_data.num_inference_steps, 8, 2):
    _check(steps, _ideogram4_guidance_schedule(steps))
  # A 48-entry schedule forwarded to a shorter run is exactly what the override prevents.
  with pytest.raises(ValueError, match="must have length"):
    _check(8, _ideogram4_guidance_schedule(_IDEOGRAM4_STEPS))


def test_default_schedule_tracks_the_pipeline_default():
  """48 steps is not arbitrary: it is the length of the pipeline's own default schedule."""
  default = inspect.signature(Ideogram4Pipeline.__call__).parameters["guidance_schedule"].default
  assert list(default) == _ideogram4_guidance_schedule(_IDEOGRAM4_STEPS)


def test_the_factory_enables_the_loader_on_the_pipeline_class():
  """The init config is a plain one now: converting the checkpoint is the loader's job."""
  before = Ideogram4Pipeline.__dict__.get("from_pretrained")

  ideogram4_example(_factory_args())

  installed = Ideogram4Pipeline.__dict__.get(loaders_base.LOADER_PATCH_ATTR)
  assert isinstance(installed, Ideogram4Loader)
  assert Ideogram4Pipeline.__dict__["from_pretrained"] is not before
  assert Ideogram4Pipeline.from_pretrained.__func__ is installed.wrapper
