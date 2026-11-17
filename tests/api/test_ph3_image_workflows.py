"""CPU-only guards for the two P-H3-Image CLI workflows.

Three things are pinned here, all of them wiring rather than model behaviour (the model itself is
covered by the acceptance run):

1. **Both workflows exist and prune to different block chains.** `text2image` and
   `image_conditioned` are entry points of the same eight-step pipeline, and the second one is
   selected only by passing `image`.
2. **The prompt upsampler is popped from the live block tree.** `pipe.blocks` returns a deepcopy, so
   popping that copy leaves the step in place and the "prompt rendered as given" contract silently
   breaks; the components also have to be unloaded, or the ~2 GiB conditioner stays resident.
3. **The cache budget matches the checkpoint's schedule.** Eight steps leave room for at most a
   two-step warmup, and a `--steps-mask` run has to switch to the static policy.
"""

import argparse

import pytest

from cache_dit.cli.examples import (
  _ph3_image_cache_config,
  _ph3_image_post_init,
  ph3_image_edit_example,
  ph3_image_example,
)
from cache_dit.cli.registers import ExampleRegister


class _FakePipe:
  """The two attributes the post-init hook touches."""

  def __init__(self):
    self._blocks = argparse.Namespace(sub_blocks={
      "prompt_upsampler": object(),
      "text_encoder": object(),
      "decode": object(),
    })
    self.unloaded = []
    self.updated = {}

  def unload_components(self, names):
    for name in names:
      self._blocks.sub_blocks.pop(name, None)
    self.unloaded.extend(names)

  def update_components(self, **kwargs):
    self.updated.update(kwargs)


def _args(**overrides) -> argparse.Namespace:
  base = dict(
    model_path=None,
    prompt=None,
    height=None,
    width=None,
    num_inference_steps=None,
    quantize=False,
    quantize_exclude_layers=None,
    cache=False,
    residual_diff_threshold=0.12,
    max_warmup_steps=8,
    max_cached_steps=-1,
    steps_mask=False,
    mask_policy=None,
    prompt_enhancer=None,
    image_path=None,
    parallel_type=None,
    config_path=None,
    example="ph3_image",
  )
  base.update(overrides)
  return argparse.Namespace(**base)


def test_both_workflows_are_registered():
  assert "ph3_image" in ExampleRegister.list_examples()
  assert "ph3_image_edit" in ExampleRegister.list_examples()
  assert ExampleRegister.get_default("ph3_image") == "PrunaAI/P-H3-Image"
  assert ExampleRegister.get_default("ph3_image_edit") == "PrunaAI/P-H3-Image"


def test_each_workflow_prunes_to_its_own_block_chain():
  from diffusers.modular_pipelines.p_h3_image import PH3ImageBlocks

  blocks = PH3ImageBlocks()
  text2image = blocks.get_workflow("text2image")
  edit = blocks.get_workflow("image_conditioned")

  text2image_components = {spec.name for spec in text2image.expected_components}
  edit_components = {spec.name for spec in edit.expected_components}
  assert "transformer" in text2image_components
  assert "prompt_upsampler" in text2image_components
  # the edit chain adds the VAE encoder path
  assert edit_components >= text2image_components


def test_examples_select_their_workflow_and_skip_the_step_count():
  text2image = ph3_image_example(_args())
  edit = ph3_image_edit_example(_args())

  assert text2image.init_config.modular_workflow == "text2image"
  assert edit.init_config.modular_workflow == "image_conditioned"
  # the checkpoint's sigma grid decides the step count
  assert text2image.input_data.use_num_inference_steps is False
  assert edit.input_data.use_num_inference_steps is False
  # the edit canvas is resolved by the pipeline, not by a client-side resize
  assert edit.input_data.skip_image_resize is True
  # `--prompt-enhancer` keeps the upsampler; the default pops it
  assert text2image.init_config.post_init_hook is not None


def test_post_init_hook_pops_the_upsampler_from_the_live_tree():
  pipe = _FakePipe()
  _ph3_image_post_init(None)(pipe)

  assert "prompt_upsampler" not in pipe._blocks.sub_blocks
  assert "decode" in pipe._blocks.sub_blocks
  assert pipe.unloaded == ["prompt_upsampler", "prompt_upsampler_processor"]
  assert pipe.updated == {}


def test_post_init_hook_injects_the_upsampler_when_a_directory_is_given(monkeypatch):
  loaded = {}

  class _FakeUpsampler:

    @classmethod
    def from_pretrained(cls, path, subfolder=None, dtype=None):
      loaded["upsampler"] = (path, subfolder, dtype)
      return cls()

  class _FakeProcessor:

    @classmethod
    def from_pretrained(cls, path, subfolder=None):
      loaded["processor"] = (path, subfolder)
      return cls()

  import transformers

  monkeypatch.setattr(transformers, "Qwen3_5ForConditionalGeneration", _FakeUpsampler)
  monkeypatch.setattr(transformers, "AutoProcessor", _FakeProcessor)

  pipe = _FakePipe()
  _ph3_image_post_init("/models/p-h3-image")(pipe)

  assert "prompt_upsampler" not in pipe._blocks.sub_blocks
  assert loaded["upsampler"][0] == "/models/p-h3-image"
  assert loaded["upsampler"][1] == "prompt_upsampler"
  assert loaded["processor"][1] == "prompt_upsampler_processor"
  assert set(pipe.updated) == {"prompt_upsampler", "prompt_upsampler_processor"}


def test_cache_config_matches_the_eight_step_grid():
  config = _ph3_image_cache_config(_args(cache=True), "text2image")

  assert config.num_inference_steps == 8
  assert config.max_warmup_steps <= 2
  assert config.enable_separate_cfg is False
  assert config.max_continuous_cached_steps == 1
  assert config.steps_computation_mask is None
  assert config.steps_computation_policy == "dynamic"


def test_steps_mask_run_switches_to_the_static_policy():
  config = _ph3_image_cache_config(_args(cache=True, steps_mask=True), "text2image")

  mask = config.steps_computation_mask
  assert len(mask) == 8
  assert mask[-1] == 1, "the last step lands on the image and always computes"
  assert mask.count(0) == 2
  assert config.steps_computation_policy == "static"


def test_cache_config_rejects_an_unknown_workflow():
  with pytest.raises(ValueError):
    _ph3_image_cache_config(_args(cache=True), "text2video")


def test_cache_is_off_without_the_flag():
  assert _ph3_image_cache_config(_args(cache=False), "text2image") is None


def test_adaln_plan_reads_the_ph3_conditioning_level():
  """`--adaln-online` must see the two-row timestep vectors of an edit request.

  P-H3-Image names the pinned conditioning level `keyframe_noise_aug`; reading only MiniMax-H3's
  `reference_noise_aug` would leave every edit step without a memoized table (slow, not wrong), so
  the fallback is asserted on a stub instead of on a loaded checkpoint.
  """
  from cache_dit.offload.adaln_plan import timestep_sets_from_pipeline

  class _Stub:
    config = argparse.Namespace(sample_sigmas=[1.0, 0.5, 0.0])
    keyframe_noise_aug = 0.999

  sets = timestep_sets_from_pipeline(_Stub())
  assert any(len(entry) == 2 for entry in sets)
  assert [0.0, 0.999] in sets
