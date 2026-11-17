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
    # The real API clears the components and never touches the block tree; faking a pop here is what
    # let the first version of the hook ship a no-op `--prompt-enhancer`.
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
    max_continuous_cached_steps=3,
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
  """The two workflows differ by the reference-image path, not by "more blocks in general"."""
  from diffusers.modular_pipelines.p_h3_image import PH3ImageBlocks

  blocks = PH3ImageBlocks()
  assert list(blocks.sub_blocks) == [
    "before_encode", "prompt_upsampler", "text_encoder", "vae_encoder", "denoise", "decode"
  ]
  assert set(blocks._workflow_map) == {"text2image", "image_conditioned"}

  text2image = list(blocks.get_workflow("text2image").sub_blocks)
  edit = list(blocks.get_workflow("image_conditioned").sub_blocks)
  # `before_encode` / `vae_encoder` are triggered by an `image` input, so the text chain drops them
  assert "before_encode" not in text2image
  assert "vae_encoder" not in text2image
  assert "before_encode" in edit
  assert "vae_encoder" in edit
  assert len(text2image) == 8
  assert len(edit) == 12


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
  before = len(pipe._blocks.sub_blocks)

  _ph3_image_post_init(None, "/models/p-h3-image")(pipe)

  assert "prompt_upsampler" not in pipe._blocks.sub_blocks
  assert len(pipe._blocks.sub_blocks) == before - 1
  assert "decode" in pipe._blocks.sub_blocks
  assert pipe.unloaded == ["prompt_upsampler", "prompt_upsampler_processor"]
  assert pipe.updated == {}


def test_the_hook_pops_the_real_tree_not_the_deepcopy():
  """`pipe.blocks` deep-copies the tree, and `unload_components` never removes a block.

  Constructed without weights: the block tree and the component specs are all this needs, and a fake
  pipeline here is what let the first version of the hook ship a no-op `--prompt-enhancer`.
  """
  from diffusers import PH3ImageModularPipeline

  pipe = PH3ImageModularPipeline()
  before = list(pipe._blocks.sub_blocks)
  assert before == [
    "before_encode", "prompt_upsampler", "text_encoder", "vae_encoder", "denoise", "decode"
  ]

  pipe.blocks.sub_blocks.pop("prompt_upsampler", None)
  assert list(pipe._blocks.sub_blocks) == before

  _ph3_image_post_init(None, "/models/p-h3-image")(pipe)
  assert list(pipe._blocks.sub_blocks) == [name for name in before if name != "prompt_upsampler"]


def _stub_component_loaders(monkeypatch, loaded):
  """Record `from_pretrained` calls and return an uninitialised instance of the requested class.

  Patching `transformers.<Name>` does not work once `_LazyModule` has materialised the class (its
  `__getattribute__` wins over the instance dict), so the patch goes on the base methods the classes
  actually use. Returning `cls.__new__(cls)` keeps the real type, i.e. `update_components` cannot
  reject it, while no weights are read.
  """
  import transformers

  def _record(cls, path, subfolder=None, dtype=None, **kwargs):
    loaded.setdefault("calls", []).append((cls.__name__, path, subfolder, dtype))
    return cls.__new__(cls)

  monkeypatch.setattr(transformers.PreTrainedModel, "from_pretrained", classmethod(_record))
  monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", classmethod(_record))


def test_post_init_hook_keeps_the_step_on_a_real_pipeline_for_a_foreign_directory(monkeypatch):
  """A foreign enhancer directory swaps the component and leaves the block tree alone.

  Real `PH3ImageModularPipeline`, no weights: the point is that the step survives a reload, because
  `update_components` cannot put a popped block back.
  """
  from diffusers import PH3ImageModularPipeline

  loaded = {}
  _stub_component_loaders(monkeypatch, loaded)

  pipe = PH3ImageModularPipeline()
  before = list(pipe._blocks.sub_blocks)
  _ph3_image_post_init("/models/p-h3-image", "/models/other-checkpoint")(pipe)

  assert list(pipe._blocks.sub_blocks) == before
  assert [(name, path, subfolder) for name, path, subfolder, _ in loaded["calls"]] == [
    ("Qwen3_5ForConditionalGeneration", "/models/p-h3-image", "prompt_upsampler"),
    ("AutoProcessor", "/models/p-h3-image", "prompt_upsampler_processor"),
  ]
  assert type(pipe.prompt_upsampler).__name__ == "Qwen3_5ForConditionalGeneration"
  assert type(pipe.prompt_upsampler_processor).__name__ == "AutoProcessor"


def test_post_init_hook_refuses_the_enhancer_under_torchrun(monkeypatch):
  """Every rank would load its own 18.8 GiB copy and no `--parallel*` option covers it."""
  monkeypatch.setenv("WORLD_SIZE", "2")
  try:
    _ph3_image_post_init("/models/p-h3-image", "/models/other-checkpoint")
  except ValueError as error:
    assert "--prompt-enhancer" in str(error)
  else:
    raise AssertionError("the enhancer must be rejected before a process group exists")

  monkeypatch.setenv("WORLD_SIZE", "1")
  assert _ph3_image_post_init("/models/p-h3-image", "/models/other-checkpoint") is not None


def test_post_init_hook_keeps_the_step_when_the_directory_is_the_checkpoint():
  """Pointing `--prompt-enhancer` at the checkpoint itself must keep the loaded components.

  Popping the step here would render the short request verbatim while still paying the upsampler's
  residency: the flag would look accepted and change the prompt distribution anyway. No loader runs
  on this path at all, so the test needs no stubbing: `unloaded`/`updated` staying empty is the
  assertion.
  """
  pipe = _FakePipe()
  _ph3_image_post_init("/models/p-h3-image", "/models/p-h3-image")(pipe)

  assert "prompt_upsampler" in pipe._blocks.sub_blocks
  assert pipe.unloaded == []
  assert pipe.updated == {}


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
