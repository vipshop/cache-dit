"""Regression tests for the MiniMax-H3 workflow wiring (`fl2va`, and later `ref2va`).

The workflows share the caching / parallelism plumbing of `t2va`; what differs is the CLI layer and
the checkpoint partition a workflow loads. The silent failure modes covered here:

1. `call_pipe` asks a modular pipeline for the audio-video state keys only when the task is in
   `_AUDIO_VIDEO_TASKS`. A task that is missing there receives the whole `PipelineState`, and the
   parse step fails only after the full inference has already run.
2. `MiniMax-H3` loads `transformer/` for `t2va` / `fl2va` and `transformer_ref/` for `ref2va`, so a
   `ref2va` pipeline has no `transformer` attribute at all.
3. The pipeline derives its own canvas from the source aspect ratio, so the CLI must not resize the
   conditioning frames.
4. The conditioning workflows pin their conditioning rows every step, which dilutes the residual
   mean DBCache compares against; their threshold preset must differ from `t2va` while an explicit
   `--residual-diff-threshold` still wins.
"""
import argparse
import importlib
from contextlib import contextmanager
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from cache_dit.cli import registers
from cache_dit.cli.examples import (
  _MINIMAX_H3_CACHE_THRESHOLDS,
  _MINIMAX_H3_DEFAULT_THRESHOLD,
  _minimax_h3_cache_config,
)
from cache_dit.cli.registers import (
  ExampleInputData,
  ExampleOutputData,
  ExampleType,
  _AUDIO_VIDEO_TASKS,
  Example,
  parse_audio_video_output,
)
from cache_dit.caching.block_adapters.adapters import minimax_h3_adapter
from cache_dit.cli.utils import create_profiler_from_args, get_args
from cache_dit.layerwise_nvtx import stage_range
from cache_dit.profiler import profile_steps
from cache_dit.utils import TRANSFORMER_ATTR_NAMES, assign_transformer, resolve_transformer


def _data_args(**overrides) -> argparse.Namespace:
  """Namespace with the attributes `ExampleInputData.data` reads."""
  values = dict(
    model_path=None,
    cache=False,
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
    # `Example.__init__` asks the distributed helpers whether this run is parallel
    parallel_type=None,
    config_path=None,
  )
  values.update(overrides)
  return argparse.Namespace(**values)


def _write_image(tmp_path, name: str, size=(80, 40)) -> str:
  path = tmp_path / name
  Image.new("RGB", size, color=(120, 30, 200)).save(path)
  return str(path)


def test_fl2va_belongs_to_the_audio_video_tasks():
  """`call_pipe` and `run` share this set, so a new enum member has to be wired here."""
  assert _AUDIO_VIDEO_TASKS == frozenset({ExampleType.T2VA, ExampleType.FL2VA, ExampleType.REF2VA})


def test_references_survive_the_input_data_round_trip():
  """`references` must not go through `asdict`, which would flatten them into plain dicts."""
  marker = object()
  data = ExampleInputData().set_references([marker]).data(_data_args())
  assert data["references"][0] is marker


def test_reference_paths_are_loaded_in_pipeline_order(tmp_path):
  """`--reference-*-path` builds the media objects, in image -> video -> audio order."""
  pytest.importorskip("av")
  image = _write_image(tmp_path, "ref.png")
  data = ExampleInputData().data(
    _data_args(reference_image_path=[image, image],
               reference_video_path=[_reference_video_asset()]))
  kinds = [reference.kind for reference in data["references"]]
  # exact order, so a reversed implementation cannot pass by having both kinds present
  assert kinds == ["image", "image", "video"]


def test_ref2va_has_its_own_conservative_preset():
  config = _minimax_h3_cache_config(_cache_args(), "ref2va")
  assert config.residual_diff_threshold == _MINIMAX_H3_CACHE_THRESHOLDS["ref2va"]
  assert config.residual_diff_threshold < _MINIMAX_H3_CACHE_THRESHOLDS["fl2va"]
  assert config.max_continuous_cached_steps == 1


def test_unknown_workflow_has_no_cache_preset():
  with pytest.raises(ValueError, match="cache preset"):
    _minimax_h3_cache_config(_cache_args(), "v2v")


def test_call_pipe_requests_audio_video_state(monkeypatch):
  monkeypatch.setattr(registers, "is_modular_pipeline", lambda pipe: True)
  stub = SimpleNamespace(init_config=SimpleNamespace(task_type=ExampleType.FL2VA))
  seen = {}

  def _pipe(**kwargs):
    seen.update(kwargs)
    return {"videos": [None], "audio": None, "sampling_rate": 32000}

  output = Example.call_pipe(stub, _pipe, {"prompt": "a fox"})
  assert seen["output"] == ["videos", "audio", "sampling_rate"]
  assert output["sampling_rate"] == 32000


def test_last_image_keeps_the_source_geometry(tmp_path):
  """With `skip_image_resize` the pipeline receives the frames it will crop itself."""
  data = ExampleInputData(
    skip_image_resize=True,
    height=448,
    width=768,
  ).data(_data_args(last_image_path=_write_image(tmp_path, "last.jpg")))
  assert data["last_image"].size == (80, 40)


def test_last_image_is_resized_without_the_flag(tmp_path):
  data = ExampleInputData(height=448, width=768).data(
    _data_args(last_image_path=_write_image(tmp_path, "last.jpg")))
  assert data["last_image"].size == (768, 448)


def test_skip_image_resize_never_reaches_the_pipeline(tmp_path):
  """`asdict` keeps `False`, and the `is not None` filter would forward it as a kwarg."""
  data = ExampleInputData(skip_image_resize=False).data(_data_args())
  assert "skip_image_resize" not in data


def _reference_video_asset() -> str:
  """Path of the bundled reference clip, used by the reference-path test."""
  from cache_dit.cli.examples import _minimax_h3_demo_asset

  return _minimax_h3_demo_asset("reference_video.mp4")


def _cache_args(**overrides) -> argparse.Namespace:
  values = dict(
    cache=True,
    num_inference_steps=50,
    residual_diff_threshold=_MINIMAX_H3_DEFAULT_THRESHOLD,
    max_warmup_steps=8,
    max_cached_steps=-1,
  )
  values.update(overrides)
  return argparse.Namespace(**values)


def test_cache_threshold_preset_differs_per_workflow():
  t2va = _minimax_h3_cache_config(_cache_args(), "t2va")
  fl2va = _minimax_h3_cache_config(_cache_args(), "fl2va")
  assert fl2va.residual_diff_threshold < t2va.residual_diff_threshold
  assert fl2va.residual_diff_threshold == _MINIMAX_H3_CACHE_THRESHOLDS["fl2va"]
  # one consecutive step at most, for every workflow
  assert fl2va.max_continuous_cached_steps == 1


def test_explicit_threshold_wins_over_the_preset():
  config = _minimax_h3_cache_config(_cache_args(residual_diff_threshold=0.3), "fl2va")
  assert config.residual_diff_threshold == 0.3


def test_resolve_transformer_follows_the_loaded_partition():
  transformer = torch.nn.Linear(2, 2)
  ref_pipe = SimpleNamespace(transformer_ref=transformer)
  assert resolve_transformer(ref_pipe) == ("transformer_ref", transformer)
  assert not hasattr(ref_pipe, "transformer")

  main_pipe = SimpleNamespace(transformer=transformer)
  assert resolve_transformer(main_pipe) == ("transformer", transformer)

  empty = SimpleNamespace()
  assert resolve_transformer(empty) == (None, None)


def test_assign_transformer_writes_the_loaded_partition():
  """Replacing the weights must not create a second, unused `transformer` attribute."""
  replacement = torch.nn.Linear(4, 4)
  pipe = SimpleNamespace(transformer_ref=torch.nn.Linear(2, 2))
  written = assign_transformer(pipe, replacement)
  assert written == "transformer_ref"
  assert pipe.transformer_ref is replacement
  assert not hasattr(pipe, "transformer")


def _tiny_minimax_h3_transformer():
  from diffusers import MiniMaxH3Transformer3DModel

  return MiniMaxH3Transformer3DModel(
    num_attention_heads=4,
    attention_head_dim=8,
    hidden_size=32,
    num_layers=1,
    num_refiner_layers=1,
    ffn_dim=64,
    in_channels=2,
    audio_in_channels=2,
    patch_size=(1, 2, 2),
    text_dim=4,
    freq_dim=8,
    time_embed_hidden_dim=32,
    time_embed_dim=16,
    rope_freq_dim=1,
  )


def test_adapter_accepts_a_pipeline_that_only_has_transformer_ref():
  transformer = _tiny_minimax_h3_transformer()
  pipe = SimpleNamespace(transformer_ref=transformer)
  adapter = minimax_h3_adapter(pipe)
  assert adapter.transformer is transformer
  assert adapter.blocks is transformer.transformer_blocks
  assert adapter.has_separate_cfg is False


def test_adapter_reports_a_missing_partition():
  with pytest.raises(ValueError, match="transformer_ref"):
    minimax_h3_adapter(SimpleNamespace())


def test_fl2va_example_defaults_to_the_demo_frames(monkeypatch):
  """The example ships a first and a *different* last frame, and skips the client-side resize."""
  from cache_dit.cli.examples import _minimax_h3_demo_asset, minimax_h3_fl2va_example
  from cache_dit.cli.generate import get_example_args

  first = _minimax_h3_demo_asset("first_frame.png")
  last = _minimax_h3_demo_asset("last_frame.png")
  assert os.path.exists(first) and os.path.exists(last), (first, last)

  # `Example.__init__` performs the (single-process) distributed init, so the namespace has to come
  # from the real parser rather than from the minimal stub the input tests use.
  monkeypatch.setattr(sys, "argv", ["cache_dit.generate", "minimax_h3_fl2va", "--steps", "1"])
  args = get_example_args()
  example = minimax_h3_fl2va_example(args)
  assert example.init_config.task_type is ExampleType.FL2VA
  assert example.init_config.modular_workflow == "fl2va"
  data = example.input_data.data(args)
  assert isinstance(data["image"], Image.Image)
  assert isinstance(data["last_image"], Image.Image)
  # both frames reach the pipeline at their own resolution: the model derives the canvas itself
  assert data["image"].size != data["last_image"].size
  assert (data["height"], data["width"]) == (448, 768)
  assert data["image"].size != (768, 448)
  assert "skip_image_resize" not in data


def test_transformer_attr_names_cover_both_partitions():
  """Each partition is reachable, and a pipeline that has both resolves the primary one."""
  assert "transformer" in TRANSFORMER_ATTR_NAMES
  assert "transformer_ref" in TRANSFORMER_ATTR_NAMES
  both = SimpleNamespace(
    transformer=torch.nn.Linear(2, 2),
    transformer_ref=torch.nn.Linear(2, 2),
  )
  assert resolve_transformer(both)[0] == TRANSFORMER_ATTR_NAMES[0]


def test_parse_audio_video_output_accepts_both_shapes():
  video = torch.zeros(3, 2, 2)
  parsed = parse_audio_video_output(
    {
      "videos": [video],
      "audio": torch.zeros(1, 2, 32000),
      "sampling_rate": 32000
    },
    ExampleType.FL2VA,
    native_fps=24,
  )
  assert parsed["video"] is video
  assert parsed["audio"].shape == (2, 32000)
  assert parsed["audio_sampling_rate"] == 32000
  assert parsed["video_fps"] == 24

  # a regular pipeline returns an object with attributes instead of a dict
  parsed = parse_audio_video_output(SimpleNamespace(videos=video, audio=None, sampling_rate=None),
                                    ExampleType.T2VA)
  assert parsed["video"] is video
  assert parsed["audio"] is None


def test_parse_audio_video_output_names_the_task_it_wanted():
  with pytest.raises(ValueError, match="FL2VA"):
    parse_audio_video_output(SimpleNamespace(videos=None), ExampleType.FL2VA)


def test_image_and_last_image_paths_are_independent(tmp_path):
  """First and last frame come from two flags, not from one shared path."""
  data = ExampleInputData(height=64, width=64).data(
    _data_args(image_path=_write_image(tmp_path, "first.png", (40, 20)),
               last_image_path=_write_image(tmp_path, "last.png", (20, 40))))
  assert data["image"].size == (64, 64)
  assert data["last_image"].size == (64, 64)
  # the two sources are distinct files, so the sizes differ before the resize
  raw = ExampleInputData().data(
    _data_args(image_path=_write_image(tmp_path, "a.png", (40, 20)),
               last_image_path=_write_image(tmp_path, "b.png", (20, 40))))
  assert raw["image"].size != raw["last_image"].size


def test_skip_image_resize_refuses_a_mask(tmp_path):
  """A mask has to line up with the image pixel for pixel, so it cannot stay unscaled."""
  with pytest.raises(ValueError, match="skip_image_resize"):
    ExampleInputData(skip_image_resize=True).data(
      _data_args(image_path=_write_image(tmp_path, "img.png"),
                 mask_image_path=_write_image(tmp_path, "mask.png")))


def test_vision_tower_is_kept_resident_and_excluded():
  """A vision-language conditioner keeps its tower: its forward reads the parameter device."""
  from cache_dit.cli.utils import _keep_vision_tower_resident, _stream_filter_excluding

  conditioner = torch.nn.Module()
  conditioner.model = torch.nn.Module()
  conditioner.model.visual = torch.nn.Sequential(torch.nn.Linear(4, 4))
  conditioner.model.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

  prefix = _keep_vision_tower_resident(conditioner, torch.device("cpu"))
  assert prefix == "model.visual"

  keep = _stream_filter_excluding(prefix)
  tower_leaf = conditioner.model.visual[0]
  decoder_leaf = conditioner.model.layers[0]
  assert keep("model.visual.0", tower_leaf) is False
  assert keep("model.layers.0", decoder_leaf) is True
  # the root itself is never a leaf target
  assert keep("", conditioner) is False


def test_text_only_conditioner_has_no_vision_tower():
  from cache_dit.cli.utils import _keep_vision_tower_resident

  conditioner = torch.nn.Module()
  conditioner.model = torch.nn.Module()
  conditioner.model.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
  assert _keep_vision_tower_resident(conditioner, torch.device("cpu")) is None


def test_reference_flags_are_ignored_by_a_workflow_that_takes_none(monkeypatch):
  """A global flag must not decode media (or import loaders) for a task that has no references."""
  from cache_dit.cli import registers as registers_mod

  def _explode(*args, **kwargs):
    raise AssertionError("references must not be built for a workflow that takes none")

  monkeypatch.setattr(registers_mod, "_references_from_args", _explode)
  inputs = ExampleInputData().set_references(["sentinel"])
  inputs.allow_references = False

  data = inputs.data(_data_args(reference_image_path=["a.png"]))
  assert "references" not in data

  # `prepare_input_data` is what sets the flag, from the task type
  example = Example(
    args=_data_args(reference_image_path=["a.png"]),
    init_config=SimpleNamespace(task_type=ExampleType.T2VA, extra_optimize_kwargs={}),
    input_data=ExampleInputData(),
  )
  kwargs, _ = example.prepare_input_data()
  assert "references" not in kwargs


def test_fl2va_defaults_skip_the_demo_asset_when_overridden(tmp_path, monkeypatch):
  """Passing both frames must not resolve (or download) the bundled pair at all."""
  from cache_dit.cli import examples as examples_mod

  def _explode(name: str):
    raise AssertionError(f"demo asset '{name}' must not be resolved when an override is given")

  monkeypatch.setattr(examples_mod, "_minimax_h3_demo_asset", _explode)
  args = _data_args(image_path=_write_image(tmp_path, "first.png"),
                    last_image_path=_write_image(tmp_path, "last.png"))
  args.quantize = False
  example = examples_mod.minimax_h3_fl2va_example(args)
  data = example.input_data.data(args)
  assert isinstance(data["image"], Image.Image)
  assert isinstance(data["last_image"], Image.Image)


def test_ref2va_defaults_skip_the_demo_media_when_overridden(tmp_path, monkeypatch):
  """A reference override replaces the bundled pair before it is decoded."""
  pytest.importorskip("av")
  from cache_dit.cli import examples as examples_mod

  def _explode(name: str):
    raise AssertionError(f"demo asset '{name}' must not be resolved when references are given")

  monkeypatch.setattr(examples_mod, "_minimax_h3_demo_asset", _explode)
  args = _data_args(reference_image_path=[_write_image(tmp_path, "ref.png")])
  args.quantize = False
  example = examples_mod.minimax_h3_ref2va_example(args)
  data = example.input_data.data(args)
  assert len(data["references"]) == 1


def test_compile_targets_the_partition_the_pipeline_actually_has(monkeypatch):
  """`ref2va` has no `transformer`, so a compile that only looks for one would silently no-op."""
  from cache_dit.cli import utils as utils_mod

  compiled = []
  monkeypatch.setattr(utils_mod, "set_compile_configs", lambda **kwargs: None)
  monkeypatch.setattr(utils_mod, "_should_defer_few_shot_compile", lambda args, module: False)
  monkeypatch.setattr(utils_mod, "_compile_transformer_module",
                      lambda args, pipe, module, name: compiled.append(name))

  transformer_ref = torch.nn.Linear(2, 2)
  pipe = SimpleNamespace(transformer_ref=transformer_ref)
  args = _data_args()
  args.compile = True
  args.svdq_few_shot_compile = False
  args.cuda_graph = False
  args.ulysses_anything = False

  utils_mod.maybe_compile_transformer(args, pipe)
  assert compiled == ["transformer_ref"]


def test_quantize_targets_the_partition_the_pipeline_actually_has(monkeypatch):
  """`ref2va` has no `transformer`; quantizing must not silently leave it bf16."""
  from cache_dit.cli import utils as utils_mod

  quantized = []
  marker = torch.nn.Linear(2, 2)

  def _fake_quantize(module, quantize_config=None):
    quantized.append(module)
    return marker

  monkeypatch.setattr(utils_mod, "quantize", _fake_quantize)
  args = utils_mod.get_base_args(parse=False).parse_args([])
  args.quantize = True
  args.quantize_type = "float8_per_row"
  args.compile = False
  args.cuda_graph = False

  original = torch.nn.Linear(2, 2)
  pipe = SimpleNamespace(transformer_ref=original)
  utils_mod.maybe_quantize_transformer(args, pipe)

  assert quantized == [original]
  assert pipe.transformer_ref is marker


def test_summary_reads_the_partition_the_pipeline_actually_has(monkeypatch):
  """`strify` runs on every save, so a ref-only pipeline must not raise there."""
  import importlib

  # `cache_dit.summary` is both a module and a re-exported function, so import the module itself
  summary_mod = importlib.import_module("cache_dit.summary")
  monkeypatch.setattr(summary_mod, "is_pipeline", lambda obj: True)
  stats = summary_mod.summary(SimpleNamespace(transformer_ref=torch.nn.Linear(2, 2)), logging=False)
  # an untracked ref partition reports the empty default rather than raising
  assert len(stats) == 1


def test_cache_teardown_visits_the_partition_the_pipeline_actually_has(monkeypatch):
  """`disable_cache` has to unpatch `transformer_ref.forward` as well."""
  from cache_dit.caching.cache_adapters import cache_adapter as cache_adapter_mod

  monkeypatch.setattr(cache_adapter_mod, "is_pipeline", lambda obj: True)
  released = []
  module = torch.nn.Linear(2, 2)
  pipe = SimpleNamespace(transformer_ref=module)

  cache_adapter_mod.CachedAdapter.release_hooks(pipe, _release_transformer=released.append)
  assert released == [module]


@pytest.mark.parametrize("flags,collector,stack,nvtx,layerwise", [
  ([], "torch", True, False, False),
  (["--profile-with-stack"], "torch", True, False, False),
  (["--no-profile-with-stack"], "torch", False, False, False),
  (["--profile-collector", "nsys"], "nsys", True, False, False),
  (["--nvtx"], "torch", True, True, False),
  (["--layerwise-nvtx"], "torch", True, False, True),
  (["--nvtx", "--layerwise-nvtx"], "torch", True, True, True),
])
def test_profiler_flags_forward_configuration_without_enabling_collection(
    flags, collector, stack, nvtx, layerwise):
  args = get_args(parse=False).parse_args(flags)
  request = create_profiler_from_args(args)
  assert (request.collector, request.with_stack, args.nvtx,
          args.layerwise_nvtx) == (collector, stack, nvtx, layerwise)
  assert not request.enabled


@pytest.fixture(params=["t2va"])
def profiled_workflow(request, monkeypatch, tmp_path):
  """Exercise the CLI lifecycle once, sharing the collector and annotation probes."""
  from cache_dit.cli import examples

  workflow = request.param
  root_name = "transformer_ref" if workflow == "ref2va" else "transformer"
  state = SimpleNamespace(events=[], requests=[], marks=[], stack=[], messages=[])
  modules = [torch.nn.Linear(2, 2) for _ in range(4)]
  args = get_args(parse=False).parse_args([
    "--prompt",
    "formal",
    "--warmup-prompt",
    "warmup",
    "--warmup",
    "1",
    "--repeat",
    "2",
    "--steps",
    "5",
    "--warmup-num-inference-steps",
    "2",
    "--seed",
    "11",
    "--warmup-seed",
    "23",
    "--generator-device",
    "cpu",
    "--profile",
    "--profile-ranges",
    "0,2",
    "--profile-name",
    "fixed",
    "--profile-dir",
    str(tmp_path),
    "--profile-activities",
    "CPU",
    "--no-profile-with-stack",
  ])
  args.image_path = args.last_image_path = _write_image(tmp_path, "provided.png")
  args.reference_image_path = [args.image_path]
  args.example = "minimax_h3" if workflow == "t2va" else f"minimax_h3_{workflow}"
  args.track_memory = args.cache_summary = False

  class Pipe:

    def set_progress_bar_config(self, **kwargs):
      pass

    def __call__(self, **kwargs):
      state.events.append(("call", kwargs["prompt"], kwargs["num_inference_steps"],
                           kwargs["generator"].initial_seed()))
      state.events.append(("hooks", tuple(getattr(self, root_name)._forward_pre_hooks),
                           tuple(self.transformer_2._forward_pre_hooks)))
      with stage_range("test_request"):
        getattr(self, root_name)(torch.ones(1, 2))
        self.transformer_2(torch.ones(1, 2))
        with profile_steps(range(kwargs["num_inference_steps"])) as steps:
          list(steps)
      if kwargs["prompt"] == "warmup":
        setattr(self, root_name, modules[2])
      return {"videos": [[Image.new("RGB", (1, 1))]], "audio": None}

  pipe = Pipe()
  setattr(pipe, root_name, modules[0])
  pipe.transformer_2 = modules[3]

  def optimize(*args, **kwargs):
    state.events.append(("optimize", ))
    setattr(pipe, root_name, modules[1])

  original_roots = registers.iter_transformer_modules

  def roots(pipe):
    state.events.append(("roots", ))
    assert getattr(pipe, root_name) is modules[2]
    return original_roots(pipe)

  @contextmanager
  def create_request(args, **kwargs):
    request = create_profiler_from_args(args, **kwargs)
    state.requests.append(request)
    state.events.append(("enter", ))
    try:
      with request:
        yield request
    finally:
      state.events.append(("exit", ))

  def push(name):
    state.stack.append(name)
    state.marks.append(name)

  module = importlib.import_module("cache_dit.profiler")
  monkeypatch.setattr(
    module, "current_platform",
    SimpleNamespace(device_type="cuda",
                    is_accelerator_available=lambda: True,
                    synchronize=lambda device: None))
  nvtx_module = importlib.import_module("cache_dit.layerwise_nvtx")
  monkeypatch.setattr(nvtx_module, "current_platform", SimpleNamespace(device_type="cuda"))
  monkeypatch.setattr(torch.cuda.nvtx, "range_push", push)
  monkeypatch.setattr(torch.cuda.nvtx, "range_pop", state.stack.pop)
  monkeypatch.setattr(registers, "maybe_init_distributed", lambda args: (0, "cpu"))
  monkeypatch.setattr(registers, "get_rank_device", lambda: (0, torch.device("cpu")))
  monkeypatch.setattr(registers, "maybe_apply_optimization", optimize)
  monkeypatch.setattr(registers, "maybe_finalize_deferred_svdq_pipe_move",
                      lambda pipe: state.events.append(("finalize", )))
  monkeypatch.setattr(registers, "maybe_destroy_distributed", lambda: None)
  monkeypatch.setattr(registers, "iter_transformer_modules", roots)
  monkeypatch.setattr(registers, "create_profiler_from_args", create_request)
  monkeypatch.setattr(registers, "strify", lambda *args: "test")
  monkeypatch.setattr(
    registers.logger, "info", lambda message, *args: state.messages.append(message % args
                                                                           if args else message))
  monkeypatch.setattr(ExampleOutputData, "save", lambda *args: None)
  monkeypatch.setattr(ExampleOutputData, "summary", lambda *args: None)
  example_name = "minimax_h3_example" if workflow == "t2va" else f"minimax_h3_{workflow}_example"
  state.example = getattr(examples, example_name)(args)
  monkeypatch.setattr(state.example.init_config, "get_pipe", lambda args: pipe)
  monkeypatch.setattr(state.example, "prepare_input_data", lambda: ({
    "prompt": "formal",
    "num_inference_steps": 5
  }, {}))
  state.modules = modules
  return state


@pytest.mark.parametrize("profiled_workflow,mode,profile", [
  ("t2va", "disabled", False),
  ("t2va", "disabled", True),
  ("t2va", "stage", False),
  ("t2va", "stage", True),
  ("t2va", "layerwise", False),
  ("t2va", "layerwise", True),
  ("fl2va", "layerwise", False),
  ("ref2va", "layerwise", True),
],
                         indirect=["profiled_workflow"])
def test_profiler_cli_preserves_workflow_and_warmup_order(mode, profile, profiled_workflow):
  from cache_dit.models.pipelines.minimax_h3 import minimax_h3_blocks

  state = profiled_workflow
  example = state.example
  workflow = example.init_config.modular_workflow
  assert example.init_config.modular_blocks_factory is minimax_h3_blocks
  assert workflow == ("t2va" if example.args.example == "minimax_h3" else
                      example.args.example.removeprefix("minimax_h3_"))
  root_name = "transformer_ref" if workflow == "ref2va" else "transformer"
  example.args.profile = profile
  example.args.nvtx = mode == "stage"
  example.args.layerwise_nvtx = mode == "layerwise"
  example.run()
  assert [e for e in state.events if e[0] in ("optimize", "call", "finalize", "enter", "exit")] == [
    ("optimize", ),
    ("call", "warmup", 2, 23),
    ("finalize", ),
    ("enter", ),
    ("call", "formal", 5, 11),
    ("finalize", ),
    ("exit", ),
    ("enter", ),
    ("call", "formal", 5, 11),
    ("finalize", ),
    ("exit", ),
  ]
  assert len(state.requests) == 2
  assert all(r.enabled is profile and r.devices == (example.device, ) for r in state.requests)
  assert state.events.count(("roots", )) == int(mode == "layerwise")
  hooks = [e[1:] for e in state.events if e[0] == "hooks"]
  assert hooks[0] == ((), ())
  assert hooks[1] == hooks[2]
  assert all(bool(ids) is (mode == "layerwise") for ids in hooks[1])
  assert state.marks.count("test_request") == (0 if mode == "disabled" else 2)
  assert sum(name.startswith(f"{root_name} (Linear)")
             for name in state.marks) == (2 if mode == "layerwise" else 0)
  assert sum(name.startswith("transformer_2 (Linear)")
             for name in state.marks) == (2 if mode == "layerwise" else 0)
  assert not state.stack
  assert all(not m._forward_pre_hooks and not m._forward_hooks for m in state.modules)
  paths = [r.trace_path for request in state.requests for r in request.results]
  assert len(paths) == len(set(paths)) == (4 if profile else 0)
  assert all(p.is_file() and p.name.startswith("fixed-") for p in paths)
  assert all(r.actual_range == r.requested_range == (i, i) for request in state.requests
             for r, i in zip(request.results, (0, 2)))
  assert sum(m.startswith("Profiler loop=") for m in state.messages) == len(paths)
  marks = list(state.marks)
  with stage_range("outside"):
    state.modules[2](torch.ones(1, 2))
  assert state.marks == marks


@pytest.mark.parametrize("ranges,count", [("all", 2), ("10-14", 0)])
def test_cli_logs_whole_and_unmatched_requests(ranges, count, profiled_workflow):
  state = profiled_workflow
  state.example.args.profile_ranges = ranges
  state.example.run()
  windows = [m for m in state.messages if m.startswith("Profiler loop=")]
  assert len(windows) == count
  if count:
    assert all(
      str(r.trace_path) in " ".join(windows) for request in state.requests for r in request.results)
  else:
    assert sum(
      m.startswith("Profiler produced no windows: No denoising steps matched")
      for m in state.messages) == 2
    assert all(not r.results and r.trace_path is None for r in state.requests)


def test_balanced_cli_provides_all_visible_devices(profiled_workflow, monkeypatch):
  state = profiled_workflow
  state.example.args.device_map_balance = True
  monkeypatch.setattr(registers, "current_platform", SimpleNamespace(device_count=lambda: 2))
  state.example.run()
  assert all(request.devices == (0, 1) for request in state.requests)


def test_profiler_cli_factory_accepts_existing_namespaces_and_explicit_devices():
  context = create_profiler_from_args(argparse.Namespace(profile=False), devices=[0, 1])
  assert context.ranges == "all" and context.devices == (0, 1)


@pytest.mark.parametrize("layerwise,profile", [(False, False), (True, True)])
def test_cli_model_error_cleans_annotations_and_requests(layerwise, profile, profiled_workflow,
                                                         monkeypatch):
  state = profiled_workflow
  state.example.args.nvtx = not layerwise
  state.example.args.layerwise_nvtx = layerwise
  state.example.args.profile = profile
  error = ValueError("formal inference failed")

  def fail(*args):
    raise error

  monkeypatch.setattr(state.modules[2], "forward", fail)
  with pytest.raises(ValueError) as caught:
    state.example.run()
  assert caught.value is error
  assert len(state.requests) == 1
  assert not state.stack
  assert all(not m._forward_pre_hooks and not m._forward_hooks for m in state.modules)
  assert all(r.profiler is None for r in state.requests)
  marks = list(state.marks)
  with stage_range("outside"):
    pass
  assert state.marks == marks
