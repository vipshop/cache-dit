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
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from cache_dit._utils import registers
from cache_dit._utils.examples import (
  _MINIMAX_H3_CACHE_THRESHOLDS,
  _MINIMAX_H3_DEFAULT_THRESHOLD,
  _minimax_h3_cache_config,
)
from cache_dit._utils.registers import (
  ExampleInputData,
  ExampleType,
  _AUDIO_VIDEO_TASKS,
  Example,
  parse_audio_video_output,
)
from cache_dit.caching.block_adapters.adapters import minimax_h3_adapter
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
  from cache_dit._utils.examples import _minimax_h3_demo_asset

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
  from cache_dit._utils.examples import _minimax_h3_demo_asset, minimax_h3_fl2va_example
  from cache_dit._utils.generate import get_example_args

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
  from cache_dit._utils.utils import _keep_vision_tower_resident, _stream_filter_excluding

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
  from cache_dit._utils.utils import _keep_vision_tower_resident

  conditioner = torch.nn.Module()
  conditioner.model = torch.nn.Module()
  conditioner.model.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
  assert _keep_vision_tower_resident(conditioner, torch.device("cpu")) is None
