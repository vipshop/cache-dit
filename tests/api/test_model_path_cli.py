"""`cache_dit.generate {MODEL_PATH}` matching, `--task`, and how the path reaches an example."""

import sys

import pytest
from PIL import Image

from cache_dit.cli import examples as examples_mod
from cache_dit.cli import generate
from cache_dit.cli.registers import EXAMPLE_TASK_TYPES, ExampleRegister, ExampleType, same_model
from cache_dit.cli.utils import get_base_args, maybe_postprocess_args

LOCAL = "/cache-dit-tests/local"
SLOT_KEYS = {
  "lora_path": "lora",
  "controlnet_path": "controlnet",
  "transformer_path": "transformer"
}


def cli(argv, monkeypatch):
  """Run `generate.get_example_args` on `argv` and return the parsed namespace."""

  monkeypatch.setattr(sys, "argv", ["cache_dit.generate", *argv])
  return generate.get_example_args()


# ----------------------------------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [
  "black-forest-labs/FLUX.1-dev",
  "/data/FLUX.1-dev",
  "/data/flux.1-dev/",
  "~/models/FLUX.1-dev",
  "$HOME/FLUX.1-dev",
  "/home/u/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev",
  "/home/u/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/abc",
])
def test_same_model_accepts_local_dirs_repo_ids_and_hf_cache(path):
  assert same_model(path, "black-forest-labs/FLUX.1-dev")
  assert ExampleRegister.select_example(path)[0] == "flux"


@pytest.mark.parametrize("path", [
  "/data/FLUX.1-Fill-dev",
  "/data/FLUX.1-dev-optimized",
  "/data/runs/FLUX.1-dev/ckpt",
  "/data/models--other-org--FLUX.1-dev",
  "",
])
def test_same_model_rejects_other_models(path):
  assert not same_model(path, "black-forest-labs/FLUX.1-dev")


def test_hf_cache_matching_follows_the_repo_layout():
  assert same_model("/hf/models--Qwen--Qwen-Image-2.1/snapshots/x", "Qwen/Qwen-Image-2.1")
  assert not same_model("/hf/models--Qwen/snapshots/x", "Qwen/Qwen-Image-2.1")


# ----------------------------------------------------------------------------------------------
# Default task and --task
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path, task, expected", [
  ("/m/LTX-2", None, "ltx2_t2v"),
  ("/m/LTX-2", ExampleType.I2V, "ltx2_i2v"),
  ("/m/LTX-2.5-Diffusers", None, "ltx25_t2v"),
  ("/m/LTX-2.5-Diffusers", ExampleType.I2V, "ltx25_i2v"),
  ("/m/MiniMax-H3", None, "minimax_h3"),
  ("/m/MiniMax-H3", ExampleType.FL2VA, "minimax_h3_fl2va"),
  ("/m/MiniMax-H3", ExampleType.REF2VA, "minimax_h3_ref2va"),
  ("/m/Qwen-Image-2.1", None, "qwen_image_21"),
  ("/m/Qwen-Image-2.1", ExampleType.IE2I, "qwen_image_21_edit"),
  ("/m/Qwen-Image-2.1-Turbo", None, "qwen_image_21_turbo"),
  ("/m/Qwen-Image-2.1-Turbo", ExampleType.IE2I, "qwen_image_21_turbo_edit"),
  ("/m/flux-3-image-klein", None, "flux3"),
  ("/m/flux-3-image-klein", ExampleType.IE2I, "flux3_edit"),
  ("/m/FLUX.2-klein-4B", None, "flux2_klein_4b"),
  ("/m/FLUX.2-klein-4B", ExampleType.IE2I, "flux2_klein_4b_edit"),
  ("/m/Cosmos3-Nano", None, "cosmos3_omni_t2i"),
  ("/m/Cosmos3-Nano", ExampleType.T2V, "cosmos3_omni_t2v"),
  ("/m/Pika-H3-Flex", None, "pika_h3_flex"),
  ("/m/Pika-H3-Flex", ExampleType.FL2VA, "pika_h3_flex_fl2va"),
])
def test_default_task_and_task_selection(path, task, expected):
  assert ExampleRegister.select_example(path, task)[0] == expected


def test_ties_keep_the_registration_order():
  # Both MAGI-2 examples drive T2VA.
  assert ExampleRegister.select_example("/m/MAGI-2-preview")[0] == "magi2_t2va"
  assert ExampleRegister.select_example("/m/MAGI-2-preview", ExampleType.T2VA)[0] == "magi2_t2va"


def test_errors_name_the_candidates():
  with pytest.raises(KeyError, match="No example matches model path '/m/nope'"):
    ExampleRegister.select_example("/m/nope")
  with pytest.raises(LookupError,
                     match=r"task IE2I; available: ltx2_t2v \(T2V\), ltx2_i2v \(I2V\)"):
    ExampleRegister.select_example("/m/LTX-2", ExampleType.IE2I)


# ----------------------------------------------------------------------------------------------
# Registry metadata
# ----------------------------------------------------------------------------------------------


def test_task_table_matches_registry_and_priority_covers_every_used_task():
  assert set(EXAMPLE_TASK_TYPES) == set(ExampleRegister.list_examples())
  used = set(EXAMPLE_TASK_TYPES.values())
  priority = generate.ExampleRegister.select_example.__globals__["_DEFAULT_TASK_PRIORITY"]
  assert set(priority) >= used
  assert len(set(priority)) == len(priority)


def test_every_shared_default_is_reachable_by_task():
  defaults = {}
  for name in ExampleRegister.list_examples():
    defaults.setdefault(ExampleRegister.get_default(name).lower(), []).append(name)
  for default, names in defaults.items():
    path = f"/m/{default.rsplit('/', 1)[-1]}"
    for task in {EXAMPLE_TASK_TYPES[n] for n in names}:
      chosen, candidates = ExampleRegister.select_example(path, task)
      assert chosen in candidates
      assert EXAMPLE_TASK_TYPES[chosen] is task


# ----------------------------------------------------------------------------------------------
# CLI resolution
# ----------------------------------------------------------------------------------------------


def test_path_positional_selects_the_example_and_sets_the_model_path(monkeypatch):
  args = cli([f"{LOCAL}/FLUX.1-dev", "--cache"], monkeypatch)
  assert args.task == "generate"
  assert args.example == "flux"
  assert args.model_path == f"{LOCAL}/FLUX.1-dev"
  assert args.cache is True


def test_task_flag_selects_among_shared_models(monkeypatch):
  args = cli([f"{LOCAL}/LTX-2", "--task", "i2v"], monkeypatch)
  assert (args.example, args.task_type) == ("ltx2_i2v", "I2V")


def test_path_of_a_variant_goes_to_the_slot_its_weights_are_loaded_from(monkeypatch):
  args = cli([f"{LOCAL}/nunchaku-flux.1-dev"], monkeypatch)
  assert (args.example, args.transformer_path) == ("flux_nunchaku", f"{LOCAL}/nunchaku-flux.1-dev")
  assert args.model_path is None
  args = cli([f"{LOCAL}/Qwen-Image-Lightning", "--task", "ie2i"], monkeypatch)
  assert (args.example, args.lora_path) == ("qwen_image_edit_lightning",
                                            f"{LOCAL}/Qwen-Image-Lightning")
  assert args.model_path is None


@pytest.mark.parametrize("argv", [
  [f"{LOCAL}/nope"],
  [f"{LOCAL}/LTX-2", "--task", "T2I"],
  [f"{LOCAL}/FLUX.1-dev", "flux"],
  [f"{LOCAL}/FLUX.1-dev", "--model-path", "/other"],
  [f"{LOCAL}/nunchaku-flux.1-dev", "--transformer-path", "/other"],
  ["ltx2_i2v", "--task", "t2v"],
  ["generate", "nope"],
  ["nope"],
  ["flux", "bar"],
  ["--task", "T2I"],
  ["--task", "nope"],
  ["--task"],
])
def test_bad_input_exits_with_an_error(argv, monkeypatch, capsys):
  with pytest.raises(SystemExit) as exit_info:
    cli(argv, monkeypatch)
  assert exit_info.value.code == 2
  assert "error:" in capsys.readouterr().err


def test_path_environment_variables_are_expanded(monkeypatch):
  monkeypatch.setenv("MY_FLUX_DIR", f"{LOCAL}/FLUX.1-dev")
  args = cli(["$MY_FLUX_DIR"], monkeypatch)
  assert (args.example, args.model_path) == ("flux", f"{LOCAL}/FLUX.1-dev")


def test_model_path_flag_still_moves_the_base_checkpoint_of_a_slot_example(monkeypatch):
  args = cli([f"{LOCAL}/nunchaku-flux.1-dev", "--model-path", f"{LOCAL}/FLUX.1-dev"], monkeypatch)
  assert args.example == "flux_nunchaku"
  assert args.transformer_path == f"{LOCAL}/nunchaku-flux.1-dev"
  assert args.model_path == f"{LOCAL}/FLUX.1-dev"


@pytest.mark.parametrize("argv, expected", [
  (["flux"], "flux"),
  (["generate", "flux"], "flux"),
  (["flux_nunchaku"], "flux_nunchaku"),
])
def test_example_name_form_is_unchanged(argv, expected, monkeypatch):
  args = cli(argv, monkeypatch)
  assert (args.task, args.example, args.model_path) == ("generate", expected, None)


def test_example_name_with_matching_task_is_accepted(monkeypatch):
  args = cli(["ltx2_i2v", "--task", "I2V"], monkeypatch)
  assert args.example == "ltx2_i2v"


def test_list_task(monkeypatch):
  assert cli(["list"], monkeypatch).task == "list"


# ----------------------------------------------------------------------------------------------
# How the model path reaches an example
# ----------------------------------------------------------------------------------------------


class _Capture:
  """Stands in for `Example` so a factory can run without the distributed init."""

  def __init__(self, **kwargs):
    self.kwargs = kwargs


BASE = "/cache-dit-tests/base"


def run_factory(name, monkeypatch, sentinel, image_path, calls, base=None):
  """Build one example with `sentinel` in the slot a model path goes to; log the `_path` calls."""

  args = maybe_postprocess_args(get_base_args(parse=False).parse_args([]))
  args.example = name
  args.image_path = image_path
  args.last_image_path = image_path
  if base is not None:
    args.model_path = base
  setattr(args, generate._PATH_SLOT_EXAMPLES.get(name, "model_path"), str(sentinel))
  original = examples_mod._path

  def spy(default, *more, **kwargs):
    result = original(default, *more, **kwargs)
    slot = next((SLOT_KEYS[key] for key, on in (
      ("lora_path", "lora"),
      ("controlnet_path", "controlnet"),
      ("transformer_path", "transformer"),
    ) if kwargs.get(on)), None)
    calls.append((default, slot, result))
    return result

  monkeypatch.setattr(examples_mod, "_path", spy)
  monkeypatch.setattr(examples_mod, "Example", _Capture)
  monkeypatch.setattr(examples_mod, "load_image", lambda *a, **k: Image.new("RGB", (8, 8)))
  monkeypatch.setattr(examples_mod,
                      "load_video",
                      lambda *a, **k: [Image.new("RGB", (8, 8))] * 4,
                      raising=False)
  return ExampleRegister.get_example(args, name)


def error_chain(error):
  texts = []
  while error is not None and str(error) not in texts:
    texts.append(str(error))
    error = error.__cause__ or error.__context__
  return texts


# Factories that read a real checkpoint (config / weights / adapter file) while building the
# example. A fake directory stops them there, after the `_path` call that took the model path, so
# for these the checks run on the `_path` calls made before the load.
_LOADS_CHECKPOINT_WHILE_BUILDING = frozenset({
  "helios_t2v",
  "helios_t2v_distill",
  "pika_h3_flex",
  "pika_h3_flex_vendored",
  "pika_h3_flex_fl2va",
  "pika_h3_flex_ref2va",
  "pika_h3_flex_fl2va_vendored",
  "pika_h3_flex_ref2va_vendored",
  "qwen_image_controlnet",
  "skyreels_v2",
  "wan2.1_vace",
  "wan2.2_vace",
  "zimage_turbo_controlnet_2.0",
  "zimage_turbo_controlnet_2.1",
})


def init_config_of(example):
  return example.kwargs["init_config"] if isinstance(example, _Capture) else example.init_config


@pytest.mark.parametrize("name", ExampleRegister.list_examples())
def test_every_example_loads_the_model_path_from_its_slot(monkeypatch, name, tmp_path):
  sentinel = tmp_path / name
  # A directory that exists: the model loaders then fail on its content, naming the path.
  sentinel.mkdir()
  sentinel = str(sentinel)
  image_path = str(tmp_path / "frame.png")
  Image.new("RGB", (8, 8)).save(image_path)
  calls = []
  base = BASE if name in examples_mod._MODEL_PATH_BASE_DEFAULTS else None
  example = None
  try:
    example = run_factory(name, monkeypatch, sentinel, image_path, calls, base)
  except ModuleNotFoundError as error:
    pytest.skip(f"{name} needs the optional package {error.name}")
  except (OSError, ValueError) as error:
    assert name in _LOADS_CHECKPOINT_WHILE_BUILDING, f"{name}: {error_chain(error)[-1]}"
  registered = ExampleRegister.get_default(name)
  hits = [call for call in calls if call[2] == sentinel]
  assert hits, f"no `_path` call of {name} loads the model path: {calls}"
  # The path lands on exactly the registered model, in the slot its weights are loaded from.
  slot = generate._PATH_SLOT_EXAMPLES.get(name, "model_path")
  assert {hit[0] for hit in hits} == {registered}, calls
  assert {hit[1] for hit in hits} == {SLOT_KEYS.get(slot)}, calls
  if base is not None and example is not None:
    # `--model-path` moves the base checkpoint of an example registered with other weights.
    base_hits = [call for call in calls if call[2] == base]
    assert base_hits and all(hit[1] is None and hit[0] != registered for hit in base_hits), calls
  if example is not None:
    # The task table mirrors what the factory sets.
    assert init_config_of(example).task_type is EXAMPLE_TASK_TYPES[name]


@pytest.mark.parametrize("name, base_default", [
  ("flux_nunchaku", "black-forest-labs/FLUX.1-dev"),
  ("qwen_image_controlnet", "Qwen/Qwen-Image"),
  ("zimage_turbo_controlnet_2.0", "Tongyi-MAI/Z-Image-Turbo"),
  ("zimage_turbo_controlnet_2.1", "Tongyi-MAI/Z-Image-Turbo"),
])
def test_model_path_flag_reaches_the_base_of_slot_examples(monkeypatch, name, base_default):
  # Their factories load the registered weights before the base `_path` call, so check `_path`
  # directly: a plain call borrows the args, the registered weights' call does not.
  args = maybe_postprocess_args(get_base_args(parse=False).parse_args(["--model-path", BASE]))
  monkeypatch.setattr(ExampleRegister, "_active_example", (args, name))
  assert examples_mod._path(base_default) == BASE
  assert examples_mod._path("google/gemma-3-12b-it") == "google/gemma-3-12b-it"
  args.model_path = None
  # Without a model path the env var / default lookup is what it was.
  assert examples_mod._path(base_default) != BASE


def test_auxiliary_models_and_sibling_variants_keep_their_defaults(monkeypatch):
  # gemma is the Echo conditioner, the Wan2.1 base is not the CausVid student.
  for name, kept in (("echo_t2va", "google/gemma-3-12b-it"), ("causvid_wan",
                                                              "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")):
    calls = []
    run_factory(name, monkeypatch, f"/cache-dit-tests/{name}", "frame.png", calls)
    assert [call[2] for call in calls if call[0] == kept] == [kept]


def test_sibling_variant_keeps_its_own_checkpoint(monkeypatch):
  calls = []
  run_factory("flux2_klein_4b", monkeypatch, "/cache-dit-tests/flux2_klein_4b", "frame.png", calls)
  taken = [call[0] for call in calls if call[2] == "/cache-dit-tests/flux2_klein_4b"]
  assert taken == ["black-forest-labs/FLUX.2-klein-4B"]
  assert all("4B" in call[0] or "4b" in call[0] for call in calls
             if call[1] is None and call[2] != call[0])


def test_factory_without_an_active_example_ignores_the_model_path(monkeypatch):
  # Direct factory calls (tests, bench) keep the pre-existing behavior of `_path`.
  monkeypatch.delenv("FLUX_DIR", raising=False)
  assert examples_mod._path("black-forest-labs/FLUX.1-dev") == "black-forest-labs/FLUX.1-dev"
  assert ExampleRegister.active_example() is None


def test_model_path_flag_now_reaches_examples_that_did_not_thread_args(monkeypatch):
  sentinel = "/cache-dit-tests/flux"
  args = maybe_postprocess_args(get_base_args(parse=False).parse_args(["--model-path", sentinel]))
  args.example = "flux"
  monkeypatch.setattr(examples_mod, "Example", _Capture)
  init_config = ExampleRegister.get_example(args, "flux").kwargs["init_config"]
  assert init_config.model_name_or_path == sentinel
