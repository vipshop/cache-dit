from argparse import Namespace
from types import SimpleNamespace

import torch
from PIL import Image

from cache_dit._utils import registers as example_registers
from cache_dit._utils.registers import Example
from cache_dit._utils.registers import ExampleInitConfig
from cache_dit._utils.registers import ExampleInputData
from cache_dit._utils.registers import ExampleType
from cache_dit._utils.utils import get_args
from cache_dit._utils.utils import maybe_postprocess_args


def test_examples_cli_accepts_warmup_seed() -> None:
  parser = get_args(parse=False)

  args = maybe_postprocess_args(parser.parse_args(["--warmup-seed", "123"]))

  assert args.warmup_seed == 123


def test_examples_cli_accepts_warmup_prompt() -> None:
  parser = get_args(parse=False)

  args = maybe_postprocess_args(parser.parse_args(["--warmup-prompt", "warmup text"]))

  assert args.warmup_prompt == "warmup text"


def test_example_new_generator_uses_warmup_seed_only_for_warmup(monkeypatch, ) -> None:
  monkeypatch.setattr(example_registers, "maybe_init_distributed", lambda _args: (0, "cpu"))
  monkeypatch.setattr(example_registers, "get_rank_device", lambda: (0, torch.device("cpu")))
  args = Namespace(generator_device=None, seed=11, warmup_seed=23)
  example = Example(args=args, input_data=ExampleInputData(seed=7))

  warmup_kwargs = example.new_generator({}, args, warmup=True)
  inference_kwargs = example.new_generator({}, args, warmup=False)

  assert warmup_kwargs["generator"].initial_seed() == 23
  assert inference_kwargs["generator"].initial_seed() == 11


def test_example_new_generator_warmup_falls_back_to_regular_seed_when_unset(monkeypatch, ) -> None:
  monkeypatch.setattr(example_registers, "maybe_init_distributed", lambda _args: (0, "cpu"))
  monkeypatch.setattr(example_registers, "get_rank_device", lambda: (0, torch.device("cpu")))
  args = Namespace(generator_device=None, seed=11, warmup_seed=None)
  example = Example(args=args, input_data=ExampleInputData(seed=7))

  warmup_kwargs = example.new_generator({}, args, warmup=True)

  assert warmup_kwargs["generator"].initial_seed() == 11


def test_example_run_uses_warmup_prompt_only_for_warmup(monkeypatch) -> None:
  parser = get_args(parse=False)
  args = maybe_postprocess_args(
    parser.parse_args([
      "--prompt",
      "formal prompt",
      "--warmup-prompt",
      "warmup prompt",
      "--warmup",
      "1",
      "--repeat",
      "1",
    ]))
  args.track_memory = False
  args.profile = False
  args.cache_summary = False
  args.example = None

  class _DummyPipe:

    def __init__(self):
      self.prompts = []

    def set_progress_bar_config(self, **_kwargs) -> None:
      return None

    def __call__(self, **kwargs):
      self.prompts.append(kwargs["prompt"])
      return Namespace(images=[Image.new("RGB", (1, 1))])

  pipe = _DummyPipe()

  monkeypatch.setattr(example_registers, "maybe_init_distributed", lambda _args: (1, "cpu"))
  monkeypatch.setattr(example_registers, "get_rank_device", lambda: (1, torch.device("cpu")))
  monkeypatch.setattr(example_registers, "maybe_apply_optimization",
                      lambda _args, _pipe, **_kwargs: _pipe)
  monkeypatch.setattr(example_registers, "maybe_destroy_distributed", lambda: None)
  monkeypatch.setattr(example_registers, "strify", lambda _args, _pipe: "test")

  init_config = ExampleInitConfig(
    task_type=ExampleType.T2I,
    model_name_or_path="dummy",
    pipeline_class=None,
  )
  monkeypatch.setattr(init_config, "get_pipe", lambda _args: pipe)

  example = Example(
    args=args,
    init_config=init_config,
    input_data=ExampleInputData(prompt="default prompt", num_inference_steps=2),
  )

  example.run()

  assert pipe.prompts == ["warmup prompt", "formal prompt"]


def test_example_init_config_summary_includes_warmup_overrides() -> None:
  init_config = ExampleInitConfig(
    task_type=ExampleType.T2I,
    model_name_or_path="dummy-model",
    pipeline_class=None,
  )
  args = Namespace(
    model_path=None,
    warmup_seed=23,
    warmup_prompt="warmup prompt",
  )

  summary = init_config.summary(args)

  assert "- Warmup Seed: 23" in summary
  assert "- Warmup Prompt: warmup prompt" in summary


def test_example_input_data_defaults_generator_to_rank_accelerator(monkeypatch) -> None:
  monkeypatch.setattr(example_registers, "get_rank_device", lambda: (0, torch.device("cuda", 0)))

  data = ExampleInputData(prompt="x")

  assert data.resolved_gen_device() == "cuda:0"


def test_example_input_data_keeps_explicit_generator_device(monkeypatch) -> None:
  monkeypatch.setattr(example_registers, "get_rank_device", lambda: (0, torch.device("cuda", 0)))

  data = ExampleInputData(prompt="x", gen_device="cpu")

  assert data.resolved_gen_device() == "cpu"


def test_example_input_data_falls_back_to_cpu_on_non_cuda_accelerator(monkeypatch) -> None:
  # A bare device stand-in: `torch.device("npu")` only parses when torch_npu is installed.
  monkeypatch.setattr(example_registers, "get_rank_device", lambda:
                      (0, SimpleNamespace(type="npu")))

  data = ExampleInputData(prompt="x")

  assert data.resolved_gen_device() == "cpu"
