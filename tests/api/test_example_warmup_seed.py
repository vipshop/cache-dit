from argparse import Namespace

from cache_dit._utils import registers as example_registers
from cache_dit._utils.registers import Example
from cache_dit._utils.registers import ExampleInputData
from cache_dit._utils.utils import get_args
from cache_dit._utils.utils import maybe_postprocess_args


def test_examples_cli_accepts_warmup_seed() -> None:
  parser = get_args(parse=False)

  args = maybe_postprocess_args(parser.parse_args(["--warmup-seed", "123"]))

  assert args.warmup_seed == 123


def test_example_new_generator_uses_warmup_seed_only_for_warmup(monkeypatch, ) -> None:
  monkeypatch.setattr(example_registers, "maybe_init_distributed", lambda _args: (0, "cpu"))
  args = Namespace(generator_device=None, seed=11, warmup_seed=23)
  example = Example(args=args, input_data=ExampleInputData(seed=7))

  warmup_kwargs = example.new_generator({}, args, warmup=True)
  inference_kwargs = example.new_generator({}, args, warmup=False)

  assert warmup_kwargs["generator"].initial_seed() == 23
  assert inference_kwargs["generator"].initial_seed() == 11


def test_example_new_generator_warmup_falls_back_to_regular_seed_when_unset(monkeypatch, ) -> None:
  monkeypatch.setattr(example_registers, "maybe_init_distributed", lambda _args: (0, "cpu"))
  args = Namespace(generator_device=None, seed=11, warmup_seed=None)
  example = Example(args=args, input_data=ExampleInputData(seed=7))

  warmup_kwargs = example.new_generator({}, args, warmup=True)

  assert warmup_kwargs["generator"].initial_seed() == 11
