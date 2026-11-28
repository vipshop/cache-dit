"""CLI coverage for `--parallel-params`."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cache_dit.cli import optimize, parallel
from cache_dit.cli import utils as cli_utils

parser = cli_utils.get_args(parse=False)


def resolve(argv):
  return cli_utils.maybe_postprocess_args(parser.parse_args(argv))


@pytest.mark.parametrize("text, expected", [
  ("ulysses=2,ring=1,tp=1", ("ulysses", ("ulysses", ), {
    "ulysses": 2,
    "ring": 1,
    "tp": 1
  })),
  ("ulysses=2", ("ulysses", ("ulysses", ), {
    "ulysses": 2,
    "ring": 1,
    "tp": 1
  })),
  ("ulysses=1", ("ulysses", ("ulysses", ), {
    "ulysses": 1,
    "ring": 1,
    "tp": 1
  })),
  ("tp=1", ("tp", ("tp", ), {
    "ulysses": 1,
    "ring": 1,
    "tp": 1
  })),
  ("ring=4", ("ring", ("ring", ), {
    "ulysses": 1,
    "ring": 4,
    "tp": 1
  })),
  ("ulysses=2,ring=2", ("usp", ("ulysses", "ring"), {
    "ulysses": 2,
    "ring": 2,
    "tp": 1
  })),
  ("ulysses=2,tp=2", ("ulysses_tp", ("ulysses", "tp"), {
    "ulysses": 2,
    "ring": 1,
    "tp": 2
  })),
  ("ring=2,tp=2", ("ring_tp", ("ring", "tp"), {
    "ulysses": 1,
    "ring": 2,
    "tp": 2
  })),
  ("ulysses=2,ring=2,tp=2", ("usp_tp", ("ulysses", "ring", "tp"), {
    "ulysses": 2,
    "ring": 2,
    "tp": 2
  })),
])
def test_degrees_name_the_parallelism_type(text, expected) -> None:
  args = resolve(["--parallel-params", text])
  assert (args.parallel_type, args.parallel_axes, args.parallel_degrees) == expected


def test_default_is_no_parallelism() -> None:
  args = resolve([])
  assert args.parallel_type is None
  assert args.parallel_axes == ()
  assert (args.parallel_vae, args.parallel_text_encoder, args.parallel_controlnet) == (False, ) * 3
  assert args.local_weight_shards is False
  assert (args.ulysses_anything, args.ulysses_async) == (False, False)
  assert (args.ring_rotate_method, args.ring_no_convert_to_fp32) == ("p2p", False)


def test_components_and_options() -> None:
  args = resolve([
    "--parallel-params", "ulysses=4,vae=true,text=true,controlnet=true,uaa=true,uaqkv=true,"
    "rotate=allgather,ring_convert_to_fp32=false"
  ])
  assert args.parallel_type == "ulysses"
  assert (args.parallel_vae, args.parallel_text_encoder, args.parallel_controlnet) == (True, ) * 3
  assert (args.ulysses_anything, args.ulysses_async) == (True, True)
  assert (args.ring_rotate_method, args.ring_no_convert_to_fp32) == ("allgather", True)


def test_degrees_must_multiply_to_the_world_size(monkeypatch) -> None:
  monkeypatch.setenv("WORLD_SIZE", "4")
  assert resolve(["--parallel-params", "ulysses=2,ring=2"]).parallel_type == "usp"
  with pytest.raises(ValueError, match=r"multiply to 8, which must equal the world size 4"):
    resolve(["--parallel-params", "ulysses=2,tp=4"])
  # Without torchrun the check runs where the groups are built, not here.
  monkeypatch.delenv("WORLD_SIZE")
  assert resolve(["--parallel-params", "ulysses=2,tp=4"]).parallel_type == "ulysses_tp"


def test_components_without_degrees_do_not_enable_parallel_groups() -> None:
  args = resolve(["--parallel-params", "vae=true"])
  assert (args.parallel_type, args.parallel_vae) == (None, True)


def test_local_weight_shards_reaches_core_config(monkeypatch) -> None:
  args = resolve(["--parallel-config", "ulysses=2,text=true,local_weight_shards=true"])
  monkeypatch.setattr(optimize.dist, "is_initialized", lambda: True)
  monkeypatch.setattr(optimize.dist, "get_world_size", lambda: 2)
  pipe = SimpleNamespace(text_encoder=object())
  enable_cache = Mock(side_effect=RuntimeError("stop at Core API"))
  monkeypatch.setattr(optimize, "enable_cache", enable_cache)
  with pytest.raises(RuntimeError, match="stop at Core API"):
    optimize.maybe_apply_optimization(args, pipe)

  config = enable_cache.call_args.kwargs["parallelism_config"]
  assert config.local_weight_shards is True
  assert (config.ulysses_size, config.tp_size) == (2, None)
  assert config.extra_parallel_modules == [pipe.text_encoder]


@pytest.mark.parametrize("value", ["true", "false"])
def test_local_weight_shards_conflicts_with_yaml_config(value) -> None:
  with pytest.raises(ValueError, match="YAML parallelism_config"):
    resolve(["--config", "example.yaml", "--parallel-params", f"tp=2,local_weight_shards={value}"])


def test_all_degrees_of_one_do_not_enable_parallel_groups(caplog, monkeypatch) -> None:
  monkeypatch.setattr(parallel.logger, "handlers", [caplog.handler])
  args = resolve(["--parallel-params", "ulysses=1,ring=1,tp=1"])
  assert args.parallel_type is None
  assert "parallelism is not enabled" in caplog.text


@pytest.mark.parametrize("text, message", [
  ("", "needs at least one"),
  ("ulysses", "Expected 'key=value' pairs"),
])
def test_syntax_errors_are_rejected_by_argparse(text, message, capsys) -> None:
  with pytest.raises(SystemExit):
    parser.parse_args(["--parallel-params", text])
  assert message in capsys.readouterr().err


@pytest.mark.parametrize("text, message", [
  ("ulysses=0", "expects an integer >= 1"),
  ("ulysses=2,uaa=true,ulysses_anything=false", "is given twice"),
  ("nope=1", "Unknown --parallel-params keys"),
  ("rotate=diag", "Invalid --parallel-params 'ring_rotate_method=diag'"),
  ("vae=maybe", "Invalid --parallel-params 'vae=maybe'"),
  ("tp=1,local_weight_shards=true", "requires tp > 1 or text=true"),
  ("text=true,local_weight_shards=true", "requires tp > 1 or text=true"),
  ("ulysses=2,local_weight_shards=true", "requires tp > 1 or text=true"),
])
def test_bad_values_are_rejected(text, message) -> None:
  with pytest.raises(ValueError, match=message):
    resolve(["--parallel-params", text])


@pytest.mark.parametrize("flag", [
  "--parallel",
  "--parallel-type",
  "--parallel-vae",
  "--parallel-text",
  "--parallel-controlnet",
  "--ulysses-anything",
  "--uaa",
  "--ulysses-async",
  "--uaqkv",
  "--ring-rotate-method",
  "--rotate",
  "--ring-no-convert-to-fp32",
  "--no-fp32",
])
def test_removed_parallel_flags_are_rejected(flag) -> None:
  with pytest.raises(SystemExit):
    parser.parse_args([flag, "ulysses"])


def test_strify_reflects_the_ring_and_component_options() -> None:
  base = cli_utils.strify(resolve([]), {})
  assert cli_utils.strify(resolve(["--parallel-params", "ulysses=2"]), {}) == base
  assert cli_utils.strify(resolve(["--parallel-params", "ring=2,rotate=allgather"]),
                          {}).endswith("_rotated_allgather")
  assert cli_utils.strify(resolve(["--parallel-params", "ulysses=2,tp=2,vae=true"]),
                          {}).endswith("_VAEP")
