"""CLI coverage for the non-torchrun parallel launcher."""

from types import SimpleNamespace

import pytest

from cache_dit.cli import ray as launcher
from cache_dit.cli import utils as cli_utils
from cache_dit.cli.entrypoint import get_args

parser = get_args(parse=False)


def _args(ulysses=2, ring=1, tp=1, launcher="auto"):
  return SimpleNamespace(
    parallel_type="ulysses" if ulysses > 1 else None,
    parallel_degrees={
      "ulysses": ulysses,
      "ring": ring,
      "tp": tp,
    },
    launcher=launcher,
  )


def test_rank_env_matches_what_torchrun_exports():
  env = launcher._rank_env(1, 2, 29500, "/work")
  assert env == {
    "RANK": "1",
    "LOCAL_RANK": "1",
    "WORLD_SIZE": "2",
    "LOCAL_WORLD_SIZE": "2",
    "MASTER_ADDR": "127.0.0.1",
    "MASTER_PORT": "29500",
    "CACHE_DIT_LAUNCH_CWD": "/work",
  }


def test_visible_devices_follows_cuda_visible_devices(monkeypatch):
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
  assert launcher._visible_devices() == ["4", "5"]


def test_world_size_is_the_degree_product():
  assert launcher._world_size(_args(ulysses=2, ring=2, tp=2)) == 8


def test_no_launch_without_parallelism(monkeypatch):
  monkeypatch.delenv("WORLD_SIZE", raising=False)
  monkeypatch.setattr(launcher, "_visible_devices", lambda: ["4", "5"])
  assert launcher.maybe_launch_parallel(_args(ulysses=1), []) is False


def test_no_launch_under_torchrun(monkeypatch):
  monkeypatch.setenv("WORLD_SIZE", "2")
  monkeypatch.setattr(launcher, "_visible_devices", lambda: ["4", "5"])
  assert launcher.maybe_launch_parallel(_args(ulysses=2), []) is False


def test_degree_product_must_match_the_visible_gpus(monkeypatch):
  monkeypatch.delenv("WORLD_SIZE", raising=False)
  monkeypatch.setattr(launcher, "_visible_devices", lambda: ["4", "5", "6", "7"])
  with pytest.raises(ValueError, match="multiply to 2, but 4 GPU"):
    launcher.maybe_launch_parallel(_args(ulysses=2), [])


def test_local_launcher_receives_rank_count_and_argv(monkeypatch):
  monkeypatch.delenv("WORLD_SIZE", raising=False)
  monkeypatch.setattr(launcher, "_visible_devices", lambda: ["4", "5"])
  monkeypatch.setattr(launcher, "_free_port", lambda: 29500)
  calls = []
  monkeypatch.setattr(launcher, "_run_local", lambda argv, world_size, port: calls.append(
    (argv, world_size, port)))

  assert launcher.maybe_launch_parallel(_args(ulysses=2, launcher="local"), ["flux"]) is True
  assert calls == [(["flux"], 2, 29500)]


def test_ray_launcher_is_selected(monkeypatch):
  monkeypatch.delenv("WORLD_SIZE", raising=False)
  monkeypatch.setattr(launcher, "_visible_devices", lambda: ["4", "5"])
  monkeypatch.setattr(launcher, "_free_port", lambda: 29500)
  calls = []
  monkeypatch.setattr(launcher, "_run_ray", lambda argv, world_size, port: calls.append(
    (argv, world_size, port)))

  assert launcher.maybe_launch_parallel(_args(ulysses=2, launcher="ray"), ["flux"]) is True
  assert calls == [(["flux"], 2, 29500)]


@pytest.mark.parametrize("launcher_name", ["auto", "ray", "local"])
def test_cli_accepts_the_launcher_choices(launcher_name):
  args = cli_utils.maybe_postprocess_args(
    parser.parse_args(["--parallel-params", "ulysses=2", "--launcher", launcher_name]))
  assert args.launcher == launcher_name


def test_cli_rejects_an_unknown_launcher():
  with pytest.raises(SystemExit):
    parser.parse_args(["--launcher", "slurm"])
