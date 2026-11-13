"""Public profiler contracts; real model and CLI coverage lives in the H3 tests."""

import importlib
from contextlib import nullcontext
from contextvars import Context
from types import SimpleNamespace

import pytest

from cache_dit.profiler import ProfilerContext, profile_function, profile_steps

profiler_module = importlib.import_module("cache_dit.profiler")


@pytest.fixture
def cpu_collector(monkeypatch):
  """Use the real CPU collector while replacing accelerator checks and synchronization."""
  monkeypatch.setattr(
    profiler_module, "current_platform",
    SimpleNamespace(device_type="cuda",
                    is_accelerator_available=lambda: True,
                    synchronize=lambda device: None))


@profile_function(enabled=False, ranges="invalid")
def test_disabled_request_does_not_probe_platform(monkeypatch):
  monkeypatch.setattr(profiler_module, "current_platform", None)
  with ProfilerContext(enabled=False, ranges="invalid") as request:
    with profile_steps([7]) as steps:
      assert list(steps) == [(0, 7)]
  assert request.results == [] and request.trace_path is None


@pytest.mark.parametrize("options,error", [
  ({
    "ranges": "-1"
  }, ValueError),
  ({
    "activities": ["invalid"],
    "ranges": "99"
  }, ValueError),
  ({
    "devices": []
  }, ValueError),
  ({
    "ranges": "0"
  }, RuntimeError),
])
def test_invalid_or_unconnected_request(options, error, cpu_collector):
  with pytest.raises(error):
    with ProfilerContext(**options):
      pass


def test_unsupported_platform_fails_at_entry(monkeypatch):
  monkeypatch.setattr(profiler_module, "current_platform",
                      SimpleNamespace(device_type="cpu", is_accelerator_available=lambda: False))
  with pytest.raises(RuntimeError):
    with ProfilerContext():
      pass


@pytest.mark.parametrize("status", ["completed", "interrupted", "error", "exhausted"])
def test_windows_and_loop_exit(status, cpu_collector, tmp_path):
  request = ProfilerContext(activities=["CPU"],
                            ranges="4-5,2,1,2",
                            output_dir=tmp_path,
                            with_stack=False)
  with pytest.raises(LookupError) if status == "error" else nullcontext():
    with request:
      for _ in range(2):
        with profile_steps(range(2 if status == "exhausted" else 6)) as steps:
          for i, _ in steps:
            if i == 1 and status in ("interrupted", "error"):
              if status == "error":
                raise LookupError("model failed")
              break
  expected = [(1, 2),
              (4, 5)] * 2 if status == "completed" else [(1, 1)] * (1 if status == "error" else 2)
  assert [r.actual_range for r in request.results] == expected
  assert all(r.status == status and r.last_step_complete == (status in ("completed", "exhausted"))
             for r in request.results)
  paths = [r.trace_path for r in request.results]
  assert len(set(paths)) == len(expected) and all(p.is_file() for p in paths)
  assert request.trace_path == paths[-1]


def test_request_excludes_other_contexts_even_between_windows(cpu_collector, tmp_path):

  def visit():
    with profile_steps([7]) as steps:
      return list(steps)

  with ProfilerContext(activities=["CPU"], ranges="99", output_dir=tmp_path) as request:
    assert Context().run(visit) == [(0, 7)]
    with pytest.raises(RuntimeError, match="already active"):
      Context().run(request.__enter__)
    with profile_steps([]) as steps:
      list(steps)
  assert request.results == [] and "empty" in request.empty_reason
  with ProfilerContext(activities=["CPU"], output_dir=tmp_path, with_stack=False):
    pass
