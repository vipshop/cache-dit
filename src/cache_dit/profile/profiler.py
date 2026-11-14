"""Request and complete-step collection for Torch and external Nsight Systems.

Reference: Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_one_batch.py
"""

import logging
import os
import re
import time
import socket
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import List, Optional, Tuple
from uuid import uuid4

import torch
from ..platforms import current_platform

logger = logging.getLogger(__name__)

# Default profiler directory
PROFILER_DIR = os.getenv("CACHE_DIT_TORCH_PROFILER_DIR", "/tmp/cache_dit_profiles")

# NPU profiling depth (Level0 | Level1 | Level2 | none). Higher levels collect more
# CANN/AscendCL data (e.g. communication.json, api_statistic.csv, aic metrics) at the
# cost of larger output and slower parsing. See the Ascend PyTorch Profiler docs.
NPU_PROFILER_LEVEL_ENV = "CACHE_DIT_NPU_PROFILER_LEVEL"

# Loops resolve their own request; collector ownership is shared across the process.
_CURRENT_REQUEST = ContextVar("cache_dit_profiler", default=None)
_COLLECTOR_LOCK = Lock()


def _normalize_ranges(ranges: str):
  if ranges.strip() == "all":
    return None
  intervals = []
  for item in ranges.split(","):
    match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", item)
    if match is None:
      raise ValueError(f"Invalid profiler range: {item!r}; use 'all' or '0-4,10-14'.")
    start = int(match[1])
    end = int(match[2] or match[1])
    if start > end:
      raise ValueError(f"Profiler range starts after its end: {item!r}.")
    intervals.append((start, end))
  merged = []
  for start, end in sorted(intervals):
    if merged and start <= merged[-1][1] + 1:
      merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    else:
      merged.append((start, end))
  return merged


@dataclass
class ProfileResult:
  """One collector window and its successfully exported artifacts."""

  request_id: str
  loop_id: Optional[int]
  window_id: int
  requested_range: Optional[Tuple[int, int]]
  actual_range: Optional[Tuple[int, int]] = None
  last_step_complete: Optional[bool] = None
  status: str = "running"
  trace_path: Optional[Path] = None
  memory_snapshot_path: Optional[Path] = None
  memory_summary_path: Optional[Path] = None


@contextmanager
def profile_steps(timesteps):
  """Enumerate a real denoising loop, closing its window on every exit.

  :param timesteps: The original loop iterable, with one item per logical step.
  :yields: An iterator of ``(step_index, timestep)`` pairs.
  """

  request = _CURRENT_REQUEST.get()
  if request is None or request._ranges is None:
    yield enumerate(timesteps)
    return
  loop_id = request._loop_count
  request._loop_count += 1
  steps = request._steps(timesteps, loop_id)
  try:
    yield steps
  except BaseException:
    request._close_window("error", loop_id=loop_id, suppress_errors=True)
    raise
  else:
    request._close_window("interrupted", loop_id=loop_id)
  finally:
    steps.close()


def _resolve_profiler_backend() -> Tuple:
  """Select the profiler backend for the current platform.

  :returns: The profiler module and its accelerator activity enum.
  """

  if current_platform.device_type == "npu":
    import torch_npu.profiler as npu_profiler  # noqa: WPS433

    return npu_profiler, npu_profiler.ProfilerActivity.NPU
  from torch import profiler  # noqa: WPS433

  return profiler, profiler.ProfilerActivity.CUDA


def _npu_profiler_level(npu_profiler):
  """Resolve the NPU profiler level from the env (default Level0).

  :param npu_profiler: The ``torch_npu.profiler`` module.
  :returns: A ``ProfilerLevel`` enum value.
  """

  raw = os.getenv(NPU_PROFILER_LEVEL_ENV, "Level0").strip().lower()
  table = {
    "none": npu_profiler.ProfilerLevel.Level_none,
    "level_none": npu_profiler.ProfilerLevel.Level_none,
    "0": npu_profiler.ProfilerLevel.Level0,
    "level0": npu_profiler.ProfilerLevel.Level0,
    "1": npu_profiler.ProfilerLevel.Level1,
    "level1": npu_profiler.ProfilerLevel.Level1,
    "2": npu_profiler.ProfilerLevel.Level2,
    "level2": npu_profiler.ProfilerLevel.Level2,
  }
  return table.get(raw, npu_profiler.ProfilerLevel.Level0)


def _npu_experimental_config(npu_profiler):
  """Build the NPU experimental config that produces the full output directory.

  ``export_type=Text`` emits the ``.json``/``.csv`` timeline and summary files plus the
  aggregate ``.db`` files; ``data_simplification=False`` keeps every generated file
  (kernel_details.csv, step_trace_time.csv, trace_view.json, op_summary.csv, the
  ``PROF_XXX`` raw data, ``FRAMEWORK``/``logs``) instead of pruning them.

  :param npu_profiler: The ``torch_npu.profiler`` module.
  :returns: The experimental config.
  """

  level = _npu_profiler_level(npu_profiler)
  # aic_metrics must match the level, otherwise torch_npu resets it with a warning.
  aic_metrics = (npu_profiler.AiCMetrics.PipeUtilization if level
                 in (npu_profiler.ProfilerLevel.Level1,
                     npu_profiler.ProfilerLevel.Level2) else npu_profiler.AiCMetrics.AiCoreNone)
  return npu_profiler._ExperimentalConfig(
    export_type=[npu_profiler.ExportType.Text],
    profiler_level=level,
    aic_metrics=aic_metrics,
    data_simplification=False,
  )


class ProfilerContext:
  """Profile a request or selected complete denoising steps.

  Torch exports CUDA gzip traces/memory or full Ascend CANN directories. Nsight uses CUDA gates,
  ignores Torch capture options and leaves reports to the external collector. Distributed Nsight
  requests use an existing unbound CPU/Gloo default group.
  """

  def __init__(
    self,
    enabled: bool = True,
    activities: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    profile_name: Optional[str] = None,
    with_stack: bool = True,
    record_shapes: bool = True,
    *,
    collector: str = "torch",
    ranges: str = "all",
    devices=None,
  ):
    """Configure a profiler session.

    :param enabled: Whether profiling should actually be activated.
    :param activities: Activity names such as `CPU`, `GPU`, or `MEM`. `GPU` maps to the
      platform accelerator activity (CUDA on CUDA devices, NPU on Ascend devices).
    :param output_dir: Directory where traces and memory snapshots are written.
    :param profile_name: Base name used for profiler output files.
    :param with_stack: Whether to capture Python stacks for profiled ops.
    :param record_shapes: Whether to record tensor shapes in the profiler trace.
    :param collector: ``torch`` for trace export or ``nsys`` for external CUDA collection.
    :param ranges: ``all`` or inclusive step ranges, applied independently to each loop.
    :param devices: Devices to synchronize; None uses the current device.
    """

    self.enabled = enabled
    self.activities = activities or ["CPU", "GPU"]
    self.output_dir = Path(output_dir or get_profiler_output_dir()).expanduser()
    self.profile_name = profile_name or f"profile_{int(time.time())}"
    self.with_stack = with_stack
    self.record_shapes = record_shapes
    self.collector = collector
    self.ranges = ranges
    self.devices = tuple(devices) if devices is not None else (None, )
    self._token = None
    self._window = None
    self.profiler = None
    self.trace_path = None
    self.memory_snapshot_path = None
    self.results = []
    self.empty_reason = None

  def __enter__(self):
    if not self.enabled:
      return self

    if self.collector not in ("torch", "nsys"):
      raise ValueError(f"Invalid profiler collector: {self.collector!r}; use 'torch' or 'nsys'.")
    if self.collector == "torch" and not current_platform.is_accelerator_available():
      raise RuntimeError("Torch ProfilerContext only supports CUDA or Ascend NPU devices, "
                         f"got device_type={current_platform.device_type!r}.")
    if self.collector == "nsys" and current_platform.device_type != "cuda":
      raise RuntimeError("Nsight collection requires a CUDA platform.")
    intervals = _normalize_ranges(self.ranges)
    if not self.devices:
      raise ValueError("Profiler devices must include every device used by the request.")
    # Reject overlapping requests even during gaps between collection windows.
    if not _COLLECTOR_LOCK.acquire(blocking=False):
      raise RuntimeError("Another profiler request is already active in this process.")
    try:
      self.request_id = uuid4().hex
      if self.collector == "torch":
        self._backend, accelerator = _resolve_profiler_backend()
        activity_map = {
          "CPU": self._backend.ProfilerActivity.CPU,
          "GPU": accelerator,
          current_platform.device_type.upper(): accelerator,
        }
        invalid = set(self.activities) - activity_map.keys() - {"MEM"}
        if invalid:
          raise ValueError(f"Invalid profiler activity: {invalid}.")
        self._torch_activities = [activity_map[a]
                                  for a in self.activities if a != "MEM"] or [accelerator]
        self._track_memory = "MEM" in self.activities
        rank = (torch.distributed.get_rank() if torch.distributed.is_initialized() else os.getenv(
          "RANK", "0"))
        self._request_name = re.sub(
          r"[^A-Za-z0-9_-]", "_",
          f"{self.profile_name}-{socket.gethostname()}-pid{os.getpid()}-rank{rank}-{self.request_id}"
        )
      self._token = _CURRENT_REQUEST.set(self)
      self._ranges = intervals
      self._loop_count = 0
      self._has_steps = False
      self.results = []
      self.empty_reason = None
      self.trace_path = None
      self.memory_snapshot_path = None
      if self._ranges is None:
        self._open_window(None, None)
    except BaseException:
      self._release_request(suppress_errors=True)
      raise
    return self

  def _release_request(self, *, suppress_errors):
    token, self._token = self._token, None
    try:
      if token is not None:
        _CURRENT_REQUEST.reset(token)
    except BaseException as error:
      if not suppress_errors:
        raise
      logger.warning("Profiler request cleanup failed while preserving the original exception: %s",
                     error)
    finally:
      _COLLECTOR_LOCK.release()

  def _open_window(self, loop_id, requested_range):
    if self._window is not None:
      raise RuntimeError("Profiler windows from overlapping denoising loops are not supported.")
    result = ProfileResult(self.request_id, loop_id, len(self.results), requested_range)
    self.results.append(result)
    self._window = result
    self._memory_started = False
    self._collector_started = False
    try:
      for device in self.devices:
        current_platform.synchronize(device)
      if self.collector == "nsys":
        distributed = torch.distributed.is_initialized()
        if distributed:
          torch.distributed.barrier()
        torch.cuda.profiler.start()
        self.profiler = torch.cuda.profiler
        if distributed:
          torch.distributed.barrier()
        return result
      self._window_name = f"{self._request_name}-loop{loop_id if loop_id is not None else 'all'}-window{result.window_id}"
      self.output_dir.mkdir(parents=True, exist_ok=True)
      options = {}
      if current_platform.device_type == "npu":
        options = dict(
          profile_memory=self._track_memory,
          experimental_config=_npu_experimental_config(self._backend),
          on_trace_ready=self._backend.tensorboard_trace_handler(
            dir_name=str(self.output_dir),
            worker_name=self._window_name,
            analyse_flag=True,
            async_mode=False,
          ),
        )
      elif self._track_memory:
        torch.cuda.memory._record_memory_history(max_entries=100000)
        self._memory_started = True
      self.profiler = self._backend.profile(
        activities=self._torch_activities,
        with_stack=self.with_stack,
        record_shapes=self.record_shapes,
        **options,
      )
      self.profiler.start()
      self._collector_started = True
      logger.info("Started profiling window %s (activities: %s). Output directory: %s",
                  self._window_name, self.activities, self.output_dir)
    except BaseException:
      self._close_window("error", suppress_errors=True)
      raise
    return result

  def _steps(self, timesteps, loop_id):
    intervals = iter(self._ranges)
    start, end = next(intervals)
    result = None
    for i, timestep in enumerate(timesteps):
      self._has_steps = True
      if i == start:
        result = self._open_window(loop_id, (start, end))
      if result is None:
        yield i, timestep
        continue
      result.actual_range = (start, i)
      result.last_step_complete = False
      yield i, timestep
      result.last_step_complete = True
      if i == end:
        self._close_window("completed", loop_id=loop_id)
        result = None
        start, end = next(intervals, (None, None))
    self._close_window("exhausted", loop_id=loop_id)

  def __exit__(self, exc_type, exc_val, exc_tb):
    if self._token is None:
      return
    suppress_errors = exc_type is not None
    try:
      self._close_window("error" if suppress_errors else "completed",
                         suppress_errors=suppress_errors)
      if not self.results:
        self.empty_reason = ("The denoising loops were empty or unconsumed." if not self._has_steps
                             else "No denoising steps matched the requested ranges.")
        if self._loop_count == 0 and not suppress_errors:
          raise RuntimeError(
            "Numeric profiling ranges require the pipeline to enter profile_steps().")
    except BaseException:
      suppress_errors = True
      raise
    finally:
      self._release_request(suppress_errors=suppress_errors)

  def _close_window(self, status, *, loop_id=None, suppress_errors=False):
    result = self._window
    if result is None or (loop_id is not None and result.loop_id != loop_id):
      return
    self._window = None
    result.status = status
    errors = []

    @contextmanager
    def finish():
      try:
        yield
      except BaseException as error:
        errors.append(error)

    if self.profiler is not None:
      for device in self.devices:
        with finish():
          current_platform.synchronize(device)
      distributed = (self.collector == "nsys" and torch.distributed.is_initialized()
                     and not suppress_errors and not errors)
      if distributed:
        with finish():
          torch.distributed.barrier()
      with finish():
        self.profiler.stop()
      if distributed and not errors:
        with finish():
          torch.distributed.barrier()
      if self.collector == "torch":
        with finish():
          path = self._export_trace()
          if not errors and self._collector_started:
            result.trace_path = self.trace_path = path
    if self._memory_started:
      with finish():
        path = self.output_dir / f"{self._window_name}-memory.pickle"
        torch.cuda.memory._dump_snapshot(path)
        result.memory_snapshot_path = self.memory_snapshot_path = path
      with finish():
        path = self.output_dir / f"{self._window_name}-memory.txt"
        path.write_text(torch.cuda.memory_summary())
        result.memory_summary_path = path
      with finish():
        torch.cuda.memory._record_memory_history(enabled=None)
    self.profiler = None
    if errors:
      result.status = "error"
      if not suppress_errors:
        raise errors[0]
      logger.warning("Profiler cleanup failed while preserving the original exception: %s", errors)

  def _export_trace(self):
    if current_platform.device_type == "cuda":
      path = self.output_dir / f"{self._window_name}.trace.json.gz"
      self.profiler.export_chrome_trace(str(path))
      return path
    # NPU parsing is synchronous; poll briefly for slow disk finalization.
    pattern = f"{self._window_name}_*_ascend_pt"
    for _ in range(10):
      for path in self.output_dir.glob(pattern):
        if (path / "ASCEND_PROFILER_OUTPUT").is_dir():
          return path
      time.sleep(0.5)
    raise FileNotFoundError(f"NPU profiler output was not produced: {self.output_dir / pattern}")


def profile_function(
  enabled: bool = True,
  activities: Optional[List[str]] = None,
  output_dir: Optional[str] = None,
  profile_name: Optional[str] = None,
  with_stack: bool = False,
  record_shapes: bool = True,
  *,
  ranges: str = "all",
  devices=None,
):
  """Decorator factory that profiles one function call with `ProfilerContext`.

  :param enabled: Whether the feature is enabled.
  :param activities: Profiler activities to capture.
  :param output_dir: Directory for generated outputs.
  :param profile_name: Base name for the profiler output.
  :param with_stack: Whether to capture Python stack traces.
  :param record_shapes: Whether to record tensor shapes.
  :param ranges: ``all`` or inclusive denoising step ranges.
  :param devices: Devices used by the function; None uses the current device.
  """

  def decorator(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
      name = profile_name or func.__name__
      with ProfilerContext(
          enabled=enabled,
          activities=activities,
          output_dir=output_dir,
          profile_name=name,
          with_stack=with_stack,
          record_shapes=record_shapes,
          ranges=ranges,
          devices=devices,
      ):
        return func(*args, **kwargs)

    return wrapper

  return decorator


def create_profiler_context(
  enabled: bool = False,
  activities: Optional[List[str]] = None,
  output_dir: Optional[str] = None,
  profile_name: Optional[str] = None,
  **kwargs,
) -> ProfilerContext:
  """Convenience helper to build a `ProfilerContext` instance.

  :param enabled: Whether the feature is enabled.
  :param activities: Profiler activities to capture.
  :param output_dir: Directory for generated outputs.
  :param profile_name: Base name for the profiler output.
  :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
  :returns: A configured `ProfilerContext` instance.
  """

  return ProfilerContext(
    enabled=enabled,
    activities=activities,
    output_dir=output_dir,
    profile_name=profile_name,
    **kwargs,
  )


def get_profiler_output_dir() -> str:
  """Return the default profiler output directory from the environment.

  :returns: The resolved profiler output dir.
  """

  return os.environ.get("CACHE_DIT_TORCH_PROFILER_DIR", PROFILER_DIR)


def set_profiler_output_dir(path: str):
  """Override the default profiler output directory for future sessions.

  :param path: Path to the input resource.
  """

  os.environ["CACHE_DIT_TORCH_PROFILER_DIR"] = path
