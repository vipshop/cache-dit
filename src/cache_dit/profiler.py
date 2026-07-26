"""Torch Profiler for cache-dit.

Reference: Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_one_batch.py
"""

import glob
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from .platforms import current_platform

logger = logging.getLogger(__name__)

# Default profiler directory
PROFILER_DIR = os.getenv("CACHE_DIT_TORCH_PROFILER_DIR", "/tmp/cache_dit_profiles")

# NPU profiling depth (Level0 | Level1 | Level2 | none). Higher levels collect more
# CANN/AscendCL data (e.g. communication.json, api_statistic.csv, aic metrics) at the
# cost of larger output and slower parsing. See the Ascend PyTorch Profiler docs.
NPU_PROFILER_LEVEL_ENV = "CACHE_DIT_NPU_PROFILER_LEVEL"


def _supported_device() -> bool:
  """Whether the profiler supports the current device.

  CUDA devices are profiled with ``torch.profiler`` and Ascend NPU devices with
  ``torch_npu.profiler``; other device types are not supported.

  :returns: True if the current device is a supported accelerator.
  """

  if not current_platform.is_accelerator_available():
    return False
  return current_platform.device_type in ("cuda", "npu")


def _resolve_profiler_backend() -> Tuple:
  """Select the profiler backend for the current platform.

  ``torch_npu.profiler`` mirrors the ``torch.profiler`` API, so both backends are
  driven the same way; only the ``ProfilerActivity`` enum and the accelerator
  activity name differ.

  :returns: A tuple of (profile callable, ProfilerActivity enum, accelerator activity name).
  """

  if current_platform.device_type == "npu":
    import torch_npu.profiler as npu_profiler  # noqa: WPS433

    return npu_profiler.profile, npu_profiler.ProfilerActivity, "NPU"
  from torch.profiler import ProfilerActivity, profile  # noqa: WPS433

  return profile, ProfilerActivity, "CUDA"


def _npu_profiler_module():
  """Import and return the ``torch_npu.profiler`` module (NPU only)."""

  import torch_npu.profiler as npu_profiler  # noqa: WPS433

  return npu_profiler


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
  :returns: A tuple of (experimental config, resolved profiler level).
  """

  level = _npu_profiler_level(npu_profiler)
  # aic_metrics must match the level, otherwise torch_npu resets it with a warning.
  aic_metrics = (npu_profiler.AiCMetrics.PipeUtilization if level
                 in (npu_profiler.ProfilerLevel.Level1,
                     npu_profiler.ProfilerLevel.Level2) else npu_profiler.AiCMetrics.AiCoreNone)
  config = npu_profiler._ExperimentalConfig(
    export_type=[npu_profiler.ExportType.Text],
    profiler_level=level,
    aic_metrics=aic_metrics,
    data_simplification=False,
  )
  return config, level


def _npu_worker_name(profile_name: str, rank: int, world_size: int) -> str:
  """Build a valid NPU ``worker_name`` from the profile name.

  ``worker_name`` only allows letters, digits, underscores and hyphens, and becomes the
  ``{worker_name}_{timestamp}_ascend_pt`` output directory name.

  :param profile_name: The base profile name.
  :param rank: The distributed rank.
  :param world_size: The distributed world size.
  :returns: A sanitized worker name.
  """

  safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(profile_name or "cache_dit")).strip("_")
  safe = safe or "cache_dit"
  if world_size > 1:
    safe = f"{safe}-rank{rank}"
  return safe


class ProfilerContext:
  """Context manager wrapper around ``torch.profiler`` / ``torch_npu.profiler``.

  It centralizes trace-file naming, optional accelerator memory-history capture, and
  multi-rank output layout so profiling can be enabled consistently from scripts or
  helper decorators.

  On CUDA it exports a gzip chrome trace and optional ``torch.cuda.memory`` snapshots.
  On Ascend NPU it drives ``on_trace_ready=tensorboard_trace_handler`` so the full CANN
  profiling output directory is produced (``ASCEND_PROFILER_OUTPUT`` with
  ``kernel_details.csv`` / ``step_trace_time.csv`` / ``trace_view.json`` / ``op_summary.csv``
  plus the ``PROF_XXX`` raw data, ``FRAMEWORK`` and ``logs``), not just a single trace.
  """

  def __init__(
    self,
    enabled: bool = True,
    activities: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    profile_name: Optional[str] = None,
    with_stack: bool = True,
    record_shapes: bool = True,
  ):
    """Configure a profiler session.

    :param enabled: Whether profiling should actually be activated.
    :param activities: Activity names such as `CPU`, `GPU`, or `MEM`. `GPU` maps to the
      platform accelerator activity (CUDA on CUDA devices, NPU on Ascend devices).
    :param output_dir: Directory where traces and memory snapshots are written.
    :param profile_name: Base name used for profiler output files.
    :param with_stack: Whether to capture Python stacks for profiled ops.
    :param record_shapes: Whether to record tensor shapes in the profiler trace.
    """

    assert _supported_device(), (
      "Torch ProfilerContext currently only supports CUDA or Ascend NPU devices, "
      f"got device_type={current_platform.device_type!r}.")
    self.enabled = enabled
    self.activities = activities or ["CPU", "GPU"]
    self.output_dir = Path(output_dir or PROFILER_DIR).expanduser()
    self.profile_name = profile_name or f"profile_{int(time.time())}"
    self.with_stack = with_stack
    self.record_shapes = record_shapes
    # NPU records memory through the profiler (profile_memory=True); CUDA uses the
    # separate torch.cuda.memory history mechanism instead.
    self._is_npu = current_platform.device_type == "npu"
    self._track_memory = "MEM" in self.activities
    # NPU: worker name used for the {worker_name}_{ts}_ascend_pt output directory.
    self._npu_worker = None

    self.profiler = None
    self.trace_path = None
    self.memory_snapshot_path = None

  def __enter__(self):
    if not self.enabled:
      return self

    assert _supported_device(), (
      "Torch ProfilerContext currently only supports CUDA or Ascend NPU devices, "
      f"got device_type={current_platform.device_type!r}.")

    self.output_dir.mkdir(parents=True, exist_ok=True)

    profile_fn, activity_enum, accelerator_activity = _resolve_profiler_backend()
    # "GPU" maps to the platform accelerator activity (CUDA on CUDA, NPU on NPU);
    # the accelerator name (e.g. "NPU") is also accepted as an explicit synonym.
    accelerator_activity_enum = getattr(activity_enum, accelerator_activity)
    activity_map = {
      "CPU": activity_enum.CPU,
      "GPU": accelerator_activity_enum,
      accelerator_activity: accelerator_activity_enum,
    }
    torch_activities = [activity_map[a] for a in self.activities if a in activity_map]

    rank = 0
    world_size = 1
    if torch.distributed.is_initialized():
      rank = torch.distributed.get_rank()
      world_size = torch.distributed.get_world_size()

    if not torch_activities:
      return self

    if self._is_npu:
      self._start_npu_profiler(profile_fn, rank, world_size, torch_activities)
    else:
      self._start_cuda_profiler(profile_fn, rank, world_size, torch_activities)

    return self

  def _start_npu_profiler(self, profile_fn, rank, world_size, torch_activities):
    """Configure and start the NPU profiler with full-directory export."""

    npu_profiler = _npu_profiler_module()
    self._npu_worker = _npu_worker_name(self.profile_name, rank, world_size)
    experimental_config, level = _npu_experimental_config(npu_profiler)
    trace_handler = npu_profiler.tensorboard_trace_handler(
      dir_name=str(self.output_dir),
      worker_name=self._npu_worker,
      analyse_flag=True,
      async_mode=False,
    )
    self.profiler = profile_fn(
      activities=torch_activities,
      with_stack=self.with_stack,
      record_shapes=self.record_shapes,
      profile_memory=self._track_memory,
      on_trace_ready=trace_handler,
      experimental_config=experimental_config,
    )
    self.profiler.start()
    logger.info(f"Started NPU profiling. Full CANN output will be saved under "
                f"{self.output_dir}/{self._npu_worker}_<timestamp>_ascend_pt "
                f"(profiler_level={level}, activities: {self.activities})")

  def _start_cuda_profiler(self, profile_fn, rank, world_size, torch_activities):
    """Configure and start the CUDA profiler (gzip chrome trace)."""

    filename_parts = [self.profile_name]
    if world_size > 1:
      filename_parts.append(f"rank{rank}")
    filename = "-".join(filename_parts) + ".trace.json.gz"
    self.trace_path = self.output_dir / filename

    if self._track_memory and torch.cuda.is_available():
      torch.cuda.memory._record_memory_history(max_entries=100000)
      logger.info("Started CUDA memory profiling")

    self.profiler = profile_fn(
      activities=torch_activities,
      with_stack=self.with_stack,
      record_shapes=self.record_shapes,
    )
    self.profiler.start()
    logger.info(f"Started profiling. Traces will be saved to: {self.output_dir} "
                f"(activities: {self.activities})")

  def __exit__(self, exc_type, exc_val, exc_tb):
    if not self.enabled:
      return

    if self.profiler is not None:
      if current_platform.is_accelerator_available():
        current_platform.synchronize()

      self.profiler.stop()

      if self._is_npu:
        self.trace_path = self._locate_npu_output()
        logger.info(f"Profiling completed. NPU profile saved to: {self.trace_path}")
      else:
        logger.info(f"Exporting trace to: {self.trace_path}")
        self.profiler.export_chrome_trace(str(self.trace_path))
        logger.info(f"Profiling completed. Trace saved to: {self.trace_path}")

    if self._track_memory and not self._is_npu and torch.cuda.is_available():
      timestamp = int(time.time())
      rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
      memory_snapshot_path = (self.output_dir /
                              f"{self.profile_name}-rank{rank}-memory-{timestamp}.pickle")
      torch.cuda.memory._dump_snapshot(str(memory_snapshot_path))
      torch.cuda.memory._record_memory_history(enabled=None)
      logger.info(f"Memory snapshot saved to: {memory_snapshot_path}")

      memory_summary_path = (self.output_dir /
                             f"{self.profile_name}-rank{rank}-memory-{timestamp}.txt")
      with open(memory_summary_path, "w") as f:
        f.write(torch.cuda.memory_summary())
      logger.info(f"Memory summary saved to: {memory_summary_path}")

  def _locate_npu_output(self) -> Path:
    """Find the ``{worker_name}_{timestamp}_ascend_pt`` directory produced by the handler.

    Parsing is synchronous (async_mode=False), so it is normally ready right after
    ``stop()``; a short poll covers slow disk finalization.

    :returns: The produced output directory (falls back to ``output_dir`` if not found).
    """

    pattern = str(self.output_dir / f"{self._npu_worker}_*_ascend_pt")
    for _ in range(10):
      ready = [
        c for c in glob.glob(pattern) if os.path.isdir(os.path.join(c, "ASCEND_PROFILER_OUTPUT"))
      ]
      if ready:
        return Path(max(ready, key=os.path.getmtime))
      time.sleep(0.5)
    candidates = glob.glob(pattern)
    if candidates:
      return Path(max(candidates, key=os.path.getmtime))
    return self.output_dir


def profile_function(
  enabled: bool = True,
  activities: Optional[List[str]] = None,
  output_dir: Optional[str] = None,
  profile_name: Optional[str] = None,
  with_stack: bool = False,
  record_shapes: bool = True,
):
  """Decorator factory that profiles one function call with `ProfilerContext`.

  :param enabled: Whether the feature is enabled.
  :param activities: Profiler activities to capture.
  :param output_dir: Directory for generated outputs.
  :param profile_name: Base name for the profiler output.
  :param with_stack: Whether to capture Python stack traces.
  :param record_shapes: Whether to record tensor shapes.
  """

  def decorator(func):

    def wrapper(*args, **kwargs):
      name = profile_name or func.__name__
      with ProfilerContext(
          enabled=enabled,
          activities=activities,
          output_dir=output_dir,
          profile_name=name,
          with_stack=with_stack,
          record_shapes=record_shapes,
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
