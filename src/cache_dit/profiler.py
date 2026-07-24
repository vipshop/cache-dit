"""Torch Profiler for cache-dit.

Reference: Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_one_batch.py
"""

import gzip
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from .platforms import current_platform

logger = logging.getLogger(__name__)

# Default profiler directory
PROFILER_DIR = os.getenv("CACHE_DIT_TORCH_PROFILER_DIR", "/tmp/cache_dit_profiles")


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


class ProfilerContext:
  """Context manager wrapper around ``torch.profiler`` / ``torch_npu.profiler``.

  It centralizes trace-file naming, optional accelerator memory-history capture, and
  multi-rank output layout so profiling can be enabled consistently from scripts or
  helper decorators. CUDA devices capture memory snapshots via ``torch.cuda.memory``;
  Ascend NPU devices record memory through the profiler's own ``profile_memory`` option.
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

    filename_parts = [self.profile_name]
    if world_size > 1:
      filename_parts.append(f"rank{rank}")
    filename = "-".join(filename_parts) + ".trace.json.gz"
    self.trace_path = self.output_dir / filename

    if self._track_memory and not self._is_npu and torch.cuda.is_available():
      torch.cuda.memory._record_memory_history(max_entries=100000)
      logger.info("Started CUDA memory profiling")

    if torch_activities:
      profiler_kwargs = dict(
        activities=torch_activities,
        with_stack=self.with_stack,
        record_shapes=self.record_shapes,
      )
      if self._is_npu and self._track_memory:
        profiler_kwargs["profile_memory"] = True
      self.profiler = profile_fn(**profiler_kwargs)

      self.profiler.start()
      logger.info(f"Started profiling. Traces will be saved to: {self.output_dir} "
                  f"(activities: {self.activities})")

    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    if not self.enabled:
      return

    if self.profiler is not None:
      if current_platform.is_accelerator_available():
        current_platform.synchronize()

      self.profiler.stop()

      logger.info(f"Exporting trace to: {self.trace_path}")
      if self._is_npu:
        # torch_npu.profiler.export_chrome_trace can only write a plain .json file;
        # export to a temporary json then gzip it so the .trace.json.gz artifact
        # convention used on CUDA is preserved. with_suffix("") strips the trailing
        # ".gz" so the temp path keeps a ".json" suffix the exporter accepts.
        tmp_trace = self.trace_path.with_suffix("")
        self.profiler.export_chrome_trace(str(tmp_trace))
        with open(tmp_trace, "rb") as src, gzip.open(self.trace_path, "wb") as dst:
          dst.writelines(src)
        tmp_trace.unlink(missing_ok=True)
      else:
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
