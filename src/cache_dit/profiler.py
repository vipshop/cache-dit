"""
Torch Profiler for cache-dit.

Reference: Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_one_batch.py
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import torch
from torch.profiler import ProfilerActivity, profile
from .platforms import current_platform

logger = logging.getLogger(__name__)

# Default profiler directory
PROFILER_DIR = os.getenv("CACHE_DIT_TORCH_PROFILER_DIR", "/tmp/cache_dit_profiles")


class ProfilerContext:

    def __init__(
        self,
        enabled: bool = True,
        activities: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        profile_name: Optional[str] = None,
        with_stack: bool = True,
        record_shapes: bool = True,
    ):
        assert (
            current_platform.is_accelerator_available() and current_platform.device_type == "cuda"
        ), "Torch ProfilerContext currently only supports CUDA devices."
        self.enabled = enabled
        self.activities = activities or ["CPU", "GPU"]
        self.output_dir = Path(output_dir or PROFILER_DIR).expanduser()
        self.profile_name = profile_name or f"profile_{int(time.time())}"
        self.with_stack = with_stack
        self.record_shapes = record_shapes

        self.profiler = None
        self.trace_path = None
        self.memory_snapshot_path = None

    def __enter__(self):
        if not self.enabled:
            return self

        assert (
            current_platform.is_accelerator_available() and current_platform.device_type == "cuda"
        ), "Torch ProfilerContext currently only supports CUDA devices."

        self.output_dir.mkdir(parents=True, exist_ok=True)

        activity_map = {
            "CPU": ProfilerActivity.CPU,
            "GPU": ProfilerActivity.CUDA,
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

        if "MEM" in self.activities and torch.cuda.is_available():
            torch.cuda.memory._record_memory_history(max_entries=100000)
            logger.info("Started CUDA memory profiling")

        if torch_activities:
            self.profiler = profile(
                activities=torch_activities,
                with_stack=self.with_stack,
                record_shapes=self.record_shapes,
            )

            self.profiler.start()
            logger.info(
                f"Started profiling. Traces will be saved to: {self.output_dir} "
                f"(activities: {self.activities})"
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return

        if self.profiler is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            self.profiler.stop()

            logger.info(f"Exporting trace to: {self.trace_path}")
            self.profiler.export_chrome_trace(str(self.trace_path))

            logger.info(f"Profiling completed. Trace saved to: {self.trace_path}")

        if "MEM" in self.activities and torch.cuda.is_available():
            timestamp = int(time.time())
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            memory_snapshot_path = (
                self.output_dir / f"{self.profile_name}-rank{rank}-memory-{timestamp}.pickle"
            )
            torch.cuda.memory._dump_snapshot(str(memory_snapshot_path))
            torch.cuda.memory._record_memory_history(enabled=None)
            logger.info(f"Memory snapshot saved to: {memory_snapshot_path}")

            memory_summary_path = (
                self.output_dir / f"{self.profile_name}-rank{rank}-memory-{timestamp}.txt"
            )
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
    return ProfilerContext(
        enabled=enabled,
        activities=activities,
        output_dir=output_dir,
        profile_name=profile_name,
        **kwargs,
    )


def get_profiler_output_dir() -> str:
    return os.environ.get("CACHE_DIT_TORCH_PROFILER_DIR", PROFILER_DIR)


def set_profiler_output_dir(path: str):
    os.environ["CACHE_DIT_TORCH_PROFILER_DIR"] = path
