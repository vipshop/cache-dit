"""Launch the generate CLI on several GPUs without torchrun.

`--parallel-params` already carries the explicit degrees, so when the user does not start under
torchrun this module derives the world size from them and starts one worker per visible GPU with the
environment torchrun would have exported. Ray schedules the workers when it is installed (each task
reserves one GPU); otherwise the same ranks run as local subprocesses.
"""

import os
import socket
import subprocess
import sys

from ..logger import init_logger

logger = init_logger(__name__)


def _visible_devices() -> list[str]:
  """Return the GPU ids the parent process was given, in order."""

  raw = os.environ.get("CUDA_VISIBLE_DEVICES")
  if raw is not None:
    return [device.strip() for device in raw.split(",") if device.strip()]
  import torch

  return [str(index) for index in range(torch.cuda.device_count())]


def _free_port() -> int:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    return sock.getsockname()[1]


def _rank_env(rank: int, world_size: int, port: int, cwd: str) -> dict[str, str]:
  """The environment torchrun exports for one rank: `dist.init_process_group()` reads it."""

  return {
    "RANK": str(rank),
    "LOCAL_RANK": str(rank),
    "WORLD_SIZE": str(world_size),
    "LOCAL_WORLD_SIZE": str(world_size),
    "MASTER_ADDR": "127.0.0.1",
    "MASTER_PORT": str(port),
    "CACHE_DIT_LAUNCH_CWD": cwd,
  }


def _world_size(args) -> int:
  degrees = args.parallel_degrees
  return degrees["ulysses"] * degrees["ring"] * degrees["tp"]


def _run_rank_in_process(rank: int, argv: list[str]) -> None:
  """Run one rank in the current process (a Ray worker owns exactly one GPU)."""

  cwd = os.environ.get("CACHE_DIT_LAUNCH_CWD")
  if cwd:
    os.chdir(cwd)
  sys.argv = ["cache_dit.generate", *argv]
  from .generate import entrypoint

  entrypoint()


def _run_local(argv: list[str], world_size: int, port: int) -> None:
  cwd = os.getcwd()
  devices = _visible_devices()
  processes = []
  for rank in range(world_size):
    env = os.environ.copy()
    env.update(_rank_env(rank, world_size, port, cwd))
    env["CUDA_VISIBLE_DEVICES"] = devices[rank]
    logger.info("Launching rank %d/%d on GPU %s ...", rank, world_size, devices[rank])
    processes.append(subprocess.Popen([sys.executable, "-m", "cache_dit.generate", *argv], env=env))
  for rank, process in enumerate(processes):
    return_code = process.wait()
    if return_code != 0:
      for other in processes:
        if other.poll() is None:
          other.terminate()
      raise SystemExit(f"rank {rank} exited with code {return_code}")


def _run_ray(argv: list[str], world_size: int, port: int) -> None:
  try:
    import ray
  except ImportError as error:
    raise ImportError("--launcher ray requires the 'ray' package; install it or pass "
                      "--launcher local.") from error

  ray.init(include_dashboard=False, log_to_driver=True)
  try:
    available = int(ray.cluster_resources().get("GPU", 0))
    if available < world_size:
      raise ValueError(f"Ray reports {available} GPU(s) but --parallel-params needs "
                       f"{world_size} workers.")
    cwd = os.getcwd()

    @ray.remote(num_gpus=1)
    def _worker(rank: int) -> None:
      _run_rank_in_process(rank, argv)

    tasks = [
      _worker.options(runtime_env={
        "env_vars": _rank_env(rank, world_size, port, cwd)
      }).remote(rank) for rank in range(world_size)
    ]
    try:
      ray.get(tasks)
    except Exception as error:  # RayTaskError carries the worker traceback in its message
      logger.error("A parallel worker failed: %s", error)
      raise SystemExit(1) from error
  finally:
    ray.shutdown()


def maybe_launch_parallel(args, argv: list[str]) -> bool:
  """Start the parallel workers and return True when this process is only the launcher.

  Under torchrun the workers already exist (`WORLD_SIZE` is exported), so the normal single-process
  path continues. Without it, the explicit degrees must match the number of visible GPUs: the same
  rule `--parallel-params` enforces against `WORLD_SIZE`.

  :param args: Parsed CLI namespace.
  :param argv: The original command line, replayed by every worker.
  :returns: Whether workers were launched (the caller must then stop).
  """

  if args.parallel_type is None or os.environ.get("WORLD_SIZE"):
    return False
  world_size = _world_size(args)
  if world_size <= 1:
    return False

  devices = _visible_devices()
  if len(devices) != world_size:
    raise ValueError(
      f"--parallel-params degrees (ulysses={args.parallel_degrees['ulysses']},"
      f"ring={args.parallel_degrees['ring']},tp={args.parallel_degrees['tp']}) multiply to "
      f"{world_size}, but {len(devices)} GPU(s) are visible; set CUDA_VISIBLE_DEVICES to exactly "
      f"{world_size} devices or use torchrun.")

  port = _free_port()
  launcher = getattr(args, "launcher", "auto")
  if launcher == "local":
    _run_local(argv, world_size, port)
    return True
  if launcher == "auto":
    try:
      import ray  # noqa: F401
    except ImportError:
      logger.info("Ray is not installed; launching %d local processes instead.", world_size)
      _run_local(argv, world_size, port)
      return True
  _run_ray(argv, world_size, port)
  return True
