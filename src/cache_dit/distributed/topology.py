"""Read-only NUMA topology diagnostics for context-parallel process groups."""

import os
import re
import subprocess
from functools import lru_cache

from ..logger import init_logger

logger = init_logger(__name__)

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
_GPU_ROW = re.compile(r"^GPU(\d+)$")


def _parse_topology(text: str) -> tuple[list[list[str]], bool]:
  """Parse `nvidia-smi topo -m` into a GPU-to-GPU link matrix.

  :param text: Raw command output, optionally containing ANSI color sequences.
  :returns: The square link matrix and whether any NVLink cell is present.
  :raises ValueError: If no complete GPU matrix can be found.
  """
  rows: dict[int, list[str]] = {}
  for line in _ANSI_ESCAPE.sub("", text).splitlines():
    tokens = line.strip().split()
    if not tokens or _GPU_ROW.fullmatch(tokens[0]) is None:
      continue
    index = int(tokens[0][3:])
    rows[index] = tokens[1:]
  if not rows:
    raise ValueError("no GPU rows found in topology output")
  count = len(rows)
  if sorted(rows) != list(range(count)):
    raise ValueError("GPU indices are not contiguous")
  matrix = []
  for index in range(count):
    cells = rows[index]
    if len(cells) < count:
      raise ValueError(f"incomplete topology row for GPU{index}")
    matrix.append(cells[:count])
  has_nvlink = any(re.fullmatch(r"NV\d+", cell) is not None for row in matrix for cell in row)
  return matrix, has_nvlink


def _cross_numa_pairs(matrix: list[list[str]], gpu_indices: list[int]) -> list[tuple[int, int]]:
  """Return unordered GPU pairs connected through the host SMP interconnect.

  :param matrix: Link matrix produced by `_parse_topology`.
  :param gpu_indices: Physical GPU indices belonging to one CP group.
  :returns: Unique `(gpu_a, gpu_b)` pairs whose link is `SYS`.
  """
  pairs = set()
  for offset, gpu_a in enumerate(gpu_indices):
    for gpu_b in gpu_indices[offset + 1:]:
      if matrix[gpu_a][gpu_b] == "SYS":
        pairs.add((min(gpu_a, gpu_b), max(gpu_a, gpu_b)))
  return sorted(pairs)


def _visible_gpu_indices() -> list[int]:
  """Map logical device ordinals to physical GPU indices when CUDA_VISIBLE_DEVICES is set.

  :returns: Physical GPU indices, or an empty list when the mapping is not numeric.
  """
  visible = os.environ.get("CUDA_VISIBLE_DEVICES")
  if visible is None or not visible.strip():
    import torch

    return list(range(torch.cuda.device_count()))
  devices = [item.strip() for item in visible.split(",")]
  if not all(item.isdigit() for item in devices):
    return []
  return [int(item) for item in devices]


@lru_cache(maxsize=1)
def _query_topology() -> tuple[list[list[str]], bool]:
  """Run and parse `nvidia-smi topo -m` once per process.

  :returns: Parsed link matrix and NVLink presence.
  """
  result = subprocess.run(
    ["nvidia-smi", "topo", "-m"],
    check=True,
    capture_output=True,
    text=True,
    timeout=2,
  )
  return _parse_topology(result.stdout)


def warn_if_cross_numa(cp_ranks: list[int]) -> None:
  """Warn when a CP group contains PCIe SYS links between different NUMA nodes.

  :param cp_ranks: Global distributed ranks in one CP group. This function never creates or modifies
    a process group, and all discovery failures are intentionally fail-open.
  """
  if len(cp_ranks) < 2:
    return
  try:
    matrix, _ = _query_topology()
    visible = _visible_gpu_indices()
    if not visible:
      logger.debug("Skip NUMA warning because CUDA_VISIBLE_DEVICES is not a numeric GPU list.")
      return
    if max(cp_ranks) >= len(visible):
      logger.debug("Skip NUMA warning because not all ranks belong to the local node.")
      return
    gpu_indices = [visible[rank] for rank in cp_ranks]
    pairs = _cross_numa_pairs(matrix, gpu_indices)
  except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
    logger.debug(f"Skip NUMA topology warning: {exc}")
    return
  if pairs:
    logger.warning_once(
      f"Context-parallel group ranks {cp_ranks} spans NUMA nodes through PCIe SYS links "
      f"(GPU pairs: {pairs}) without NVLink. Keep each CP group within one NUMA node when "
      "possible; cross-NUMA all-to-all bandwidth can be substantially lower.")


__all__ = ["warn_if_cross_numa"]
