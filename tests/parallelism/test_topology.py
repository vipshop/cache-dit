import pytest

import cache_dit.distributed.topology as topology

_TOPOLOGY_TEXT = """\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\x1b[0m
GPU0\t X \tNODE\tSYS\tSYS\t0-383
GPU1\tNODE\t X \tSYS\tSYS\t0-383
GPU2\tSYS\tSYS\t X \tNODE\t384-767
GPU3\tSYS\tSYS\tNODE\t X \t384-767
"""


def test_parse_topology_and_find_cross_numa_pairs():
  matrix, has_nvlink = topology._parse_topology(_TOPOLOGY_TEXT)

  assert not has_nvlink
  assert matrix[0][1] == "NODE"
  assert matrix[0][2] == "SYS"
  assert topology._cross_numa_pairs(matrix, [0, 1]) == []
  assert topology._cross_numa_pairs(matrix, [0, 1, 2, 3]) == [(0, 2), (0, 3), (1, 2), (1, 3)]


def test_parse_topology_rejects_unknown_format():
  with pytest.raises(ValueError, match="no GPU rows"):
    topology._parse_topology("not a topology table")


def test_cross_numa_warning_only_without_nvlink(monkeypatch):
  warnings = []
  monkeypatch.setattr(topology, "_query_topology", lambda: topology._parse_topology(_TOPOLOGY_TEXT))
  monkeypatch.setattr(topology, "_visible_gpu_indices", lambda: [0, 1, 2, 3])
  monkeypatch.setattr(topology.logger, "warning_once", warnings.append)

  topology.warn_if_cross_numa([0, 1])
  assert warnings == []
  topology.warn_if_cross_numa([0, 1, 2, 3])
  assert len(warnings) == 1
  assert "spans NUMA nodes" in warnings[0]


def test_nvlink_suppresses_cross_numa_warning(monkeypatch):
  matrix, _ = topology._parse_topology(_TOPOLOGY_TEXT)
  matrix[0][2] = matrix[2][0] = "NV2"
  monkeypatch.setattr(topology, "_query_topology", lambda: (matrix, True))
  monkeypatch.setattr(topology, "_visible_gpu_indices", lambda: [0, 1, 2, 3])
  warnings = []
  monkeypatch.setattr(topology.logger, "warning_once", warnings.append)

  topology.warn_if_cross_numa([0, 2])
  assert warnings == []


def test_unrelated_nvlink_does_not_suppress_sys_warning(monkeypatch):
  matrix, _ = topology._parse_topology(_TOPOLOGY_TEXT)
  matrix[0][1] = matrix[1][0] = "NV2"
  monkeypatch.setattr(topology, "_query_topology", lambda: (matrix, True))
  monkeypatch.setattr(topology, "_visible_gpu_indices", lambda: [0, 1, 2, 3])
  warnings = []
  monkeypatch.setattr(topology.logger, "warning_once", warnings.append)

  topology.warn_if_cross_numa([0, 2])
  assert len(warnings) == 1


def test_nonlocal_ranks_skip_topology_mapping(monkeypatch):
  monkeypatch.setattr(topology, "_query_topology", lambda: topology._parse_topology(_TOPOLOGY_TEXT))
  monkeypatch.setattr(topology, "_visible_gpu_indices", lambda: [0, 1, 2, 3])
  warnings = []
  monkeypatch.setattr(topology.logger, "warning_once", warnings.append)

  topology.warn_if_cross_numa([4, 5])
  assert warnings == []


def test_topology_discovery_is_fail_open(monkeypatch):

  def _fail():
    raise OSError("nvidia-smi is unavailable")

  monkeypatch.setattr(topology, "_query_topology", _fail)
  warnings = []
  monkeypatch.setattr(topology.logger, "warning_once", warnings.append)

  topology.warn_if_cross_numa([0, 1])
  assert warnings == []


def test_non_numeric_visible_devices_skip_warning(monkeypatch):
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc,GPU-def")
  assert topology._visible_gpu_indices() == []
