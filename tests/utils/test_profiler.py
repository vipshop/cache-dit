import pytest

from cache_dit.platforms import current_platform
from cache_dit.profiler import (
  ProfilerContext,
  _npu_worker_name,
  _resolve_profiler_backend,
  _supported_device,
)


class TestProfilerBackendSelection:

  def test_supported_device_returns_bool(self):
    assert isinstance(_supported_device(), bool)

  def test_supported_device_matches_platform(self):
    # CPU is not a supported profiler device; CUDA/NPU are.
    if current_platform.device_type == "cpu":
      assert _supported_device() is False
    else:
      assert _supported_device() is current_platform.is_accelerator_available()

  def test_resolve_backend_returns_cuda_on_non_npu(self):
    # On non-NPU devices the CUDA profiler backend is selected; NPU devices select
    # torch_npu.profiler. Either way the tuple shape is (callable, enum, str).
    profile_fn, activity_enum, accelerator = _resolve_profiler_backend()
    assert callable(profile_fn)
    assert hasattr(activity_enum, "CPU")
    assert isinstance(accelerator, str)
    if current_platform.device_type == "npu":
      assert accelerator == "NPU"
    else:
      assert accelerator == "CUDA"


class TestNpuWorkerName:

  def test_sanitizes_disallowed_chars(self):
    # worker_name only allows [A-Za-z0-9_-]; dots and spaces must be replaced.
    assert _npu_worker_name("wan2.2_t2v profile", 0, 1) == "wan2_2_t2v_profile"

  def test_appends_rank_for_distributed(self):
    assert _npu_worker_name("flux", 3, 8) == "flux-rank3"

  def test_single_rank_omits_suffix(self):
    assert _npu_worker_name("flux", 0, 1) == "flux"

  def test_falls_back_for_empty_name(self):
    assert _npu_worker_name("", 0, 1) == "cache_dit"
    # All-disallowed chars sanitize to empty and fall back to the default.
    assert _npu_worker_name("...", 0, 1) == "cache_dit"


class TestProfilerContextGuard:

  def test_construct_raises_on_unsupported_device(self):
    # ProfilerContext must refuse unsupported devices (e.g. CPU) instead of
    # silently producing an empty/invalid trace.
    if _supported_device():
      pytest.skip("device is supported by the profiler")
    with pytest.raises(AssertionError):
      ProfilerContext(enabled=True)
