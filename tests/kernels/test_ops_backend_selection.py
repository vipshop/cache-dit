import pytest

from cache_dit.kernels.backend import KernelBackend
from cache_dit.kernels.ops import _get_backend_selector
from cache_dit.kernels.ops import _select_kernel_backend


def test_get_backend_selector_returns_registered_selector() -> None:
  assert _get_backend_selector("fused_merge_attn_states") is _select_kernel_backend
  assert _get_backend_selector("fp8_comm_per_token_quant") is _select_kernel_backend
  assert _get_backend_selector("fp8_comm_per_token_dequant") is _select_kernel_backend
  assert _get_backend_selector("fp8_comm_qkv_permute_quant") is _select_kernel_backend
  assert _get_backend_selector("fp8_comm_qkv_permute_dequant") is _select_kernel_backend


def test_get_backend_selector_rejects_unknown_op() -> None:
  with pytest.raises(KeyError, match="No backend selector registered"):
    _get_backend_selector("unknown_op")


def _patch_supported_backends(monkeypatch: pytest.MonkeyPatch, cutedsl_enabled: bool) -> None:

  def _is_supported(cls, backend: KernelBackend) -> bool:
    if backend == KernelBackend.CUTEDSL:
      return cutedsl_enabled
    if backend == KernelBackend.TRITON:
      return True
    return False

  monkeypatch.setattr(KernelBackend, "is_supported", classmethod(_is_supported))


def test_select_kernel_backend_prefers_cutedsl_when_available(
  monkeypatch: pytest.MonkeyPatch, ) -> None:
  _patch_supported_backends(monkeypatch, cutedsl_enabled=True)

  assert _select_kernel_backend() == KernelBackend.CUTEDSL


def test_select_kernel_backend_falls_back_to_triton_when_cutedsl_missing(
  monkeypatch: pytest.MonkeyPatch, ) -> None:
  _patch_supported_backends(monkeypatch, cutedsl_enabled=False)

  assert _select_kernel_backend() == KernelBackend.TRITON
