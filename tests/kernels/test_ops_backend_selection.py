import pytest

from cache_dit.kernels.backend import KernelBackend
from cache_dit.kernels.ops import _get_backend_selector
from cache_dit.kernels.ops import _resolve_backend
from cache_dit.kernels.ops import _select_cuda_backend

_COMM_OPS = [
  "fp8_comm_per_token_quant",
  "fp8_comm_per_token_dequant",
  "fp8_comm_qkv_permute_quant",
  "fp8_comm_qkv_permute_dequant",
  "fused_merge_attn_states",
]


@pytest.mark.parametrize("op_name", _COMM_OPS)
def test_comm_ops_use_cuda_selector(op_name: str) -> None:
  assert _get_backend_selector(op_name) is _select_cuda_backend


def test_get_backend_selector_rejects_unknown_op() -> None:
  with pytest.raises(KeyError, match="No backend selector registered"):
    _get_backend_selector("unknown_op")


def test_select_cuda_backend() -> None:
  assert _select_cuda_backend() == KernelBackend.CUDA


@pytest.mark.parametrize("op_name", _COMM_OPS)
def test_resolve_comm_backend(op_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(KernelBackend, "is_supported", classmethod(lambda cls, backend: True))
  assert _resolve_backend(op_name) == KernelBackend.CUDA
