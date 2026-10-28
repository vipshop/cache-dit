import importlib

import pytest
import torch

import cache_dit.kernels._cuda_ops as cuda_kernels
from cache_dit.kernels import ops


@pytest.mark.parametrize(
  "op_name,args",
  [
    ("fp8_comm_per_token_quant", (torch.empty(0), )),
    ("fp8_comm_per_token_dequant", (torch.empty(0), )),
    ("fp8_comm_qkv_permute_quant", (torch.empty(0), )),
    ("fp8_comm_qkv_permute_dequant", (torch.empty(0), )),
  ],
)
def test_public_ops_dispatch_directly_to_cuda(
  op_name: str,
  args: tuple[torch.Tensor, ...],
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  sentinel = object()

  def fake_cuda_op(*_args):
    return sentinel

  monkeypatch.setattr(cuda_kernels, op_name, fake_cuda_op)
  assert getattr(ops, op_name)(*args) is sentinel


@pytest.mark.parametrize(
  "symbol",
  [
    "KernelBackend",
    "_ERROR_TEMPLATE",
    "_KERNEL_BE_SELECTOR_MAP",
    "_get_backend_selector",
    "_resolve_backend",
    "_select_cuda_backend",
    "_select_mindiesd_backend",
  ],
)
def test_backend_dispatch_layer_removed(symbol: str) -> None:
  assert not hasattr(ops, symbol)


def test_backend_module_removed() -> None:
  with pytest.raises(ModuleNotFoundError):
    importlib.import_module("cache_dit.kernels.backend")


def test_merge_states_wrapper_is_dedicated() -> None:
  from cache_dit.kernels import _merge_states

  assert ops.fused_merge_attn_states is _merge_states.fused_merge_attn_states
  assert not hasattr(cuda_kernels, "fused_merge_attn_states")
