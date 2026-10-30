import importlib

import pytest

import cache_dit.kernels as kernels
import cache_dit.kernels._attn as attn_ops
import cache_dit.kernels._comm as comm_ops
import cache_dit.kernels._svdquant as svdq_ops


@pytest.mark.parametrize(
  "op_name,source_module",
  [
    ("fused_merge_attn_states", attn_ops),
    ("fp8_comm_per_token_quant", comm_ops),
    ("fp8_comm_per_token_dequant", comm_ops),
    ("fp8_comm_qkv_permute_quant", comm_ops),
    ("fp8_comm_qkv_permute_dequant", comm_ops),
    ("svdq_get_load_error", svdq_ops),
    ("svdq_extension_is_available", svdq_ops),
    ("svdq_gemm_w4a4", svdq_ops),
    ("svdq_gemm_w4a4_v2", svdq_ops),
    ("svdq_gemm_w4a4_ext", svdq_ops),
    ("svdq_quantize_w4a4_act_fuse_lora", svdq_ops),
    ("svdq_quantize_w4a4_wgt", svdq_ops),
    ("svdq_set_faster_i2f_mode", svdq_ops),
    ("svdq_set_log_level", svdq_ops),
  ],
)
def test_public_api_is_reexported_from_domain_module(op_name: str, source_module) -> None:
  assert getattr(kernels, op_name) is getattr(source_module, op_name)


def test_public_api_matches_all() -> None:
  assert sorted(kernels.__all__) == sorted(name for name in vars(kernels)
                                           if not name.startswith("_") and name != "kernels")
  assert all(getattr(kernels, name) is not None for name in kernels.__all__)


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
  assert not hasattr(kernels, symbol)


@pytest.mark.parametrize("module", ["backend", "ops", "_cuda_ops", "_ops_registery"])
def test_removed_dispatch_modules(module: str) -> None:
  with pytest.raises(ModuleNotFoundError):
    importlib.import_module(f"cache_dit.kernels.{module}")
