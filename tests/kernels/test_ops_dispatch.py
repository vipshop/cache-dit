import importlib

import pytest

import cache_dit.kernels as kernels
import cache_dit.kernels._attn as attn_ops
import cache_dit.kernels._comm as comm_ops
import cache_dit.kernels._cuda_ext as cuda_ext
import cache_dit.kernels._models as models_ops
import cache_dit.kernels._svdquant as svdq_ops


@pytest.mark.parametrize(
  "op_name,source_module",
  [
    ("fused_merge_attn_states", attn_ops),
    ("fp8_comm_per_token_quant", comm_ops),
    ("fp8_comm_per_token_dequant", comm_ops),
    ("fp8_comm_qkv_permute_quant", comm_ops),
    ("fp8_comm_qkv_permute_dequant", comm_ops),
    ("cuda_ext_is_available", cuda_ext),
    ("cuda_ext_get_load_error", cuda_ext),
    ("boogu_scaled_rms_norm", models_ops),
    ("boogu_gated_rms_norm", models_ops),
    ("boogu_fused_qk_norm_rope", models_ops),
    ("svdq_quantize_w4a4", svdq_ops),
    ("svdq_gemm_w4a4", svdq_ops),
  ],
)
def test_public_api_is_reexported_from_domain_module(op_name: str, source_module) -> None:
  assert getattr(kernels, op_name) is getattr(source_module, op_name)


def test_public_api_matches_all() -> None:
  assert sorted(kernels.__all__) == sorted(name for name in vars(kernels)
                                           if not name.startswith("_") and name != "kernels")
  assert all(getattr(kernels, name) is not None for name in kernels.__all__)


@pytest.mark.parametrize("module", [kernels, attn_ops, comm_ops, models_ops, svdq_ops])
@pytest.mark.parametrize(
  "symbol",
  [
    "comm_extension_is_available",
    "comm_get_load_error",
    "attn_extension_is_available",
    "attn_get_load_error",
    "models_extension_is_available",
    "models_get_load_error",
    "svdq_is_available",
    "svdq_get_load_error",
  ],
)
def test_per_family_availability_getters_removed(module, symbol: str) -> None:
  # The package-level check alone would still pass if a domain module kept its own copy.
  assert not hasattr(module, symbol)


def test_single_extension_module_name() -> None:
  assert cuda_ext._EXTENSION_MODULE_NAME == "cache_dit._C_ops"


def test_svdq_has_single_public_gemm() -> None:
  assert not hasattr(kernels, "svdq_gemm_w4a4_ext")
  assert not hasattr(svdq_ops, "_svdq_gemm_w4a4_ext")
  assert not hasattr(svdq_ops, "_call_svdq_gemm_w4a4_native")


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
