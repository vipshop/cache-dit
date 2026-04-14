"""Cd cache-dit pytest tests/quantization/test_svdquant_imports.py -v -s."""

import pytest
import torch

import cache_dit.quantization.svdquant.linear as svdq_linear_module
from cache_dit.kernels import svdq_extension_is_available
from cache_dit.kernels import svdq_get_load_error
from cache_dit.kernels.cutedsl.svdquant.mma import int4_mma_opcode
from cache_dit.kernels.cutedsl.svdquant.mma import mma_bf16xbf16_f32
from cache_dit.kernels.cutedsl.svdquant.mma import mma_f16_or_bf16_f32
from cache_dit.kernels.cutedsl.svdquant.mma import mma_f16xf16_f32
from cache_dit.kernels.cutedsl.svdquant.mma import mma_m16n8k16_f32bf16bf16f32
from cache_dit.kernels.cutedsl.svdquant.mma import mma_m16n8k16_f32f16f16f32
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import cp_async_ca_16
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import cp_async_commit
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import cp_async_wait
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import exp2_approx_f32
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import fdivide_approx_f32
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import gelu_bf16x2_f32
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import h2div_bf16x2_f32
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import int2float_fast
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import int2half2_fast_4096_rn
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import ldmatrix_x4_m8n8_shared_b16
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import movmatrix_m8n8_trans_b16
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import pack_int4_pairs_to_word
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import quantize_float2_fp4
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import quantize_f32x8_to_int4_word_signed
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import quantize_f32x8_to_int4_word_unsigned
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import quantize_float2_int4_signed
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import quantize_float2_int4_unsigned
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import quantize_float2_int8_signed
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import quantize_float4_fp8
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import rcp_approx_f32
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import reduce_add_f32
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import round_f32x2_to_bf16x2_f32
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import round_f32x2_to_f16x2_f32
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import rsqrt_approx_f32
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import sigmoid_approx_f32
from cache_dit.kernels.cutedsl.svdquant.gemm_utils import tanh_approx_f32
from cache_dit.quantization.svdquant import SVDQW4A4Linear


def test_svdquant_import_safe_without_extension() -> None:
  available = svdq_extension_is_available()
  error = svdq_get_load_error()

  if available:
    assert error is None
  else:
    assert error is None or isinstance(error, Exception)


def test_svdquant_cutedsl_mma_wrappers_importable() -> None:
  assert callable(mma_m16n8k16_f32f16f16f32)
  assert callable(mma_m16n8k16_f32bf16bf16f32)
  assert callable(mma_f16xf16_f32)
  assert callable(mma_bf16xbf16_f32)
  assert callable(mma_f16_or_bf16_f32)
  assert int4_mma_opcode(act_unsigned=False) == "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32"
  assert int4_mma_opcode(act_unsigned=True) == "mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32"


def test_svdquant_cutedsl_gemm_utils_importable() -> None:
  assert callable(quantize_float2_int4_signed)
  assert callable(quantize_float2_int4_unsigned)
  assert callable(pack_int4_pairs_to_word)
  assert callable(quantize_f32x8_to_int4_word_signed)
  assert callable(quantize_f32x8_to_int4_word_unsigned)
  assert callable(rcp_approx_f32)
  assert callable(fdivide_approx_f32)
  assert callable(round_f32x2_to_f16x2_f32)
  assert callable(round_f32x2_to_bf16x2_f32)
  assert callable(quantize_float2_int8_signed)
  assert callable(quantize_float2_fp4)
  assert callable(quantize_float4_fp8)
  assert callable(tanh_approx_f32)
  assert callable(exp2_approx_f32)
  assert callable(rsqrt_approx_f32)
  assert callable(sigmoid_approx_f32)
  assert callable(h2div_bf16x2_f32)
  assert callable(gelu_bf16x2_f32)
  assert callable(reduce_add_f32)
  assert callable(int2float_fast)
  assert callable(int2half2_fast_4096_rn)
  assert callable(ldmatrix_x4_m8n8_shared_b16)
  assert callable(movmatrix_m8n8_trans_b16)
  assert callable(cp_async_ca_16)
  assert callable(cp_async_commit)
  assert callable(cp_async_wait)


def test_svdq_w4a4_linear_int4_parameter_shapes() -> None:
  layer = SVDQW4A4Linear(
    in_features=128,
    out_features=64,
    rank=32,
    precision="int4",
    torch_dtype=torch.bfloat16,
  )

  assert layer.group_size == 64
  assert layer.qweight.shape == (64, 64)
  assert layer.wscales.shape == (2, 64)
  assert layer.smooth_factor.shape == (128, )
  assert layer.smooth_factor_orig.shape == (128, )
  assert layer.proj_down.shape == (128, 32)
  assert layer.proj_up.shape == (64, 32)
  assert layer.wtscale is None
  assert layer.wcscales is None
  assert layer.runtime_kernel == "v1"


def test_svdq_w4a4_linear_rejects_unsupported_geometry() -> None:
  try:
    SVDQW4A4Linear(in_features=96, out_features=64, rank=32, precision="int4")
  except ValueError as exc:
    assert "group_size" in str(exc)
  else:
    raise AssertionError("Expected unsupported INT4 geometry to raise ValueError.")


def test_svdq_w4a4_linear_accepts_runtime_kernel_v3() -> None:
  layer = SVDQW4A4Linear(
    in_features=128,
    out_features=64,
    rank=32,
    precision="int4",
    runtime_kernel="v3",
    torch_dtype=torch.bfloat16,
  )

  assert layer.runtime_kernel == "v3"


def test_svdq_w4a4_linear_rejects_unknown_runtime_kernel() -> None:
  try:
    SVDQW4A4Linear(
      in_features=128,
      out_features=64,
      rank=32,
      precision="int4",
      runtime_kernel="v4",
    )
  except ValueError as exc:
    assert "runtime_kernel" in str(exc)
  else:
    raise AssertionError("Expected unsupported runtime_kernel to raise ValueError.")


def test_svdq_w4a4_linear_forward_accepts_2d_input(monkeypatch: pytest.MonkeyPatch, ) -> None:
  layer = SVDQW4A4Linear(
    in_features=128,
    out_features=64,
    rank=32,
    precision="int4",
    torch_dtype=torch.bfloat16,
  )

  def fake_quantize(x: torch.Tensor, pad_size: int = 256):
    del pad_size
    token_count = x.shape[0]
    quantized_x = torch.zeros(token_count + 2, 64, dtype=torch.int8)
    ascales = torch.ones(token_count + 2, 1, dtype=torch.bfloat16)
    lora_act = torch.zeros(token_count + 2, layer.rank, dtype=torch.bfloat16)
    return quantized_x, ascales, lora_act

  def fake_forward_quant(
    quantized_x: torch.Tensor,
    ascales: torch.Tensor,
    lora_act: torch.Tensor,
    output: torch.Tensor | None = None,
  ) -> torch.Tensor:
    del ascales, lora_act
    result = torch.arange(
      quantized_x.shape[0] * layer.out_features,
      dtype=torch.bfloat16,
    ).reshape(quantized_x.shape[0], layer.out_features)
    if output is not None:
      output.copy_(result)
      return output
    return result

  monkeypatch.setattr(layer, "quantize", fake_quantize)
  monkeypatch.setattr(layer, "forward_quant", fake_forward_quant)

  x = torch.randn(3, 128, dtype=torch.bfloat16)
  output = layer(x)

  assert output.shape == (3, 64)
  torch.testing.assert_close(
    output,
    torch.arange(3 * 64, dtype=torch.bfloat16).reshape(3, 64),
    rtol=0.0,
    atol=0.0,
  )


def test_svdq_w4a4_linear_v3_dispatches_cutedsl_wrappers(monkeypatch: pytest.MonkeyPatch, ) -> None:
  layer = SVDQW4A4Linear(
    in_features=128,
    out_features=64,
    rank=32,
    precision="int4",
    runtime_kernel="v3",
    torch_dtype=torch.bfloat16,
  )

  called: dict[str, object] = {}

  def fake_v3_quantize(
    input: torch.Tensor,
    lora_down: torch.Tensor | None,
    smooth: torch.Tensor | None,
    fuse_glu: bool = False,
    fp4: bool = False,
    pad_size: int = 256,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del lora_down, smooth, fuse_glu, fp4, pad_size
    called["quantize"] = tuple(input.shape)
    return (
      torch.zeros(input.shape[0], input.shape[1] // 2, dtype=torch.uint8),
      torch.ones(input.shape[1] // 64, input.shape[0], dtype=torch.bfloat16),
      torch.zeros(input.shape[0], layer.rank, dtype=torch.float32),
    )

  def fake_v3_gemm(**kwargs) -> torch.Tensor:
    called["gemm"] = kwargs["stage"]
    return torch.zeros(kwargs["act"].shape[0], layer.out_features, dtype=torch.bfloat16)

  monkeypatch.setattr(svdq_linear_module, "svdq_quantize_w4a4_act_fuse_lora_v3", fake_v3_quantize)
  monkeypatch.setattr(svdq_linear_module, "svdq_gemm_w4a4_v2_v3", fake_v3_gemm)

  x = torch.randn(2, 128, dtype=torch.bfloat16)
  output = layer(x)

  assert called["quantize"] == (2, 128)
  assert called["gemm"] == 1
  assert output.shape == (2, 64)
