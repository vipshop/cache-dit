"""Runtime and fake-registration tests for Triton kernel wrappers.

Run with pytest:
    export CUDA_VISIBLE_DEVICES=7
    pytest tests/kernels/test_triton_ops.py -q
"""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode

from cache_dit.kernels import fp8_comm_per_token_dequant
from cache_dit.kernels import fp8_comm_per_token_quant
from cache_dit.kernels import fp8_comm_qkv_permute_dequant
from cache_dit.kernels import fp8_comm_qkv_permute_quant
from cache_dit.kernels import fused_merge_attn_states


def _require_cuda_runtime() -> None:
  if not torch.cuda.is_available():
    pytest.skip("CUDA is required for the Triton kernel tests.")


def _require_float8_runtime() -> None:
  _require_cuda_runtime()
  capability = torch.cuda.get_device_capability()
  if capability < (8, 9):
    pytest.skip(f"Float8 Triton kernels require compute capability >= 8.9, got {capability}.")


def _runtime_dtype() -> torch.dtype:
  return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _reference_fused_merge_attn_states(
  prev_out: torch.Tensor,
  prev_lse: torch.Tensor,
  suff_out: torch.Tensor,
  suff_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  prev_out_fp32 = prev_out.float()
  suff_out_fp32 = suff_out.float()
  prev_lse_fp32 = prev_lse.float()
  suff_lse_fp32 = suff_lse.float()
  out = prev_out_fp32 - torch.sigmoid(suff_lse_fp32 - prev_lse_fp32) * (prev_out_fp32 -
                                                                        suff_out_fp32)
  lse = prev_lse_fp32 - torch.log(torch.sigmoid(prev_lse_fp32 - suff_lse_fp32))
  return out.to(suff_out.dtype), lse.to(suff_lse.dtype)


def test_fp8_comm_per_token_fake_registration_shapes() -> None:
  with FakeTensorMode():
    x = torch.empty((2, 3, 128), device="cuda", dtype=torch.bfloat16)
    quantized = fp8_comm_per_token_quant(x)
    dequantized = fp8_comm_per_token_dequant(quantized)

  assert isinstance(quantized, FakeTensor)
  assert quantized.shape == (2, 3, 130)
  assert quantized.dtype == torch.float8_e4m3fn
  assert isinstance(dequantized, FakeTensor)
  assert dequantized.shape == x.shape
  assert dequantized.dtype == torch.bfloat16


def test_fp8_comm_qkv_fake_registration_shapes() -> None:
  with FakeTensorMode():
    x = torch.empty((2, 4, 3, 5, 64), device="cuda", dtype=torch.bfloat16)
    quantized = fp8_comm_qkv_permute_quant(x)
    dequantized = fp8_comm_qkv_permute_dequant(quantized[0])

  assert isinstance(quantized, FakeTensor)
  assert quantized.shape == (3, 4, 2, 5, 68)
  assert quantized.dtype == torch.float8_e4m3fn
  assert isinstance(dequantized, FakeTensor)
  assert dequantized.shape == (2, 4, 5, 64)
  assert dequantized.dtype == torch.bfloat16


def test_fused_merge_attn_states_fake_registration_shapes() -> None:
  with FakeTensorMode():
    prev_out = torch.empty((2, 3, 4, 16), device="cuda", dtype=torch.bfloat16)
    prev_lse = torch.empty((2, 3, 4, 1), device="cuda", dtype=torch.float32)
    suff_out = torch.empty((2, 3, 4, 16), device="cuda", dtype=torch.bfloat16)
    suff_lse = torch.empty((2, 3, 4, 1), device="cuda", dtype=torch.float32)
    out, lse = fused_merge_attn_states(prev_out, prev_lse, suff_out, suff_lse)

  assert isinstance(out, FakeTensor)
  assert isinstance(lse, FakeTensor)
  assert out.shape == prev_out.shape
  assert out.dtype == prev_out.dtype
  assert lse.shape == prev_lse.shape
  assert lse.dtype == prev_lse.dtype


def test_fp8_comm_per_token_quant_dequant_roundtrip() -> None:
  _require_float8_runtime()

  torch.manual_seed(0)
  x = torch.randn((2, 3, 128), device="cuda", dtype=torch.bfloat16)

  with torch.inference_mode():
    quantized = fp8_comm_per_token_quant(x)
    dequantized = fp8_comm_per_token_dequant(quantized)
    torch.cuda.synchronize()

  diff = (dequantized.float() - x.float()).abs()
  assert quantized.shape == (2, 3, 130)
  assert quantized.dtype == torch.float8_e4m3fn
  assert dequantized.shape == x.shape
  assert dequantized.dtype == torch.bfloat16
  assert torch.isfinite(quantized.float()).all()
  assert torch.isfinite(dequantized).all()
  assert diff.mean().item() < 0.05
  assert diff.max().item() < 0.2


def test_fp8_comm_qkv_permute_quant_dequant_tracks_selected_slice() -> None:
  _require_float8_runtime()

  torch.manual_seed(0)
  x = torch.randn((2, 4, 3, 5, 64), device="cuda", dtype=torch.bfloat16)

  with torch.inference_mode():
    quantized = fp8_comm_qkv_permute_quant(x)
    torch.cuda.synchronize()

  assert quantized.shape == (3, 4, 2, 5, 68)
  assert quantized.dtype == torch.float8_e4m3fn
  assert torch.isfinite(quantized.float()).all()

  with torch.inference_mode():
    for p_index in range(x.shape[2]):
      dequantized = fp8_comm_qkv_permute_dequant(quantized[p_index])
      torch.cuda.synchronize()

      distances = []
      for ref_index in range(x.shape[2]):
        reference = x[:, :, ref_index, :, :].float()
        distances.append((dequantized.float() - reference).abs().mean().item())

      assert dequantized.shape == (x.shape[0], x.shape[1], x.shape[3], x.shape[4])
      assert dequantized.dtype == torch.bfloat16
      assert torch.isfinite(dequantized).all()
      assert min(range(len(distances)), key=lambda index: distances[index]) == p_index


def test_fused_merge_attn_states_matches_reference() -> None:
  _require_cuda_runtime()

  torch.manual_seed(0)
  dtype = _runtime_dtype()
  prev_out = torch.randn((2, 3, 4, 16), device="cuda", dtype=dtype)
  suff_out = torch.randn((2, 3, 4, 16), device="cuda", dtype=dtype)
  prev_lse = torch.randn((2, 3, 4, 1), device="cuda", dtype=torch.float32)
  suff_lse = torch.randn((2, 3, 4, 1), device="cuda", dtype=torch.float32)

  with torch.inference_mode():
    out, lse = fused_merge_attn_states(prev_out, prev_lse, suff_out, suff_lse)
    torch.cuda.synchronize()

  reference_out, reference_lse = _reference_fused_merge_attn_states(
    prev_out,
    prev_lse,
    suff_out,
    suff_lse,
  )
  assert out.shape == prev_out.shape
  assert lse.shape == prev_lse.shape
  assert out.dtype == prev_out.dtype
  assert lse.dtype == prev_lse.dtype
  assert torch.isfinite(out).all()
  assert torch.isfinite(lse).all()
  torch.testing.assert_close(out, reference_out, rtol=0.0, atol=0.01)
  torch.testing.assert_close(lse, reference_lse, rtol=0.0, atol=1e-5)
