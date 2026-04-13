"""Runtime and fake-registration tests for the CuTe DSL fused attention op.

Run with pytest:
    export CUDA_VISIBLE_DEVICES=7
    pytest tests/kernels/test_cutedsl_ops.py -q
"""

import importlib

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode

from cache_dit.kernels.cutedsl import fused_merge_attn_states as cutedsl_fused_merge_attn_states
from cache_dit.kernels.triton import fused_merge_attn_states as triton_fused_merge_attn_states

_TYPICAL_TEST_SHAPES = [
  (1, 512, 16, 32),
  (1, 1024, 32, 64),
  (1, 2048, 48, 128),
]
_UNSUPPORTED_TEST_SHAPE = (1, 129, 5, 70)


def _require_cuda_runtime() -> None:
  if not torch.cuda.is_available():
    pytest.skip("CUDA is required for the CuTe DSL kernel tests.")


def _require_cutedsl_runtime() -> None:
  _require_cuda_runtime()
  if importlib.util.find_spec("cutlass.cute") is None:
    pytest.skip("CuTe DSL runtime is not available.")


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


@pytest.mark.parametrize(("batch", "seq_len", "num_heads", "head_size"), _TYPICAL_TEST_SHAPES)
def test_fused_merge_attn_states_fake_registration_shapes(
  batch: int,
  seq_len: int,
  num_heads: int,
  head_size: int,
) -> None:
  _require_cutedsl_runtime()

  with FakeTensorMode():
    prev_out = torch.empty((batch, seq_len, num_heads, head_size),
                           device="cuda",
                           dtype=torch.bfloat16)
    prev_lse = torch.empty((batch, seq_len, num_heads, 1), device="cuda", dtype=torch.float32)
    suff_out = torch.empty((batch, seq_len, num_heads, head_size),
                           device="cuda",
                           dtype=torch.bfloat16)
    suff_lse = torch.empty((batch, seq_len, num_heads, 1), device="cuda", dtype=torch.float32)
    out, lse = cutedsl_fused_merge_attn_states(prev_out, prev_lse, suff_out, suff_lse)

  assert isinstance(out, FakeTensor)
  assert isinstance(lse, FakeTensor)
  assert out.shape == prev_out.shape
  assert out.dtype == prev_out.dtype
  assert lse.shape == prev_lse.shape
  assert lse.dtype == prev_lse.dtype


@pytest.mark.parametrize(("batch", "seq_len", "num_heads", "head_size"), _TYPICAL_TEST_SHAPES)
def test_fused_merge_attn_states_matches_reference_and_triton(
  batch: int,
  seq_len: int,
  num_heads: int,
  head_size: int,
) -> None:
  _require_cutedsl_runtime()

  torch.manual_seed(0)
  dtype = _runtime_dtype()
  prev_out = torch.randn((batch, seq_len, num_heads, head_size), device="cuda", dtype=dtype)
  suff_out = torch.randn((batch, seq_len, num_heads, head_size), device="cuda", dtype=dtype)
  prev_lse = torch.randn((batch, seq_len, num_heads, 1), device="cuda", dtype=torch.float32)
  suff_lse = torch.randn((batch, seq_len, num_heads, 1), device="cuda", dtype=torch.float32)

  with torch.inference_mode():
    cutedsl_out, cutedsl_lse = cutedsl_fused_merge_attn_states(
      prev_out,
      prev_lse,
      suff_out,
      suff_lse,
    )
    triton_out, triton_lse = triton_fused_merge_attn_states(
      prev_out,
      prev_lse,
      suff_out,
      suff_lse,
    )
    torch.cuda.synchronize()

  reference_out, reference_lse = _reference_fused_merge_attn_states(
    prev_out,
    prev_lse,
    suff_out,
    suff_lse,
  )

  assert cutedsl_out.shape == prev_out.shape
  assert cutedsl_lse.shape == prev_lse.shape
  assert cutedsl_out.dtype == prev_out.dtype
  assert cutedsl_lse.dtype == prev_lse.dtype
  assert torch.isfinite(cutedsl_out).all()
  assert torch.isfinite(cutedsl_lse).all()

  torch.testing.assert_close(cutedsl_out, reference_out, rtol=0.0, atol=0.01)
  torch.testing.assert_close(cutedsl_lse, reference_lse, rtol=0.0, atol=1e-5)
  torch.testing.assert_close(cutedsl_out, triton_out, rtol=0.0, atol=0.01)
  torch.testing.assert_close(cutedsl_lse, triton_lse, rtol=0.0, atol=1e-5)


def test_fused_merge_attn_states_rejects_unsupported_head_size() -> None:
  _require_cutedsl_runtime()

  batch, seq_len, num_heads, head_size = _UNSUPPORTED_TEST_SHAPE
  torch.manual_seed(0)
  dtype = _runtime_dtype()
  prev_out = torch.randn((batch, seq_len, num_heads, head_size), device="cuda", dtype=dtype)
  suff_out = torch.randn((batch, seq_len, num_heads, head_size), device="cuda", dtype=dtype)
  prev_lse = torch.randn((batch, seq_len, num_heads, 1), device="cuda", dtype=torch.float32)
  suff_lse = torch.randn((batch, seq_len, num_heads, 1), device="cuda", dtype=torch.float32)

  with pytest.raises(AssertionError, match="only supports D"):
    cutedsl_fused_merge_attn_states(
      prev_out,
      prev_lse,
      suff_out,
      suff_lse,
    )


def test_fused_merge_attn_states_rejects_mismatched_value_dtypes() -> None:
  _require_cutedsl_runtime()

  shape = (1, 64, 4, 32)
  prev_out = torch.randn(shape, device="cuda", dtype=torch.float16)
  suff_out = torch.randn(shape, device="cuda", dtype=torch.float32)
  prev_lse = torch.randn((1, 64, 4, 1), device="cuda", dtype=torch.float32)
  suff_lse = torch.randn((1, 64, 4, 1), device="cuda", dtype=torch.float32)

  with pytest.raises(AssertionError, match="share dtype"):
    cutedsl_fused_merge_attn_states(
      prev_out,
      prev_lse,
      suff_out,
      suff_lse,
    )
