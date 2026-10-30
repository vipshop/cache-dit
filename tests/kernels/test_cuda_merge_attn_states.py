import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode

from cache_dit.kernels._attn import attn_extension_is_available
from cache_dit.kernels._attn import fused_merge_attn_states


def _require_runtime() -> None:
  if not torch.cuda.is_available() or not attn_extension_is_available():
    pytest.skip("CUDA and the cache-dit attention extension are required.")


def _reference_merge(prev_out, prev_lse, suff_out, suff_lse):
  prev_out = prev_out.float()
  suff_out = suff_out.float()
  prev_lse = prev_lse.float()
  suff_lse = suff_lse.float()
  prev_lse = torch.where(torch.isinf(prev_lse), torch.full_like(prev_lse, -torch.inf), prev_lse)
  suff_lse = torch.where(torch.isinf(suff_lse), torch.full_like(suff_lse, -torch.inf), suff_lse)
  max_lse = torch.maximum(prev_lse, suff_lse)
  both_empty = torch.isneginf(max_lse)
  prev_scale = torch.exp(prev_lse - max_lse)
  suff_scale = torch.exp(suff_lse - max_lse)
  denom = prev_scale + suff_scale
  safe_denom = torch.where(both_empty, torch.ones_like(denom), denom)
  prev_weight = torch.where(both_empty, torch.ones_like(denom), prev_scale / safe_denom)
  suff_weight = torch.where(both_empty, torch.zeros_like(denom), suff_scale / safe_denom)
  out = prev_out * prev_weight + suff_out * suff_weight
  lse = torch.where(both_empty, max_lse, max_lse + torch.log(safe_denom))
  return out, lse


def test_fused_merge_attn_states_fake_registration() -> None:
  with FakeTensorMode():
    prev_out = torch.empty((2, 3, 4, 64), device="cuda", dtype=torch.bfloat16)
    suff_out = torch.empty_like(prev_out)
    prev_lse = torch.empty((2, 3, 4, 1), device="cuda", dtype=torch.float32)
    suff_lse = torch.empty_like(prev_lse)
    out, lse = fused_merge_attn_states(prev_out, prev_lse, suff_out, suff_lse)

  assert isinstance(out, FakeTensor)
  assert out.shape == prev_out.shape
  assert out.dtype == prev_out.dtype
  assert out.is_contiguous()
  assert lse.shape == prev_lse.shape
  assert lse.dtype == prev_lse.dtype
  assert lse.is_contiguous()


@pytest.mark.parametrize("head_size", [31, 32, 64, 96, 128, 192, 256])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("lse_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fused_merge_attn_states_matches_stable_reference(
  head_size: int,
  out_dtype: torch.dtype,
  lse_dtype: torch.dtype,
) -> None:
  _require_runtime()
  torch.manual_seed(head_size)
  shape = (2, 3, 4, head_size)
  prev_out = torch.randn(shape, device="cuda", dtype=out_dtype)
  suff_out = torch.randn(shape, device="cuda", dtype=out_dtype)
  prev_lse = torch.randn((2, 3, 4, 1), device="cuda", dtype=lse_dtype) * 20
  suff_lse = torch.randn((2, 3, 4, 1), device="cuda", dtype=lse_dtype) * 20

  out, lse = fused_merge_attn_states(prev_out, prev_lse, suff_out, suff_lse)
  expected_out, expected_lse = _reference_merge(prev_out, prev_lse, suff_out, suff_lse)
  out_atol = 2e-5 if out_dtype == torch.float32 else 0.02
  lse_atol = 2e-5 if lse_dtype == torch.float32 else 0.05
  torch.testing.assert_close(out.float(), expected_out.to(out_dtype).float(), rtol=0, atol=out_atol)
  torch.testing.assert_close(lse.float(), expected_lse.to(lse_dtype).float(), rtol=0, atol=lse_atol)


def test_fused_merge_attn_states_empty_segments() -> None:
  _require_runtime()
  shape = (1, 2, 3, 64)
  prev_out = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
  suff_out = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
  prev_lse = torch.randn((1, 2, 3, 1), device="cuda", dtype=torch.float32)
  suff_lse = torch.randn_like(prev_lse)
  prev_lse[0, 0, 0] = torch.inf
  suff_lse[0, 0, 1] = -torch.inf
  prev_lse[0, 1, 0] = -torch.inf
  suff_lse[0, 1, 0] = -torch.inf
  prev_lse[0, 0, 2] = torch.nan

  out, lse = fused_merge_attn_states(prev_out, prev_lse, suff_out, suff_lse)
  expected_out, expected_lse = _reference_merge(prev_out, prev_lse, suff_out, suff_lse)
  torch.testing.assert_close(out.float(), expected_out, rtol=0, atol=0.02, equal_nan=True)
  torch.testing.assert_close(lse, expected_lse, rtol=0, atol=2e-5, equal_nan=True)
  torch.testing.assert_close(out[0, 0, 0], suff_out[0, 0, 0], rtol=0, atol=0)
  torch.testing.assert_close(out[0, 1, 0], prev_out[0, 1, 0], rtol=0, atol=0)
  assert torch.isneginf(lse[0, 1, 0]).all()
  assert torch.isnan(out[0, 0, 2]).all()
  assert torch.isnan(lse[0, 0, 2]).all()


def test_fused_merge_attn_states_empty_input() -> None:
  _require_runtime()
  shape = (0, 2, 3, 64)
  prev_out = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
  suff_out = torch.empty_like(prev_out)
  prev_lse = torch.empty((0, 2, 3, 1), device="cuda", dtype=torch.float32)
  suff_lse = torch.empty_like(prev_lse)
  out, lse = fused_merge_attn_states(prev_out, prev_lse, suff_out, suff_lse)
  assert out.shape == prev_out.shape
  assert lse.shape == prev_lse.shape
  assert out.numel() == 0
  assert lse.numel() == 0


def test_fused_merge_attn_states_compile_smoke() -> None:
  _require_runtime()
  shape = (1, 2, 3, 64)
  prev_out = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
  suff_out = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
  prev_lse = torch.randn((1, 2, 3, 1), device="cuda", dtype=torch.float32)
  suff_lse = torch.randn_like(prev_lse)
  compiled = torch.compile(fused_merge_attn_states, fullgraph=True)
  out, lse = compiled(prev_out, prev_lse, suff_out, suff_lse)
  expected_out, expected_lse = _reference_merge(prev_out, prev_lse, suff_out, suff_lse)
  torch.testing.assert_close(out.float(), expected_out, rtol=0, atol=0.02)
  torch.testing.assert_close(lse, expected_lse, rtol=0, atol=2e-5)
