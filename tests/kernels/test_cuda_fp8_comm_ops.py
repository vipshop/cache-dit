import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode

from cache_dit.kernels.cuda import fp8_comm_per_token_dequant
from cache_dit.kernels.cuda import fp8_comm_per_token_quant
from cache_dit.kernels.cuda import fp8_comm_qkv_permute_dequant
from cache_dit.kernels.cuda import fp8_comm_qkv_permute_quant
from cache_dit.kernels.cuda._comm import comm_extension_is_available


def _require_runtime() -> None:
  if not torch.cuda.is_available() or not comm_extension_is_available():
    pytest.skip("CUDA and the cache-dit communication extension are required.")


def _pack_rows(rows: torch.Tensor, eps: float, scale_dtype: torch.dtype) -> torch.Tensor:
  rows_fp32 = rows.float()
  head_size = rows.shape[-1]
  scale = (rows_fp32.abs().amax(dim=-1, keepdim=True) / 448.0).clamp_min(eps)
  quantized = (rows_fp32 * scale.reciprocal()).clamp(-448.0, 448.0).half()
  quantized = quantized.to(torch.float8_e4m3fn).view(torch.uint8)
  scale_bytes = scale.to(scale_dtype).contiguous().view(torch.uint8).reshape(rows.shape[0], -1)
  packed = torch.empty((rows.shape[0], head_size + scale_bytes.shape[1]),
                       device=rows.device,
                       dtype=torch.uint8)
  packed[:, :head_size] = quantized
  packed[:, head_size:] = scale_bytes
  return packed.view(torch.float8_e4m3fn)


def _reference_per_token_quant(x: torch.Tensor) -> torch.Tensor:
  return _pack_rows(x.reshape(-1, x.shape[-1]), 1e-4,
                    torch.bfloat16).reshape(*x.shape[:-1], x.shape[-1] + 2)


def _reference_qkv_quant(x: torch.Tensor) -> torch.Tensor:
  batch, seq_len, partitions, num_heads, head_size = x.shape
  rows = x.permute(2, 1, 0, 3, 4).reshape(-1, head_size)
  return _pack_rows(rows, 1e-6, torch.float32).reshape(partitions, seq_len, batch, num_heads,
                                                       head_size + 4)


def test_fp8_comm_fake_registration_shapes() -> None:
  with FakeTensorMode():
    x = torch.empty((2, 3, 128), device="cuda", dtype=torch.bfloat16)
    quantized = fp8_comm_per_token_quant(x)
    dequantized = fp8_comm_per_token_dequant(quantized)
    qkv = torch.empty((2, 4, 3, 5, 64), device="cuda", dtype=torch.bfloat16)
    quantized_qkv = fp8_comm_qkv_permute_quant(qkv)
    dequantized_qkv = fp8_comm_qkv_permute_dequant(quantized_qkv[0])

  assert isinstance(quantized, FakeTensor)
  assert quantized.shape == (2, 3, 130)
  assert quantized.dtype == torch.float8_e4m3fn
  assert dequantized.shape == x.shape
  assert dequantized.dtype == torch.bfloat16
  assert quantized_qkv.shape == (3, 4, 2, 5, 68)
  assert dequantized_qkv.shape == (2, 4, 5, 64)
  assert dequantized_qkv.is_contiguous()


@pytest.mark.parametrize("head_size", [64, 66, 96, 128, 192, 256])
def test_fp8_comm_per_token_matches_packing_reference(head_size: int) -> None:
  _require_runtime()
  torch.manual_seed(head_size)
  x = torch.randn((3, 5, head_size), device="cuda", dtype=torch.bfloat16)
  x[0, 0].zero_()
  x[0, 1].uniform_(-0.01, 0.01)
  x[0, 2].fill_(448.0)

  quantized = fp8_comm_per_token_quant(x)
  expected = _reference_per_token_quant(x)
  torch.testing.assert_close(quantized.view(torch.uint8),
                             expected.view(torch.uint8),
                             rtol=0,
                             atol=0)

  dequantized = fp8_comm_per_token_dequant(quantized)
  assert dequantized.shape == x.shape
  assert dequantized.dtype == torch.bfloat16
  assert dequantized.is_contiguous()
  torch.testing.assert_close(dequantized.float(), x.float(), rtol=0.08, atol=0.08)


@pytest.mark.parametrize("head_size", [64, 68, 96, 128, 192])
def test_fp8_comm_qkv_matches_packing_reference_and_strides(head_size: int) -> None:
  _require_runtime()
  torch.manual_seed(head_size)
  x_storage = torch.randn((4, 2, 3, 5, head_size), device="cuda", dtype=torch.bfloat16)
  x = x_storage.transpose(0, 1)
  assert not x.is_contiguous()
  x[0, 0, 0, 0].zero_()
  x[0, 0, 0, 1].uniform_(-0.01, 0.01)

  quantized = fp8_comm_qkv_permute_quant(x)
  expected = _reference_qkv_quant(x)
  torch.testing.assert_close(
    quantized[..., :head_size].view(torch.uint8),
    expected[..., :head_size].view(torch.uint8),
    rtol=0,
    atol=0,
  )
  actual_scale = quantized[..., head_size:].contiguous().view(torch.float32)
  expected_scale = expected[..., head_size:].contiguous().view(torch.float32)
  torch.testing.assert_close(actual_scale, expected_scale, rtol=1e-7, atol=1e-9)

  for partition in range(x.shape[2]):
    dequantized = fp8_comm_qkv_permute_dequant(quantized[partition])
    reference = x[:, :, partition].float()
    assert dequantized.is_contiguous()
    torch.testing.assert_close(dequantized.float(), reference, rtol=0.08, atol=0.08)


def test_fp8_comm_qkv_dequant_supports_strided_sequence() -> None:
  _require_runtime()
  torch.manual_seed(0)
  x = torch.randn((2, 6, 3, 4, 64), device="cuda", dtype=torch.bfloat16)
  quantized = fp8_comm_qkv_permute_quant(x)
  dequantized = fp8_comm_qkv_permute_dequant(quantized[1, ::2])
  torch.testing.assert_close(dequantized.float(), x[:, ::2, 1].float(), rtol=0.08, atol=0.08)


@pytest.mark.parametrize("head_size", [62, 63, 66])
def test_fp8_comm_rejects_undefined_packed_layout(head_size: int) -> None:
  _require_runtime()
  if head_size % 2:
    per_token = torch.randn((2, head_size), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="positive and even"):
      fp8_comm_per_token_quant(per_token)
  if head_size % 4:
    qkv = torch.randn((1, 2, 3, 4, head_size), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="multiple of 4"):
      fp8_comm_qkv_permute_quant(qkv)


def test_fp8_comm_empty_input() -> None:
  _require_runtime()
  per_token = torch.empty((0, 128), device="cuda", dtype=torch.bfloat16)
  quantized = fp8_comm_per_token_quant(per_token)
  assert quantized.shape == (0, 130)
  assert fp8_comm_per_token_dequant(quantized).shape == per_token.shape

  qkv = torch.empty((0, 2, 3, 4, 64), device="cuda", dtype=torch.bfloat16)
  quantized_qkv = fp8_comm_qkv_permute_quant(qkv)
  assert quantized_qkv.shape == (3, 2, 0, 4, 68)
  assert fp8_comm_qkv_permute_dequant(quantized_qkv[0]).shape == (0, 2, 4, 64)


def test_fp8_comm_compile_smoke() -> None:
  _require_runtime()
  x = torch.randn((2, 3, 128), device="cuda", dtype=torch.bfloat16)
  quantized = torch.compile(fp8_comm_per_token_quant, fullgraph=True)(x)
  dequantized = torch.compile(fp8_comm_per_token_dequant, fullgraph=True)(quantized)
  qkv = torch.randn((2, 4, 3, 5, 64), device="cuda", dtype=torch.bfloat16)
  quantized_qkv = torch.compile(fp8_comm_qkv_permute_quant, fullgraph=True)(qkv)
  dequantized_qkv = torch.compile(fp8_comm_qkv_permute_dequant, fullgraph=True)(quantized_qkv[0])
  assert dequantized.shape == x.shape
  assert dequantized_qkv.shape == qkv[:, :, 0].shape
  assert dequantized_qkv.is_contiguous()
