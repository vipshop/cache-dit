import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode

from cache_dit.kernels import cuda_ext_is_available
from cache_dit.kernels.ulysses import ulysses_fp8_o_permute_dequant
from cache_dit.kernels.ulysses import ulysses_fp8_o_permute_quant
from cache_dit.kernels.ulysses import ulysses_fp8_per_token_dequant
from cache_dit.kernels.ulysses import ulysses_fp8_per_token_quant
from cache_dit.kernels.ulysses import ulysses_fp8_qkv_permute_dequant
from cache_dit.kernels.ulysses import ulysses_fp8_qkv_permute_quant


def _require_runtime() -> None:
  if not torch.cuda.is_available() or not cuda_ext_is_available():
    pytest.skip("CUDA and the cache-dit CUDA extension are required.")


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


def test_ulysses_fp8_fake_registration_shapes() -> None:
  with FakeTensorMode():
    x = torch.empty((2, 3, 128), device="cuda", dtype=torch.bfloat16)
    quantized = ulysses_fp8_per_token_quant(x)
    dequantized = ulysses_fp8_per_token_dequant(quantized)
    qkv = torch.empty((2, 4, 3, 5, 64), device="cuda", dtype=torch.bfloat16)
    quantized_qkv = ulysses_fp8_qkv_permute_quant(qkv)
    dequantized_qkv = ulysses_fp8_qkv_permute_dequant(quantized_qkv[0])
    o = torch.empty((2, 8, 3, 64), device="cuda", dtype=torch.bfloat16)
    quantized_o = ulysses_fp8_o_permute_quant(o, partitions=4, packed_head_size=68)
    dequantized_o = ulysses_fp8_o_permute_dequant(quantized_o.flatten(0, 1), head_size=64)

  assert isinstance(quantized, FakeTensor)
  assert quantized.shape == (2, 3, 130)
  assert quantized.dtype == torch.float8_e4m3fn
  assert dequantized.shape == x.shape
  assert dequantized.dtype == torch.bfloat16
  assert quantized_qkv.shape == (3, 4, 2, 5, 68)
  assert dequantized_qkv.shape == (2, 4, 5, 64)
  assert dequantized_qkv.is_contiguous()
  assert quantized_o.shape == (4, 3, 2, 2, 68)
  assert quantized_o.dtype == torch.float8_e4m3fn
  assert dequantized_o.shape == (2, 2, 12, 64)
  assert dequantized_o.dtype == torch.bfloat16
  assert dequantized_o.is_contiguous()


def test_ulysses_fp8_fake_packed_variants() -> None:
  with FakeTensorMode():
    x = torch.empty((2, 3, 64), device="cuda", dtype=torch.bfloat16)
    quantized = ulysses_fp8_per_token_quant(x, packed_head_size=80)
    dequantized = ulysses_fp8_per_token_dequant(quantized, head_size=64)
    qkv = torch.empty((2, 4, 3, 5, 64), device="cuda", dtype=torch.bfloat16)
    quantized_qkv = ulysses_fp8_qkv_permute_quant(qkv, packed_head_size=72)
    dequantized_qkv = ulysses_fp8_qkv_permute_dequant(quantized_qkv[0], head_size=64)

  assert quantized.shape == (2, 3, 80)
  assert dequantized.shape == x.shape
  assert quantized_qkv.shape == (3, 4, 2, 5, 72)
  assert dequantized_qkv.shape == (2, 4, 5, 64)


@pytest.mark.parametrize("head_size", [64, 66, 96, 128, 192, 256])
def test_ulysses_fp8_per_token_matches_packing_reference(head_size: int) -> None:
  _require_runtime()
  torch.manual_seed(head_size)
  x = torch.randn((3, 5, head_size), device="cuda", dtype=torch.bfloat16)
  x[0, 0].zero_()
  x[0, 1].uniform_(-0.01, 0.01)
  x[0, 2].fill_(448.0)

  quantized = ulysses_fp8_per_token_quant(x)
  expected = _reference_per_token_quant(x)
  torch.testing.assert_close(quantized.view(torch.uint8),
                             expected.view(torch.uint8),
                             rtol=0,
                             atol=0)

  dequantized = ulysses_fp8_per_token_dequant(quantized)
  assert dequantized.shape == x.shape
  assert dequantized.dtype == torch.bfloat16
  assert dequantized.is_contiguous()
  torch.testing.assert_close(dequantized.float(), x.float(), rtol=0.08, atol=0.08)


@pytest.mark.parametrize("head_size", [64, 68, 96, 128, 192])
def test_ulysses_fp8_qkv_matches_packing_reference_and_strides(head_size: int) -> None:
  _require_runtime()
  torch.manual_seed(head_size)
  x_storage = torch.randn((4, 2, 3, 5, head_size), device="cuda", dtype=torch.bfloat16)
  x = x_storage.transpose(0, 1)
  assert not x.is_contiguous()
  x[0, 0, 0, 0].zero_()
  x[0, 0, 0, 1].uniform_(-0.01, 0.01)

  quantized = ulysses_fp8_qkv_permute_quant(x)
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
    dequantized = ulysses_fp8_qkv_permute_dequant(quantized[partition])
    reference = x[:, :, partition].float()
    assert dequantized.is_contiguous()
    torch.testing.assert_close(dequantized.float(), reference, rtol=0.08, atol=0.08)


def test_ulysses_fp8_per_token_packed_matches_legacy_and_ignores_padding() -> None:
  _require_runtime()
  torch.manual_seed(0)
  x = torch.randn((3, 64), device="cuda", dtype=torch.bfloat16)
  expected = ulysses_fp8_per_token_quant(x)
  quantized = ulysses_fp8_per_token_quant(x, packed_head_size=80)

  torch.testing.assert_close(quantized[:, :66].view(torch.uint8),
                             expected.view(torch.uint8),
                             rtol=0,
                             atol=0)
  assert torch.count_nonzero(quantized[:, 66:].view(torch.uint8)) == 0
  quantized[:, 66:].fill_(255)
  dequantized = ulysses_fp8_per_token_dequant(quantized, head_size=64)
  expected_dequantized = ulysses_fp8_per_token_dequant(expected)
  torch.testing.assert_close(dequantized.view(torch.uint16),
                             expected_dequantized.view(torch.uint16),
                             rtol=0,
                             atol=0)


def test_ulysses_fp8_qkv_packed_matches_legacy_and_ignores_padding() -> None:
  _require_runtime()
  torch.manual_seed(0)
  x = torch.randn((2, 4, 3, 5, 64), device="cuda", dtype=torch.bfloat16)
  expected = ulysses_fp8_qkv_permute_quant(x)
  quantized = ulysses_fp8_qkv_permute_quant(x, packed_head_size=72)

  torch.testing.assert_close(quantized[..., :68].view(torch.uint8),
                             expected.view(torch.uint8),
                             rtol=0,
                             atol=0)
  assert torch.count_nonzero(quantized[..., 68:].view(torch.uint8)) == 0
  quantized[..., 68:].fill_(255)
  for partition in range(x.shape[2]):
    dequantized = ulysses_fp8_qkv_permute_dequant(quantized[partition], head_size=64)
    expected_dequantized = ulysses_fp8_qkv_permute_dequant(expected[partition])
    torch.testing.assert_close(dequantized.view(torch.uint16),
                               expected_dequantized.view(torch.uint16),
                               rtol=0,
                               atol=0)


@pytest.mark.parametrize("head_size,packed_head_size", [(64, 68), (66, 68), (128, 132), (256, 260)])
def test_ulysses_fp8_o_matches_legacy_layout_and_ignores_padding(head_size: int,
                                                                 packed_head_size: int) -> None:
  _require_runtime()
  torch.manual_seed(head_size)
  x_storage = torch.randn((8, 2, 3, head_size), device="cuda", dtype=torch.bfloat16)
  x = x_storage.transpose(0, 1)
  assert x.shape == (2, 8, 3, head_size)
  assert not x.is_contiguous()
  x[0, 0, 0].zero_()
  x[0, 1, 0].uniform_(-0.01, 0.01)
  partitions = 4

  quantized = ulysses_fp8_o_permute_quant(x, partitions, packed_head_size)
  permuted = x.reshape(2, partitions, 2, 3, head_size).permute(1, 3, 0, 2, 4).contiguous()
  expected = ulysses_fp8_per_token_quant(permuted)
  expected_padded = torch.zeros((*expected.shape[:-1], packed_head_size),
                                dtype=expected.dtype,
                                device=expected.device)
  expected_padded[..., :head_size + 2] = expected
  torch.testing.assert_close(quantized.view(torch.uint8),
                             expected_padded.view(torch.uint8),
                             rtol=0,
                             atol=0)

  expected_dequantized = ulysses_fp8_per_token_dequant(expected)
  expected_dequantized = expected_dequantized.flatten(0, 1).permute(1, 2, 0, 3).contiguous()
  quantized[..., head_size + 2:].fill_(255)
  dequantized = ulysses_fp8_o_permute_dequant(quantized.flatten(0, 1), head_size)
  assert dequantized.is_contiguous()
  torch.testing.assert_close(dequantized.view(torch.uint16),
                             expected_dequantized.view(torch.uint16),
                             rtol=0,
                             atol=0)


def test_ulysses_fp8_o_rejects_misaligned_or_invalid_layout() -> None:
  _require_runtime()
  x = torch.randn((2, 8, 3, 64), device="cuda", dtype=torch.bfloat16)
  with pytest.raises(RuntimeError, match="multiple of 16 bytes"):
    ulysses_fp8_o_permute_quant(x, partitions=4, packed_head_size=66)
  with pytest.raises(RuntimeError, match="divisible by partitions"):
    ulysses_fp8_o_permute_quant(x, partitions=3, packed_head_size=68)


def test_ulysses_fp8_qkv_dequant_supports_strided_sequence() -> None:
  _require_runtime()
  torch.manual_seed(0)
  x = torch.randn((2, 6, 3, 4, 64), device="cuda", dtype=torch.bfloat16)
  quantized = ulysses_fp8_qkv_permute_quant(x)
  dequantized = ulysses_fp8_qkv_permute_dequant(quantized[1, ::2])
  torch.testing.assert_close(dequantized.float(), x[:, ::2, 1].float(), rtol=0.08, atol=0.08)


@pytest.mark.parametrize("head_size", [62, 63, 66])
def test_ulysses_fp8_rejects_undefined_packed_layout(head_size: int) -> None:
  _require_runtime()
  if head_size % 2:
    per_token = torch.randn((2, head_size), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="positive and even"):
      ulysses_fp8_per_token_quant(per_token)
  if head_size % 4:
    qkv = torch.randn((1, 2, 3, 4, head_size), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="multiple of 4"):
      ulysses_fp8_qkv_permute_quant(qkv)


def test_ulysses_fp8_empty_input() -> None:
  _require_runtime()
  per_token = torch.empty((0, 128), device="cuda", dtype=torch.bfloat16)
  quantized = ulysses_fp8_per_token_quant(per_token)
  assert quantized.shape == (0, 130)
  assert ulysses_fp8_per_token_dequant(quantized).shape == per_token.shape

  qkv = torch.empty((0, 2, 3, 4, 64), device="cuda", dtype=torch.bfloat16)
  quantized_qkv = ulysses_fp8_qkv_permute_quant(qkv)
  assert quantized_qkv.shape == (3, 2, 0, 4, 68)
  assert ulysses_fp8_qkv_permute_dequant(quantized_qkv[0]).shape == (0, 2, 4, 64)

  o = torch.empty((2, 0, 3, 64), device="cuda", dtype=torch.bfloat16)
  quantized_o = ulysses_fp8_o_permute_quant(o, partitions=2, packed_head_size=66)
  assert quantized_o.shape == (2, 3, 2, 0, 66)
  assert ulysses_fp8_o_permute_dequant(quantized_o.flatten(0, 1), 64).shape == (2, 0, 6, 64)


def test_ulysses_fp8_compile_smoke() -> None:
  _require_runtime()
  x = torch.randn((2, 3, 128), device="cuda", dtype=torch.bfloat16)
  quantized = torch.compile(ulysses_fp8_per_token_quant, fullgraph=True)(x)
  dequantized = torch.compile(ulysses_fp8_per_token_dequant, fullgraph=True)(quantized)
  qkv = torch.randn((2, 4, 3, 5, 64), device="cuda", dtype=torch.bfloat16)
  quantized_qkv = torch.compile(ulysses_fp8_qkv_permute_quant, fullgraph=True)(qkv)
  dequantized_qkv = torch.compile(ulysses_fp8_qkv_permute_dequant, fullgraph=True)(quantized_qkv[0])
  packed = torch.compile(ulysses_fp8_per_token_quant, fullgraph=True)(x, 132)
  unpacked = torch.compile(ulysses_fp8_per_token_dequant, fullgraph=True)(packed, 128)
  packed_qkv = torch.compile(ulysses_fp8_qkv_permute_quant, fullgraph=True)(qkv, 72)
  unpacked_qkv = torch.compile(ulysses_fp8_qkv_permute_dequant, fullgraph=True)(packed_qkv[0], 64)
  o = torch.randn((2, 8, 3, 64), device="cuda", dtype=torch.bfloat16)
  quantized_o = torch.compile(ulysses_fp8_o_permute_quant, fullgraph=True)(o, 4, 68)
  dequantized_o = torch.compile(ulysses_fp8_o_permute_dequant,
                                fullgraph=True)(quantized_o.flatten(0, 1), 64)
  assert dequantized.shape == x.shape
  assert dequantized_qkv.shape == qkv[:, :, 0].shape
  assert dequantized_qkv.is_contiguous()
  assert dequantized_o.shape == (2, 2, 12, 64)
  assert dequantized_o.is_contiguous()
  assert packed.shape == (2, 3, 132)
  assert unpacked.shape == x.shape
  assert packed_qkv.shape == (3, 4, 2, 5, 72)
  assert unpacked_qkv.shape == qkv[:, :, 0].shape
