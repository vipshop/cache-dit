import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode

from cache_dit.kernels import cuda_ext_is_available
from cache_dit.kernels.ulysses import ulysses_fN_o_permute_dequant
from cache_dit.kernels.ulysses import ulysses_fN_o_permute_quant
from cache_dit.kernels.ulysses import ulysses_fN_per_token_dequant
from cache_dit.kernels.ulysses import ulysses_fN_per_token_quant
from cache_dit.kernels.ulysses import ulysses_fN_qkv_permute_dequant
from cache_dit.kernels.ulysses import ulysses_fN_qkv_permute_quant
from cache_dit.kernels.ulysses import ulysses_fp4_o_permute_dequant
from cache_dit.kernels.ulysses import ulysses_fp4_o_permute_quant
from cache_dit.kernels.ulysses import ulysses_fp4_per_token_dequant
from cache_dit.kernels.ulysses import ulysses_fp4_per_token_quant
from cache_dit.kernels.ulysses import ulysses_fp4_qkv_permute_dequant
from cache_dit.kernels.ulysses import ulysses_fp4_qkv_permute_quant
from cache_dit.quantization.mxfp4.tensor import _quant_codes
from cache_dit.quantization.svdquant.wNa8 import pack_e2m1
from cache_dit.quantization.svdquant.wNa8 import unpack_e2m1

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
                -6.0)


def _require_runtime() -> None:
  if not torch.cuda.is_available() or not cuda_ext_is_available():
    pytest.skip("CUDA and the cache-dit CUDA extension are required.")
  if torch.cuda.get_device_capability(0)[0] < 10:
    pytest.skip("FP4 E2M1 communication requires native sm100+ conversion support.")


def _pack_fp4_rows(rows: torch.Tensor, eps: float, scale_dtype: torch.dtype) -> torch.Tensor:
  rows_fp32 = rows.float()
  head_size = rows.shape[-1]
  payloads = []
  scales = []
  for start in range(0, head_size, 16):
    group = rows_fp32[:, start:min(start + 16, head_size)]
    scale = (group.abs().amax(dim=-1, keepdim=True).double() / 6.0).float().clamp_min(eps)
    payloads.append(pack_e2m1(_quant_codes(group * scale.reciprocal())))
    scales.append(scale)
  payload = torch.cat(payloads, dim=1)
  scale = torch.cat(scales, dim=1)
  scale_bytes = scale.to(scale_dtype).contiguous().view(torch.uint8).reshape(rows.shape[0], -1)
  return torch.cat((payload, scale_bytes), dim=1).contiguous()


def _dequant_fp4_rows(packed: torch.Tensor, head_size: int,
                      scale_dtype: torch.dtype) -> torch.Tensor:
  payload_width = (head_size + 1) // 2
  codes = unpack_e2m1(packed[:, :payload_width])[:, :head_size].long()
  values = torch.tensor(_E2M1_VALUES, device=packed.device)[codes]
  scales = packed[:, payload_width:].contiguous().view(scale_dtype)
  outputs = []
  for group, start in enumerate(range(0, head_size, 16)):
    end = min(start + 16, head_size)
    outputs.append((values[:, start:end] * scales[:, group:group + 1].float()).to(torch.bfloat16))
  return torch.cat(outputs, dim=1)


def _assert_round_trip(quantized, expected, dequantized, head_size, scale_dtype):
  torch.testing.assert_close(quantized.reshape(-1, quantized.shape[-1]), expected, rtol=0, atol=0)
  expected_dequantized = _dequant_fp4_rows(expected, head_size, scale_dtype)
  torch.testing.assert_close(dequantized.reshape(-1, head_size).view(torch.uint16),
                             expected_dequantized.view(torch.uint16),
                             rtol=0,
                             atol=0)


@pytest.mark.parametrize("head_size", [24, 40, 64, 66, 68, 72, 128, 264, 384])
def test_ulysses_fp4_per_token_matches_reference(head_size: int) -> None:
  _require_runtime()
  torch.manual_seed(head_size)
  x = torch.randn((2, 3, head_size), device="cuda", dtype=torch.bfloat16)
  x[0, 0].zero_()
  x[0, 1, :min(head_size, 16)].uniform_(-0.01, 0.01)
  expected = _pack_fp4_rows(x.reshape(-1, head_size), 1e-4, torch.bfloat16)
  quantized = ulysses_fp4_per_token_quant(x)
  dequantized = ulysses_fp4_per_token_dequant(quantized, head_size, 16)
  _assert_round_trip(quantized, expected, dequantized, head_size, torch.bfloat16)

  padded = ulysses_fp4_per_token_quant(x, expected.shape[-1] + 5)
  assert torch.count_nonzero(padded[..., expected.shape[-1]:]) == 0
  padded[..., expected.shape[-1]:].fill_(255)
  padded_dequantized = ulysses_fp4_per_token_dequant(padded, head_size, 16)
  torch.testing.assert_close(padded_dequantized.view(torch.uint16),
                             dequantized.view(torch.uint16),
                             rtol=0,
                             atol=0)


def test_ulysses_fp4_conversion_ties_match_reference() -> None:
  _require_runtime()
  values = torch.tensor(
    [6, .25, .75, 1.25, 1.75, 2.5, 3.5, 5, -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5, 0],
    device="cuda",
    dtype=torch.bfloat16).repeat(2, 1)
  expected = _pack_fp4_rows(values, 1e-4, torch.bfloat16)
  actual = ulysses_fp4_per_token_quant(values)
  torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("head_size", [24, 40, 64, 68, 72, 128, 264])
def test_ulysses_fp4_qkv_matches_reference(head_size: int) -> None:
  _require_runtime()
  torch.manual_seed(head_size)
  x = torch.randn((4, 2, 3, 5, head_size), device="cuda", dtype=torch.bfloat16).transpose(0, 1)
  x[0, 0, 0, 0].zero_()
  rows = x.permute(2, 1, 0, 3, 4).reshape(-1, head_size)
  expected = _pack_fp4_rows(rows, 1e-6, torch.float32).reshape(3, 4, 2, 5, -1)
  quantized = ulysses_fp4_qkv_permute_quant(x)
  torch.testing.assert_close(quantized, expected, rtol=0, atol=0)
  padded = ulysses_fp4_qkv_permute_quant(x, expected.shape[-1] + 5)
  torch.testing.assert_close(padded[..., :expected.shape[-1]], expected, rtol=0, atol=0)
  assert torch.count_nonzero(padded[..., expected.shape[-1]:]) == 0
  padded[..., expected.shape[-1]:].fill_(255)
  for partition in range(3):
    expected_rows = expected[partition].permute(1, 0, 2, 3).reshape(-1, expected.shape[-1])
    expected_dequantized = _dequant_fp4_rows(expected_rows, head_size, torch.float32)
    dequantized = ulysses_fp4_qkv_permute_dequant(padded[partition], head_size, 16)
    torch.testing.assert_close(dequantized.reshape(-1, head_size).view(torch.uint16),
                               expected_dequantized.view(torch.uint16),
                               rtol=0,
                               atol=0)


@pytest.mark.parametrize("head_size", [24, 40, 64, 66, 72, 128, 264])
def test_ulysses_fp4_o_matches_reference(head_size: int) -> None:
  _require_runtime()
  torch.manual_seed(head_size)
  x = torch.randn((2, 4, 3, head_size), device="cuda", dtype=torch.bfloat16)
  x[0, 0, 0].zero_()
  rows = x.reshape(2, 2, 2, 3, head_size).permute(1, 3, 0, 2, 4).reshape(-1, head_size)
  expected = _pack_fp4_rows(rows, 1e-4, torch.bfloat16)
  min_width = expected.shape[-1]
  packed_head_size = min_width
  while (3 * 2 * 2 * packed_head_size) % 16 != 0:
    packed_head_size += 1
  expected = expected.reshape(2, 3, 2, 2, min_width)

  quantized = ulysses_fp4_o_permute_quant(x, 2, packed_head_size)
  torch.testing.assert_close(quantized[..., :min_width], expected, rtol=0, atol=0)
  assert torch.count_nonzero(quantized[..., min_width:]) == 0
  quantized[..., min_width:].fill_(255)
  expected_dequantized = _dequant_fp4_rows(expected.reshape(-1, min_width), head_size,
                                           torch.bfloat16)
  expected_dequantized = expected_dequantized.reshape(2, 3, 2, 2, head_size)
  expected_dequantized = expected_dequantized.flatten(0, 1).permute(1, 2, 0, 3).contiguous()
  dequantized = ulysses_fp4_o_permute_dequant(quantized.flatten(0, 1), head_size, 16)
  torch.testing.assert_close(dequantized.view(torch.uint16),
                             expected_dequantized.view(torch.uint16),
                             rtol=0,
                             atol=0)


def test_ulysses_fp4_qkv_four_rows_per_warp_matches_reference() -> None:
  _require_runtime()
  torch.manual_seed(272)
  x = torch.randn((64, 2, 3, 8, 128), device="cuda", dtype=torch.bfloat16).transpose(0, 1)
  rows = x.permute(2, 1, 0, 3, 4).reshape(-1, 128)
  expected = _pack_fp4_rows(rows, 1e-6, torch.float32).reshape(3, 64, 2, 8, -1)
  quantized = ulysses_fp4_qkv_permute_quant(x)
  torch.testing.assert_close(quantized, expected, rtol=0, atol=0)


def test_ulysses_fp4_o_four_rows_per_warp_matches_reference() -> None:
  _require_runtime()
  torch.manual_seed(512)
  x = torch.randn((1, 256, 8, 128), device="cuda", dtype=torch.bfloat16)
  rows = x.reshape(1, 4, 64, 8, 128).permute(1, 3, 0, 2, 4).reshape(-1, 128)
  expected = _pack_fp4_rows(rows, 1e-4, torch.bfloat16)
  quantized = ulysses_fp4_o_permute_quant(x, 4, expected.shape[-1])
  torch.testing.assert_close(quantized.reshape(-1, expected.shape[-1]), expected, rtol=0, atol=0)
  expected_dequantized = _dequant_fp4_rows(expected, 128, torch.bfloat16)
  expected_dequantized = expected_dequantized.reshape(4, 8, 1, 64, 128)
  expected_dequantized = expected_dequantized.flatten(0, 1).permute(1, 2, 0, 3).contiguous()
  dequantized = ulysses_fp4_o_permute_dequant(quantized.flatten(0, 1), 128, 16)
  torch.testing.assert_close(dequantized.view(torch.uint16),
                             expected_dequantized.view(torch.uint16),
                             rtol=0,
                             atol=0)


def test_ulysses_fN_fp4_fake_and_compile() -> None:
  with FakeTensorMode():
    x = torch.empty((2, 3, 128), device="cuda", dtype=torch.bfloat16)
    quantized = ulysses_fN_per_token_quant(x, bits=4)
    dequantized = ulysses_fN_per_token_dequant(quantized, 128, bits=4)
    qkv = torch.empty((2, 4, 3, 5, 128), device="cuda", dtype=torch.bfloat16)
    quantized_qkv = ulysses_fN_qkv_permute_quant(qkv, bits=4)
    dequantized_qkv = ulysses_fN_qkv_permute_dequant(quantized_qkv[0], 128, bits=4)
    o = torch.empty((2, 8, 3, 128), device="cuda", dtype=torch.bfloat16)
    quantized_o = ulysses_fN_o_permute_quant(o, 4, 96, bits=4)
    dequantized_o = ulysses_fN_o_permute_dequant(quantized_o.flatten(0, 1), 128, bits=4)
  assert isinstance(quantized, FakeTensor)
  assert quantized.dtype == torch.uint8
  assert dequantized.shape == x.shape
  assert quantized_qkv.dtype == torch.uint8
  assert dequantized_qkv.shape == (2, 4, 5, 128)
  assert quantized_o.dtype == torch.uint8
  assert dequantized_o.shape == (2, 2, 12, 128)

  _require_runtime()
  x = torch.randn((2, 3, 128), device="cuda", dtype=torch.bfloat16)
  quantized = torch.compile(ulysses_fN_per_token_quant, fullgraph=True)(x, bits=4)
  dequantized = torch.compile(ulysses_fN_per_token_dequant, fullgraph=True)(quantized, 128, bits=4)
  qkv = torch.randn((2, 4, 3, 5, 128), device="cuda", dtype=torch.bfloat16)
  quantized_qkv = torch.compile(ulysses_fN_qkv_permute_quant, fullgraph=True)(qkv, bits=4)
  dequantized_qkv = torch.compile(ulysses_fN_qkv_permute_dequant, fullgraph=True)(quantized_qkv[0],
                                                                                  128,
                                                                                  bits=4)
  o = torch.randn((2, 8, 3, 128), device="cuda", dtype=torch.bfloat16)
  quantized_o = torch.compile(ulysses_fN_o_permute_quant, fullgraph=True)(o, 4, 96, bits=4)
  dequantized_o = torch.compile(ulysses_fN_o_permute_dequant,
                                fullgraph=True)(quantized_o.flatten(0, 1), 128, bits=4)
  assert dequantized.shape == x.shape
  assert dequantized_qkv.shape == (2, 4, 5, 128)
  assert dequantized_o.shape == (2, 2, 12, 128)


def test_ulysses_fp4_dequant_accepts_misaligned_packed_view() -> None:
  _require_runtime()
  x = torch.randn((2, 128), device="cuda", dtype=torch.bfloat16)
  quantized = ulysses_fp4_per_token_quant(x)
  storage = torch.empty(quantized.numel() + 1, dtype=torch.uint8, device=x.device)
  misaligned = storage[1:].view_as(quantized)
  misaligned.copy_(quantized)
  actual = ulysses_fp4_per_token_dequant(misaligned, 128, 16)
  expected = ulysses_fp4_per_token_dequant(quantized, 128, 16)
  torch.testing.assert_close(actual.view(torch.uint16), expected.view(torch.uint16), rtol=0, atol=0)


def test_ulysses_fp4_defaults_empty_and_invalid() -> None:
  _require_runtime()
  x = torch.randn((2, 128), device="cuda", dtype=torch.bfloat16)
  default = ulysses_fp4_per_token_quant(x)
  explicit = ulysses_fp4_per_token_quant(x, group_size=16)
  torch.testing.assert_close(default, explicit, rtol=0, atol=0)
  assert default.shape == (2, 80)
  empty = torch.empty((0, 128), device="cuda", dtype=torch.bfloat16)
  assert ulysses_fp4_per_token_quant(empty).shape == (0, 80)
  assert ulysses_fp4_per_token_dequant(default[:0], 128, 16).shape == (0, 128)
  for group_size in (0, 8, 32):
    with pytest.raises(ValueError, match="group_size"):
      ulysses_fp4_per_token_quant(x, group_size=group_size)
  with pytest.raises(ValueError, match="bits must be 4, 6, or 8"):
    ulysses_fN_per_token_quant(x, bits=5)
  with pytest.raises(ValueError, match="head_size is required"):
    ulysses_fN_per_token_dequant(default, bits=4)
