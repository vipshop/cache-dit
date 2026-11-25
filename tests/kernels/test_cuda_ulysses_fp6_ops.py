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
from cache_dit.kernels.ulysses import ulysses_fp6_o_permute_dequant
from cache_dit.kernels.ulysses import ulysses_fp6_o_permute_quant
from cache_dit.kernels.ulysses import ulysses_fp6_per_token_dequant
from cache_dit.kernels.ulysses import ulysses_fp6_per_token_quant
from cache_dit.kernels.ulysses import ulysses_fp6_qkv_permute_dequant
from cache_dit.kernels.ulysses import ulysses_fp6_qkv_permute_quant
from cache_dit.quantization.svdquant.w6a6 import _e2m3_magnitudes
from cache_dit.quantization.svdquant.w6a6 import pack_e2m3
from cache_dit.quantization.svdquant.w6a6 import quantize_e2m3
from cache_dit.quantization.svdquant.w6a6 import unpack_e2m3


def _require_runtime() -> None:
  if not torch.cuda.is_available() or not cuda_ext_is_available():
    pytest.skip("CUDA and the cache-dit CUDA extension are required.")
  if torch.cuda.get_device_capability(0)[0] < 10:
    pytest.skip("FP6 E2M3 communication requires native sm100+ conversion support.")


def _payload_bytes(head_size: int) -> int:
  bitstream_bytes = ((head_size + 3) // 4) * 3
  return (bitstream_bytes + 3) // 4 * 4


def _pack_fp6_rows(rows: torch.Tensor, eps: float, scale_dtype: torch.dtype,
                   group_size: int) -> torch.Tensor:
  rows_fp32 = rows.float()
  head_size = rows.shape[-1]
  bounds = ([(0, head_size)] if group_size == 0 else [(start, min(start + group_size, head_size))
                                                      for start in range(0, head_size, group_size)])
  payload = torch.zeros((rows.shape[0], _payload_bytes(head_size)),
                        dtype=torch.uint8,
                        device=rows.device)
  bitstream_width = ((head_size + 3) // 4) * 3
  padded_codes = torch.zeros((rows.shape[0], ((head_size + 3) // 4) * 4),
                             dtype=torch.uint8,
                             device=rows.device)
  scales = []
  for start, end in bounds:
    group = rows_fp32[:, start:end]
    scale = (group.abs().amax(dim=-1, keepdim=True).double() / 7.5).float().clamp_min(eps)
    padded_codes[:, start:end] = quantize_e2m3(group * scale.reciprocal())
    scales.append(scale)
  payload[:, :bitstream_width] = pack_e2m3(padded_codes)
  scale = torch.cat(scales, dim=1)
  scale_bytes = scale.to(scale_dtype).contiguous().view(torch.uint8).reshape(rows.shape[0], -1)
  packed = torch.cat((payload, scale_bytes), dim=1).contiguous()
  packed_width = (packed.shape[-1] + 3) // 4 * 4
  return torch.cat((packed, packed.new_zeros((rows.shape[0], packed_width - packed.shape[-1]))),
                   dim=1)


def _dequant_fp6_rows(packed: torch.Tensor, head_size: int, scale_dtype: torch.dtype,
                      group_size: int) -> torch.Tensor:
  groups = 1 if group_size == 0 else (head_size + group_size - 1) // group_size
  payload_width = _payload_bytes(head_size)
  bitstream_width = ((head_size + 3) // 4) * 3
  codes = unpack_e2m3(packed[:, :bitstream_width])[:, :head_size].long()
  table = _e2m3_magnitudes(packed.device)
  values = table[codes & 0x1f] * torch.where(codes >= 0x20, -1.0, 1.0)
  scales = packed[:, payload_width:].contiguous().view(scale_dtype)
  outputs = []
  for group in range(groups):
    start = group * (head_size if group_size == 0 else group_size)
    end = head_size if group_size == 0 else min(start + group_size, head_size)
    outputs.append((values[:, start:end] * scales[:, group:group + 1].float()).to(torch.bfloat16))
  return torch.cat(outputs, dim=1)


def _assert_round_trip_matches_reference(quantized, expected, dequantized, head_size, scale_dtype,
                                         group_size):
  torch.testing.assert_close(quantized.reshape(-1, quantized.shape[-1]), expected, rtol=0, atol=0)
  expected_dequantized = _dequant_fp6_rows(expected, head_size, scale_dtype, group_size)
  torch.testing.assert_close(dequantized.reshape(-1, head_size).view(torch.uint16),
                             expected_dequantized.view(torch.uint16),
                             rtol=0,
                             atol=0)


@pytest.mark.parametrize("group_size", [0, 16, 32])
@pytest.mark.parametrize("head_size", [24, 40, 64, 66, 68, 72, 128, 264, 384])
def test_ulysses_fp6_per_token_matches_reference(head_size: int, group_size: int) -> None:
  _require_runtime()
  torch.manual_seed(head_size + group_size)
  x = torch.randn((2, 3, head_size), device="cuda", dtype=torch.bfloat16)
  x[0, 0].zero_()
  x[0, 1, :min(head_size, 16)].uniform_(-0.01, 0.01)
  rows = x.reshape(-1, head_size)
  expected = _pack_fp6_rows(rows, 1e-4, torch.bfloat16, group_size)
  quantized = ulysses_fp6_per_token_quant(x, group_size=group_size)
  dequantized = ulysses_fp6_per_token_dequant(quantized, head_size, group_size)
  _assert_round_trip_matches_reference(quantized, expected, dequantized, head_size, torch.bfloat16,
                                       group_size)

  padded = ulysses_fp6_per_token_quant(x, expected.shape[-1] + 4, group_size)
  assert torch.count_nonzero(padded[..., expected.shape[-1]:]) == 0
  torch.testing.assert_close(padded.reshape(-1, padded.shape[-1])[..., :expected.shape[-1]],
                             expected,
                             rtol=0,
                             atol=0)
  padded[..., expected.shape[-1]:].fill_(255)
  padded_dequantized = ulysses_fp6_per_token_dequant(padded, head_size, group_size)
  torch.testing.assert_close(padded_dequantized.view(torch.uint16),
                             dequantized.view(torch.uint16),
                             rtol=0,
                             atol=0)


@pytest.mark.parametrize("group_size", [0, 16, 32])
@pytest.mark.parametrize("head_size", [24, 40, 64, 68, 72, 128, 264])
def test_ulysses_fp6_qkv_matches_reference(head_size: int, group_size: int) -> None:
  _require_runtime()
  torch.manual_seed(head_size + group_size)
  x = torch.randn((4, 2, 3, 5, head_size), device="cuda", dtype=torch.bfloat16).transpose(0, 1)
  x[0, 0, 0, 0].zero_()
  rows = x.permute(2, 1, 0, 3, 4).reshape(-1, head_size)
  expected = _pack_fp6_rows(rows, 1e-6, torch.float32, group_size).reshape(3, 4, 2, 5, -1)
  quantized = ulysses_fp6_qkv_permute_quant(x, group_size=group_size)
  torch.testing.assert_close(quantized, expected, rtol=0, atol=0)
  for partition in range(3):
    expected_rows = expected[partition].permute(1, 0, 2, 3).reshape(-1, expected.shape[-1])
    expected_dequantized = _dequant_fp6_rows(expected_rows, head_size, torch.float32, group_size)
    dequantized = ulysses_fp6_qkv_permute_dequant(quantized[partition], head_size, group_size)
    torch.testing.assert_close(dequantized.reshape(-1, head_size).view(torch.uint16),
                               expected_dequantized.view(torch.uint16),
                               rtol=0,
                               atol=0)


@pytest.mark.parametrize("group_size", [0, 16, 32])
@pytest.mark.parametrize("head_size", [24, 40, 64, 66, 72, 128, 264])
def test_ulysses_fp6_o_matches_reference(head_size: int, group_size: int) -> None:
  _require_runtime()
  torch.manual_seed(head_size + group_size)
  x = torch.randn((2, 4, 3, head_size), device="cuda", dtype=torch.bfloat16)
  x[0, 0, 0].zero_()
  rows = x.reshape(2, 2, 2, 3, head_size).permute(1, 3, 0, 2, 4).reshape(-1, head_size)
  expected = _pack_fp6_rows(rows, 1e-4, torch.bfloat16, group_size)
  min_width = expected.shape[-1]
  packed_head_size = min_width
  while (3 * 2 * 2 * packed_head_size) % 16 != 0:
    packed_head_size += 4
  expected = expected.reshape(2, 3, 2, 2, min_width)

  quantized = ulysses_fp6_o_permute_quant(x, 2, packed_head_size, group_size)
  torch.testing.assert_close(quantized[..., :min_width], expected, rtol=0, atol=0)
  assert torch.count_nonzero(quantized[..., min_width:]) == 0
  quantized[..., min_width:].fill_(255)
  expected_dequantized = _dequant_fp6_rows(expected.reshape(-1, min_width), head_size,
                                           torch.bfloat16, group_size)
  expected_dequantized = expected_dequantized.reshape(2, 3, 2, 2, head_size)
  expected_dequantized = expected_dequantized.flatten(0, 1).permute(1, 2, 0, 3).contiguous()
  dequantized = ulysses_fp6_o_permute_dequant(quantized.flatten(0, 1), head_size, group_size)
  torch.testing.assert_close(dequantized.view(torch.uint16),
                             expected_dequantized.view(torch.uint16),
                             rtol=0,
                             atol=0)


@pytest.mark.parametrize("group_size", [16, 32])
def test_ulysses_fp6_qkv_four_rows_per_warp_matches_reference(group_size: int) -> None:
  _require_runtime()
  torch.manual_seed(256 + group_size)
  x = torch.randn((64, 2, 3, 8, 128), device="cuda", dtype=torch.bfloat16).transpose(0, 1)
  rows = x.permute(2, 1, 0, 3, 4).reshape(-1, 128)
  expected = _pack_fp6_rows(rows, 1e-6, torch.float32, group_size).reshape(3, 64, 2, 8, -1)

  quantized = ulysses_fp6_qkv_permute_quant(x, group_size=group_size)

  torch.testing.assert_close(quantized, expected, rtol=0, atol=0)


@pytest.mark.parametrize("group_size", [16, 32])
def test_ulysses_fp6_o_four_rows_per_warp_matches_reference(group_size: int) -> None:
  _require_runtime()
  torch.manual_seed(512 + group_size)
  x = torch.randn((1, 256, 8, 128), device="cuda", dtype=torch.bfloat16)
  rows = x.reshape(1, 4, 64, 8, 128).permute(1, 3, 0, 2, 4).reshape(-1, 128)
  expected = _pack_fp6_rows(rows, 1e-4, torch.bfloat16, group_size)

  quantized = ulysses_fp6_o_permute_quant(x, 4, expected.shape[-1], group_size)

  torch.testing.assert_close(quantized.reshape(-1, expected.shape[-1]), expected, rtol=0, atol=0)
  expected_dequantized = _dequant_fp6_rows(expected, 128, torch.bfloat16, group_size)
  expected_dequantized = expected_dequantized.reshape(4, 8, 1, 64, 128)
  expected_dequantized = expected_dequantized.flatten(0, 1).permute(1, 2, 0, 3).contiguous()
  dequantized = ulysses_fp6_o_permute_dequant(quantized.flatten(0, 1), 128, group_size)
  torch.testing.assert_close(dequantized.view(torch.uint16),
                             expected_dequantized.view(torch.uint16),
                             rtol=0,
                             atol=0)


@pytest.mark.parametrize("group_size", [0, 16, 32])
def test_ulysses_fN_fake_and_compile(group_size: int) -> None:
  with FakeTensorMode():
    x = torch.empty((2, 3, 128), device="cuda", dtype=torch.bfloat16)
    quantized = ulysses_fN_per_token_quant(x, group_size=group_size, bits=6)
    dequantized = ulysses_fN_per_token_dequant(quantized, 128, group_size=group_size, bits=6)
    qkv = torch.empty((2, 4, 3, 5, 128), device="cuda", dtype=torch.bfloat16)
    quantized_qkv = ulysses_fN_qkv_permute_quant(qkv, group_size=group_size, bits=6)
    dequantized_qkv = ulysses_fN_qkv_permute_dequant(quantized_qkv[0],
                                                     128,
                                                     group_size=group_size,
                                                     bits=6)
    o = torch.empty((2, 8, 3, 128), device="cuda", dtype=torch.bfloat16)
    quantized_o = ulysses_fN_o_permute_quant(o, 4, 128, group_size=group_size, bits=6)
    dequantized_o = ulysses_fN_o_permute_dequant(quantized_o.flatten(0, 1),
                                                 128,
                                                 group_size=group_size,
                                                 bits=6)
  assert isinstance(quantized, FakeTensor)
  assert quantized.dtype == torch.uint8
  assert dequantized.shape == x.shape
  assert quantized_qkv.dtype == torch.uint8
  assert dequantized_qkv.shape == (2, 4, 5, 128)
  assert quantized_o.dtype == torch.uint8
  assert dequantized_o.shape == (2, 2, 12, 128)

  _require_runtime()
  x = torch.randn((2, 3, 128), device="cuda", dtype=torch.bfloat16)
  quantized = torch.compile(ulysses_fN_per_token_quant, fullgraph=True)(x, None, group_size, bits=6)
  dequantized = torch.compile(ulysses_fN_per_token_dequant, fullgraph=True)(quantized,
                                                                            128,
                                                                            group_size,
                                                                            bits=6)
  qkv = torch.randn((2, 4, 3, 5, 128), device="cuda", dtype=torch.bfloat16)
  quantized_qkv = torch.compile(ulysses_fN_qkv_permute_quant, fullgraph=True)(qkv,
                                                                              None,
                                                                              group_size,
                                                                              bits=6)
  dequantized_qkv = torch.compile(ulysses_fN_qkv_permute_dequant, fullgraph=True)(quantized_qkv[0],
                                                                                  128,
                                                                                  group_size,
                                                                                  bits=6)
  o = torch.randn((2, 8, 3, 128), device="cuda", dtype=torch.bfloat16)
  quantized_o = torch.compile(ulysses_fN_o_permute_quant, fullgraph=True)(o,
                                                                          4,
                                                                          128,
                                                                          group_size=group_size,
                                                                          bits=6)
  dequantized_o = torch.compile(ulysses_fN_o_permute_dequant,
                                fullgraph=True)(quantized_o.flatten(0, 1), 128, group_size, bits=6)
  assert dequantized.shape == x.shape
  assert dequantized_qkv.shape == (2, 4, 5, 128)
  assert dequantized_o.shape == (2, 2, 12, 128)


def test_ulysses_fp6_dequant_accepts_misaligned_packed_view() -> None:
  _require_runtime()
  x = torch.randn((2, 128), device="cuda", dtype=torch.bfloat16)
  quantized = ulysses_fp6_per_token_quant(x, group_size=32)
  storage = torch.empty(quantized.numel() + 1, dtype=torch.uint8, device=x.device)
  misaligned = storage[1:].view_as(quantized)
  misaligned.copy_(quantized)
  assert misaligned.is_contiguous() and misaligned.storage_offset() == 1

  actual = ulysses_fp6_per_token_dequant(misaligned, 128, 32)
  expected = ulysses_fp6_per_token_dequant(quantized, 128, 32)
  torch.testing.assert_close(actual.view(torch.uint16), expected.view(torch.uint16), rtol=0, atol=0)


def test_ulysses_fp6_defaults_empty_and_invalid() -> None:
  _require_runtime()
  x = torch.randn((2, 128), device="cuda", dtype=torch.bfloat16)
  default = ulysses_fp6_per_token_quant(x)
  explicit = ulysses_fp6_per_token_quant(x, group_size=32)
  torch.testing.assert_close(default, explicit, rtol=0, atol=0)
  assert default.shape == (2, 104)
  empty = torch.empty((0, 128), device="cuda", dtype=torch.bfloat16)
  assert ulysses_fp6_per_token_quant(empty).shape == (0, 104)
  assert ulysses_fp6_per_token_dequant(default[:0], 128, 32).shape == (0, 128)
  with pytest.raises(ValueError, match="group_size"):
    ulysses_fp6_per_token_quant(x, group_size=8)
  with pytest.raises(ValueError, match="bits must be 4, 6, or 8"):
    ulysses_fN_per_token_quant(x, bits=5)
  with pytest.raises(ValueError, match="head_size is required"):
    ulysses_fN_per_token_dequant(default, bits=6)
