import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from cache_dit.distributed.communicate.primitives import _fN_packed_row_width
from cache_dit.distributed.communicate.primitives import _fN_row_min_width
from cache_dit.kernels import cuda_ext_is_available
from cache_dit.kernels.ulysses import ulysses_fN_fused_qkv_dequant
from cache_dit.kernels.ulysses import ulysses_fN_fused_qkv_quant
from cache_dit.kernels.ulysses import ulysses_fN_qkv_permute_dequant
from cache_dit.kernels.ulysses import ulysses_fN_qkv_permute_quant


def _require_runtime(bits: int) -> None:
  if not torch.cuda.is_available() or not cuda_ext_is_available():
    pytest.skip("CUDA and the cache-dit CUDA extension are required.")
  if bits == 6 and torch.cuda.get_device_capability(0)[0] < 10:
    pytest.skip("FP6 E2M3 communication requires native sm100+ conversion support.")


def _width(head_size: int, rows: int, group_size: int, bits: int) -> int:
  min_width = _fN_row_min_width(head_size, group_size, 4, bits)
  return _fN_packed_row_width([rows], 1, min_width, width_step=4 if bits == 6 else 1)


def _legacy_quant(x: torch.Tensor, local_heads: int, width: int, group_size: int,
                  bits: int) -> torch.Tensor:
  batch, seq_len, global_heads, head_size = x.shape
  partitions = global_heads // local_heads
  x = x.reshape(batch, seq_len, partitions, local_heads, head_size)
  packed = ulysses_fN_qkv_permute_quant(x, width, group_size=group_size, bits=bits)
  return packed.view(torch.uint8)


def _legacy_bf16(x: torch.Tensor, local_heads: int) -> torch.Tensor:
  batch, seq_len, global_heads, head_size = x.shape
  partitions = global_heads // local_heads
  packed_head_size = (head_size + 7) // 8 * 8
  padded = torch.zeros((batch, seq_len, partitions, local_heads, packed_head_size),
                       dtype=torch.bfloat16,
                       device=x.device)
  padded[..., :head_size] = x.reshape(batch, seq_len, partitions, local_heads, head_size)
  return padded.permute(2, 1, 0, 3,
                        4).contiguous().view(torch.uint8).reshape(partitions, seq_len, batch,
                                                                  local_heads, packed_head_size * 2)


def _legacy_dequant(packed: torch.Tensor, head_size: int, group_size: int,
                    bits: int) -> torch.Tensor:
  if bits == 8:
    packed = packed.view(torch.float8_e4m3fn)
  return ulysses_fN_qkv_permute_dequant(packed, head_size, group_size, bits=bits)


def _reference_buffer(q, k, v, local_heads, widths, group_size, bits, quantize_k):
  partitions = q.shape[2] // local_heads[0]
  q_heads, k_heads, v_heads = local_heads
  q_width, k_width, v_width = widths
  packed_q = _legacy_quant(q, q_heads, q_width, group_size, bits)
  packed_k = (_legacy_quant(k, k_heads, k_width, group_size, bits) if quantize_k else _legacy_bf16(
    k, k_heads))
  packed_v = _legacy_quant(v, v_heads, v_width, group_size, bits)
  rows = []
  for partition in range(partitions):
    for seq in range(q.shape[1]):
      rows.append(
        torch.cat((packed_q[partition, seq].reshape(-1), packed_k[partition, seq].reshape(-1),
                   packed_v[partition, seq].reshape(-1))))
  return torch.stack(rows), (packed_q, packed_k, packed_v)


def _run_case(bits: int, group_size: int, quantize_k: bool, large: bool = False) -> None:
  _require_runtime(bits)
  torch.manual_seed(1000 + bits * 100 + group_size * 10 + int(quantize_k))
  batch, seq_len, partitions, head_size = (1, 256, 2, 128) if large else (2, 3, 2, 128)
  q_heads, k_heads, v_heads = 4, 2, 2
  q = torch.randn((batch, seq_len, partitions * q_heads, head_size),
                  device="cuda",
                  dtype=torch.bfloat16)
  k = torch.randn((batch, seq_len, partitions * k_heads, head_size),
                  device="cuda",
                  dtype=torch.bfloat16)
  v = torch.randn((batch, seq_len, partitions * v_heads, head_size),
                  device="cuda",
                  dtype=torch.bfloat16)
  rows = batch * seq_len
  widths = (_width(head_size, rows * q_heads, group_size, bits),
            _width(head_size, rows * k_heads, group_size, bits) if quantize_k else head_size * 2,
            _width(head_size, rows * v_heads, group_size, bits))
  chunk = batch * (q_heads * widths[0] + k_heads * widths[1] + v_heads * widths[2])
  expected, legacy = _reference_buffer(q, k, v, (q_heads, k_heads, v_heads), widths, group_size,
                                       bits, quantize_k)

  actual = ulysses_fN_fused_qkv_quant(q, k, v, partitions, q_heads, k_heads, v_heads, *widths,
                                      chunk, group_size, bits, quantize_k)
  torch.testing.assert_close(actual, expected, rtol=0, atol=0)

  actual_q, actual_k, actual_v = ulysses_fN_fused_qkv_dequant(actual, q_heads, k_heads, v_heads,
                                                              head_size, *widths, chunk, group_size,
                                                              bits, quantize_k)
  expected_q = _legacy_dequant(legacy[0].flatten(0, 1), head_size, group_size, bits)
  if quantize_k:
    expected_k = _legacy_dequant(legacy[1].flatten(0, 1), head_size, group_size, bits)
  else:
    expected_k = legacy[1].flatten(0, 1).view(torch.bfloat16).permute(1, 0, 2, 3)[..., :head_size]
  expected_v = _legacy_dequant(legacy[2].flatten(0, 1), head_size, group_size, bits)
  torch.testing.assert_close(actual_q.view(torch.uint16),
                             expected_q.view(torch.uint16),
                             rtol=0,
                             atol=0)
  torch.testing.assert_close(actual_k.view(torch.uint16),
                             expected_k.view(torch.uint16),
                             rtol=0,
                             atol=0)
  torch.testing.assert_close(actual_v.view(torch.uint16),
                             expected_v.view(torch.uint16),
                             rtol=0,
                             atol=0)


@pytest.mark.parametrize("bits", [6, 8])
@pytest.mark.parametrize("group_size", [0, 16, 32])
@pytest.mark.parametrize("quantize_k", [False, True])
def test_fused_qkv_matches_legacy_bytes_and_dequant(bits: int, group_size: int,
                                                    quantize_k: bool) -> None:
  _run_case(bits, group_size, quantize_k)


@pytest.mark.parametrize("bits,group_size,quantize_k", [
  (8, 0, True),
  (8, 32, False),
  (6, 0, True),
  (6, 32, False),
])
def test_fused_qkv_multirow_matches_legacy(bits: int, group_size: int, quantize_k: bool) -> None:
  _run_case(bits, group_size, quantize_k, large=True)


def test_fused_qkv_fake_and_compile() -> None:
  widths = (144, 144, 144)
  chunk = 2 * (4 * widths[0] + 2 * widths[1] + 2 * widths[2])
  with FakeTensorMode():
    q = torch.empty((2, 3, 8, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((2, 3, 4, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.empty((2, 3, 4, 128), device="cuda", dtype=torch.bfloat16)
    packed = ulysses_fN_fused_qkv_quant(q, k, v, 2, 4, 2, 2, *widths, chunk, 32, 8, True)
    outputs = ulysses_fN_fused_qkv_dequant(packed, 4, 2, 2, 128, *widths, chunk, 32, 8, True)
  assert packed.shape == (6, chunk)
  assert [x.shape for x in outputs] == [(2, 6, 4, 128), (2, 6, 2, 128), (2, 6, 2, 128)]

  _require_runtime(8)
  q = torch.randn((2, 3, 8, 128), device="cuda", dtype=torch.bfloat16)
  k = torch.randn((2, 3, 4, 128), device="cuda", dtype=torch.bfloat16)
  v = torch.randn((2, 3, 4, 128), device="cuda", dtype=torch.bfloat16)
  packed = torch.compile(
    lambda a, b, c: ulysses_fN_fused_qkv_quant(a, b, c, 2, 4, 2, 2, *widths, chunk, 32, 8, True),
    fullgraph=True)(q, k, v)
  outputs = torch.compile(
    lambda x: ulysses_fN_fused_qkv_dequant(x, 4, 2, 2, 128, *widths, chunk, 32, 8, True),
    fullgraph=True)(packed)
  assert [x.shape for x in outputs] == [(2, 6, 4, 128), (2, 6, 2, 128), (2, 6, 2, 128)]
