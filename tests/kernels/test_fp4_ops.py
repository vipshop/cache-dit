"""Numerical, boundary and compile coverage for cache-dit's native FP4 operators."""

import pytest
import torch
import torch.nn.functional as F

from cache_dit.kernels import fp4_gemm
from cache_dit.kernels import fp4_quantize
from cache_dit.kernels import fp4_quantize_swiglu

pytestmark = pytest.mark.skipif(
  not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
  reason="cache-dit native FP4 requires sm120",
)

_FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
               -6.0)


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
  """Return relative L2 error.

  :param actual: Test tensor.
  :param expected: Reference tensor.
  :returns: Relative L2 error.
  """

  return float(
    (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))


def _unswizzle_scales(scales: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
  """Restore blocked scales to row-major form for reference checks."""

  groups = cols // 16
  row_blocks = (rows + 127) // 128
  col_blocks = groups // 4
  rearranged = scales.reshape(row_blocks * col_blocks, 32, 16)
  temp = rearranged.reshape(row_blocks * col_blocks, 32, 4, 4).transpose(1, 2)
  blocks = temp.reshape(row_blocks, col_blocks, 128, 4)
  padded = blocks.permute(0, 2, 1, 3).reshape(row_blocks * 128, col_blocks * 4)
  return padded[:rows, :groups]


def _dequantize(qdata: torch.Tensor, scales: torch.Tensor,
                global_scales: torch.Tensor) -> torch.Tensor:
  """Decode one native FP4 tensor for GEMM reference checks."""

  rows, packed_cols = qdata.shape
  packed = qdata.view(torch.uint8)
  table = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=qdata.device)
  values = torch.stack((table[(packed & 0x0f).long()], table[(packed >> 4).long()]),
                       dim=-1).reshape(rows, packed_cols * 2)
  scales = _unswizzle_scales(scales, rows, packed_cols * 2).float()
  values = values * scales.repeat_interleave(16, dim=1)
  if global_scales.numel() != 0:
    values = values * global_scales[:, None]
  return values


def _reference_quantize(x: torch.Tensor, level1: bool):
  """Compute the source NVFP4 recipe with ordinary PyTorch operations."""

  rows, cols = x.shape
  fp32 = x.float()
  if level1:
    global_scales = fp32.abs().amax(dim=1) / 2688.0
    inv_global = torch.where(global_scales > 0, 1.0 / global_scales,
                             torch.zeros_like(global_scales))
    normalized = fp32 * inv_global[:, None]
  else:
    global_scales = fp32.new_empty((0, ))
    normalized = fp32
  groups = normalized.reshape(rows, cols // 16, 16)
  block_scales = (groups.abs().amax(dim=2) / 6.0).to(torch.float8_e4m3fn)
  block_fp32 = block_scales.float()
  inv_block = torch.where(block_fp32 > 0, 1.0 / block_fp32, torch.zeros_like(block_fp32))
  normalized = groups * inv_block[:, :, None]
  return normalized.reshape(rows, cols), block_scales, global_scales


def _swizzle_scales(scales: torch.Tensor) -> torch.Tensor:
  """Apply the independent 128x4 scale rearrangement used by the reference."""

  rows, groups = scales.shape
  row_blocks = (rows + 127) // 128
  col_blocks = groups // 4
  padded = scales.new_zeros((row_blocks * 128, col_blocks * 4))
  padded[:rows, :groups] = scales
  blocks = padded.view(row_blocks, 128, col_blocks, 4).permute(0, 2, 1, 3)
  return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


@pytest.mark.parametrize("level1", [True, False])
def test_quantize_matches_independent_reference(level1: bool) -> None:
  torch.manual_seed(0)
  x = torch.randn(17, 256, dtype=torch.bfloat16, device="cuda")
  x[0].zero_()
  x[1, :16] = 2.0 ** -12
  qdata, scales, global_scales = fp4_quantize(x, level1=level1)
  normalized, block_ref, global_ref = _reference_quantize(x, level1)
  assert torch.allclose(global_scales, global_ref, rtol=2e-7, atol=0)
  assert torch.equal(scales.flatten(), _swizzle_scales(block_ref))

  kernels = pytest.importorskip("torchao.prototype.mx_formats.kernels")
  expected_qdata = kernels.pack_uint4(kernels.f32_to_f4_unpacked(normalized.clamp(-6.0, 6.0)))
  assert torch.equal(qdata.view(torch.uint8), expected_qdata)
  dequantized = _dequantize(qdata, scales, global_scales)
  reference = _dequantize(expected_qdata, scales, global_scales)
  assert _relative_l2(dequantized, reference) < 2e-3
  assert torch.count_nonzero(dequantized[0]) == 0


def test_quantized_scale_padding_is_zero() -> None:
  torch.manual_seed(1)
  x = torch.randn(17, 64, dtype=torch.bfloat16, device="cuda")
  _, scales, _ = fp4_quantize(x)
  unswizzled = _unswizzle_scales(scales, 128, 64)
  assert torch.count_nonzero(unswizzled[17:]) == 0


def test_fused_quantize_large_k_matches_independent_reference() -> None:
  torch.manual_seed(10)
  rows, cols = 129, 15360
  x = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
  x[0].zero_()
  x[1, :16] = 2.0 ** -12
  qdata, scales, global_scales = fp4_quantize(x, level1=True)
  normalized, block_ref, global_ref = _reference_quantize(x, level1=True)
  assert torch.allclose(global_scales, global_ref, rtol=2e-7, atol=0)
  swizzled_ref = _swizzle_scales(block_ref)
  scale_ulp = (scales.view(torch.uint8).flatten().int() -
               swizzled_ref.view(torch.uint8).int()).abs().max()
  assert int(scale_ulp) <= 1

  kernels = pytest.importorskip("torchao.prototype.mx_formats.kernels")
  expected_qdata = kernels.pack_uint4(kernels.f32_to_f4_unpacked(normalized.clamp(-6.0, 6.0)))
  dequantized = _dequantize(qdata, scales, global_scales)
  reference = _dequantize(expected_qdata, swizzled_ref, global_scales)
  assert _relative_l2(dequantized, reference) < 1e-2
  unswizzled = _unswizzle_scales(scales, rows, cols)
  assert torch.count_nonzero(unswizzled[rows:]) == 0


def test_single_level_underflow_produces_exact_zero() -> None:
  x = torch.full((3, 64), 2.0 ** -12, dtype=torch.bfloat16, device="cuda")
  qdata, scales, global_scales = fp4_quantize(x, level1=False)
  assert global_scales.numel() == 0
  assert torch.count_nonzero(scales.float()) == 0
  assert torch.count_nonzero(_dequantize(qdata, scales, global_scales)) == 0


@pytest.mark.parametrize("m,n,k", [(1, 128, 256), (17, 256, 256), (129, 384, 512), (300, 256, 512)])
def test_gemm_matches_dequantized_reference_with_m_tails_and_bias(m: int, n: int, k: int) -> None:
  torch.manual_seed(2)
  a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
  w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
  bias = torch.randn(n, dtype=torch.bfloat16, device="cuda")
  qa, sa, ga = fp4_quantize(a)
  qw, sw, gw = fp4_quantize(w)
  actual = fp4_gemm(qa, qw, sa, sw, ga, gw, bias=bias, use_native=True)
  expected = F.linear(_dequantize(qa, sa, ga), _dequantize(qw, sw, gw), bias.float())
  assert _relative_l2(actual, expected) < 3e-3


def test_default_gemm_matches_dequantized_reference() -> None:
  torch.manual_seed(2)
  a = torch.randn(129, 256, dtype=torch.bfloat16, device="cuda")
  w = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
  bias = torch.randn(128, dtype=torch.bfloat16, device="cuda")
  qa, sa, ga = fp4_quantize(a)
  qw, sw, gw = fp4_quantize(w)
  actual = fp4_gemm(qa, qw, sa, sw, ga, gw, bias=bias)
  expected = F.linear(_dequantize(qa, sa, ga), _dequantize(qw, sw, gw), bias.float())
  assert _relative_l2(actual, expected) < 5e-3


@pytest.mark.parametrize("use_native", [True, False])
def test_unsupported_shape_is_padded_and_depadded(use_native: bool) -> None:
  torch.manual_seed(2)
  a = torch.randn(17, 192, dtype=torch.bfloat16, device="cuda")
  w = torch.randn(24, 192, dtype=torch.bfloat16, device="cuda")
  bias = torch.randn(24, dtype=torch.bfloat16, device="cuda")
  qa, sa, ga = fp4_quantize(a)
  qw, sw, gw = fp4_quantize(w)
  actual = fp4_gemm(qa, qw, sa, sw, ga, gw, bias=bias, use_native=use_native)
  expected = F.linear(_dequantize(qa, sa, ga), _dequantize(qw, sw, gw), bias.float())
  assert actual.shape == (17, 24)
  assert _relative_l2(actual, expected) < 5e-3


def test_padding_fullgraph_compile() -> None:
  torch.manual_seed(6)
  a = torch.randn(17, 192, dtype=torch.bfloat16, device="cuda")
  w = torch.randn(24, 192, dtype=torch.bfloat16, device="cuda")
  bias = torch.randn(24, dtype=torch.bfloat16, device="cuda")
  qa, sa, ga = fp4_quantize(a)
  qw, sw, gw = fp4_quantize(w)
  expected = fp4_gemm(qa, qw, sa, sw, ga, gw, bias=bias)
  actual = torch.compile(lambda: fp4_gemm(qa, qw, sa, sw, ga, gw, bias=bias), fullgraph=True)()
  assert actual.shape == expected.shape
  assert _relative_l2(actual, expected) < 1e-4


def test_two_level_collective_matches_dequantized_reference() -> None:
  torch.manual_seed(5)
  a = torch.randn(129, 256, dtype=torch.bfloat16, device="cuda")
  w = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
  bias = torch.randn(256, dtype=torch.bfloat16, device="cuda")
  qa, sa, ga = fp4_quantize(a)
  qw, sw, gw = fp4_quantize(w)
  actual = fp4_gemm(qa, qw, sa, sw, ga, gw, bias=bias, use_native=True)
  expected = F.linear(_dequantize(qa, sa, ga), _dequantize(qw, sw, gw), bias.float())
  assert _relative_l2(actual, expected) < 3e-3


def test_single_level_gemm_matches_torchao_scaled_mm() -> None:
  torch.manual_seed(4)
  a = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
  w = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
  qa, sa, ga = fp4_quantize(a, level1=False)
  qw, sw, gw = fp4_quantize(w, level1=False)
  actual = fp4_gemm(qa, qw, sa, sw, ga, gw, use_native=True)
  expected = torch._scaled_mm(qa, qw.t(), sa, sw, out_dtype=torch.bfloat16)
  assert torch.equal(actual, expected)


def test_zero_activation_rows_are_exactly_zero() -> None:
  a = torch.zeros(8, 256, dtype=torch.bfloat16, device="cuda")
  w = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
  qa, sa, ga = fp4_quantize(a)
  qw, sw, gw = fp4_quantize(w)
  actual = fp4_gemm(qa, qw, sa, sw, ga, gw)
  assert torch.isfinite(actual).all()
  assert torch.count_nonzero(actual) == 0


def test_fake_and_fullgraph_compile() -> None:
  from torch._subclasses.fake_tensor import FakeTensorMode

  m, n, k = 11, 128, 256
  with FakeTensorMode():
    a = torch.empty(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.empty(n, k, dtype=torch.bfloat16, device="cuda")
    qa, sa, ga = fp4_quantize(a)
    qw, sw, gw = fp4_quantize(w)
    out = fp4_gemm(qa, qw, sa, sw, ga, gw)
    assert qa.shape == (m, k // 2) and qa.dtype == torch.float4_e2m1fn_x2
    assert sa.dtype == torch.float8_e4m3fn and ga.shape == (m, )
    assert out.shape == (m, n) and out.dtype == torch.bfloat16

  a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
  w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

  def run(a, w):
    qa, sa, ga = fp4_quantize(a)
    qw, sw, gw = fp4_quantize(w)
    return fp4_gemm(qa, qw, sa, sw, ga, gw)

  expected = run(a, w)
  actual = torch.compile(run, fullgraph=True)(a, w)
  assert _relative_l2(actual, expected) < 1e-4


def test_cuda_graph_replay_matches_eager() -> None:
  torch.manual_seed(5)
  a = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
  w = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
  qa, sa, ga = fp4_quantize(a)
  qw, sw, gw = fp4_quantize(w)
  side_stream = torch.cuda.Stream()
  side_stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(side_stream):
    for _ in range(2):
      fp4_gemm(qa, qw, sa, sw, ga, gw)
  torch.cuda.current_stream().wait_stream(side_stream)
  graph = torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
    captured = fp4_gemm(qa, qw, sa, sw, ga, gw)
  expected = fp4_gemm(qa, qw, sa, sw, ga, gw)
  graph.replay()
  assert torch.equal(captured, expected)

  new_a = torch.randn_like(a)
  new_qa, new_sa, new_ga = fp4_quantize(new_a)
  qa.copy_(new_qa)
  sa.copy_(new_sa)
  ga.copy_(new_ga)
  expected = fp4_gemm(qa, qw, sa, sw, ga, gw)
  graph.replay()
  assert torch.equal(captured, expected)


@pytest.mark.parametrize("level1", [True, False])
@pytest.mark.parametrize("activation,approximate", [("gelu", "none"), ("gelu_tanh", "tanh")])
def test_gemm_activation_matches_unfused_and_compiles(level1: bool, activation: str,
                                                      approximate: str) -> None:
  torch.manual_seed(30)
  a = torch.randn(17, 256, dtype=torch.bfloat16, device="cuda")
  w = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
  bias = torch.randn(256, dtype=torch.bfloat16, device="cuda")
  qa, sa, ga = fp4_quantize(a, level1=level1)
  qw, sw, gw = fp4_quantize(w, level1=level1)
  expected = F.gelu(fp4_gemm(qa, qw, sa, sw, ga, gw, bias=bias), approximate=approximate)
  actual = fp4_gemm(qa, qw, sa, sw, ga, gw, bias=bias, activation=activation)
  assert _relative_l2(actual, expected) < 5e-5
  compiled = torch.compile(
    lambda: fp4_gemm(qa, qw, sa, sw, ga, gw, bias=bias, activation=activation),
    fullgraph=True,
  )()
  assert torch.equal(compiled, actual)


def test_gemm_activation_fake_metadata() -> None:
  from torch._subclasses.fake_tensor import FakeTensorMode

  with FakeTensorMode():
    a = torch.empty(11, 256, dtype=torch.bfloat16, device="cuda")
    w = torch.empty(128, 256, dtype=torch.bfloat16, device="cuda")
    qa, sa, ga = fp4_quantize(a)
    qw, sw, gw = fp4_quantize(w)
    out = fp4_gemm(qa, qw, sa, sw, ga, gw, activation="gelu_tanh")
    assert out.shape == (11, 128) and out.dtype == torch.bfloat16


def test_gemm_activation_cuda_graph_replay_matches_eager() -> None:
  torch.manual_seed(31)
  a = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
  w = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
  qa, sa, ga = fp4_quantize(a)
  qw, sw, gw = fp4_quantize(w)
  side_stream = torch.cuda.Stream()
  side_stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(side_stream):
    for _ in range(2):
      fp4_gemm(qa, qw, sa, sw, ga, gw, activation="gelu_tanh")
  torch.cuda.current_stream().wait_stream(side_stream)
  graph = torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
    captured = fp4_gemm(qa, qw, sa, sw, ga, gw, activation="gelu_tanh")
  expected = fp4_gemm(qa, qw, sa, sw, ga, gw, activation="gelu_tanh")
  graph.replay()
  assert torch.equal(captured, expected)


def test_invalid_gemm_activation_is_rejected() -> None:
  a = torch.randn(2, 256, dtype=torch.bfloat16, device="cuda")
  w = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
  qa, sa, ga = fp4_quantize(a)
  qw, sw, gw = fp4_quantize(w)
  with pytest.raises(ValueError, match="Unsupported FP4 activation"):
    fp4_gemm(qa, qw, sa, sw, ga, gw, activation="silu")
  with pytest.raises(ValueError, match="torch._scaled_mm exposes no activation"):
    fp4_gemm(qa, qw, sa, sw, ga, gw, activation="gelu", use_native=False)


@pytest.mark.parametrize("rows,cols", [(17, 256), (129, 15360)])
@pytest.mark.parametrize("level1", [True, False])
@pytest.mark.parametrize("separate_gate", [True, False])
def test_swiglu_quantize_matches_explicit_gating(rows: int, cols: int, level1: bool,
                                                 separate_gate: bool) -> None:
  torch.manual_seed(40)
  value = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
  gate = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
  value[0].zero_()
  value[1, :16] = 2.0 ** -12
  expected = fp4_quantize(value * F.silu(gate), level1=level1)
  actual = fp4_quantize_swiglu(
    value if separate_gate else torch.cat((value, gate), dim=1),
    gate if separate_gate else None,
    level1=level1,
  )
  for actual_tensor, expected_tensor in zip(actual, expected):
    assert torch.equal(actual_tensor, expected_tensor)
  assert torch.count_nonzero(_dequantize(*actual)[0]) == 0
  unswizzled = _unswizzle_scales(actual[1], rows + (128 - rows % 128) % 128, cols)
  assert torch.count_nonzero(unswizzled[rows:]) == 0

  if cols == 256:
    compiled = torch.compile(
      lambda: fp4_quantize_swiglu(
        value if separate_gate else torch.cat((value, gate), dim=1),
        gate if separate_gate else None,
        level1=level1,
      ),
      fullgraph=True,
    )()
    for actual_tensor, expected_tensor in zip(compiled, expected):
      assert torch.equal(actual_tensor, expected_tensor)


def test_swiglu_quantize_fake_metadata() -> None:
  from torch._subclasses.fake_tensor import FakeTensorMode

  with FakeTensorMode():
    value = torch.empty(11, 256, dtype=torch.bfloat16, device="cuda")
    gate = torch.empty(11, 256, dtype=torch.bfloat16, device="cuda")
    for actual in (fp4_quantize_swiglu(value,
                                       gate), fp4_quantize_swiglu(torch.cat((value, gate), dim=1))):
      assert actual[0].shape == (11, 128)
      assert actual[0].dtype == torch.float4_e2m1fn_x2
      assert actual[1].dtype == torch.float8_e4m3fn
      assert actual[2].shape == (11, ) and actual[2].dtype == torch.float32


def test_swiglu_quantize_cuda_graph_replay_matches_eager() -> None:
  torch.manual_seed(41)
  value = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
  gate = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
  side_stream = torch.cuda.Stream()
  side_stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(side_stream):
    for _ in range(2):
      fp4_quantize_swiglu(value, gate)
  torch.cuda.current_stream().wait_stream(side_stream)
  graph = torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
    captured = fp4_quantize_swiglu(value, gate)
  expected = fp4_quantize_swiglu(value, gate)
  graph.replay()
  for actual_tensor, expected_tensor in zip(captured, expected):
    assert torch.equal(actual_tensor, expected_tensor)

  value.copy_(torch.randn_like(value))
  gate.copy_(torch.randn_like(gate))
  expected = fp4_quantize_swiglu(value, gate)
  graph.replay()
  for actual_tensor, expected_tensor in zip(captured, expected):
    assert torch.equal(actual_tensor, expected_tensor)


def test_invalid_swiglu_quantize_geometry_is_rejected() -> None:
  value = torch.randn(2, 256, dtype=torch.bfloat16, device="cuda")
  with pytest.raises(ValueError, match="H divisible by 64"):
    fp4_quantize_swiglu(value[:, :96])
  with pytest.raises(ValueError, match="value and gate shapes"):
    fp4_quantize_swiglu(value, torch.randn(2, 128, dtype=torch.bfloat16, device="cuda"))
  with pytest.raises(ValueError, match="row counts must match"):
    fp4_quantize_swiglu(value, torch.randn(3, 256, dtype=torch.bfloat16, device="cuda"))
  with pytest.raises(ValueError, match="even 2H width"):
    fp4_quantize_swiglu(torch.randn(2, 255, dtype=torch.bfloat16, device="cuda"))


def test_invalid_quantize_geometry_is_rejected() -> None:
  with pytest.raises(ValueError, match="empty"):
    fp4_quantize(torch.empty(0, 64, dtype=torch.bfloat16, device="cuda"))
  with pytest.raises(ValueError, match="K % 64"):
    fp4_quantize(torch.randn(2, 32, dtype=torch.bfloat16, device="cuda"))
