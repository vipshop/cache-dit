"""Tests for the SVDQ-FP8 (W8A8) fused kernels and runtime module."""

import pytest
import torch
from torch import nn

from cache_dit import QuantizeConfig
from cache_dit.kernels import cuda_ext_is_available
from cache_dit.kernels import svdq_gemm_w8a8
from cache_dit.kernels import svdq_quantize_w8a8
from cache_dit.quantization.svdquant.wNa8 import SVDQW8A8Linear
from cache_dit.quantization.svdquant.wNa8 import _quantize_svdq_w8a8

EPS = 1e-12

pytestmark = pytest.mark.skipif(
  not torch.cuda.is_available() or not cuda_ext_is_available()
  or torch.cuda.get_device_capability(0)[0] < 12,
  reason="SVDQ-FP8 requires the CUDA extension on an sm120+ device",
)

SHAPES = ((128, 128, 128), (256, 256, 512), (512, 384, 1024))
RANKS = (0, 32, 64, 128)


def quantize_per_row(residual: torch.Tensor):
  scale = residual.abs().amax(dim=1, keepdim=True).clamp_min(EPS) / 448.0
  return (residual / scale).to(torch.float8_e4m3fn), scale.squeeze(1).contiguous()


def make_gemm_inputs(m, n, k, rank, seed=0):
  generator = torch.Generator(device="cuda").manual_seed(seed)
  qa = (torch.randn(m, k, device="cuda", generator=generator) / 4).to(torch.float8_e4m3fn)
  qb = (torch.randn(n, k, device="cuda", generator=generator) / 4).to(torch.float8_e4m3fn)
  sa = torch.rand(m, device="cuda", generator=generator) + 0.25
  sb = torch.rand(n, device="cuda", generator=generator) + 0.25
  bias = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=generator)
  d = torch.randn(m, rank, device="cuda", dtype=torch.bfloat16,
                  generator=generator) if rank else None
  up = torch.randn(n, rank, device="cuda", dtype=torch.bfloat16,
                   generator=generator) if rank else None
  return qa, qb, sa, sb, bias, d, up


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("m,n,k", SHAPES)
def test_gemm_matches_reference(m, n, k, rank):
  qa, qb, sa, sb, bias, d, up = make_gemm_inputs(m, n, k, rank)
  out = svdq_gemm_w8a8(qa, qb, sa, sb, lora_act_in=d, lora_up=up, bias=bias)
  ref = qa.float() @ qb.float().t() * (sa[:, None] * sb[None, :]) + bias.float()
  if rank:
    ref = ref + d.float() @ up.float().t()
  ref_bf16 = ref.bfloat16().float()
  rel = ((out.float() - ref).norm() / ref.norm()).item()
  bitexact = (out.float() == ref_bf16).float().mean().item()
  assert rel < 2e-3
  assert bitexact > 0.99


def test_gemm_without_bias_or_lora():
  qa, qb, sa, sb, _, _, _ = make_gemm_inputs(128, 128, 128, 0)
  out = svdq_gemm_w8a8(qa, qb, sa, sb)
  ref = qa.float() @ qb.float().t() * (sa[:, None] * sb[None, :])
  assert ((out.float() - ref).norm() / ref.norm()).item() < 2e-3


def test_gemm_rejects_unsupported_rank_and_shape():
  qa, qb, sa, sb, bias, d, up = make_gemm_inputs(128, 128, 128, 32)
  with pytest.raises(ValueError, match="rank"):
    svdq_gemm_w8a8(qa,
                   qb,
                   sa,
                   sb,
                   lora_act_in=d[:, :16].contiguous(),
                   lora_up=up[:, :16].contiguous(),
                   bias=bias)
  with pytest.raises(ValueError, match="multiples of 128"):
    svdq_gemm_w8a8(qa[:64], qb, sa[:64], sb, bias=bias)


@pytest.mark.parametrize("rank", RANKS)
def test_producer_matches_reference(rank):
  m, k = 256, 512
  generator = torch.Generator(device="cuda").manual_seed(0)
  x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=generator) * 3
  smooth = torch.rand(k, device="cuda", dtype=torch.bfloat16, generator=generator) + 0.5
  down = (torch.randn(rank, k, device="cuda", dtype=torch.bfloat16, generator=generator) /
          8 if rank else None)
  q, sa, d = svdq_quantize_w8a8(x, smooth, down)
  xs = x.float() / smooth.float()
  ref_sa = xs.abs().amax(dim=1).clamp_min(EPS) / 448.0
  ref_q = (xs / ref_sa[:, None]).to(torch.float8_e4m3fn)
  assert (sa - ref_sa).abs().max().item() / ref_sa.max().item() < 1e-6
  dequant = q.float() * sa[:, None]
  ref_dequant = ref_q.float() * ref_sa[:, None]
  # --use_fast_math lets a few E4M3 codes flip by one ULP; dequantized values must still agree.
  assert ((dequant - ref_dequant).norm() / ref_dequant.norm()).item() < 3e-3
  if rank:
    ref_d = (x.float() @ down.float().t()).bfloat16()
    assert (d == ref_d).float().mean().item() > 0.99
  else:
    assert d is None


def test_producer_handles_zero_rows_and_padding():
  m, k, rank = 130, 256, 32
  x = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
  x[100:] = torch.randn(m - 100, k, device="cuda", dtype=torch.bfloat16)
  smooth = torch.ones(k, device="cuda", dtype=torch.bfloat16)
  down = torch.randn(rank, k, device="cuda", dtype=torch.bfloat16)
  q, sa, d = svdq_quantize_w8a8(x, smooth, down)
  assert q.shape == (m, k) and sa.shape == (m, ) and d.shape == (m, rank)
  assert torch.isfinite(q.float()).all() and torch.isfinite(sa).all() and torch.isfinite(
    d.float()).all()
  assert torch.count_nonzero(q[:100].float()) == 0
  assert torch.count_nonzero(d[:100].float()) == 0


@pytest.mark.parametrize("rank", (0, 32, 64, 128))
def test_linear_module_matches_reference(rank):
  torch.manual_seed(0)
  n, k = 192, 320
  linear = nn.Linear(k, n, bias=True, dtype=torch.bfloat16, device="cuda")
  packed = _quantize_svdq_w8a8(linear, rank=rank, smooth_strategy="weight")
  x = torch.randn(3, 200, k, device="cuda", dtype=torch.bfloat16)
  out = packed(x)
  assert out.shape == (3, 200, n)
  weight = packed.dequantize_weight()
  ref = x.float() @ weight.t() + linear.bias.float()
  rel = ((out.float().reshape(-1, n) - ref.reshape(-1, n)).norm() / ref.norm()).item()
  assert rel < 0.06


def test_linear_forward_has_no_standalone_bf16_gemm(monkeypatch):
  linear = nn.Linear(256, 256, dtype=torch.bfloat16, device="cuda")
  packed = _quantize_svdq_w8a8(linear, rank=32, smooth_strategy="weight")
  x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)

  def forbidden(*args, **kwargs):
    raise AssertionError("the low-rank branch must not run as a standalone GEMM")

  monkeypatch.setattr(torch, "matmul", forbidden)
  monkeypatch.setattr(torch, "mm", forbidden)
  monkeypatch.setattr(torch.nn.functional, "linear", forbidden)
  out = packed(x)
  assert out.shape == (128, 256)


def test_low_rank_weight_beats_plain_fp8():
  torch.manual_seed(0)
  n, k, rank = 256, 512, 32
  base = torch.randn(n, 8, device="cuda") @ torch.randn(8, k, device="cuda")
  linear = nn.Linear(k, n, bias=False, dtype=torch.bfloat16, device="cuda")
  with torch.no_grad():
    linear.weight.copy_((base + 0.02 * torch.randn(n, k, device="cuda")).bfloat16())
  packed = _quantize_svdq_w8a8(linear, rank=rank, smooth_strategy="identity")
  x = torch.randn(256, k, device="cuda", dtype=torch.bfloat16)
  exact = x.float() @ linear.weight.float().t()
  svdq_err = ((packed(x).float() - exact).norm() / exact.norm()).item()
  qb, sb = quantize_per_row(linear.weight.float())
  sa = x.float().abs().amax(dim=1).clamp_min(EPS) / 448.0
  qa = (x.float() / sa[:, None]).to(torch.float8_e4m3fn)
  plain = qa.float() @ qb.float().t() * (sa[:, None] * sb[None, :])
  plain_err = ((plain - exact).norm() / exact.norm()).item()
  assert svdq_err < plain_err * 0.7


def test_fake_tensor_registration():
  from torch._subclasses.fake_tensor import FakeTensorMode

  with FakeTensorMode():
    qa = torch.empty(128, 128, device="cuda", dtype=torch.float8_e4m3fn)
    qb = torch.empty(128, 128, device="cuda", dtype=torch.float8_e4m3fn)
    sa = torch.empty(128, device="cuda", dtype=torch.float32)
    sb = torch.empty(128, device="cuda", dtype=torch.float32)
    out = svdq_gemm_w8a8(qa, qb, sa, sb)
    assert out.shape == (128, 128) and out.dtype == torch.bfloat16
    x = torch.empty(100, 128, device="cuda", dtype=torch.bfloat16)
    smooth = torch.empty(128, device="cuda", dtype=torch.bfloat16)
    q, scales, d = svdq_quantize_w8a8(x, smooth)
    assert q.shape == (100, 128) and scales.shape == (100, ) and d is None


def test_fullgraph_compile_smoke():
  linear = nn.Linear(128, 128, dtype=torch.bfloat16, device="cuda")
  packed = _quantize_svdq_w8a8(linear, rank=32, smooth_strategy="weight")
  compiled = torch.compile(packed, fullgraph=True)
  x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
  out = compiled(x)
  reference = packed(x)
  assert torch.allclose(out.float(), reference.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("quant_type,rank,is_dq", [
  ("svdq_fp8_r32", 32, False),
  ("svdq_fp8_r64_dq", 64, True),
  ("svdq_fp8_r128_dq", 128, True),
])
def test_quantize_config_parses_fp8_types(quant_type, rank, is_dq):
  config = QuantizeConfig(quant_type=quant_type,
                          calibrate_fn=None if is_dq else lambda module: None)
  assert config.get_svdq_precision() == "fp8"
  assert config.get_svdq_rank() == rank
  assert config.is_svdq_dq() == is_dq


def test_quantize_config_rejects_bad_fp8_rank():
  from cache_dit.quantization.svdquant.quantizer import validate_linear_geometry

  with pytest.raises(ValueError, match="rank"):
    validate_linear_geometry(256, 256, rank=48, precision="fp8")


@pytest.mark.parametrize("key,value", [
  ("runtime_kernel", "native"),
  ("lora_layout", "native"),
  ("fused_mlp", True),
  ("layerwise_offload", True),
  ("offload_quantized_layers_to_cpu", True),
  ("svd_refine_iters", 1),
  ("streaming", False),
])
def test_quantize_config_rejects_unsupported_fp8_kwargs(key, value):
  with pytest.raises(ValueError, match="not supported.*SVDQ-FP8"):
    QuantizeConfig(quant_type="svdq_fp8_r32_dq", svdq_kwargs={key: value})


def test_producer_empty_and_padded_composition():
  smooth = torch.ones(128, device="cuda", dtype=torch.bfloat16)
  down = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
  empty = torch.empty(0, 128, device="cuda", dtype=torch.bfloat16)
  q, sa, d = svdq_quantize_w8a8(empty, smooth, down)
  assert q.shape == (0, 128) and sa.shape == (0, ) and d.shape == (0, 32)

  x = torch.randn(130, 128, device="cuda", dtype=torch.bfloat16)
  q, sa, d = svdq_quantize_w8a8(x, smooth, down, pad_rows=True)
  qb = torch.randn(128, 128, device="cuda").to(torch.float8_e4m3fn)
  sb = torch.rand(128, device="cuda") + 0.5
  up = torch.randn(128, 32, device="cuda", dtype=torch.bfloat16)
  out = svdq_gemm_w8a8(q, qb, sa, sb, lora_act_in=d, lora_up=up)
  assert out.shape == (256, 128)


def test_producer_direct_d_without_split_k():
  m, k, rank = 32768, 128, 32
  x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
  smooth = torch.ones(k, device="cuda", dtype=torch.bfloat16)
  down = torch.randn(rank, k, device="cuda", dtype=torch.bfloat16)
  _, _, d = svdq_quantize_w8a8(x, smooth, down)
  rows = slice(0, 128)
  ref = (x[rows].float() @ down.float().t()).bfloat16()
  assert (d[rows] == ref).float().mean().item() > 0.99


def test_real_flux_shape_smoke():
  """Cover the production split-K count and a multi-wave GEMM grid."""
  m, n, k, rank = 4096, 12288, 3072, 128
  generator = torch.Generator(device="cuda").manual_seed(0)
  x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=generator)
  smooth = torch.rand(k, device="cuda", dtype=torch.bfloat16, generator=generator) + 0.5
  down = torch.randn(rank, k, device="cuda", dtype=torch.bfloat16, generator=generator) / 32
  q, sa, d = svdq_quantize_w8a8(x, smooth, down)
  ref_d = (x.float() @ down.float().t()).bfloat16()
  assert ((d.float() - ref_d.float()).norm() / ref_d.float().norm()).item() < 2e-3

  qb = (torch.randn(n, k, device="cuda", generator=generator) / 8).to(torch.float8_e4m3fn)
  sb = torch.rand(n, device="cuda", generator=generator) + 0.25
  up = torch.randn(n, rank, device="cuda", dtype=torch.bfloat16, generator=generator) / 16
  out = svdq_gemm_w8a8(q, qb, sa, sb, lora_act_in=d, lora_up=up)
  rows = slice(0, 128)
  ref = q[rows].float() @ qb.float().t() * (sa[rows, None] * sb[None, :])
  ref = ref + d[rows].float() @ up.float().t()
  rel = ((out[rows].float() - ref).norm() / ref.norm()).item()
  assert rel < 2e-3
  assert torch.isfinite(out.float()).all()


def test_quantize_svdq_fp8_honors_cuda_quantize_device():
  import cache_dit

  model = nn.Linear(128, 128, dtype=torch.bfloat16, device="cpu")
  packed = cache_dit.quantize(
    model,
    QuantizeConfig(quant_type="svdq_fp8_r32_dq",
                   svdq_kwargs={
                     "quantize_device": "cuda",
                     "smooth_strategy": "weight"
                   }))
  assert isinstance(packed, SVDQW8A8Linear)
  assert packed.qweight.is_cuda
  assert packed(torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)).shape == (16, 128)


def test_quantize_svdq_fp8_dq_replaces_linears():
  import cache_dit

  model = nn.Sequential(nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda"),
                        nn.Linear(256, 128, dtype=torch.bfloat16, device="cuda"))
  model = cache_dit.quantize(
    model, QuantizeConfig(quant_type="svdq_fp8_r32_dq", svdq_kwargs={"smooth_strategy": "weight"}))
  assert all(isinstance(layer, SVDQW8A8Linear) for layer in model)
  x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
  assert model(x).shape == (64, 128)
