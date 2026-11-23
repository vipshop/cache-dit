"""Tests for SVDQ-W4A8 packed-E2M1 kernels and runtime modules."""

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from cache_dit import QuantizeConfig
from cache_dit.kernels import cuda_ext_is_available
from cache_dit.kernels import svdq_gemm_w4a8
from cache_dit.kernels import svdq_quantize_w4a8
from cache_dit.kernels import svdq_quantize_w8a8
from cache_dit.quantization.svdquant.wNa8 import E2M1_CODEBOOK
from cache_dit.quantization.svdquant.wNa8 import dequantize_e2m1
from cache_dit.quantization.svdquant.wNa8 import pack_e2m1
from cache_dit.quantization.svdquant.wNa8 import _quantize_svdq_w4a8
from cache_dit.quantization.svdquant.wNa8 import unpack_e2m1

pytestmark = pytest.mark.skipif(
  not torch.cuda.is_available() or not cuda_ext_is_available()
  or torch.cuda.get_device_capability(0)[0] < 12,
  reason="SVDQ-W4A8 requires the CUDA extension on sm120+",
)


def test_e2m1_pack_covers_full_codebook_and_nibble_order():
  codes = torch.arange(16, dtype=torch.uint8).repeat(4)
  packed = pack_e2m1(codes)
  assert torch.equal(unpack_e2m1(packed), codes)
  assert packed[0].item() == 0x10 and packed[1].item() == 0x32
  values = torch.tensor(E2M1_CODEBOOK)[unpack_e2m1(packed).long()]
  assert torch.equal(values[:16], torch.tensor(E2M1_CODEBOOK))
  with pytest.raises(ValueError, match="even K"):
    pack_e2m1(torch.zeros(3, dtype=torch.uint8))


@pytest.mark.parametrize("rank", [0, 32, 64, 128])
@pytest.mark.parametrize("m,n,k", [(128, 128, 128), (256, 256, 512), (512, 384, 1024)])
def test_w4a8_gemm_matches_dequantized_reference(m, n, k, rank):
  generator = torch.Generator(device="cuda").manual_seed(0)
  qa = (torch.randn(m, k, device="cuda", generator=generator) / 4).to(torch.float8_e4m3fn)
  qb = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8, generator=generator)
  sa = torch.rand(m, device="cuda", generator=generator) + 0.25
  sb = torch.rand(n, device="cuda", generator=generator) + 0.25
  bias = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=generator)
  d = torch.randn(m, rank, device="cuda", dtype=torch.bfloat16,
                  generator=generator) if rank else None
  up = torch.randn(n, rank, device="cuda", dtype=torch.bfloat16,
                   generator=generator) if rank else None

  out = svdq_gemm_w4a8(qa, qb, sa, sb, lora_act_in=d, lora_up=up, bias=bias)
  weight = dequantize_e2m1(qb, sb)
  ref = qa.float() @ weight.t() * sa[:, None] + bias.float()
  if rank:
    ref = ref + d.float() @ up.float().t()
  rel = ((out.float() - ref).norm() / ref.norm()).item()
  assert rel < 2e-3


def test_w4a8_gemm_all_ones_and_optional_arguments():
  m = n = k = 128
  qa = torch.full((m, k), 0x38, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
  qb = torch.full((n, k // 2), 0x22, dtype=torch.uint8, device="cuda")
  out = svdq_gemm_w4a8(qa, qb, torch.ones(m, device="cuda"), torch.ones(n, device="cuda"))
  assert torch.equal(out, torch.full((m, n), float(k), dtype=torch.bfloat16, device="cuda"))


def test_w4a8_gemm_rejects_bad_packed_shape_and_dtype():
  qa = torch.zeros(128, 128, device="cuda", dtype=torch.float8_e4m3fn)
  qb = torch.zeros(128, 64, device="cuda", dtype=torch.uint8)
  sa = torch.ones(128, device="cuda")
  with pytest.raises(ValueError, match="K/2"):
    svdq_gemm_w4a8(qa, qb[:, :63].contiguous(), sa, sa)
  with pytest.raises(ValueError, match="uint8"):
    svdq_gemm_w4a8(qa, qb.to(torch.int8), sa, sa)


def test_w4a8_producer_matches_w8a8():
  x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
  smooth = torch.rand(512, device="cuda", dtype=torch.bfloat16) + 0.5
  down = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
  actual = svdq_quantize_w4a8(x, smooth, down)
  expected = svdq_quantize_w8a8(x, smooth, down)
  assert torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8))
  assert torch.equal(actual[1], expected[1])
  assert (actual[2] == expected[2]).float().mean().item() > 0.99


@pytest.mark.parametrize("rank", [0, 32, 64, 128])
def test_w4a8_module_matches_dequantized_reference(rank):
  torch.manual_seed(0)
  linear = nn.Linear(320, 192, dtype=torch.bfloat16, device="cuda")
  packed = _quantize_svdq_w4a8(linear, rank=rank, smooth_strategy="weight")
  weight_error = ((packed.dequantize_weight() - linear.weight.float()).norm() /
                  linear.weight.float().norm()).item()
  assert weight_error < 0.25
  x = torch.randn(3, 130, 320, device="cuda", dtype=torch.bfloat16)
  out = packed(x)
  flat = x.reshape(-1, 320)
  qact, ascales, lora_act = svdq_quantize_w4a8(
    flat, packed.smooth_factor[:320], packed.proj_down[:, :320].contiguous() if rank else None)
  weight = dequantize_e2m1(packed.qweight[:192, :160], packed.wscales[:192])
  weight = weight / packed.smooth_factor[:320].float()
  ref = qact.float() @ weight.t() * ascales[:, None] + linear.bias.float()
  if rank:
    ref = ref + lora_act.float() @ packed.proj_up[:192].float().t()
  rel = ((out.float().reshape(-1, 192) - ref.reshape(-1, 192)).norm() / ref.norm()).item()
  assert out.shape == (3, 130, 192)
  assert rel < 5e-3


def test_w4a8_forward_has_no_standalone_bf16_gemm():
  linear = nn.Linear(256, 256, dtype=torch.bfloat16, device="cuda")
  packed = _quantize_svdq_w4a8(linear, rank=32)

  class ForbidMatmul(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
      if func in (torch.ops.aten.mm.default, torch.ops.aten.matmul.default,
                  torch.ops.aten.bmm.default, torch.ops.aten.addmm.default):
        raise AssertionError("W4A8 must not launch a standalone BF16 GEMM")
      return func(*args, **(kwargs or {}))

  with ForbidMatmul():
    assert packed(torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)).shape == (128, 256)


def test_w4a8_fake_and_fullgraph_compile():
  from torch._subclasses.fake_tensor import FakeTensorMode

  with FakeTensorMode():
    act = torch.empty(128, 128, device="cuda", dtype=torch.float8_e4m3fn)
    wgt = torch.empty(128, 64, device="cuda", dtype=torch.uint8)
    scales = torch.empty(128, device="cuda", dtype=torch.float32)
    assert svdq_gemm_w4a8(act, wgt, scales, scales).shape == (128, 128)

  packed = _quantize_svdq_w4a8(nn.Linear(128, 128, dtype=torch.bfloat16, device="cuda"), rank=32)
  compiled = torch.compile(packed, fullgraph=True)
  x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
  torch.testing.assert_close(compiled(x), packed(x), atol=2e-3, rtol=2e-3)


def test_w4a8_real_flux_shape_smoke():
  m, n, k, rank = 4096, 12288, 3072, 32
  qa = (torch.randn(m, k, device="cuda") / 4).to(torch.float8_e4m3fn)
  qb = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
  sa = torch.rand(m, device="cuda") + 0.25
  sb = torch.rand(n, device="cuda") + 0.25
  d = torch.randn(m, rank, device="cuda", dtype=torch.bfloat16)
  up = torch.randn(n, rank, device="cuda", dtype=torch.bfloat16)
  out = svdq_gemm_w4a8(qa, qb, sa, sb, lora_act_in=d, lora_up=up)
  assert out.shape == (m, n) and torch.isfinite(out.float()).all()


def test_w4a8_config_and_unsupported_fused_mlp():
  config = QuantizeConfig(quant_type="svdq_w4a8_r64_dq")
  assert config.get_svdq_precision() == "w4a8"
  assert config.get_svdq_rank() == 64 and config.is_svdq_dq()
  with pytest.raises(ValueError, match="fused_mlp"):
    QuantizeConfig(quant_type="svdq_w4a8_r32_dq", svdq_kwargs={"fused_mlp": True})
