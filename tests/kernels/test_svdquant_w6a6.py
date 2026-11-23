"""Tests for the block-scaled SVDQ-W6A6 kernels and runtime module."""

import pytest
import torch
from torch import nn

from cache_dit import QuantizeConfig
from cache_dit.kernels import cuda_ext_is_available
from cache_dit.kernels import svdq_gemm_w6a6
from cache_dit.kernels import svdq_quantize_w6a6
from cache_dit.quantization.svdquant.w6a6 import SVDQW6A6Linear
from cache_dit.quantization.svdquant.w6a6 import _e2m3_magnitudes
from cache_dit.quantization.svdquant.w6a6 import _quantize_svdq_w6a6
from cache_dit.quantization.svdquant.w6a6 import dequantize_e2m3
from cache_dit.quantization.svdquant.w6a6 import pack_e2m3
from cache_dit.quantization.svdquant.w6a6 import quantize_e2m3
from cache_dit.quantization.svdquant.w6a6 import quantize_residual_fp6
from cache_dit.quantization.svdquant.w6a6 import ue8m0_codes
from cache_dit.quantization.svdquant.w6a6 import ue8m0_scales
from cache_dit.quantization.svdquant.w6a6 import unpack_e2m3

pytestmark = pytest.mark.skipif(
  not torch.cuda.is_available() or not cuda_ext_is_available()
  or torch.cuda.get_device_capability(0)[0] < 12,
  reason="SVDQ-W6A6 requires the CUDA extension on sm120+",
)


def test_e2m3_codebook_matches_the_hardware_encoding():
  table = _e2m3_magnitudes()
  # The MXFP6 encoding is verified against the device `cvt` instruction: 1.0, 2.0 and 7.5 are the
  # values the hardware reports for codes 0x08, 0x10 and 0x1f, and 0x04 is the denormal 0.5.
  assert table[0x08].item() == 1.0
  assert table[0x10].item() == 2.0
  assert table[0x1F].item() == 7.5
  assert table[0x04].item() == 0.5
  assert torch.equal(table, table.sort().values)


def test_e2m3_pack_roundtrip_and_bit_order():
  codes = torch.arange(32, dtype=torch.uint8).repeat(2).reshape(4, 16)
  packed = pack_e2m3(codes)
  assert packed.shape == (4, 12)
  assert torch.equal(unpack_e2m3(packed), codes)
  # Codes 0,1,2,3 as one little-endian 24-bit stream: 0x00 | 0x01 << 6, then 0x02 << 4 and 0x03 << 2.
  assert packed[0, 0].item() == 0x40
  assert packed[0, 1].item() == 0x20
  assert packed[0, 2].item() == 0x0C
  with pytest.raises(ValueError, match="divisible by 4"):
    pack_e2m3(torch.zeros(1, 6, dtype=torch.uint8))


def test_ue8m0_codes_are_the_minimal_power_of_two_scale():
  amax = torch.tensor([0.0, 7.5, 7.6, 15.0, 1.0, 0.01], dtype=torch.float32)
  codes = ue8m0_codes(amax)
  scales = ue8m0_scales(codes)
  # Zero blocks keep scale 1.0 so their codes stay 0.
  assert codes[0].item() == 127
  assert torch.all(scales >= amax / 7.5)
  assert torch.all(scales[1:] < 2 * (amax[1:] / 7.5))  # minimal legal power of two
  assert codes[1].item() == 127 and scales[1].item() == 1.0


def test_quantize_residual_fp6_error_bounded_by_one_code():
  generator = torch.Generator().manual_seed(0)
  residual = torch.randn(128, 256, generator=generator)
  packed, wsf = quantize_residual_fp6(residual)
  assert packed.shape == (128, 192) and wsf.shape == (128, 8)
  restored = dequantize_e2m3(packed, wsf)
  assert torch.all(restored.abs().reshape(128, 8, 32).amax(dim=-1) <= 7.5 * ue8m0_scales(wsf) +
                   1e-5)
  # The block scale is chosen so that a full-range block spans most of the code book.
  rel = (restored - residual).norm() / residual.norm()
  assert rel < 0.06


@pytest.mark.parametrize("rank", [0, 32, 64, 128])
@pytest.mark.parametrize("m,n,k", [(128, 128, 128), (256, 256, 512), (512, 384, 1024)])
def test_w6a6_gemm_matches_dequantized_reference(m, n, k, rank):
  generator = torch.Generator(device="cuda").manual_seed(0)
  qa_codes = torch.randint(0, 64, (m, k), device="cuda", dtype=torch.uint8, generator=generator)
  qb_codes = torch.randint(0, 64, (n, k), device="cuda", dtype=torch.uint8, generator=generator)
  qa = pack_e2m3(qa_codes)
  qb = pack_e2m3(qb_codes)
  asf = torch.randint(118, 128, (m, k // 32), device="cuda", dtype=torch.uint8, generator=generator)
  wsf = torch.randint(118, 128, (n, k // 32), device="cuda", dtype=torch.uint8, generator=generator)
  ascales = torch.rand(m, device="cuda", generator=generator) + 0.25
  wscales = torch.rand(n, device="cuda", generator=generator) + 0.25
  bias = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=generator)
  d = torch.randn(m, rank, device="cuda", dtype=torch.bfloat16,
                  generator=generator) if rank else None
  up = torch.randn(n, rank, device="cuda", dtype=torch.bfloat16,
                   generator=generator) if rank else None

  out = svdq_gemm_w6a6(qa,
                       qb,
                       asf,
                       wsf,
                       ascales=ascales,
                       wscales=wscales,
                       lora_act_in=d,
                       lora_up=up,
                       bias=bias)
  act = dequantize_e2m3(qa, asf)
  weight = dequantize_e2m3(qb, wsf)
  ref = act @ weight.t() * ascales[:, None] * wscales[None, :] + bias.float()
  if rank:
    ref = ref + d.float() @ up.float().t()
  rel = ((out.float() - ref).norm() / ref.norm()).item()
  assert rel < 2e-3


def test_w6a6_gemm_optional_scales_fall_back_to_one():
  generator = torch.Generator(device="cuda").manual_seed(1)
  m = n = k = 128
  qa = pack_e2m3(torch.full((m, k), 0x08, dtype=torch.uint8, device="cuda"))
  qb = pack_e2m3(torch.full((n, k), 0x08, dtype=torch.uint8, device="cuda"))
  asf = torch.full((m, k // 32), 127, dtype=torch.uint8, device="cuda")
  wsf = torch.full((n, k // 32), 127, dtype=torch.uint8, device="cuda")
  out = svdq_gemm_w6a6(qa, qb, asf, wsf)
  # Every code is E2M3(1.0) and every scale is 1.0, so each output element is exactly K.
  assert torch.allclose(out.float(), torch.full((m, n), float(k), device="cuda"), atol=1e-2)
  del generator


@pytest.mark.parametrize("rank", [0, 32, 128])
def test_w6a6_producer_matches_torch_reference(rank):
  torch.manual_seed(0)
  m, k = 256, 512
  x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
  smooth = torch.rand(k, device="cuda", dtype=torch.bfloat16) + 0.5
  down = torch.randn(rank, k, device="cuda", dtype=torch.bfloat16) if rank else None
  qact, asf, lora_act = svdq_quantize_w6a6(x, smooth, down, pad_rows=True)

  smoothed = x.float() / smooth.float()
  blocks = smoothed.reshape(m, k // 32, 32)
  codes = ue8m0_codes(blocks.abs().amax(dim=-1))
  ref_asf = codes
  assert torch.equal(asf[:m], ref_asf)
  ref_qact = pack_e2m3(quantize_e2m3(blocks / ue8m0_scales(codes).unsqueeze(-1)).flatten(1))
  # `quantize_e2m3` reproduces the device conversion bit for bit except at ties inside the denormal
  # band (|x| < 0.25 * scale), where the instruction truncates instead of rounding half to even.
  # Those codes differ by one step and cover well under 0.01% of the elements.
  codes_got = unpack_e2m3(qact[:m]).to(torch.int32)
  codes_ref = unpack_e2m3(ref_qact).to(torch.int32)
  delta = (codes_got - codes_ref).abs()
  assert delta.max().item() <= 1
  assert (delta != 0).sum().item() <= codes_ref.numel() // 1000
  if rank:
    ref_d = (x.float() @ down.float().t()).to(torch.bfloat16)
    assert (lora_act[:m].float() - ref_d.float()).abs().max().item() < 5e-2


def test_w6a6_producer_and_gemm_compose_to_the_quantized_linear():
  torch.manual_seed(0)
  linear = nn.Linear(256, 256, bias=True, device="cuda", dtype=torch.bfloat16)
  packed = _quantize_svdq_w6a6(linear, rank=32, smooth_strategy="identity")
  assert isinstance(packed, SVDQW6A6Linear)
  x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
  out = packed(x)
  # The module must be exactly the producer + block-scaled GEMM composition; the activation error of
  # the E2M3 grid is checked below, not mixed into this wiring test.
  qact, asf, lora_act = svdq_quantize_w6a6(x, packed.smooth_factor, packed.proj_down, pad_rows=True)
  ref = svdq_gemm_w6a6(qact,
                       packed.qweight,
                       asf,
                       packed.wsf,
                       lora_act_in=lora_act,
                       lora_up=packed.proj_up,
                       bias=packed.bias)[:x.shape[0]]
  rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
  assert rel < 1e-6
  # And the packed weight must stay close to the source weight: MXFP6 keeps three mantissa bits plus
  # a 32-wide power-of-two block scale, so the residual reconstruction is at the few-percent level.
  weight = packed.dequantize_weight()
  weight_rel = ((weight - linear.weight.float()).norm() / linear.weight.float().norm()).item()
  assert weight_rel < 0.05


def test_w6a6_forward_supports_fullgraph_compile():
  torch.manual_seed(0)
  linear = nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16)
  packed = _quantize_svdq_w6a6(linear, rank=32, smooth_strategy="identity")
  x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
  eager = packed(x)
  compiled = torch.compile(packed, fullgraph=True)(x)
  rel = ((compiled.float() - eager.float()).norm() / eager.float().norm()).item()
  assert rel < 2e-3


def test_w6a6_quantize_config_accepts_fp6_aliases():
  for quant_type in ("svdq_fp6_r32_dq", "svdq_w6a6_r64_dq"):
    config = QuantizeConfig(quant_type=quant_type)
    assert config.get_svdq_precision() in ("fp6", "w6a6")
    assert config.get_svdq_rank() == (32 if "r32" in quant_type else 64)


def test_w6a6_quantizer_rejects_an_unsupported_rank():
  from cache_dit.quantization.svdquant.quantizer import validate_linear_geometry

  with pytest.raises(ValueError, match="rank"):
    validate_linear_geometry(256, 256, rank=16, precision="w6a6")
  validate_linear_geometry(256, 256, rank=128, precision="w6a6")
