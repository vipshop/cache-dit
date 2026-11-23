import pytest
import torch
from torch import nn

from cache_dit.kernels import cuda_ext_is_available
from cache_dit.kernels import expand_wna4
from cache_dit.kernels import svdq_gemm_w4a4
from cache_dit.kernels import svdq_gemm_wna4
from cache_dit.kernels import svdq_quantize_w4a4
from cache_dit.quantization.svdquant.wNa4 import pack_codes
from cache_dit.quantization.svdquant.wNa4 import _quantize_svdq_wNa4
from cache_dit.quantization.svdquant.wNa4 import unpack_codes


@pytest.mark.parametrize("bits", [1, 2, 3])
def test_packed_codes_roundtrip(bits: int) -> None:
  codes = torch.arange(128, dtype=torch.int64).view(1, 128) % (1 << bits)
  packed = pack_codes(codes, bits)
  if bits == 3:
    actual = unpack_codes(packed[0], bits, packed[1])
  else:
    actual = unpack_codes(packed, bits)
  assert torch.equal(actual.reshape(-1), codes.reshape(-1))


def test_wNa4_preserves_fp8_scale_bytes(monkeypatch) -> None:
  import cache_dit.quantization.svdquant.wNa4 as wNa4

  def fake_decompose(weight, rank, **kwargs):
    n, k = weight.shape
    return (torch.zeros((rank, k), dtype=weight.dtype), torch.zeros(
      (n, rank), dtype=weight.dtype), torch.ones((n, k), dtype=weight.dtype))

  monkeypatch.setattr(wNa4, "decompose_lowrank_residual", fake_decompose)
  linear = nn.Linear(128, 256, dtype=torch.bfloat16)
  module = _quantize_svdq_wNa4(linear, 2, rank=32)
  # A unit residual produces FP8(448) = 0x7e. Numeric FP8 -> uint8 conversion would produce 255.
  assert module.wscales.unique().tolist() == [126]


@pytest.mark.parametrize("bits,limit", [(1, 1.1), (2, 0.55), (3, 0.4)])
def test_wNa4_weight_reconstruction_envelope(bits: int, limit: float) -> None:
  torch.manual_seed(0)
  linear = nn.Linear(128, 256, dtype=torch.bfloat16)
  module = _quantize_svdq_wNa4(linear, bits, rank=32)
  actual = module.dequantize_weight()
  expected = linear.weight.float()
  error = ((actual - expected).norm() / expected.norm()).item()
  assert error < limit


def _expanded_nibbles(module) -> torch.Tensor:
  from cache_dit.quantization.svdquant.wNa4 import _PACKED_A4_3STAGE_NIBBLES

  codes = unpack_codes(module.qweight, module.bits,
                       module.qweight_hi if module.bits == 3 else None).long()
  if module.stages == 3:
    pidx = torch.stack([(module.wmicroscales >> (2 * index)) & 3 for index in range(4)],
                       dim=-1).flatten(-2).long()
    pidx = pidx.repeat_interleave(module.inner_group, dim=-1)
    table = torch.tensor(_PACKED_A4_3STAGE_NIBBLES[module.bits], device=codes.device)
    nibbles = table[pidx, codes]
  elif module.bits == 1:
    nibbles = codes * 8 + 3
  elif module.bits == 2:
    nibbles = (codes >> 1) * 8 + (codes & 1) * 2 + 1
  else:
    nibbles = (codes >> 2) * 8 + (codes & 3) * 2 + 1
  return (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).to(torch.uint8).view(torch.int8)


@pytest.mark.skipif(not cuda_ext_is_available(), reason="requires the cache-dit CUDA extension")
@pytest.mark.parametrize("bits", [1, 2, 3])
def test_packed_gemm_is_bit_exact_with_expanded_w4a4(bits: int) -> None:
  torch.manual_seed(0)
  linear = nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
  module = _quantize_svdq_wNa4(linear, bits, rank=32)
  x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
  qact, ascales, lora_act = svdq_quantize_w4a4(x,
                                               module.proj_down,
                                               module.smooth_factor,
                                               lora_layout="row_major",
                                               fp4=True,
                                               pad_size=256)
  reference = svdq_gemm_w4a4(qact,
                             _expanded_nibbles(module).contiguous(),
                             ascales,
                             module.wscales.view(torch.float8_e4m3fn),
                             lora_act_in=lora_act,
                             lora_up=module.proj_up,
                             bias=module.bias,
                             wcscales=module.wcscales,
                             lora_layout="row_major",
                             runtime="cute")
  actual = svdq_gemm_wna4(qact,
                          module.qweight,
                          module.qweight_hi,
                          ascales,
                          module.wscales.view(torch.float8_e4m3fn),
                          lora_act_in=lora_act,
                          lora_up=module.proj_up,
                          bias=module.bias,
                          wcscales=module.wcscales,
                          wgt_micro=module.wmicroscales,
                          bits=bits)
  assert torch.equal(actual, reference)


@pytest.mark.parametrize("bits", [1, 2])
def test_w12a4_three_stage_reduces_weight_error(bits: int) -> None:
  torch.manual_seed(0)
  linear = nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
  two_stage = _quantize_svdq_wNa4(linear, bits, rank=32, stages=2)
  four_stage = _quantize_svdq_wNa4(linear, bits, rank=32, stages=3)
  two_group = _quantize_svdq_wNa4(linear, bits, rank=32, stages=3, inner_group=2)
  expected = linear.weight.float()
  two_error = ((two_stage.dequantize_weight() - expected).norm() / expected.norm()).item()
  four_error = ((four_stage.dequantize_weight() - expected).norm() / expected.norm()).item()
  two_group_error = ((two_group.dequantize_weight() - expected).norm() / expected.norm()).item()
  assert four_stage.wmicroscales is not None
  assert two_stage.wmicroscales is None
  assert two_group.wmicroscales.shape[1] == four_stage.wmicroscales.shape[1] * 2
  assert four_error < two_error
  assert two_group_error < four_error


@pytest.mark.parametrize("bits", [1, 2])
def test_w12a4_defaults_to_three_stage(bits: int) -> None:
  from cache_dit import QuantizeConfig

  config = QuantizeConfig(quant_type=f"svdq_w{bits}a4_r32_dq")
  assert config.svdq_kwargs["packed_a4_stages"] == 3
  assert config.svdq_kwargs["packed_a4_inner_group"] == 4
  legacy = QuantizeConfig(quant_type=f"svdq_w{bits}a4_r32_dq", svdq_kwargs={"packed_a4_stages": 2})
  assert legacy.svdq_kwargs["packed_a4_stages"] == 2
  module = _quantize_svdq_wNa4(nn.Linear(128, 256, dtype=torch.bfloat16), bits, rank=32)
  assert module.stages == 3 and module.wmicroscales is not None
  assert module.inner_group == 4


def test_w2a4_rejects_reserved_inner_exponent() -> None:
  from cache_dit.quantization.svdquant.wNa4 import SVDQW2A4Linear

  module = SVDQW2A4Linear(128, 256, rank=32, stages=3)
  module.wmicroscales[0, 0] = 3
  with pytest.raises(ValueError, match="reserved exponent"):
    module.dequantize_weight()
  restored = SVDQW2A4Linear(128, 256, rank=32, stages=3)
  with pytest.raises(ValueError, match="reserved exponent"):
    restored.load_state_dict(module.state_dict())


@pytest.mark.skipif(not cuda_ext_is_available(), reason="requires the cache-dit CUDA extension")
@pytest.mark.parametrize("bits", [1, 2])
@pytest.mark.parametrize("inner_group", [4, 2])
def test_w12a4_three_stage_bit_layout_is_independent(bits: int, inner_group: int) -> None:
  rows, k = 16, 64
  codes = torch.arange(rows * k, device="cuda").view(rows, k)
  codes = (codes if bits == 1 else codes % 4).to(torch.uint8)
  allowed = 4 if bits == 1 else 3
  sub = (torch.arange(k // inner_group, device="cuda") % allowed).view(1, -1).expand(rows,
                                                                                     -1).long()
  grouped = sub.view(rows, -1, 4)
  micro = torch.zeros((rows, grouped.shape[1]), dtype=torch.uint8, device="cuda")
  for index in range(4):
    micro |= grouped[..., index].to(torch.uint8) << (2 * index)
  pidx = sub.repeat_interleave(inner_group, dim=-1)
  if bits == 1:
    nibbles = 1 + 2 * pidx + 8 * (codes.long() & 1)
  else:
    base = codes.long() & 1
    pidx0 = torch.where(base == 0, 1, 3)
    pidx1 = torch.where(base == 0, 2, 5)
    pidx2 = torch.where(base == 0, 4, 7)
    magnitude = torch.where(pidx == 0, pidx0, torch.where(pidx == 1, pidx1, pidx2))
    nibbles = magnitude + 8 * ((codes.long() >> 1) & 1)
  expected = (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).to(torch.uint8).view(torch.int8)
  packed = pack_codes(codes, bits)
  actual = expand_wna4(packed, None, bits, micro, inner_group=inner_group)
  assert torch.equal(actual, expected)
  exponents = torch.stack([(micro >> (2 * index)) & 3 for index in range(4)]).flatten()
  assert exponents.unique().numel() == allowed


@pytest.mark.skipif(not cuda_ext_is_available(), reason="requires the cache-dit CUDA extension")
@pytest.mark.parametrize("bits", [1, 2])
@pytest.mark.parametrize("inner_group", [4, 2])
def test_w12a4_three_stage_matches_expanded_w4a4(bits: int, inner_group: int) -> None:
  torch.manual_seed(0)
  linear = nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
  module = _quantize_svdq_wNa4(linear, bits, rank=32, stages=3, inner_group=inner_group)
  x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
  qact, ascales, lora_act = svdq_quantize_w4a4(x,
                                               module.proj_down,
                                               module.smooth_factor,
                                               lora_layout="row_major",
                                               fp4=True,
                                               pad_size=256)
  expanded = expand_wna4(module.qweight,
                         module.qweight_hi,
                         bits,
                         module.wmicroscales,
                         inner_group=inner_group)
  assert torch.equal(expanded, _expanded_nibbles(module))
  reference = svdq_gemm_w4a4(qact,
                             expanded,
                             ascales,
                             module.wscales.view(torch.float8_e4m3fn),
                             lora_act_in=lora_act,
                             lora_up=module.proj_up,
                             bias=module.bias,
                             wcscales=module.wcscales,
                             lora_layout="row_major",
                             runtime="cute")
  actual = svdq_gemm_wna4(qact,
                          module.qweight,
                          module.qweight_hi,
                          ascales,
                          module.wscales.view(torch.float8_e4m3fn),
                          lora_act_in=lora_act,
                          lora_up=module.proj_up,
                          bias=module.bias,
                          wcscales=module.wcscales,
                          wgt_micro=module.wmicroscales,
                          inner_group=inner_group,
                          bits=bits)
  assert torch.equal(actual, reference)
  assert torch.equal(module(x), reference)


@pytest.mark.skipif(not cuda_ext_is_available(), reason="requires the cache-dit CUDA extension")
@pytest.mark.parametrize("bits", [1, 2])
@pytest.mark.parametrize("rank", [256, 512])
def test_w12a4_high_rank_matches_expanded_w4a4(bits: int, rank: int) -> None:
  torch.manual_seed(0)
  linear = nn.Linear(rank, rank, dtype=torch.bfloat16, device="cuda")
  module = _quantize_svdq_wNa4(linear, bits, rank=rank)
  assert module.stages == 3
  x = torch.randn(256, rank, device="cuda", dtype=torch.bfloat16)
  qact, ascales, lora_act = svdq_quantize_w4a4(x,
                                               module.proj_down,
                                               module.smooth_factor,
                                               lora_layout="row_major",
                                               fp4=True,
                                               pad_size=256)
  reference = svdq_gemm_w4a4(qact,
                             _expanded_nibbles(module).contiguous(),
                             ascales,
                             module.wscales.view(torch.float8_e4m3fn),
                             lora_act_in=lora_act,
                             lora_up=module.proj_up,
                             bias=module.bias,
                             wcscales=module.wcscales,
                             lora_layout="row_major",
                             runtime="cute")
  actual = svdq_gemm_wna4(qact,
                          module.qweight,
                          module.qweight_hi,
                          ascales,
                          module.wscales.view(torch.float8_e4m3fn),
                          lora_act_in=lora_act,
                          lora_up=module.proj_up,
                          bias=module.bias,
                          wcscales=module.wcscales,
                          wgt_micro=module.wmicroscales,
                          bits=bits)
  assert module.proj_down.shape == (rank, rank)
  assert module.proj_up.shape == (rank, rank)
  assert torch.equal(actual, reference)
  assert torch.equal(module(x), reference)


@pytest.mark.skipif(not cuda_ext_is_available(), reason="requires the cache-dit CUDA extension")
@pytest.mark.parametrize("bits", [1, 2, 3])
def test_module_expansion_path_matches_w4a4(bits: int) -> None:
  torch.manual_seed(0)
  linear = nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
  module = _quantize_svdq_wNa4(linear, bits, rank=32)
  x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
  qact, ascales, lora_act = svdq_quantize_w4a4(x,
                                               module.proj_down,
                                               module.smooth_factor,
                                               lora_layout="row_major",
                                               fp4=True,
                                               pad_size=256)
  reference = svdq_gemm_w4a4(qact,
                             _expanded_nibbles(module).contiguous(),
                             ascales,
                             module.wscales.view(torch.float8_e4m3fn),
                             lora_act_in=lora_act,
                             lora_up=module.proj_up,
                             bias=module.bias,
                             wcscales=module.wcscales,
                             lora_layout="row_major",
                             runtime="cute")
  assert torch.equal(module(x), reference)


def _dequantize_activations(qact: torch.Tensor, ascales: torch.Tensor) -> torch.Tensor:
  rows = torch.arange(qact.shape[0],
                      device=qact.device).view(-1, 1).expand(qact.shape[0], qact.shape[1] * 2)
  cols = torch.arange(qact.shape[1] * 2,
                      device=qact.device).view(1, -1).expand(qact.shape[0], qact.shape[1] * 2)
  kt = cols // 64
  kk = cols % 64
  local = 16 * (rows % 8) + ((rows % 16) // 8) + 128 * ((rows % 32) // 16)
  base = ((rows // 256) * (qact.shape[1] // 32) + kt) * 8 + (rows % 256) // 32
  word_index = base * 256 + local + (kk // 32) * 2 + ((kk % 32) // 8) * 4
  words = qact.contiguous().view(torch.int32).flatten()[word_index]
  codes = ((words >> ((kk % 8) * 4)) & 0xF).long()
  scale_lane = 4 * (rows % 8) + (rows % 32) // 8
  sf_flat = ascales.contiguous().view(torch.uint8).flatten()
  sf_bytes = sf_flat[(base * 32 + scale_lane) * 4 + (kk // 16)]
  table = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    device=qact.device)
  return table[codes] * sf_bytes.view(torch.float8_e4m3fn).float()


@pytest.mark.skipif(not cuda_ext_is_available(), reason="requires the cache-dit CUDA extension")
@pytest.mark.parametrize("bits", [1, 2, 3])
def test_wNa4_forward_matches_dequant_reference(bits: int) -> None:
  torch.manual_seed(0)
  linear = nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
  module = _quantize_svdq_wNa4(linear, bits, rank=32)
  x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
  qact, ascales, lora_act = svdq_quantize_w4a4(
    x,
    module.proj_down,
    module.smooth_factor,
    lora_layout="row_major",
    fp4=True,
    pad_size=256,
  )
  activation = _dequantize_activations(qact, ascales)
  residual = module.dequantize_weight()
  lora_term = 0.0
  if module.rank:
    # The producer computes LoRA-down from the original input, not from the FP4 activation.
    residual = residual - module.proj_up.float() @ module.proj_down.float().t()
    lora_term = lora_act.float() @ module.proj_up.float().t()
  expected = activation @ residual.t() + lora_term + module.bias.float()
  actual = module(x).float()
  relative_error = ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()
  assert relative_error < 2e-3


@pytest.mark.parametrize("bits", [1, 2, 3])
def test_wNa4_zero_group_quantization_is_finite(bits: int) -> None:
  from cache_dit.quantization.svdquant.wNa4 import quantize_residual

  packed, scales, coarse = quantize_residual(torch.zeros(256, 128), bits)
  assert torch.isfinite(coarse.float()).all()
  assert scales.view(torch.uint8).max().item() == 0
  codes = unpack_codes(packed[0], bits, packed[1]) if bits == 3 else unpack_codes(packed, bits)
  assert codes.max().item() == 0


@pytest.mark.skipif(not cuda_ext_is_available(), reason="requires the cache-dit CUDA extension")
@pytest.mark.parametrize("bits", [1, 2, 3])
@pytest.mark.parametrize("mode", [None, "reduce-overhead"])
def test_wNa4_fullgraph_compile(bits: int, mode: str | None) -> None:
  linear = nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
  module = _quantize_svdq_wNa4(linear, bits, rank=32)
  x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
  expected = module(x)
  compiled = torch.compile(module, fullgraph=True, mode=mode)
  for _ in range(3):
    actual = compiled(x)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not cuda_ext_is_available(), reason="requires the cache-dit CUDA extension")
@pytest.mark.parametrize("bits", [1, 2])
@pytest.mark.parametrize("inner_group", [4, 2])
def test_w12a4_three_stage_fullgraph_compile(bits: int, inner_group: int) -> None:
  linear = nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
  module = _quantize_svdq_wNa4(linear, bits, rank=32, stages=3, inner_group=inner_group)
  x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
  expected = module(x)
  torch._dynamo.reset()
  compiled = torch.compile(module, fullgraph=True)
  assert torch.equal(compiled(x), expected)


@pytest.mark.skipif(not cuda_ext_is_available(), reason="requires the cache-dit CUDA extension")
@pytest.mark.parametrize("bits", [1, 2])
@pytest.mark.parametrize("rank", [256, 512])
def test_w12a4_high_rank_fullgraph_compile(bits: int, rank: int) -> None:
  linear = nn.Linear(rank, rank, dtype=torch.bfloat16, device="cuda")
  module = _quantize_svdq_wNa4(linear, bits, rank=rank)
  x = torch.randn(256, rank, device="cuda", dtype=torch.bfloat16)
  expected = module(x)
  torch._dynamo.reset()
  compiled = torch.compile(module, fullgraph=True)
  assert torch.equal(compiled(x), expected)


@pytest.mark.parametrize("bits", [1, 2, 3])
def test_wNa4_dq_checkpoint_roundtrip(bits: int, tmp_path) -> None:
  import copy

  import cache_dit
  from cache_dit import QuantizeConfig
  from cache_dit.quantization.svdquant.wNa4 import SVDQWNA4Linear

  model = nn.Sequential(nn.Linear(130, 192, bias=False, dtype=torch.bfloat16),
                        nn.Linear(192, 130, dtype=torch.bfloat16))
  fresh = copy.deepcopy(model)
  config = QuantizeConfig(quant_type=f"svdq_w{bits}a4_r32_dq",
                          serialize_to=str(tmp_path / "checkpoint"),
                          svdq_kwargs={"quantize_device": "cpu"})
  quantized = cache_dit.quantize(model, config)
  loaded = cache_dit.load(fresh, str(tmp_path / "checkpoint"))
  assert all(isinstance(layer, SVDQWNA4Linear) for layer in loaded)
  for actual, expected in zip(loaded, quantized):
    for key, value in actual.state_dict().items():
      assert torch.equal(value, expected.state_dict()[key]), key


@pytest.mark.parametrize("bits", [1, 2])
def test_w12a4_rank512_dq_checkpoint_roundtrip(bits: int, tmp_path) -> None:
  import copy

  import cache_dit
  from cache_dit import QuantizeConfig
  from cache_dit.quantization.svdquant.wNa4 import SVDQWNA4Linear

  model = nn.Sequential(nn.Linear(512, 512, bias=False, dtype=torch.bfloat16))
  fresh = copy.deepcopy(model)
  config = QuantizeConfig(quant_type=f"svdq_w{bits}a4_r512_dq",
                          serialize_to=str(tmp_path / "checkpoint"),
                          svdq_kwargs={"quantize_device": "cpu"})
  quantized = cache_dit.quantize(model, config)
  loaded = cache_dit.load(fresh, str(tmp_path / "checkpoint"))
  assert isinstance(loaded[0], SVDQWNA4Linear)
  assert loaded[0].rank == 512
  for key, value in loaded[0].state_dict().items():
    assert torch.equal(value, quantized[0].state_dict()[key]), key


def test_w3a4_rejects_rank512() -> None:
  from cache_dit import QuantizeConfig

  with pytest.raises(ValueError, match="supports rank"):
    QuantizeConfig(quant_type="svdq_w3a4_r512_dq")


@pytest.mark.parametrize("bits", [1, 2])
@pytest.mark.parametrize("inner_group", [4, 2])
def test_w12a4_three_stage_checkpoint_roundtrip(bits: int, inner_group: int, tmp_path) -> None:
  import copy

  import cache_dit
  from cache_dit import QuantizeConfig
  from cache_dit.quantization.svdquant.wNa4 import SVDQWNA4Linear

  model = nn.Sequential(nn.Linear(130, 192, bias=False, dtype=torch.bfloat16))
  fresh = copy.deepcopy(model)
  config = QuantizeConfig(quant_type=f"svdq_w{bits}a4_r32_dq",
                          serialize_to=str(tmp_path / "checkpoint"),
                          svdq_kwargs={
                            "quantize_device": "cpu",
                            "packed_a4_stages": 3,
                            "packed_a4_inner_group": inner_group,
                          })
  quantized = cache_dit.quantize(model, config)
  loaded = cache_dit.load(fresh, str(tmp_path / "checkpoint"))
  assert isinstance(loaded[0], SVDQWNA4Linear)
  assert loaded[0].stages == 3
  assert loaded[0].inner_group == inner_group
  assert loaded[0].wmicroscales.shape == quantized[0].wmicroscales.shape
  for key, value in loaded[0].state_dict().items():
    assert torch.equal(value, quantized[0].state_dict()[key]), key


def test_three_stage_rejects_w3a4() -> None:
  from cache_dit import QuantizeConfig

  with pytest.raises(ValueError, match="packed_a4_stages"):
    QuantizeConfig(quant_type="svdq_w3a4_r32_dq", svdq_kwargs={"packed_a4_stages": 3})


def test_inner_group_two_requires_three_stage_w1_w2() -> None:
  from cache_dit import QuantizeConfig

  config = QuantizeConfig(quant_type="svdq_w2a4_r32_dq", svdq_kwargs={"packed_a4_inner_group": 2})
  assert config.svdq_kwargs["packed_a4_inner_group"] == 2
  with pytest.raises(ValueError, match="packed_a4_inner_group"):
    QuantizeConfig(quant_type="svdq_w3a4_r32_dq", svdq_kwargs={"packed_a4_inner_group": 2})
  with pytest.raises(ValueError, match="requires"):
    QuantizeConfig(quant_type="svdq_w2a4_r32_dq",
                   svdq_kwargs={
                     "packed_a4_stages": 2,
                     "packed_a4_inner_group": 2,
                   })


def test_packed_a4_cli_shortcuts() -> None:
  from cache_dit.cli.utils import get_args

  parser = get_args(parse=False)
  args = parser.parse_args(
    ["--svdq-w2a4-r256-dq", "--svdq-packed-a4-stages", "3", "--svdq-packed-a4-inner-group", "2"])
  assert args.svdq_w2a4_r256_dq
  assert args.svdq_packed_a4_stages == 3
  assert args.svdq_packed_a4_inner_group == 2
  defaults = parser.parse_args([])
  assert defaults.svdq_packed_a4_stages is None
  assert defaults.svdq_packed_a4_inner_group is None


@pytest.mark.parametrize("rank", [144, 512])
def test_standard_w4a4_rejects_unscheduled_rank(rank: int) -> None:
  from cache_dit.quantization.svdquant.quantizer import validate_linear_geometry

  with pytest.raises(ValueError, match="runtime requires rank"):
    validate_linear_geometry(256, 256, rank=rank, precision="nvfp4")


@pytest.mark.parametrize("bits", [1, 2, 3])
def test_wNa4_ptq_partial_calibration_checkpoint(bits: int, tmp_path) -> None:
  import copy

  import cache_dit
  from cache_dit import QuantizeConfig

  model = nn.Sequential(nn.Linear(130, 192, dtype=torch.bfloat16),
                        nn.Linear(192, 130, dtype=torch.bfloat16))
  fresh = copy.deepcopy(model)

  def calibrate_fn(module) -> None:
    module[0](torch.randn(4, 130, dtype=torch.bfloat16))

  config = QuantizeConfig(quant_type=f"svdq_w{bits}a4_r32",
                          calibrate_fn=calibrate_fn,
                          serialize_to=str(tmp_path / "ptq"),
                          svdq_kwargs={"quantize_device": "cpu"})
  quantized = cache_dit.quantize(model, config)
  assert isinstance(quantized[1], nn.Linear)
  loaded = cache_dit.load(fresh, str(tmp_path / "ptq"))
  assert isinstance(loaded[1], nn.Linear)
  for key, value in loaded[0].state_dict().items():
    assert torch.equal(value, quantized[0].state_dict()[key]), key
