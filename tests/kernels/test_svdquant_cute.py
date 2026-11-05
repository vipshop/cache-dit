from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cache_dit.kernels import svdq_quantize_w4a4_act_fuse_lora
from cache_dit.kernels import svdq_quantize_w4a4_cute
from cache_dit.quantization.svdquant.linear import SVDQW4A4Linear
from cache_dit.quantization.svdquant.quantizer import quantize_linear_svdq_w4a4


def _require_sm120() -> None:
  if not torch.cuda.is_available():
    pytest.skip("CUDA is required")
  if torch.cuda.get_device_capability(0)[0] < 12:
    pytest.skip("SM120 is required")


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
  return ((actual.float() - expected.float()).norm() /
          expected.float().norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize("rank,scales", [(32, [0.5, 2.0]), (128, [1.0] * 8)])
def test_cute_quantize_producer(rank: int, scales: list[float]) -> None:
  _require_sm120()
  torch.manual_seed(1234)
  m, k = 256, 128
  x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda:0")
  smooth = torch.rand(k, dtype=torch.bfloat16, device="cuda:0") + 0.5
  lora_down = torch.randn(k, rank, dtype=torch.bfloat16, device="cuda:0") * 0.02
  qact, ascales, lora_act = svdq_quantize_w4a4_cute(x, lora_down, smooth, scales)

  expected_d = (x.float() @ lora_down.float()) * torch.tensor(scales,
                                                              device=x.device).repeat_interleave(16)
  assert qact.dtype == torch.uint8 and qact.shape == (m, k // 2)
  assert ascales.dtype == torch.float8_e4m3fn
  assert lora_act.dtype == torch.bfloat16 and lora_act.shape == (m, rank)
  assert _relative_l2(lora_act, expected_d) < 0.01

  rows = torch.arange(m, device=x.device).view(-1, 1).expand(m, k)
  cols = torch.arange(k, device=x.device).view(1, -1).expand(m, k)
  kt = cols // 64
  kk = cols % 64
  local = 16 * (rows % 8) + ((rows % 16) // 8) + 128 * ((rows % 32) // 16)
  base = ((rows // 256) * (k // 64) + kt) * 8 + (rows % 256) // 32
  word_index = base * 256 + local + (kk // 32) * 2 + ((kk % 32) // 8) * 4
  word = qact.view(torch.int32).flatten()[word_index]
  signed = ((word >> ((kk % 8) * 4)) & 0xF).long()
  scale_lane = 4 * (rows % 8) + (rows % 32) // 8
  sf_byte = ascales.view(torch.uint8).flatten()[(base * 32 + scale_lane) * 4 + (kk // 16)]
  sf = sf_byte.view(torch.float8_e4m3fn).float()
  table = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    device=x.device,
  )
  dequant = table[signed] * sf
  assert _relative_l2(dequant, x.float() / smooth.float()) < 0.12


def test_cute_producer_matches_native_v1_pack() -> None:
  _require_sm120()
  torch.manual_seed(20261110)
  m, k = 256, 128
  x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda:0")
  smooth = torch.ones(k, dtype=torch.bfloat16, device="cuda:0")
  q_native, sf_native, _ = svdq_quantize_w4a4_act_fuse_lora(x, None, smooth, False, True, 256)
  q_cute, sf_cute, _ = svdq_quantize_w4a4_cute(x, None, smooth, [1.0], 256)
  assert torch.equal(q_cute, q_native)
  # native stores scales as [K/16, M], while the v1 pack is flat; the values must match.
  assert torch.equal(
    sf_cute.view(torch.uint8).flatten().sort().values,
    sf_native.view(torch.uint8).flatten().sort().values)


def test_cute_producer_rejects_non_256_padded_batch() -> None:
  _require_sm120()
  x = torch.randn(288, 128, dtype=torch.bfloat16, device="cuda:0")
  smooth = torch.ones(128, dtype=torch.bfloat16, device="cuda:0")
  with pytest.raises(RuntimeError, match="multiple of 256"):
    svdq_quantize_w4a4_cute(x, None, smooth, [1.0], 128)


def test_cute_gemm_matches_native_runtime() -> None:
  _require_sm120()
  torch.manual_seed(20261112)
  in_features = out_features = 512
  rank = 32
  linear = nn.Linear(in_features, out_features, bias=True).cuda().to(torch.bfloat16)
  calibration = torch.randn(256, in_features, dtype=torch.bfloat16, device="cuda:0")
  state_dict = quantize_linear_svdq_w4a4(
    linear,
    [calibration],
    rank=rank,
    precision="nvfp4",
    device="cuda:0",
    torch_dtype=torch.bfloat16,
    return_state_dict=True,
  )
  native = SVDQW4A4Linear(
    in_features,
    out_features,
    rank=rank,
    precision="nvfp4",
    runtime_kernel="native",
    torch_dtype=torch.bfloat16,
  ).cuda().eval()
  cute = SVDQW4A4Linear(
    in_features,
    out_features,
    rank=rank,
    precision="nvfp4",
    runtime_kernel="cute",
    torch_dtype=torch.bfloat16,
  ).cuda().eval()
  native.load_state_dict(state_dict)
  cute.load_state_dict(state_dict)
  x = torch.randn(256, in_features, dtype=torch.bfloat16, device="cuda:0")
  with torch.inference_mode():
    y_native = native(x)
    y_cute = cute(x)
  assert torch.isfinite(y_native).all() and torch.isfinite(y_cute).all()
  assert _relative_l2(y_cute, y_native) < 5e-3


def test_cute_pack_cache_preserves_v1_state_dict() -> None:
  module = SVDQW4A4Linear(
    128,
    128,
    rank=32,
    precision="nvfp4",
    runtime_kernel="cute",
    torch_dtype=torch.bfloat16,
  )
  with torch.no_grad():
    module.qweight.copy_(torch.randint(-128, 127, module.qweight.shape, dtype=torch.int8))
    module.wscales.copy_(
      torch.arange(module.wscales.numel(), dtype=torch.float32).view(module.wscales.shape) * 1e-4)
    module.proj_up.copy_(
      torch.arange(module.proj_up.numel(), dtype=torch.float32).view(module.proj_up.shape) * 1e-4)
    module.bias.copy_(torch.arange(module.bias.numel(), dtype=torch.float32) * 1e-4)
    module.wcscales.copy_(torch.arange(module.wcscales.numel(), dtype=torch.float32) * 1e-4)
    module.smooth_factor.fill_(1.0)
    module.smooth_factor_orig.fill_(1.0)
  original = {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}
  requires_grad = module.bias.requires_grad
  module._prepare_cute_pack()
  assert module._cute_pack_ready
  assert module._cute_pack_version == 1
  assert module.bias.requires_grad == requires_grad
  for name, tensor in module.state_dict().items():
    assert torch.equal(tensor.view(torch.int16), original[name].view(torch.int16)), name

  reloaded = SVDQW4A4Linear(128, 128, rank=32, precision="nvfp4", runtime_kernel="cute")
  reloaded.load_state_dict(module.state_dict())
  assert not reloaded._cute_pack_ready
  for name, tensor in reloaded.state_dict().items():
    assert torch.equal(tensor.view(torch.int16), original[name].view(torch.int16)), name


def test_cute_runtime_validation() -> None:
  with pytest.raises(ValueError, match="only supported for precision='nvfp4'"):
    SVDQW4A4Linear(128, 128, precision="int4", runtime_kernel="cute")
  with pytest.raises(ValueError, match="torch.bfloat16"):
    SVDQW4A4Linear(128, 128, precision="nvfp4", runtime_kernel="cute", torch_dtype=torch.float16)


def test_runtime_kernel_auto_selection_and_deprecated_values() -> None:
  assert SVDQW4A4Linear(128, 128, precision="int4").runtime_kernel == "native"
  assert SVDQW4A4Linear(128, 128, precision="nvfp4").runtime_kernel == "native"
  assert SVDQW4A4Linear(128, 128, precision="nvfp4", runtime_kernel="cute").runtime_kernel == "cute"
  for deprecated in ("v1", "v2"):
    with pytest.raises(ValueError, match="deprecated"):
      SVDQW4A4Linear(128, 128, runtime_kernel=deprecated)


@pytest.mark.parametrize("runtime_kernel", [None, "native"])
def test_native_int4_dispatches_native_v2(monkeypatch, runtime_kernel: str | None) -> None:
  import cache_dit.quantization.svdquant.linear as linear_module

  module = SVDQW4A4Linear(128, 128, rank=0, precision="int4", runtime_kernel=runtime_kernel)
  sentinel = object()

  def unexpected_native_v1(**kwargs):
    raise AssertionError("INT4 native must not call deprecated native v1")

  monkeypatch.setattr(linear_module, "svdq_gemm_w4a4_native", unexpected_native_v1)
  monkeypatch.setattr(linear_module, "svdq_gemm_w4a4_native_v2", lambda **kwargs: sentinel)
  result = module.forward_quant(torch.empty(0), torch.empty(0), torch.empty(0))
  assert result is sentinel


def test_cute_fake_registration() -> None:
  act = torch.empty(256, 64, dtype=torch.int8, device="meta")
  wgt = torch.empty(128, 64, dtype=torch.int8, device="meta")
  ascales = torch.empty(256 * 128 // 16, dtype=torch.float8_e4m3fn, device="meta")
  wscales = torch.empty(128 * 128 // 16, dtype=torch.float8_e4m3fn, device="meta")
  output = torch.ops.cache_dit_cuda_ops.svdq_gemm_w4a4_cute(act, wgt, ascales, wscales, None, None,
                                                            None, None, 1.0)
  assert output.shape == (256, 128)
  assert output.dtype == torch.bfloat16
  input = torch.empty(256, 128, dtype=torch.bfloat16, device="meta")
  lora_down = torch.empty(128, 32, dtype=torch.bfloat16, device="meta")
  smooth = torch.empty(128, dtype=torch.bfloat16, device="meta")
  quantized, scales, lora_act = torch.ops.cache_dit_cuda_ops.svdq_quantize_w4a4_cute(
    input, lora_down, smooth, [1.0, 1.0], 256)
  assert quantized.shape == (256, 64)
  assert quantized.dtype == torch.uint8
  assert scales.shape == (256 * 8, )
  assert scales.dtype == torch.float8_e4m3fn
  assert lora_act.shape == (256, 32)
  assert lora_act.dtype == torch.bfloat16
