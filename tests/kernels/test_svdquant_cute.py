from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cache_dit.kernels import svdq_gemm_w4a4
from cache_dit.kernels import svdq_quantize_w4a4
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


def _decode_fragment_d(fragment_d: torch.Tensor) -> torch.Tensor:
  """Decode the native FP32 fragment D into its logical row-major `[M, R]` view.

  :param fragment_d: Flat native fragment buffer produced by the activation producer.
  :returns: Row-major `[M, R]` FP32 view of the same values.
  """

  rank = fragment_d.shape[1]
  rows = torch.arange(fragment_d.shape[0], device=fragment_d.device)[:, None]
  cols = torch.arange(rank, device=fragment_d.device)[None, :]
  rr = rows % 256
  cc = cols % 16
  warp = rr // 32
  mt = (rr % 32) // 16
  lane = (rr % 8) * 4 + (cc % 8) // 2
  j = (cc // 8) * 4 + ((rr % 16) // 8) * 2 + cc % 2
  indices = ((rows // 256 * (rank // 16) + cols // 16) * 4096 + warp * 512 + mt * 256 + j * 32 +
             lane)
  return fragment_d.flatten()[indices]


@pytest.mark.parametrize("rank,scales", [(32, [0.5, 2.0]), (64, [1.0] * 4), (128, [1.0] * 8)])
def test_row_major_quantize_producer(rank: int, scales: list[float]) -> None:
  _require_sm120()
  torch.manual_seed(1234)
  m, k = 256, 128
  x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda:0")
  smooth = torch.rand(k, dtype=torch.bfloat16, device="cuda:0") + 0.5
  lora_down = torch.randn(k, rank, dtype=torch.bfloat16, device="cuda:0") * 0.02
  qact, ascales, lora_act = svdq_quantize_w4a4(x,
                                               lora_down,
                                               smooth,
                                               lora_layout="row_major",
                                               lora_scales=scales)

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


def test_row_major_producer_matches_native_v1_pack() -> None:
  _require_sm120()
  torch.manual_seed(20261110)
  m, k = 256, 128
  x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda:0")
  smooth = torch.ones(k, dtype=torch.bfloat16, device="cuda:0")
  q_native, sf_native, _ = svdq_quantize_w4a4(x, None, smooth, lora_layout="native", fp4=True)
  q_row_major, sf_row_major, _ = svdq_quantize_w4a4(x,
                                                    None,
                                                    smooth,
                                                    lora_layout="row_major",
                                                    lora_scales=[1.0],
                                                    pad_size=256)
  assert torch.equal(q_row_major, q_native)
  # native stores scales as [K/16, M], while the v1 pack is flat; the values must match.
  assert torch.equal(
    sf_row_major.view(torch.uint8).flatten().sort().values,
    sf_native.view(torch.uint8).flatten().sort().values)


def test_row_major_producer_rejects_non_256_padded_batch() -> None:
  _require_sm120()
  x = torch.randn(288, 128, dtype=torch.bfloat16, device="cuda:0")
  smooth = torch.ones(128, dtype=torch.bfloat16, device="cuda:0")
  with pytest.raises(RuntimeError, match="multiple of 256"):
    svdq_quantize_w4a4(x, None, smooth, lora_layout="row_major", lora_scales=[1.0], pad_size=128)


@pytest.mark.parametrize("rank", [32, 64, 128])
def test_cute_gemm_matches_native_runtime(rank: int) -> None:
  _require_sm120()
  torch.manual_seed(20261112)
  in_features = out_features = 512
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


def test_cute_gemm_native_layout_applies_rank_scales() -> None:
  _require_sm120()
  torch.manual_seed(20261113)
  in_features, out_features, rank = 256, 384, 32
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
  native = SVDQW4A4Linear(in_features,
                          out_features,
                          rank=rank,
                          precision="nvfp4",
                          runtime_kernel="native",
                          torch_dtype=torch.bfloat16).cuda().eval()
  cute = SVDQW4A4Linear(in_features,
                        out_features,
                        rank=rank,
                        precision="nvfp4",
                        runtime_kernel="cute",
                        torch_dtype=torch.bfloat16).cuda().eval()
  for module in (native, cute):
    module.load_state_dict(state_dict)

  # N=384 selects BN128/BK64 and M=384 is padded to 512. The expected value below applies the
  # rank16 scales directly to the decoded native D, independently of the kernel's gather.
  scales = [0.5, 2.0]
  cute._prepare_cute_pack()
  x = torch.randn(384, in_features, dtype=torch.bfloat16, device="cuda:0")
  with torch.inference_mode():
    qact, ascales, fragment_d = native.quantize(x)
    baseline_actual = svdq_gemm_w4a4(qact,
                                     cute._cute_qweight,
                                     ascales,
                                     cute._cute_wscales,
                                     fragment_d,
                                     cute._cute_proj_up,
                                     cute._cute_bias,
                                     cute._cute_wcscales,
                                     runtime="cute",
                                     lora_layout="native",
                                     fp4=True,
                                     alpha=cute.wtscale,
                                     lora_scales=[1.0, 1.0])
    baseline_expected = native(x)
    assert _relative_l2(baseline_actual[:x.shape[0]], baseline_expected) < 5e-3
    actual = svdq_gemm_w4a4(qact,
                            cute._cute_qweight,
                            ascales,
                            cute._cute_wscales,
                            fragment_d,
                            cute._cute_proj_up,
                            cute._cute_bias,
                            cute._cute_wcscales,
                            runtime="cute",
                            lora_layout="native",
                            fp4=True,
                            alpha=cute.wtscale,
                            lora_scales=scales)
  logical_d = _decode_fragment_d(fragment_d)
  delta_d = logical_d * torch.tensor([-0.5, 1.0], device="cuda:0").repeat_interleave(16)
  expected = baseline_expected.float() + (delta_d @ cute._cute_proj_up.float().T)[:x.shape[0]]
  assert fragment_d.dtype == torch.float32 and actual.shape == (512, out_features)
  assert torch.isfinite(actual).all()
  assert _relative_l2(actual[:x.shape[0]], expected) < 5e-3

  # The same entry point also accepts row-major BF16 D. Folding the rank scales into D must
  # reproduce the native-fragment result, and the assertion targets an independent torch reference
  # rather than only the sibling kernel path.
  d_bf16 = (logical_d * torch.tensor(scales, device="cuda:0").repeat_interleave(16)).bfloat16()
  bf16_actual = svdq_gemm_w4a4(qact,
                               cute._cute_qweight,
                               ascales,
                               cute._cute_wscales,
                               d_bf16,
                               cute._cute_proj_up,
                               cute._cute_bias,
                               cute._cute_wcscales,
                               runtime="cute",
                               lora_layout="row_major",
                               fp4=True,
                               alpha=cute.wtscale)
  # `native(x)` already carries the LoRA term at scale 1, so the reference only adds the delta that
  # the re-scaled D introduces.
  bf16_delta = d_bf16.float() - logical_d
  bf16_expected = baseline_expected.float() + (
    bf16_delta @ cute._cute_proj_up.float().T)[:x.shape[0]]
  assert bf16_actual.shape == actual.shape
  assert _relative_l2(bf16_actual[:x.shape[0]], bf16_expected) < 5e-3
  assert _relative_l2(bf16_actual, actual) < 2e-3


@pytest.mark.parametrize("lora_layout", ["native", "row_major"])
def test_cute_gemm_large_m_dispatch(lora_layout: str) -> None:
  """Cover the M >= 16384 main-first schedule for both LoRA-D layouts."""

  _require_sm120()
  torch.manual_seed(20261114)
  in_features = out_features = 512
  rank, tokens = 32, 16384
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
  native = SVDQW4A4Linear(in_features,
                          out_features,
                          rank=rank,
                          precision="nvfp4",
                          runtime_kernel="native",
                          torch_dtype=torch.bfloat16).cuda().eval()
  cute = SVDQW4A4Linear(in_features,
                        out_features,
                        rank=rank,
                        precision="nvfp4",
                        runtime_kernel="cute",
                        torch_dtype=torch.bfloat16).cuda().eval()
  for module in (native, cute):
    module.load_state_dict(state_dict)
  cute._prepare_cute_pack()
  x = torch.randn(tokens, in_features, dtype=torch.bfloat16, device="cuda:0")
  with torch.inference_mode():
    qact, ascales, fragment_d = native.quantize(x)
    expected = native(x)
    row_major = lora_layout == "row_major"
    lora_act_in = _decode_fragment_d(fragment_d).bfloat16() if row_major else fragment_d
    actual = svdq_gemm_w4a4(qact,
                            cute._cute_qweight,
                            ascales,
                            cute._cute_wscales,
                            lora_act_in,
                            cute._cute_proj_up,
                            cute._cute_bias,
                            cute._cute_wcscales,
                            runtime="cute",
                            lora_layout=lora_layout,
                            fp4=True,
                            alpha=cute.wtscale)
  assert actual.shape == (tokens, out_features)
  assert torch.isfinite(actual).all()
  assert _relative_l2(actual, expected) < 5e-3


def test_gemm_w4a4_validates_lora_arguments() -> None:
  _require_sm120()
  act = torch.empty(256, 64, dtype=torch.int8, device="cuda:0")
  wgt = torch.empty(128, 64, dtype=torch.int8, device="cuda:0")
  ascales = torch.empty(256 * 128 // 16, dtype=torch.float8_e4m3fn, device="cuda:0")
  wscales = torch.empty(128 * 128 // 16, dtype=torch.float8_e4m3fn, device="cuda:0")
  bf16_d = torch.empty(256, 32, dtype=torch.bfloat16, device="cuda:0")
  fp32_d = torch.empty(256, 32, dtype=torch.float32, device="cuda:0")
  lora_up = torch.empty(128, 32, dtype=torch.bfloat16, device="cuda:0")
  with pytest.raises(ValueError, match="unknown runtime|runtime must be"):
    svdq_gemm_w4a4(act, wgt, ascales, wscales, bf16_d, lora_up, runtime="v1")
  with pytest.raises(ValueError, match="only supports lora_layout='native'"):
    svdq_gemm_w4a4(act,
                   wgt,
                   ascales,
                   wscales,
                   bf16_d,
                   lora_up,
                   runtime="native",
                   lora_layout="row_major")
  # Row-major D carries the scales, so a per-rank scale list has to be rejected instead of dropped.
  with pytest.raises(ValueError, match="folded into D"):
    svdq_gemm_w4a4(act,
                   wgt,
                   ascales,
                   wscales,
                   bf16_d,
                   lora_up,
                   runtime="cute",
                   lora_layout="row_major",
                   lora_scales=[1.0, 1.0])
  # The CuTe runtime is NVFP4-only and always returns BF16.
  with pytest.raises(ValueError, match="NVFP4 path only"):
    svdq_gemm_w4a4(act, wgt, ascales, wscales, bf16_d, lora_up, runtime="cute", fp4=False)
  with pytest.raises(ValueError, match="always produces BF16"):
    svdq_gemm_w4a4(act,
                   wgt,
                   ascales,
                   wscales,
                   bf16_d,
                   lora_up,
                   runtime="cute",
                   output_dtype=torch.float16)
  # The native layout requires the packed FP32 fragment D, so a BF16 D has to fail instead of
  # being reinterpreted.
  with pytest.raises(RuntimeError, match="packed FP32 fragment D"):
    svdq_gemm_w4a4(act,
                   wgt,
                   ascales,
                   wscales,
                   bf16_d,
                   lora_up,
                   runtime="cute",
                   lora_layout="native")
  # One scale per 16-rank group is mandatory on the native layout.
  with pytest.raises(RuntimeError, match="one value per 16-rank group"):
    svdq_gemm_w4a4(act,
                   wgt,
                   ascales,
                   wscales,
                   fp32_d,
                   lora_up,
                   runtime="cute",
                   lora_layout="native",
                   lora_scales=[1.0])
  # The native layout carries a fixed float[8] scale table, so R is capped at 128.
  wide_d = torch.empty(256, 256, dtype=torch.float32, device="cuda:0")
  wide_up = torch.empty(128, 256, dtype=torch.bfloat16, device="cuda:0")
  with pytest.raises(RuntimeError, match="at most 8 rank groups"):
    svdq_gemm_w4a4(act,
                   wgt,
                   ascales,
                   wscales,
                   wide_d,
                   wide_up,
                   runtime="cute",
                   lora_layout="native",
                   lora_scales=[1.0] * 16)


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
  # Loading is what materializes the pack, so a self-reload is enough to prime the cache.
  module.load_state_dict(module.state_dict())
  assert module._cute_pack_ready
  assert module._cute_pack_version == 1
  assert module.bias.requires_grad == requires_grad
  for name, tensor in module.state_dict().items():
    assert torch.equal(tensor.view(torch.int16), original[name].view(torch.int16)), name

  reloaded = SVDQW4A4Linear(128, 128, rank=32, precision="nvfp4", runtime_kernel="cute")
  assert not reloaded._cute_pack_ready
  reloaded.load_state_dict(module.state_dict())
  # The pack is built once at load time, so a reloaded CuTe module is forward-ready.
  assert reloaded._cute_pack_ready
  for name, tensor in reloaded.state_dict().items():
    assert torch.equal(tensor.view(torch.int16), original[name].view(torch.int16)), name


def test_cute_pack_follows_device_moves() -> None:
  if not torch.cuda.is_available():
    pytest.skip("CUDA is required")
  module = SVDQW4A4Linear(128, 128, rank=32, precision="nvfp4", runtime_kernel="cute")
  module.load_state_dict(module.state_dict())
  assert module._cute_qweight.device.type == "cpu"
  module = module.cuda()
  assert module._cute_pack_ready
  assert module._cute_qweight.is_cuda and module._cute_proj_up.is_cuda


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
  if not torch.cuda.is_available():
    pytest.skip("CUDA is required")
  import cache_dit.kernels._svdquant as svdq_ops

  module = SVDQW4A4Linear(128, 128, rank=0, precision="int4", runtime_kernel=runtime_kernel)
  seen: list[str] = []
  monkeypatch.setattr(svdq_ops, "_call_svdq_gemm_w4a4_native",
                      lambda *args, **kwargs: seen.append("v1"))
  monkeypatch.setattr(svdq_ops, "_call_svdq_gemm_w4a4_native_v2",
                      lambda *args, **kwargs: seen.append("v2"))
  module.forward_quant(torch.empty(0, 128, device="cuda:0"), torch.empty(0, device="cuda:0"),
                       torch.empty(0, device="cuda:0"))
  assert seen == ["v2"]


def test_gemm_w4a4_fake_registration() -> None:
  act = torch.empty(256, 64, dtype=torch.int8, device="meta")
  wgt = torch.empty(128, 64, dtype=torch.int8, device="meta")
  ascales = torch.empty(256 * 128 // 16, dtype=torch.float8_e4m3fn, device="meta")
  wscales = torch.empty(128 * 128 // 16, dtype=torch.float8_e4m3fn, device="meta")
  for runtime in ("cute", "native"):
    output = torch.ops.cache_dit_cuda_ops.svdq_gemm_w4a4(act, wgt, ascales, wscales, None, None,
                                                         None, None, runtime, True, 1.0, False, 1,
                                                         1, [], "native")
    assert output.shape == (256, 128)
    assert output.dtype == torch.bfloat16
  input = torch.empty(256, 128, dtype=torch.bfloat16, device="meta")
  lora_down = torch.empty(128, 32, dtype=torch.bfloat16, device="meta")
  smooth = torch.empty(128, dtype=torch.bfloat16, device="meta")
  for lora_layout in ("row_major", "native"):
    quantized, scales, lora_act = torch.ops.cache_dit_cuda_ops.svdq_quantize_w4a4(
      input, lora_down, smooth, lora_layout, False, True, [1.0, 1.0], 256)
    assert quantized.shape == (256, 64)
    assert quantized.dtype == torch.uint8
    assert scales.dtype == torch.float8_e4m3fn
    assert lora_act.shape == (256, 32)
    if lora_layout == "row_major":
      # The CuTe producer emits the flat v1 scale pack and row-major BF16 D.
      assert scales.shape == (256 * 8, )
      assert lora_act.dtype == torch.bfloat16
    else:
      # The native producer keeps the [K / 16, M] scale layout and the FP32 fragment D.
      assert scales.shape == (8, 256)
      assert lora_act.dtype == torch.float32
