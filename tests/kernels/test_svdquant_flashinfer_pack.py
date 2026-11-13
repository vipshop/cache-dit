"""FlashInfer pack derivation tests that do not need FlashInfer or a GPU."""

from __future__ import annotations

import pytest
import torch

from cache_dit.quantization.svdquant.linear import SVDQW4A4Linear
from cache_dit.quantization.svdquant.packing import convert_svdq_nvfp4_pack_to_flashinfer
from cache_dit.quantization.svdquant.packing import swizzle_fp4_scales
from cache_dit.quantization.svdquant.packing import unswizzle_fp4_scales

_E2M1_VALUES = torch.tensor(
  [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _decode_e2m1(packed: torch.Tensor, device: torch.device) -> torch.Tensor:
  """Expand packed E2M1 bytes into their `[N, K]` float values."""
  low = (packed.to(torch.int16) & 0xF).to(torch.int64)
  high = ((packed.to(torch.int16) >> 4) & 0xF).to(torch.int64)
  nibbles = torch.stack([low, high], dim=-1).reshape(packed.shape[0], -1)
  return _E2M1_VALUES.to(device)[nibbles]


def _make_module(n: int = 256, k: int = 384, rank: int = 32) -> SVDQW4A4Linear:
  from cache_dit.quantization.svdquant.packing import SVDQWeightPacker

  torch.manual_seed(0)
  module = SVDQW4A4Linear(k,
                          n,
                          rank=rank,
                          precision="nvfp4",
                          runtime_kernel="flashinfer",
                          device="cpu")
  packer = SVDQWeightPacker(bits=4)
  smooth = (torch.rand(k, device="cpu") + 0.5).to(torch.bfloat16)
  packed_smooth = packer.pack_scale(packer.pad_scale(smooth.view(-1, 1), group_size=-1),
                                    group_size=-1)
  with torch.no_grad():
    module.qweight.copy_(
      torch.randint(-128, 127, module.qweight.shape, dtype=torch.int8, device="cpu"))
    module.wscales.copy_(
      (torch.rand(module.wscales.shape, device="cpu") * 0.05 + 0.01).to(torch.float8_e4m3fn))
    module.wcscales.copy_(torch.rand(module.wcscales.shape, device="cpu") * 0.5 + 0.75)
    module.smooth_factor.copy_(packed_smooth.view(module.smooth_factor.shape))
    module.smooth_factor_orig.copy_(module.smooth_factor)
    module.proj_up.copy_(torch.randn(module.proj_up.shape, device="cpu") * 0.02)
    module.proj_down.copy_(torch.randn(module.proj_down.shape, device="cpu") * 0.02)
    module.bias.copy_(torch.randn(module.bias.shape, device="cpu") * 0.02)
  module._unpacked_smooth = smooth
  return module


def _reference_dequant(module: SVDQW4A4Linear) -> torch.Tensor:
  """Dequantize the native pack using `_prepare_cute_pack`'s row-major conversion."""
  from cache_dit.quantization.svdquant.packing import convert_svdq_nvfp4_pack_to_cute

  qweight, flat_scales, _, _, _, wcscales, _ = convert_svdq_nvfp4_pack_to_cute(
    module.qweight, module.wscales, module.proj_up, module.proj_down, module.bias, module.wcscales,
    module.smooth_factor)
  n, k2 = qweight.shape
  scales = unswizzle_fp4_scales(flat_scales.view(torch.uint8).reshape(-1), n, (k2 * 2) // 16)
  return (_decode_e2m1(qweight.view(torch.uint8), module.qweight.device) *
          scales.float().repeat_interleave(16, dim=1) * wcscales.float().unsqueeze(1))


def test_fp4_scale_swizzle_roundtrip() -> None:
  scales = (torch.rand(256, 24) * 0.1).to(torch.float8_e4m3fn)
  flat = swizzle_fp4_scales(scales)
  assert flat.dtype == torch.uint8
  assert flat.numel() == 256 * 24
  restored = unswizzle_fp4_scales(flat, 256, 24)
  assert torch.equal(restored.view(torch.uint8), scales.view(torch.uint8))


def test_flashinfer_pack_folds_the_per_channel_subscale() -> None:
  module = _make_module()
  qweight, w_sf, proj_up, proj_down, bias, pre_quant_scale = (convert_svdq_nvfp4_pack_to_flashinfer(
    module.qweight, module.wscales, module.proj_up, module.proj_down, module.bias, module.wcscales,
    module.smooth_factor))
  n, k = module.out_features, module.in_features

  assert qweight.dtype == torch.uint8 and qweight.shape == (n, k // 2)
  assert w_sf.dtype == torch.uint8 and w_sf.numel() == n * (k // 16)
  assert proj_up.shape == (n, module.rank) and proj_up.dtype == torch.bfloat16
  assert proj_down.shape == (k, module.rank) and bias is not None

  # cache-dit stores `W * smooth`, so FlashInfer must divide by the same vector.
  expected_scale = (1.0 / module._unpacked_smooth.float()).to(torch.bfloat16)
  assert torch.equal(pre_quant_scale, expected_scale)

  reference = _reference_dequant(module)
  derived = (_decode_e2m1(qweight, qweight.device) *
             unswizzle_fp4_scales(w_sf, n, k // 16).float().repeat_interleave(16, dim=1))
  rel = float((derived - reference).norm() / reference.norm())
  # Folding two scale levels into one E4M3 byte re-rounds once; this bound is lossy by design.
  assert rel < 5e-2, rel


def test_flashinfer_pack_reconstructs_the_source_weight() -> None:
  """The derived pack must reconstruct the original weight through the native runtime formula.

  The reference is the original float weight rather than another converted pack, so a wrong nibble
  permutation cannot cancel out. The reconstruction also pins the smoothing convention:
  the pack holds the *smoothed* residual `smooth * (W - L1 @ L2)`, while the LoRA branch runs on
  the raw activation with the pre-divided `proj_down`.
  """
  from torch import nn

  from cache_dit.quantization.svdquant.quantizer import quantize_linear_svdq_w4a4

  torch.manual_seed(0)
  k, n, rank = 384, 256, 32
  linear = nn.Linear(k, n, bias=False, dtype=torch.bfloat16)
  x = torch.randn(64, k, dtype=torch.bfloat16) / (k ** 0.25)
  module = quantize_linear_svdq_w4a4(linear, [x], rank=rank, precision="nvfp4", device="cpu")
  qweight, w_sf, proj_up, proj_down, _, pre_quant_scale = convert_svdq_nvfp4_pack_to_flashinfer(
    module.qweight, module.wscales, module.proj_up, module.proj_down, module.bias, module.wcscales,
    module.smooth_factor)

  decoded = (_decode_e2m1(qweight, qweight.device) *
             unswizzle_fp4_scales(w_sf, n, k // 16).float().repeat_interleave(16, dim=1))
  smooth = 1.0 / pre_quant_scale.float()
  reconstructed = decoded / smooth.unsqueeze(0) + proj_up.float() @ proj_down.float().t()
  weight = linear.weight.detach().float()
  rel = float((reconstructed - weight).norm() / weight.norm())
  # NVFP4 block quantization of the rank-32 residual dominates this budget (~8%); a nibble
  # permutation or a mis-mapped scale would land far above it.
  assert rel < 0.15, rel


def test_fp4_scale_swizzle_roundtrip_with_odd_columns() -> None:
  scales = (torch.rand(128, 3) * 0.1).to(torch.float8_e4m3fn)
  flat = swizzle_fp4_scales(scales)
  assert flat.numel() == 128 * 4
  restored = unswizzle_fp4_scales(flat, 128, 3)
  assert torch.equal(restored.view(torch.uint8), scales.view(torch.uint8))


def test_flashinfer_pack_requires_channel_scales() -> None:
  module = _make_module()
  with pytest.raises(ValueError, match="wcscales"):
    convert_svdq_nvfp4_pack_to_flashinfer(module.qweight, module.wscales, module.proj_up,
                                          module.proj_down, module.bias, None, module.smooth_factor)


def test_flashinfer_runtime_validation() -> None:
  for runtime in ("flashinfer", "flashinfer+cute-dsl", "flashinfer+cutlass-sm120"):
    with pytest.raises(ValueError, match="only supported for precision='nvfp4'"):
      SVDQW4A4Linear(128, 128, precision="int4", runtime_kernel=runtime)
    with pytest.raises(ValueError, match="torch.bfloat16"):
      SVDQW4A4Linear(128, 128, precision="nvfp4", runtime_kernel=runtime, torch_dtype=torch.float16)
  with pytest.raises(ValueError, match="runtime_kernel must be"):
    SVDQW4A4Linear(128, 128, precision="nvfp4", runtime_kernel="flashinfer+cute-dsl-unfused")


def test_flashinfer_runtime_spellings_normalize() -> None:
  from cache_dit.kernels.flashinfer import CUTE_DSL_RUNTIME, CUTLASS_SM120_RUNTIME, resolve_runtime

  # Bare `flashinfer` names the fused-prefix chain; the CuTe chain is always explicit.
  assert resolve_runtime("flashinfer") == CUTLASS_SM120_RUNTIME
  assert resolve_runtime(CUTLASS_SM120_RUNTIME) == CUTLASS_SM120_RUNTIME
  assert resolve_runtime(CUTE_DSL_RUNTIME) == CUTE_DSL_RUNTIME
  assert SVDQW4A4Linear(128, 128, precision="nvfp4",
                        runtime_kernel="flashinfer").runtime_kernel == CUTLASS_SM120_RUNTIME
  assert SVDQW4A4Linear(128, 128, precision="nvfp4",
                        runtime_kernel=CUTE_DSL_RUNTIME).runtime_kernel == CUTE_DSL_RUNTIME


def test_flashinfer_cutlass_runtime_rejects_uncovered_ranks() -> None:
  from cache_dit.kernels.flashinfer import CUTE_DSL_RUNTIME, CUTLASS_SM120_LORA_RANKS

  assert CUTLASS_SM120_LORA_RANKS == (32, 64)
  for rank in (96, 128):
    with pytest.raises(ValueError, match="only supports LoRA ranks"):
      SVDQW4A4Linear(128,
                     128,
                     rank=rank,
                     precision="nvfp4",
                     runtime_kernel="flashinfer+cutlass-sm120")
    # The CuTe chain covers every rank the cache-dit runtime allows.
    assert SVDQW4A4Linear(128, 128, rank=rank, precision="nvfp4",
                          runtime_kernel=CUTE_DSL_RUNTIME).rank == rank
  # Rank granularity is enforced for both chains, before the chain coverage check for cutlass.
  with pytest.raises(ValueError, match="rank > 0 and a multiple of 32"):
    SVDQW4A4Linear(128, 128, rank=16, precision="nvfp4", runtime_kernel=CUTE_DSL_RUNTIME)


def test_flashinfer_pack_preserves_the_legacy_state_dict() -> None:
  module = _make_module()
  original = {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}
  module.load_state_dict(module.state_dict())
  assert module._flashinfer_pack_ready
  assert module._fi_qweight is not None and module._fi_w_sf is not None
  assert set(module.state_dict()) == set(original)
  for name, tensor in module.state_dict().items():
    assert torch.equal(tensor.view(torch.int16), original[name].view(torch.int16)), name


def test_flashinfer_pack_follows_device_moves() -> None:
  if not torch.cuda.is_available():
    pytest.skip("CUDA is required")
  module = _make_module()
  module.load_state_dict(module.state_dict())
  assert module._fi_qweight.device.type == "cpu"
  module = module.cuda()
  assert module._flashinfer_pack_ready
  for name in ("_fi_qweight", "_fi_w_sf", "_fi_proj_up", "_fi_proj_down", "_fi_pre_quant_scale"):
    assert getattr(module, name).is_cuda, name


def test_flashinfer_runtime_is_never_the_default() -> None:
  from cache_dit.kernels.flashinfer import CUTLASS_SM120_RUNTIME

  assert SVDQW4A4Linear(128, 128, precision="nvfp4").runtime_kernel == "native"
  assert SVDQW4A4Linear(128, 128, precision="nvfp4", runtime_kernel="cute").runtime_kernel == "cute"
  assert SVDQW4A4Linear(128, 128, precision="nvfp4",
                        runtime_kernel="flashinfer").runtime_kernel == CUTLASS_SM120_RUNTIME


def test_flashinfer_runtime_validation_covers_rank_granularity() -> None:
  from cache_dit.kernels.flashinfer import CUTE_DSL_RUNTIME

  with pytest.raises(ValueError, match="rank > 0 and a multiple of 32"):
    SVDQW4A4Linear(128, 128, rank=0, precision="nvfp4", runtime_kernel=CUTE_DSL_RUNTIME)
  with pytest.raises(ValueError, match="rank > 0 and a multiple of 32"):
    SVDQW4A4Linear(128, 128, rank=48, precision="nvfp4", runtime_kernel=CUTE_DSL_RUNTIME)
