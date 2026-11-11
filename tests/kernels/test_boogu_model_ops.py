import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode

from cache_dit.kernels import cuda_ext_is_available
from cache_dit.kernels.models import boogu_fused_qk_norm_rope
from cache_dit.kernels.models import boogu_gated_rms_norm
from cache_dit.kernels.models import boogu_scaled_rms_norm


def _require_runtime() -> None:
  if not torch.cuda.is_available() or not cuda_ext_is_available():
    pytest.skip("CUDA and the cache-dit CUDA extension are required.")


def _ref_scaled_rms_norm(x, weight, eps, scale=None, zero_centered=False):
  xf = x.float()
  rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
  w = weight.float() + 1.0 if zero_centered else weight.float()
  out = xf * rstd * w
  if scale is not None:
    out = out * (1.0 + scale.float())
  return out.to(x.dtype)


def _ref_gated_rms_norm(x, weight, eps, gate, residual, zero_centered=False):
  xf = x.float()
  rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
  w = weight.float() + 1.0 if zero_centered else weight.float()
  out = residual.float() + torch.tanh(gate.float()) * (xf * rstd * w)
  return out.to(x.dtype)


def _ref_qk_norm_rope(x, freqs_cis, weight, eps):
  xf = x.float()
  rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
  normed = xf * rstd * weight.float()
  pairs = torch.view_as_complex(normed.reshape(*normed.shape[:-1], -1, 2))
  rotated = torch.view_as_real(pairs * freqs_cis.unsqueeze(2)).flatten(-2)
  return rotated.to(x.dtype)


def _max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
  af, bf = a.float(), b.float()
  denom = bf.abs().clamp_min(1e-3)
  return float(((af - bf).abs() / denom).max())


def test_boogu_models_fake_registration() -> None:
  with FakeTensorMode():
    x = torch.empty((1, 16, 64), device="cuda", dtype=torch.bfloat16)
    weight = torch.empty((64, ), device="cuda", dtype=torch.bfloat16)
    scale = torch.empty((64, ), device="cuda", dtype=torch.bfloat16)
    gate = torch.empty((64, ), device="cuda", dtype=torch.bfloat16)
    q = torch.empty((1, 16, 4, 64), device="cuda", dtype=torch.bfloat16)
    freqs = torch.empty((1, 16, 32), device="cuda", dtype=torch.complex64)

    scaled = boogu_scaled_rms_norm(x, weight, 1e-5, scale, False)
    gated = boogu_gated_rms_norm(x, weight, 1e-5, gate, x, False)
    rope = boogu_fused_qk_norm_rope(q, freqs, weight, 1e-5)

  assert isinstance(scaled, FakeTensor) and scaled.shape == x.shape
  assert isinstance(gated, FakeTensor) and gated.shape == x.shape
  assert isinstance(rope, FakeTensor) and rope.shape == q.shape
  assert rope.dtype == q.dtype


@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("with_scale", [False, True])
def test_boogu_scaled_rms_norm_matches_reference(zero_centered: bool, with_scale: bool) -> None:
  _require_runtime()
  torch.manual_seed(0)
  x = torch.randn((2, 128, 512), device="cuda", dtype=torch.bfloat16)
  weight = torch.randn((512, ), device="cuda", dtype=torch.bfloat16) * 0.1
  scale = torch.randn((512, ), device="cuda", dtype=torch.bfloat16) * 0.1 if with_scale else None

  out = boogu_scaled_rms_norm(x, weight, 1e-5, scale, zero_centered)
  ref = _ref_scaled_rms_norm(x, weight, 1e-5, scale, zero_centered)

  assert out.dtype == x.dtype and out.shape == x.shape
  assert _max_rel(out, ref) < 3e-2


@pytest.mark.parametrize("zero_centered", [False, True])
def test_boogu_gated_rms_norm_matches_reference(zero_centered: bool) -> None:
  _require_runtime()
  torch.manual_seed(0)
  x = torch.randn((1, 64, 1024), device="cuda", dtype=torch.bfloat16)
  residual = torch.randn_like(x)
  weight = torch.randn((1024, ), device="cuda", dtype=torch.bfloat16) * 0.1
  gate = torch.randn((1024, ), device="cuda", dtype=torch.bfloat16)

  out = boogu_gated_rms_norm(x, weight, 1e-5, gate, residual, zero_centered)
  ref = _ref_gated_rms_norm(x, weight, 1e-5, gate, residual, zero_centered)

  assert out.dtype == x.dtype and out.shape == x.shape
  assert _max_rel(out, ref) < 3e-2


@pytest.mark.parametrize("freq_dtype", [torch.complex64, torch.complex128])
def test_boogu_qk_norm_rope_matches_reference(freq_dtype: torch.dtype) -> None:
  _require_runtime()
  torch.manual_seed(0)
  # Boogu's head_dim is 120, which exercises the two-pass non-power-of-two row.
  x = torch.randn((1, 96, 4, 120), device="cuda", dtype=torch.bfloat16)
  weight = torch.randn((120, ), device="cuda", dtype=torch.bfloat16) * 0.1
  angles = torch.randn((1, 96, 60), device="cuda", dtype=torch.float32)
  freqs = torch.polar(torch.ones_like(angles), angles).to(freq_dtype)

  out = boogu_fused_qk_norm_rope(x, freqs, weight, 1e-5)
  ref = _ref_qk_norm_rope(x, freqs, weight, 1e-5)

  assert out.dtype == x.dtype and out.shape == x.shape
  assert _max_rel(out, ref) < 3e-2


def test_boogu_ops_reject_wrong_layout() -> None:
  _require_runtime()
  x = torch.randn((2, 32, 12), device="cuda", dtype=torch.bfloat16)
  weight = torch.randn((12, ), device="cuda", dtype=torch.bfloat16)
  with pytest.raises((RuntimeError, ValueError)):
    # 12 is not a multiple of 8.
    boogu_scaled_rms_norm(x, weight, 1e-5, None, False)


def test_boogu_gated_rms_norm_rejects_mismatched_residual() -> None:
  _require_runtime()
  x = torch.randn((1, 8, 64), device="cuda", dtype=torch.bfloat16)
  weight = torch.randn((64, ), device="cuda", dtype=torch.bfloat16)
  gate = torch.randn((64, ), device="cuda", dtype=torch.bfloat16)
  with pytest.raises(ValueError):
    boogu_gated_rms_norm(x, weight, 1e-5, gate, x[:, :4], False)


def test_boogu_fused_ops_switch_is_model_scoped() -> None:
  from cache_dit.models.pipelines.boogu_image import fused_ops

  assert fused_ops._BOOGU_FUSED_OPS_ENV == "CACHE_DIT_BOOGU_FUSED_OPS"
  assert fused_ops._BOOGU_FUSED_OPS_ENABLED is True


def test_boogu_fused_ops_env_parsing(monkeypatch) -> None:
  from cache_dit.models.pipelines.boogu_image import fused_ops

  name = fused_ops._BOOGU_FUSED_OPS_ENV
  for raw, expected in (("0", False), ("off", False), ("false", False), ("1", True), ("on", True),
                        ("nonsense", True)):
    monkeypatch.setenv(name, raw)
    assert fused_ops._env_enabled(name, True) is expected, raw
  monkeypatch.delenv(name)
  assert fused_ops._env_enabled(name, True) is True


def test_boogu_model_fused_path_matches_eager_chain() -> None:
  _require_runtime()
  from cache_dit.models.pipelines.boogu_image import fused_ops
  from cache_dit.models.pipelines.boogu_image.rope import BooguImageRotaryPosEmbed
  from cache_dit.models.pipelines.boogu_image.transformer_boogu import (
    BooguImageTransformer2DModel, )

  torch.manual_seed(0)
  device, dtype = "cuda", torch.bfloat16
  model = BooguImageTransformer2DModel(
    patch_size=2,
    in_channels=4,
    hidden_size=96,
    num_layers=3,
    num_double_stream_layers=1,
    num_refiner_layers=1,
    num_attention_heads=4,
    num_kv_heads=2,
    multiple_of=32,
    axes_dim_rope=(8, 8, 8),
    axes_lens=(64, 64, 64),
    instruction_feature_configs=dict(instruction_feat_dim=32,
                                     reduce_type="mean",
                                     num_instruction_feat_layers=1),
  ).to(device=device, dtype=dtype).eval()

  freqs = [
    f.to(device) for f in BooguImageRotaryPosEmbed.get_freqs_cis((8, 8, 8), (64, 64, 64), 10000)
  ]
  latent = torch.randn(1, 4, 16, 16, device=device, dtype=dtype)
  ref = torch.randn(4, 16, 16, device=device, dtype=dtype)
  instruct = torch.randn(1, 8, 32, device=device, dtype=dtype)
  mask = torch.ones(1, 8, device=device, dtype=torch.long)

  def run():
    with torch.no_grad():
      out = model(
        hidden_states=latent.clone(),
        timestep=torch.tensor([500.0], device=device),
        instruction_hidden_states=instruct.clone(),
        freqs_cis=freqs,
        instruction_attention_mask=mask.clone(),
        ref_image_hidden_states=[[ref.clone()]],
        return_dict=False,
      )
    return out[0] if isinstance(out, (list, tuple)) else out

  fused_ops.set_boogu_fused_ops_enabled(False)
  eager = run()
  fused_ops.set_boogu_fused_ops_enabled(True)
  fused = run()

  assert torch.isfinite(fused).all()
  # The fused kernels round once instead of twice, so the two paths differ by a
  # couple of bf16 ulps rather than bit-exactly.
  rel = (fused.float() -
         eager.float()).abs().max().item() / eager.float().abs().max().clamp_min(1e-6).item()
  assert rel < 2e-2


def test_boogu_scaled_rms_norm_compile_smoke() -> None:
  _require_runtime()
  x = torch.randn((1, 32, 256), device="cuda", dtype=torch.bfloat16)
  weight = torch.randn((256, ), device="cuda", dtype=torch.bfloat16) * 0.1
  scale = torch.randn((256, ), device="cuda", dtype=torch.bfloat16) * 0.1

  def fn(t):
    return boogu_scaled_rms_norm(t + 1.0, weight, 1e-5, scale, False)

  eager = fn(x)
  compiled = torch.compile(fn, fullgraph=True)(x)
  assert torch.allclose(eager, compiled, atol=0, rtol=0)
