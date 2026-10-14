"""Tests for FFPA attention backend registration, dispatch, GQA, and numerics."""
import os

import pytest
import torch
import torch.nn.functional as F

from diffusers.models.modeling_utils import ModelMixin

from cache_dit.attention import set_attn_backend
from cache_dit.attention.backends import ffpa as ffpa_backend
from cache_dit.attention.backends.register import (
  _AttnBackend,
  _AttnBackendRegistry,
  _dispatch_attention_fn,
)

_FFPA_BACKENDS = ("ffpa", "ffpa_fp8", "ffpa_fp4", "ffpa_fp8_per_block")


def _is_sm120_cuda() -> bool:
  if not torch.cuda.is_available():
    return False
  major, _ = torch.cuda.get_device_capability()
  return major == 12


requires_sm120 = pytest.mark.skipif(not _is_sm120_cuda(), reason="requires an sm_120 CUDA device")


class _DummyProcessor(torch.nn.Module):
  pass


class _DummyAttentionModule(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.processor = _DummyProcessor()
    self._attention_backend = None


class _DummyNonDiffusersTransformer(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.attn = _DummyAttentionModule()


class _DummyDiffusersModule(ModelMixin):

  def __init__(self):
    super().__init__()
    self.backends: list[str] = []

  def set_attention_backend(self, backend: str) -> None:
    self.backends.append(backend)


class ToyAttentionModel(torch.nn.Module):
  """Mimics a transformer attention layer dispatching via cache-dit."""

  def __init__(self, backend: str):
    super().__init__()
    self._attention_backend = backend

  def forward(self, query, key, value, **kwargs):
    return _dispatch_attention_fn(query, key, value, backend=self._attention_backend, **kwargs)


class _FFPAFuncSpy:
  """Wraps ffpa_attn_func to capture the forward_backend kwarg per call."""

  def __init__(self):
    self.calls: list = []
    self.arg_calls: list = []
    self.kwarg_calls: list = []
    self._orig = ffpa_backend.ffpa_attn_func

  def __call__(self, *args, **kwargs):
    self.calls.append(kwargs.get("forward_backend"))
    self.arg_calls.append(args)
    self.kwarg_calls.append(kwargs)
    return self._orig(*args, **kwargs)


def _sdpa_ref(q: torch.Tensor,
              k: torch.Tensor,
              v: torch.Tensor,
              enable_gqa: bool = False,
              attn_mask=None) -> torch.Tensor:
  return F.scaled_dot_product_attention(
    q.permute(0, 2, 1, 3),
    k.permute(0, 2, 1, 3),
    v.permute(0, 2, 1, 3),
    attn_mask=attn_mask,
    enable_gqa=enable_gqa,
  ).permute(0, 2, 1, 3)


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
  return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _make_qkv(B=1, N=1024, H=16, H_kv=None, D=128, dtype=torch.bfloat16):
  H_kv = H if H_kv is None else H_kv
  q = torch.randn(B, N, H, D, dtype=dtype, device="cuda")
  k = torch.randn(B, N, H_kv, D, dtype=dtype, device="cuda")
  v = torch.randn(B, N, H_kv, D, dtype=dtype, device="cuda")
  return q, k, v


# ---------------------------------------------------------------- registration


@pytest.mark.parametrize("name", _FFPA_BACKENDS)
def test_ffpa_backends_are_registered(name: str):
  assert _AttnBackend(name).value == name
  assert _AttnBackendRegistry.get_backend(name) is not None
  assert _AttnBackendRegistry.is_context_parallel_available(name)


@pytest.mark.parametrize("name", _FFPA_BACKENDS)
def test_set_attn_backend_ffpa(name: str):
  module = _DummyDiffusersModule()
  set_attn_backend(module, name)
  assert module.backends == [name]

  transformer = _DummyNonDiffusersTransformer()
  set_attn_backend(transformer, name)
  assert transformer.attn._attention_backend == name
  assert transformer.attn.processor._attention_backend == name


def test_ffpa_sets_small_d_env():
  assert os.environ.get("FFPA_CUDA_ALLOW_SMALL_D") == "1"


@pytest.mark.parametrize("name", _FFPA_BACKENDS)
def test_ffpa_bridged_to_diffusers(name: str):
  from cache_dit.attention import _maybe_register_custom_attn_backends
  _maybe_register_custom_attn_backends()
  from diffusers.models.attention_dispatch import (
    AttentionBackendName,
    _AttentionBackendRegistry,
  )
  assert _AttentionBackendRegistry._backends.get(AttentionBackendName(name)) is not None


# ------------------------------------------------------------- input validation


def _cpu_qkv():
  return (torch.randn(1, 512, 8, 128, dtype=torch.bfloat16) for _ in range(3))


def test_ffpa_attn_mask_validation():
  q, k, v = _cpu_qkv()
  # The mask last dimension must match the key sequence length.
  with pytest.raises(ValueError, match="key's sequence length"):
    _dispatch_attention_fn(q, k, v, backend="ffpa", attn_mask=torch.zeros(1, 1, 512, 256))
  # Only 1-D to 4-D masks are broadcast-compatible.
  with pytest.raises(ValueError, match="1-D to 4-D"):
    _dispatch_attention_fn(q, k, v, backend="ffpa", attn_mask=torch.zeros(1, 1, 1, 512, 512))
  # Explicit masks and is_causal are mutually exclusive.
  with pytest.raises(ValueError, match="mutually exclusive"):
    _dispatch_attention_fn(q, k, v, backend="ffpa", attn_mask=torch.zeros(1, 512), is_causal=True)


def test_ffpa_rejects_dropout_and_return_lse():
  q, k, v = _cpu_qkv()
  with pytest.raises(ValueError, match="dropout_p"):
    _dispatch_attention_fn(q, k, v, backend="ffpa_fp8", dropout_p=0.1)
  with pytest.raises(ValueError, match="return_lse"):
    _dispatch_attention_fn(q, k, v, backend="ffpa_fp4", attention_kwargs={"return_lse": True})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize(
  "cap,name,allowed",
  [
    ((12, 0), "ffpa", True),
    ((12, 0), "ffpa_fp8", True),
    ((12, 0), "ffpa_fp8_per_block", True),
    ((12, 0), "ffpa_fp4", True),
    ((8, 9), "ffpa", True),
    ((8, 9), "ffpa_fp8", True),
    ((8, 9), "ffpa_fp8_per_block", True),
    # fp4 ships the CUTE_TMA_FP4_SM_120 family only.
    ((8, 9), "ffpa_fp4", False),
    ((8, 6), "ffpa", False),
    ((8, 6), "ffpa_fp8", False),
    ((9, 0), "ffpa", False),
    ((10, 0), "ffpa_fp8", False),
    ((10, 0), "ffpa_fp4", False),
    ((11, 0), "ffpa_fp8", False),
  ])
def test_ffpa_arch_gate_matrix(cap, name, allowed, monkeypatch):
  """Fp16/fp8 admit sm_120 and sm_89+; fp4 admits sm_120 only."""
  monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **kw: cap)
  # Stub the kernel entry: the arch gate is pure Python, no kernel needed.
  monkeypatch.setattr(ffpa_backend, "ffpa_attn_func", lambda query, *a, **kw: query)
  q, k, v = (x.cuda() for x in _cpu_qkv())
  if allowed:
    out = _dispatch_attention_fn(q, k, v, backend=name)
    assert out.shape == q.shape
  else:
    with pytest.raises(RuntimeError, match="sm_"):
      _dispatch_attention_fn(q, k, v, backend=name)


# ------------------------------------------------------------ 5090/5080 config


def test_is_geforce_5090_or_5080(monkeypatch):
  cases = [
    ("NVIDIA GeForce RTX 5090", True),
    ("NVIDIA GeForce RTX 5080", True),
    ("NVIDIA RTX PRO 5000 Blackwell Server Edition", False),
    ("NVIDIA RTX PRO 6000 Blackwell", False),
    ("NVIDIA H800", False),
  ]
  for name, expected in cases:
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, _n=name, **kw: _n)
    assert ffpa_backend._is_geforce_5090_or_5080(torch.device("cuda", 0)) is expected


@requires_sm120
def test_build_fp8_backend_consumer_config(monkeypatch):
  ffpa_backend._ffpa_backend_cache.clear()
  monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **kw: "NVIDIA GeForce RTX 5090")
  backend = ffpa_backend._build_ffpa_cuda_backend(torch.device("cuda"), enable_fp8=True)
  assert backend.enable_fp8 and not backend.enable_fp4
  assert backend.fp8_qk_mm_type == "int8"
  assert backend.fp8_pv_acc_type == "f16"
  assert backend.fp8_q_quant_method == "per_thread"
  assert backend.fp8_v_quant_method == "per_channel"


@requires_sm120
def test_build_fp8_backend_pro_config():
  ffpa_backend._ffpa_backend_cache.clear()
  backend = ffpa_backend._build_ffpa_cuda_backend(torch.device("cuda"), enable_fp8=True)
  assert backend.enable_fp8 and not backend.enable_fp4
  assert backend.fp8_qk_mm_type == "int8"
  assert backend.fp8_pv_acc_type == "f16"
  assert backend.fp8_q_quant_method == "per_thread"
  assert backend.fp8_k_quant_method == "per_thread"
  assert backend.fp8_v_quant_method == "per_channel"
  assert backend.fp8_hybrid is False
  assert backend.fp8_hybrid_n_early == 256
  causal = ffpa_backend._build_ffpa_cuda_backend(torch.device("cuda"),
                                                 enable_fp8=True,
                                                 is_causal=True)
  assert causal.fp8_hybrid_n_early == 128


@requires_sm120
def test_build_fp8_per_block_backend_config():
  ffpa_backend._ffpa_backend_cache.clear()
  backend = ffpa_backend._build_ffpa_cuda_backend(torch.device("cuda"),
                                                  enable_fp8=True,
                                                  fp8_preset="per_block")
  assert backend.enable_fp8 and not backend.enable_fp4
  assert backend.fp8_qk_mm_type == "int8"
  assert backend.fp8_pv_acc_type == "f32"
  assert backend.fp8_q_quant_method == "per_block"
  assert backend.fp8_k_quant_method == "per_block"
  assert backend.fp8_v_quant_method == "per_channel"
  assert backend.fp8_smooth_k is True
  assert backend.fp8_smooth_v is False
  assert backend.fp8_hybrid is False


@requires_sm120
def test_build_fp4_backend_hadamard_override():
  ffpa_backend._ffpa_backend_cache.clear()
  backend = ffpa_backend._build_ffpa_cuda_backend(torch.device("cuda"), enable_fp4=True)
  assert backend.enable_fp4 and backend.fp4_hadamard is False
  ffpa_backend.set_ffpa_fp4_hadamard(True)
  try:
    backend = ffpa_backend._build_ffpa_cuda_backend(torch.device("cuda"), enable_fp4=True)
    assert backend.fp4_hadamard is True
    # distinct cache entries per override state
    assert len(ffpa_backend._ffpa_backend_cache) == 2
    # fp8 backends keep the switch off
    fp8 = ffpa_backend._build_ffpa_cuda_backend(torch.device("cuda"), enable_fp8=True)
    assert fp8.fp4_hadamard is False
  finally:
    ffpa_backend.set_ffpa_fp4_hadamard(False)
  backend = ffpa_backend._build_ffpa_cuda_backend(torch.device("cuda"), enable_fp4=True)
  assert backend.fp4_hadamard is False


@requires_sm120
def test_build_fp8_backend_hadamard_override():
  ffpa_backend._ffpa_backend_cache.clear()
  backend = ffpa_backend._build_ffpa_cuda_backend(torch.device("cuda"), enable_fp8=True)
  assert backend.enable_fp8 and backend.fp8_hadamard is False
  ffpa_backend.set_ffpa_fp8_hadamard(True)
  try:
    backend = ffpa_backend._build_ffpa_cuda_backend(torch.device("cuda"), enable_fp8=True)
    assert backend.fp8_hadamard is True
    # distinct cache entries per override state
    assert len(ffpa_backend._ffpa_backend_cache) == 2
    # fp4 backends keep the switch off
    fp4 = ffpa_backend._build_ffpa_cuda_backend(torch.device("cuda"), enable_fp4=True)
    assert fp4.fp8_hadamard is False
  finally:
    ffpa_backend.set_ffpa_fp8_hadamard(False)
  backend = ffpa_backend._build_ffpa_cuda_backend(torch.device("cuda"), enable_fp8=True)
  assert backend.fp8_hadamard is False


@requires_sm120
def test_build_fp4_backend_pv_mm_type_smooth_v_override():
  ffpa_backend._ffpa_backend_cache.clear()
  backend = ffpa_backend._build_ffpa_cuda_backend(torch.device("cuda"), enable_fp4=True)
  assert backend.enable_fp4 and backend.fp4_pv_mm_type == "fp4"
  assert backend.fp4_smooth_v is False
  ffpa_backend.set_ffpa_fp4_pv_mm_type("fp8")
  ffpa_backend.set_ffpa_fp4_smooth_v(True)
  try:
    backend = ffpa_backend._build_ffpa_cuda_backend(torch.device("cuda"), enable_fp4=True)
    assert backend.fp4_pv_mm_type == "fp8"
    assert backend.fp4_smooth_v is True
    # distinct cache entries per override state
    assert len(ffpa_backend._ffpa_backend_cache) == 2
    # fp8 backends keep the fp4 defaults
    fp8 = ffpa_backend._build_ffpa_cuda_backend(torch.device("cuda"), enable_fp8=True)
    assert fp8.fp4_pv_mm_type == "fp4"
    assert fp8.fp4_smooth_v is False
  finally:
    ffpa_backend.set_ffpa_fp4_pv_mm_type(None)
    ffpa_backend.set_ffpa_fp4_smooth_v(False)
  backend = ffpa_backend._build_ffpa_cuda_backend(torch.device("cuda"), enable_fp4=True)
  assert backend.fp4_pv_mm_type == "fp4"
  assert backend.fp4_smooth_v is False


@requires_sm120
def test_build_fp8_backend_smooth_v_override():
  ffpa_backend._ffpa_backend_cache.clear()
  backend = ffpa_backend._build_ffpa_cuda_backend(torch.device("cuda"), enable_fp8=True)
  assert backend.fp8_smooth_v is False
  ffpa_backend.set_ffpa_fp8_smooth_v(True)
  try:
    backend = ffpa_backend._build_ffpa_cuda_backend(torch.device("cuda"), enable_fp8=True)
    assert backend.fp8_smooth_v is True
    per_block = ffpa_backend._build_ffpa_cuda_backend(torch.device("cuda"),
                                                      enable_fp8=True,
                                                      fp8_preset="per_block")
    assert per_block.fp8_smooth_v is True
    # distinct cache entries per override state / preset
    assert len(ffpa_backend._ffpa_backend_cache) == 3
    # fp4 backends keep the switch off
    fp4 = ffpa_backend._build_ffpa_cuda_backend(torch.device("cuda"), enable_fp4=True)
    assert fp4.fp8_smooth_v is False
  finally:
    ffpa_backend.set_ffpa_fp8_smooth_v(False)
  backend = ffpa_backend._build_ffpa_cuda_backend(torch.device("cuda"), enable_fp8=True)
  assert backend.fp8_smooth_v is False


@requires_sm120
def test_toy_model_dispatch_ffpa_fp8_head_dim_120(monkeypatch):
  # Non-32-multiple head_dim: Q/K per_thread and V per_channel quant are
  # D_og-aware and must still run the real fp8 kernel (D pads 120->128).
  spy = _FFPAFuncSpy()
  monkeypatch.setattr(ffpa_backend, "ffpa_attn_func", spy)

  model = ToyAttentionModel("ffpa_fp8")
  q, k, v = _make_qkv(D=120)
  out = model(q, k, v)

  assert len(spy.calls) == 1
  backend = spy.calls[0]
  assert backend.fp8_q_quant_method == "per_thread"
  assert backend.fp8_v_quant_method == "per_channel"

  ref = _sdpa_ref(q, k, v)
  assert _cos_sim(out, ref) > 0.99


@requires_sm120
def test_build_backend_rejects_fp8_and_fp4_together():
  with pytest.raises(ValueError, match="mutually exclusive"):
    ffpa_backend._build_ffpa_cuda_backend(torch.device("cuda"), enable_fp8=True, enable_fp4=True)


# ------------------------------------------------------ toy model dispatch tests


@requires_sm120
def test_toy_model_dispatch_ffpa_fp16(monkeypatch):
  spy = _FFPAFuncSpy()
  monkeypatch.setattr(ffpa_backend, "ffpa_attn_func", spy)

  model = ToyAttentionModel("ffpa")
  q, k, v = _make_qkv()
  out = model(q, k, v)

  assert len(spy.calls) == 1
  backend = spy.calls[0]
  assert backend.enable_fp8 is False and backend.enable_fp4 is False
  assert out.shape == q.shape

  ref = _sdpa_ref(q, k, v)
  assert _cos_sim(out, ref) > 0.999
  assert (out.float() - ref.float()).abs().mean().item() < 0.01


@requires_sm120
def test_toy_model_dispatch_ffpa_fp8(monkeypatch):
  spy = _FFPAFuncSpy()
  monkeypatch.setattr(ffpa_backend, "ffpa_attn_func", spy)

  model = ToyAttentionModel("ffpa_fp8")
  q, k, v = _make_qkv()
  out = model(q, k, v)

  assert len(spy.calls) == 1
  backend = spy.calls[0]
  assert backend.enable_fp8 is True and backend.enable_fp4 is False

  ref = _sdpa_ref(q, k, v)
  assert _cos_sim(out, ref) > 0.99
  assert (out.float() - ref.float()).abs().mean().item() < 0.05


@requires_sm120
def test_toy_model_dispatch_ffpa_fp8_noncontiguous_v(monkeypatch):
  # Single-stream blocks (e.g. FLUX.2) slice a fused QKV projection, so V
  # arrives as an interleaved chunk view (row stride wider than H*D). The
  # fp8 persist-D relaxed gate consumes such strided-NHD views zero-copy,
  # so the backend must pass the strided V through to ffpa_attn_func
  # untouched instead of materializing it.
  spy = _FFPAFuncSpy()
  monkeypatch.setattr(ffpa_backend, "ffpa_attn_func", spy)

  model = ToyAttentionModel("ffpa_fp8")
  B, N, H, D = 1, 1024, 16, 128
  proj = torch.randn(B, N, 3 * H * D, dtype=torch.bfloat16, device="cuda")
  q, k, v = proj.chunk(3, dim=-1)
  q = q.unflatten(-1, (H, D)).contiguous()
  k = k.unflatten(-1, (H, D)).contiguous()
  v = v.unflatten(-1, (H, D))
  assert not v.is_contiguous()

  # Inference semantics (pipeline runs under torch.no_grad) enable the NHD gate.
  with torch.no_grad():
    out = model(q, k, v)

  assert len(spy.calls) == 1
  assert spy.calls[0].tensor_layout == "NHD"
  # Zero-copy: the strided V view reaches ffpa_attn_func unmaterialized.
  v_passed = spy.arg_calls[0][2]
  assert not v_passed.is_contiguous()
  assert v_passed.data_ptr() == v.data_ptr() and v_passed.stride() == v.stride()
  # NHD direct output is packed; the old BHND fallback returned a strided view.
  assert out.shape == q.shape and out.is_contiguous()

  with torch.no_grad():
    ref = model(q, k, v.contiguous())
  assert torch.equal(out, ref)
  assert _cos_sim(out, _sdpa_ref(q, k, v.contiguous())) > 0.99


@requires_sm120
def test_toy_model_dispatch_ffpa_fp8_per_block(monkeypatch):
  spy = _FFPAFuncSpy()
  monkeypatch.setattr(ffpa_backend, "ffpa_attn_func", spy)

  model = ToyAttentionModel("ffpa_fp8_per_block")
  q, k, v = _make_qkv()
  out = model(q, k, v)

  assert len(spy.calls) == 1
  backend = spy.calls[0]
  assert backend.enable_fp8 is True and backend.enable_fp4 is False
  assert backend.fp8_qk_mm_type == "int8"
  assert backend.fp8_pv_acc_type == "f32"
  assert backend.fp8_q_quant_method == "per_block"
  assert backend.fp8_k_quant_method == "per_block"
  assert backend.fp8_v_quant_method == "per_channel"
  assert backend.fp8_smooth_v is False
  assert backend.fp8_hybrid is False

  # per_block is the lowest-precision fp8 config; use fp4-level tolerances.
  ref = _sdpa_ref(q, k, v)
  assert _cos_sim(out, ref) > 0.9
  assert (out.float() - ref.float()).abs().mean().item() < 0.15


@requires_sm120
def test_toy_model_dispatch_ffpa_fp4(monkeypatch):
  spy = _FFPAFuncSpy()
  monkeypatch.setattr(ffpa_backend, "ffpa_attn_func", spy)

  model = ToyAttentionModel("ffpa_fp4")
  q, k, v = _make_qkv()
  out = model(q, k, v)

  assert len(spy.calls) == 1
  backend = spy.calls[0]
  assert backend.enable_fp4 is True and backend.enable_fp8 is False

  ref = _sdpa_ref(q, k, v)
  assert _cos_sim(out, ref) > 0.9
  assert (out.float() - ref.float()).abs().mean().item() < 0.15


@requires_sm120
@pytest.mark.parametrize("name", _FFPA_BACKENDS)
def test_toy_model_dispatch_gqa(monkeypatch, name: str):
  spy = _FFPAFuncSpy()
  monkeypatch.setattr(ffpa_backend, "ffpa_attn_func", spy)

  model = ToyAttentionModel(name)
  # 4:1 GQA: 16 query heads over 4 kv heads.
  q, k, v = _make_qkv(H=16, H_kv=4)
  out = model(q, k, v, enable_gqa=True)

  assert len(spy.calls) == 1
  assert out.shape == q.shape
  assert not out.isnan().any()

  ref = _sdpa_ref(q, k, v, enable_gqa=True)
  assert _cos_sim(out, ref) > 0.9


@requires_sm120
@pytest.mark.parametrize("name", _FFPA_BACKENDS)
def test_ffpa_no_silent_sdpa_fallback(monkeypatch, name: str):
  # ffpa_attn_func silently routes to SDPA when the CUDA path cannot run
  # (e.g. small D without the env flag, or N < 512). FFPAAttnFunc.apply is
  # only reached on the real FFPA kernel path.
  import ffpa_attn.functional as ffpa_functional

  calls: list = []
  orig_apply = ffpa_functional.FFPAAttnFunc.apply

  def _spy_apply(*args, **kwargs):
    calls.append(args)
    return orig_apply(*args, **kwargs)

  monkeypatch.setattr(ffpa_functional.FFPAAttnFunc, "apply", staticmethod(_spy_apply))

  model = ToyAttentionModel(name)
  q, k, v = _make_qkv()
  out = model(q, k, v)
  torch.cuda.synchronize()

  assert len(calls) >= 1, f"{name} silently fell back to SDPA"
  assert out.shape == q.shape


# ------------------------------------------------------------ attn mask dispatch


def _masked_qkv(B=1, N=1024, H=16, D=128, pad=128):
  q, k, v = _make_qkv(B=B, N=N, H=H, D=D)
  mask = torch.ones(B, N, dtype=torch.bool, device="cuda")
  mask[:, -pad:] = False
  return q, k, v, mask


# Calibrated on RTX PRO 5000 (sm_120) against the SDPA reference, worst case
# over 5 seeds with a tail-128 key-padding mask (B=1, N=1024, H=16, D=128):
# cos >= 0.99999 (fp16), 0.9993 (fp8 / per_block), 0.9816 (fp4); mean-abs
# <= 1.2e-4 / 1.7e-3 / 1.7e-3 / 8.3e-3. A window-shifted wrong mask scores
# cos ~0.84-0.86 (negative control margin >= 0.14).
_MASK_PARITY_TOLERANCES = {
  "ffpa": (0.9999, 5e-4),
  "ffpa_fp8": (0.998, 5e-3),
  "ffpa_fp8_per_block": (0.995, 5e-3),
  "ffpa_fp4": (0.96, 2e-2),
}


@requires_sm120
@pytest.mark.parametrize("name", _FFPA_BACKENDS)
@pytest.mark.parametrize("mask_ndim", [2, 4])
def test_toy_model_dispatch_ffpa_attn_mask(monkeypatch, name: str, mask_ndim: int):
  # Masked calls must (1) run the real FFPA kernels instead of a silent
  # SDPA fallback, (2) stay on the HND route — the NHD fast path is
  # bias-free — (3) forward a normalized [B, 1, 1, Nkv] mask, (4) actually
  # change the result, and (5) match the SDPA reference with the same mask
  # while a window-shifted negative control stays clearly worse.
  import ffpa_attn.functional as ffpa_functional

  spy = _FFPAFuncSpy()
  monkeypatch.setattr(ffpa_backend, "ffpa_attn_func", spy)
  kernel_calls: list = []
  orig_apply = ffpa_functional.FFPAAttnFunc.apply

  def _spy_apply(*args, **kwargs):
    kernel_calls.append(args)
    return orig_apply(*args, **kwargs)

  monkeypatch.setattr(ffpa_functional.FFPAAttnFunc, "apply", staticmethod(_spy_apply))

  model = ToyAttentionModel(name)
  q, k, v, mask = _masked_qkv()
  if mask_ndim == 4:
    mask = mask[:, None, None, :]

  out = model(q, k, v, attn_mask=mask)
  torch.cuda.synchronize()

  assert len(kernel_calls) >= 1, f"{name} silently fell back to SDPA"
  assert len(spy.calls) == 1
  assert spy.calls[0].tensor_layout == "HND"
  assert tuple(spy.kwarg_calls[0]["attn_mask"].shape) == (1, 1, 1, 1024)

  unmasked = model(q, k, v)
  assert not torch.allclose(out.float(), unmasked.float())

  mask4d = mask if mask.ndim == 4 else mask[:, None, None, :]
  ref = _sdpa_ref(q, k, v, attn_mask=mask4d)
  cos_min, mad_max = _MASK_PARITY_TOLERANCES[name]
  cos = _cos_sim(out, ref)
  assert cos > cos_min
  assert (out.float() - ref.float()).abs().mean().item() < mad_max

  # Negative control: masking the wrong (shifted) window must score clearly
  # worse, proving the parity check actually probes mask positions.
  wrong = torch.ones_like(mask4d)
  wrong[:, :, :, -(2 * 128):-128] = False
  ref_wrong = _sdpa_ref(q, k, v, attn_mask=wrong)
  assert cos > _cos_sim(out, ref_wrong) + 0.05


@requires_sm120
def test_toy_model_dispatch_ffpa_attn_mask_float(monkeypatch):
  # Float masks are additive biases; fp32 masks are downcast by ffpa-attn.
  spy = _FFPAFuncSpy()
  monkeypatch.setattr(ffpa_backend, "ffpa_attn_func", spy)

  model = ToyAttentionModel("ffpa")
  q, k, v, _ = _masked_qkv()
  float_mask = torch.zeros(1, 1, 1, 1024, dtype=torch.float32, device="cuda")
  float_mask[:, :, :, -128:] = float("-inf")

  out = model(q, k, v, attn_mask=float_mask)

  # The fp32 mask reaches ffpa_attn_func untouched (ffpa-attn downcasts it).
  assert spy.kwarg_calls[0]["attn_mask"].dtype == torch.float32
  assert tuple(spy.kwarg_calls[0]["attn_mask"].shape) == (1, 1, 1, 1024)
  # SDPA requires the bias dtype to match the query, mirroring the downcast.
  ref = _sdpa_ref(q, k, v, attn_mask=float_mask.to(q.dtype))
  assert _cos_sim(out, ref) > 0.999


@requires_sm120
def test_ffpa_attn_mask_shape_normalization(monkeypatch):
  # 1-D / 2-D masks expand to the [B, 1, 1, Nkv] SDPA form; 3-D masks pass
  # through untouched (already broadcast-compatible).
  spy = _FFPAFuncSpy()
  monkeypatch.setattr(ffpa_backend, "ffpa_attn_func", spy)

  model = ToyAttentionModel("ffpa")
  q, k, v = _make_qkv()
  mask1d = torch.ones(1024, dtype=torch.bool, device="cuda")
  mask1d[-128:] = False
  mask3d = torch.ones(1, 1024, 1024, dtype=torch.bool, device="cuda")
  mask3d[:, :, -128:] = False

  cases = [
    (mask1d, (1, 1, 1, 1024)),
    (mask1d[None, :], (1, 1, 1, 1024)),
    (mask3d, (1, 1024, 1024)),
  ]
  for mask, expected in cases:
    model(q, k, v, attn_mask=mask)
    assert tuple(spy.kwarg_calls[-1]["attn_mask"].shape) == expected


@requires_sm120
def test_toy_model_dispatch_ffpa_attn_mask_batch(monkeypatch):
  # A 2-D [B, Nkv] mask must expand along the batch axis, not into Nq.
  spy = _FFPAFuncSpy()
  monkeypatch.setattr(ffpa_backend, "ffpa_attn_func", spy)

  model = ToyAttentionModel("ffpa")
  q, k, v, mask = _masked_qkv(B=2)
  out = model(q, k, v, attn_mask=mask)

  assert tuple(spy.kwarg_calls[0]["attn_mask"].shape) == (2, 1, 1, 1024)
  ref = _sdpa_ref(q, k, v, attn_mask=mask[:, None, None, :])
  assert _cos_sim(out, ref) > 0.9999


@requires_sm120
def test_toy_model_dispatch_ffpa_attn_mask_native_family(monkeypatch):
  # On sm_89 the TMA / CUTE_TMA hints do not resolve (they require sm_120+),
  # so the fp16 route lands on the cp.async NATIVE family. Exercise that same
  # family with a mask here as a proxy for the Ada route (shared templates).
  from ffpa_attn import CUDABackend

  native = CUDABackend(backward=False, enable_tma=False, enable_cute=False)
  monkeypatch.setattr(ffpa_backend, "_build_ffpa_cuda_backend", lambda *a, **kw: native)

  model = ToyAttentionModel("ffpa")
  q, k, v, mask = _masked_qkv()
  out = model(q, k, v, attn_mask=mask)

  ref = _sdpa_ref(q, k, v, attn_mask=mask[:, None, None, :])
  assert _cos_sim(out, ref) > 0.9999
  assert (out.float() - ref.float()).abs().mean().item() < 5e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_ffpa_sm89_keeps_hnd_route(monkeypatch):
  # The NHD O-store branch lives in the sm_120 CUTE_TMA family only, so a
  # (faked) sm_89 device must stay on the HND route.
  monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **kw: (8, 9))
  spy = _FFPAFuncSpy()
  monkeypatch.setattr(ffpa_backend, "ffpa_attn_func", spy)

  q, k, v = (x.cuda() for x in _cpu_qkv())
  with torch.no_grad():
    ffpa_backend._ffpa_attention_impl(q, k, v, enable_fp8=True)

  assert len(spy.calls) == 1
  assert spy.calls[0].tensor_layout == "HND"
  assert spy.calls[0].enable_fp8 is True


@requires_sm120
def test_ffpa_sm120_maskless_keeps_nhd_route(monkeypatch):
  # Pair of test_ffpa_sm89_keeps_hnd_route: on sm_120 the maskless inference
  # path must still take the NHD fast path (real kernel).
  spy = _FFPAFuncSpy()
  monkeypatch.setattr(ffpa_backend, "ffpa_attn_func", spy)

  q, k, v, _ = _masked_qkv()
  with torch.no_grad():
    ffpa_backend._ffpa_attention_impl(q, k, v)

  assert len(spy.calls) == 1
  assert spy.calls[0].tensor_layout == "NHD"


@requires_sm120
@pytest.mark.parametrize("name", _FFPA_BACKENDS)
def test_ffpa_cp_with_attn_mask_raises(name: str):
  # Templated CP has no mask shard/order contract for ffpa, so the mask must
  # stay a loud error instead of being silently reordered.
  from cache_dit.attention.backends.register import _ContextParallelConfig

  q, k, v, mask = _masked_qkv()
  with pytest.raises(ValueError, match="attn_mask"):
    _dispatch_attention_fn(
      q,
      k,
      v,
      backend=name,
      attn_mask=mask,
      cp_config=_ContextParallelConfig(ulysses_degree=2),
    )
