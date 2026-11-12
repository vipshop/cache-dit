"""Accuracy and dispatch tests for the FLUX.3 Klein operator family.

The operators replace expressions diffusers already evaluates. Unlike the Qwen-Image 2.1
family, the reference here is **not** the eager chain: the compiled step (which is the arm the
fused operators are measured against) fuses each of these chains into one kernel and keeps the
intermediate in fp32, so eager rounds to bf16 more often than the compiled graph does. Every
kernel therefore keeps the chain in fp32 with a single bf16 store, and the tests compare
against an explicit fp32-chain reference at a tolerance that only the `rsqrtf` fast-math form
can consume. The pure multiply/add operators come out far inside it.

The eager transcription in :mod:`cache_dit.models.pipelines.flux3_klein.fused_ops` is still
tested, because that is what runs when a precondition fails: it must equal the stock modules
bit for bit.
"""

import pytest
import torch

from cache_dit.kernels import cuda_ext_is_available
from cache_dit.models.pipelines.flux3_klein import fused_ops

pytestmark = [
  pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
  pytest.mark.skipif(not cuda_ext_is_available(), reason="needs cache_dit._C_ops"),
  pytest.mark.skipif(not fused_ops._kernel_ops_ready(),
                     reason="needs the FLUX.3 Klein entry points in cache_dit._C_ops"),
]

# These tests cover the kernels, and the family starts disabled: `install_fused_modules()` is what
# turns it on, and it is not run here. Force it on at import - a module-level `pytestmark` skip
# never executes a fixture, so an autouse one would silently not run - or every `apply_*` assertion
# would compare the eager expression with itself. `tests/api/test_env.py` pins the switch.
fused_ops.FUSED_OPS_ENABLED = True

DEVICE = torch.device("cuda")
HEAD_DIM = 128


def _fp32_chain_qk_norm_rope(x, weight, cos, sin, eps):
  """Reference: fp32 RMSNorm, interleaved rotation, one bf16 rounding."""

  normed = torch.nn.functional.rms_norm(x.float(), (HEAD_DIM, ), weight, eps)
  real, imag = normed.reshape(*normed.shape[:-1], -1, 2).unbind(-1)
  rotated = torch.stack([-imag, real], dim=-1).flatten(3)
  out = normed * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)
  return out.to(x.dtype)


def _fp32_chain_ln_modulate(x, scale, shift, eps):
  """Reference: LayerNorm then the modulation, all in fp32, one bf16 store."""

  centered = x.float() - x.float().mean(-1, keepdim=True)
  variance = centered.pow(2).mean(-1, keepdim=True)
  normed = centered * torch.rsqrt(variance + eps)
  return (normed * (1 + scale.float()) + shift.float()).to(x.dtype)


def test_qk_norm_rope_matches_the_fp32_chain():
  from cache_dit.kernels import flux3_klein_qk_norm_rope

  torch.manual_seed(0)
  seq = 64
  x = torch.randn(1, seq, 4, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
  weight = (1 + 0.1 * torch.randn(HEAD_DIM, device=DEVICE)).to(torch.float32)
  angles = torch.randn(seq, HEAD_DIM // 2, device=DEVICE, dtype=torch.float64)
  cos = angles.cos().repeat_interleave(2, dim=1).float()
  sin = angles.sin().repeat_interleave(2, dim=1).float()

  reference = _fp32_chain_qk_norm_rope(x, weight, cos, sin, 1e-6)
  actual = flux3_klein_qk_norm_rope(x, weight, cos, sin, 1e-6)
  assert fused_ops.relative_l2(actual, reference) <= fused_ops.REL_L2_TOL


def test_qk_norm_rope_eager_fallback_is_bit_exact():
  x = torch.randn(1, 8, 2, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
  weight = torch.ones(HEAD_DIM, device=DEVICE, dtype=torch.float32)
  cos = torch.ones(8, HEAD_DIM, device=DEVICE, dtype=torch.float32)
  sin = torch.zeros(8, HEAD_DIM, device=DEVICE, dtype=torch.float32)

  reference = torch.nn.functional.rms_norm(x.float(), (HEAD_DIM, ), weight, 1e-6).to(x.dtype)
  assert torch.equal(fused_ops._eager_qk_norm_rope(x, weight, cos, sin, 1e-6), reference)


def test_ln_modulate_matches_the_fp32_chain():
  from cache_dit.kernels import flux3_klein_ln_modulate

  torch.manual_seed(0)
  dim = 1024
  x = torch.randn(2, 32, dim, device=DEVICE, dtype=torch.bfloat16)
  scale = torch.randn(2, 32, dim, device=DEVICE, dtype=torch.bfloat16) * 0.1
  shift = torch.randn(2, 32, dim, device=DEVICE, dtype=torch.bfloat16) * 0.1

  reference = _fp32_chain_ln_modulate(x, scale, shift, 1e-6)
  actual = flux3_klein_ln_modulate(x, scale, shift, 1e-6)
  # A tighter class than the rotation: nothing transcendental, so the only freedom is the
  # reduction order of the mean and the variance. That surfaces as a bf16 ulp on a few elements.
  assert fused_ops.relative_l2(actual, reference) <= 3e-6
  assert (actual != reference).float().mean() < 0.05


def test_ln_modulate_accepts_a_per_sample_row():
  from cache_dit.kernels import flux3_klein_ln_modulate

  torch.manual_seed(0)
  dim = 1024
  x = torch.randn(2, 16, dim, device=DEVICE, dtype=torch.bfloat16)
  scale = torch.randn(2, 1, dim, device=DEVICE, dtype=torch.bfloat16) * 0.1
  shift = torch.randn(2, 1, dim, device=DEVICE, dtype=torch.bfloat16) * 0.1

  reference = _fp32_chain_ln_modulate(x, scale, shift, 1e-6)
  actual = flux3_klein_ln_modulate(x, scale, shift, 1e-6)
  assert fused_ops.relative_l2(actual, reference) <= fused_ops.REL_L2_TOL


def test_ln_modulate_cat_puts_the_text_half_first():
  from cache_dit.kernels import flux3_klein_ln_modulate_cat

  torch.manual_seed(0)
  dim = 1024
  hidden = torch.randn(1, 24, dim, device=DEVICE, dtype=torch.bfloat16)
  encoder = torch.randn(1, 5, dim, device=DEVICE, dtype=torch.bfloat16)
  scale = torch.randn(1, 24, dim, device=DEVICE, dtype=torch.bfloat16) * 0.1
  shift = torch.randn(1, 24, dim, device=DEVICE, dtype=torch.bfloat16) * 0.1
  context_scale = torch.randn(1, 5, dim, device=DEVICE, dtype=torch.bfloat16) * 0.1
  context_shift = torch.randn(1, 5, dim, device=DEVICE, dtype=torch.bfloat16) * 0.1

  reference = torch.cat([
    _fp32_chain_ln_modulate(encoder, context_scale, context_shift, 1e-6),
    _fp32_chain_ln_modulate(hidden, scale, shift, 1e-6),
  ],
                        dim=1)
  actual = flux3_klein_ln_modulate_cat(hidden, encoder, scale, shift, context_scale, context_shift,
                                       1e-6)
  assert actual.shape == (1, 29, dim)
  assert fused_ops.relative_l2(actual, reference) <= fused_ops.REL_L2_TOL


def test_gated_residual_is_exact():
  from cache_dit.kernels import flux3_klein_gated_residual

  torch.manual_seed(0)
  dim = 1024
  hidden = torch.randn(1, 32, dim, device=DEVICE, dtype=torch.bfloat16)
  gate = torch.randn(1, 32, dim, device=DEVICE, dtype=torch.bfloat16) * 0.1
  attn = torch.randn(1, 32, dim, device=DEVICE, dtype=torch.bfloat16)
  ff = torch.randn(1, 32, dim, device=DEVICE, dtype=torch.bfloat16)

  reference = fused_ops._compiled_gated_residual(hidden, gate, attn, ff)
  assert torch.equal(flux3_klein_gated_residual(hidden, gate, attn, ff), reference)


def test_gated_residual_split_matches_the_upstream_split():
  from cache_dit.kernels import flux3_klein_gated_residual_split

  torch.manual_seed(0)
  dim = 1024
  prefix, content = 5, 24
  hidden = torch.randn(1, content, dim, device=DEVICE, dtype=torch.bfloat16)
  encoder = torch.randn(1, prefix, dim, device=DEVICE, dtype=torch.bfloat16)
  gate = torch.randn(1, content, dim, device=DEVICE, dtype=torch.bfloat16) * 0.1
  context_gate = torch.randn(1, prefix, dim, device=DEVICE, dtype=torch.bfloat16) * 0.1
  attn = torch.randn(1, prefix + content, dim, device=DEVICE, dtype=torch.bfloat16)
  ff = torch.randn(1, prefix + content, dim, device=DEVICE, dtype=torch.bfloat16)

  # The oracle is the chain the *compiled* step runs: the branch sum is materialised (and so
  # rounded once), the gated add stays in fp32 with one store rounding.
  expected_encoder, expected_hidden = fused_ops._compiled_gated_residual_split(
    hidden, encoder, gate, context_gate, attn, ff, prefix)

  encoder_out, hidden_out = flux3_klein_gated_residual_split(hidden, encoder, gate, context_gate,
                                                             attn, ff, prefix)
  assert torch.equal(encoder_out, expected_encoder)
  assert torch.equal(hidden_out, expected_hidden)


def test_non_contiguous_input_falls_back_to_the_eager_chain():
  torch.manual_seed(0)
  dim = 1024
  x = torch.randn(1, dim, 16, device=DEVICE, dtype=torch.bfloat16).transpose(1, 2)
  scale = torch.zeros(1, 16, dim, device=DEVICE, dtype=torch.bfloat16)
  shift = torch.zeros(1, 16, dim, device=DEVICE, dtype=torch.bfloat16)

  assert not x.is_contiguous()
  actual = fused_ops.apply_ln_modulate(x, scale, shift, 1e-6)
  assert torch.equal(actual, fused_ops._eager_ln_modulate(x, scale, shift, 1e-6))


def test_dispatch_is_counted_so_installed_never_implies_ran():
  dim = 1024
  x = torch.randn(1, 8, dim, device=DEVICE, dtype=torch.bfloat16)
  scale = torch.zeros(1, 8, dim, device=DEVICE, dtype=torch.bfloat16)
  shift = torch.zeros(1, 8, dim, device=DEVICE, dtype=torch.bfloat16)
  fused_ops.reset_fused_ops_stats()
  fused_ops.apply_ln_modulate(x, scale, shift, 1e-6)
  stats = fused_ops.fused_ops_stats()
  assert stats["hits"]["ln_modulate"] == 1
  assert stats["fallbacks"]["ln_modulate"] == 0

  # A layout the kernels cannot take must be counted as a fallback, never as a hit.
  transposed = torch.randn(1, dim, 8, device=DEVICE, dtype=torch.bfloat16).transpose(1, 2)
  assert not transposed.is_contiguous()
  fused_ops.apply_ln_modulate(transposed, scale, shift, 1e-6)
  stats = fused_ops.fused_ops_stats()
  assert stats["hits"]["ln_modulate"] == 1
  assert stats["fallbacks"]["ln_modulate"] == 1


def test_narrowing_the_allow_list_really_skips_the_kernel(monkeypatch):
  """The subset is a code constant, and the dispatch paths read it - not just a list somewhere."""

  assert fused_ops.ACTIVE_OPS == frozenset(fused_ops.ALL_OPS)

  dim = 1024
  x = torch.randn(1, 8, dim, device=DEVICE, dtype=torch.bfloat16)
  scale = torch.zeros(1, 8, dim, device=DEVICE, dtype=torch.bfloat16)
  shift = torch.zeros(1, 8, dim, device=DEVICE, dtype=torch.bfloat16)

  monkeypatch.setattr(fused_ops, "ACTIVE_OPS", frozenset())
  fused_ops.reset_fused_ops_stats()
  assert torch.equal(fused_ops.apply_ln_modulate(x, scale, shift, 1e-6),
                     fused_ops._eager_ln_modulate(x, scale, shift, 1e-6))
  stats = fused_ops.fused_ops_stats()
  assert stats["hits"]["ln_modulate"] == 0
  # Counted, so "the operator was listed" can never be mistaken for "the kernel ran".
  assert stats["fallbacks"]["ln_modulate"] == 1


def _tiny_transformer():
  """A FLUX.3 transformer small enough to run on a CPU-sized budget."""

  from diffusers import Flux3Transformer2DModel

  torch.manual_seed(0)
  transformer = Flux3Transformer2DModel(
    in_channels=8,
    cond_in_channels=8,
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=HEAD_DIM,
    num_attention_heads=8,
    joint_attention_dim=64,
    timestep_channels=8,
    flow_map=True,
  ).to(DEVICE, torch.bfloat16)
  # `norm_q` / `norm_k` live in `_keep_in_fp32_modules`, so `from_pretrained` leaves their weight
  # in fp32 while everything else is bf16. A blanket `.to(bfloat16)` would not reproduce that,
  # and the kernel precondition would (correctly) reject it.
  for module in transformer.modules():
    if type(module).__name__ == "RMSNorm" and module.weight is not None:
      module.weight.data = module.weight.data.float()
  return transformer


def _tiny_inputs(dim):
  image_tokens, text_tokens = 16, 5
  return dict(
    hidden_states=torch.randn(1, image_tokens, 8, device=DEVICE, dtype=torch.bfloat16),
    encoder_hidden_states=torch.randn(1, text_tokens, 64, device=DEVICE, dtype=torch.bfloat16),
    timestep=torch.full((1, ), 0.5, device=DEVICE, dtype=torch.bfloat16),
    img_ids=torch.zeros(image_tokens, 4, device=DEVICE),
    txt_ids=torch.zeros(text_tokens, 4, device=DEVICE),
    target_timestep=torch.full((1, ), 0.25, device=DEVICE, dtype=torch.bfloat16),
    return_dict=False,
  )


def test_model_level_fused_pass_matches_the_stock_forward_and_restores():
  from cache_dit.models.pipelines.flux3_klein import _Flux3KleinTransformer2DModel

  transformer = _tiny_transformer()
  inputs = _tiny_inputs(transformer.inner_dim)
  with torch.no_grad():
    reference = transformer(**inputs)[0]

  transformer.__class__ = _Flux3KleinTransformer2DModel
  transformer.apply_fused_ops()
  fused_ops.reset_fused_ops_stats()
  with torch.no_grad():
    fused = transformer(**inputs)[0]
  # Every operator has to be observed on the kernel path: a precondition that silently rejects
  # the real tensor shapes would otherwise leave the test green on the eager transcription.
  stats = fused_ops.fused_ops_stats()
  for op in ("qk_norm_rope", "ln_modulate", "gated_residual"):
    assert stats["hits"][op] > 0, f"{op} never reached its kernel: {stats}"
    assert stats["fallbacks"][op] == 0, f"{op} fell back: {stats}"
  assert fused_ops.relative_l2(fused, reference) <= fused_ops.REL_L2_TOL

  transformer.remove_fused_ops()
  with torch.no_grad():
    restored = transformer(**inputs)[0]
  assert torch.equal(restored, reference)


def test_fused_forward_survives_fullgraph_compile():
  from cache_dit.models.pipelines.flux3_klein import _Flux3KleinTransformer2DModel

  transformer = _tiny_transformer()
  transformer.__class__ = _Flux3KleinTransformer2DModel
  transformer.apply_fused_ops()
  inputs = _tiny_inputs(transformer.inner_dim)

  compiled = torch.compile(transformer.forward, fullgraph=True, dynamic=False)
  with torch.no_grad():
    out = compiled(**inputs)[0]
  assert out.shape == inputs["hidden_states"].shape
