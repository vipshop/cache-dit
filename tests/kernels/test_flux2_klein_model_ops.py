"""Parity and dispatch tests for the FLUX.2-Klein operator family.

The operators replace expressions diffusers already evaluates, so each test compares against
the eager branch in :mod:`cache_dit.models.pipelines.flux2_klein.fused_ops`. Where the chain is
pure multiply/add the assertion is `torch.equal`, because the only way those kernels can be
wrong is a missing bf16 rounding point or an FMA that torch does not emit - a tolerance would
hide exactly that. The normalisations can only differ in their reduction tree, so they are
gated at a relative L2 that one bf16 ulp satisfies (measured 1.1e-5).
"""

import pytest
import torch

from cache_dit.kernels import cuda_ext_is_available
from cache_dit.models.pipelines.flux2_klein import fused_ops

pytestmark = [
  pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
  pytest.mark.skipif(not cuda_ext_is_available(), reason="needs cache_dit._C_ops"),
]

DEVICE = torch.device("cuda")
DIM = 3072
HEAD_DIM = 128
HEADS = 24


def _rows(count: int = 1) -> torch.Tensor:
  return torch.randn(count, DIM, device=DEVICE, dtype=torch.bfloat16)


def test_ln_modulate_matches_the_eager_chain():
  from cache_dit.kernels import flux2_klein_ln_modulate

  torch.manual_seed(0)
  x = torch.randn(1, 64, DIM, device=DEVICE, dtype=torch.bfloat16)
  one_plus = (1 + torch.randn(1, 1, DIM, device=DEVICE, dtype=torch.bfloat16)).reshape(1, DIM)
  shift = torch.randn(1, 1, DIM, device=DEVICE, dtype=torch.bfloat16).reshape(1, DIM)

  kernel = flux2_klein_ln_modulate(x, one_plus, shift, 1e-6)
  reference = fused_ops._eager_ln_modulate(x, one_plus, shift, 1e-6)
  assert fused_ops.relative_l2(kernel, reference) <= fused_ops.NORM_REL_L2_TOL


def test_residual_ln_modulate_matches_the_eager_chain():
  from cache_dit.kernels import flux2_klein_residual_ln_modulate

  torch.manual_seed(0)
  hidden = torch.randn(1, 64, DIM, device=DEVICE, dtype=torch.bfloat16)
  delta = torch.randn(1, 64, DIM, device=DEVICE, dtype=torch.bfloat16)
  gate = _rows()
  one_plus = _rows()
  shift = _rows()

  hidden_out, mod_out = flux2_klein_residual_ln_modulate(hidden, gate, delta, one_plus, shift, 1e-6)
  eager_hidden, eager_mod = fused_ops._eager_residual_ln_modulate(hidden, gate, delta, one_plus,
                                                                  shift, 1e-6)
  # The residual add is pure bf16 multiply/add, so it has to be bit-exact.
  assert torch.equal(hidden_out, eager_hidden)
  assert fused_ops.relative_l2(mod_out, eager_mod) <= fused_ops.NORM_REL_L2_TOL


def test_gated_residual_is_bit_exact():
  from cache_dit.kernels import flux2_klein_gated_residual

  torch.manual_seed(0)
  hidden = torch.randn(1, 32, DIM, device=DEVICE, dtype=torch.bfloat16)
  update = torch.randn(1, 32, DIM, device=DEVICE, dtype=torch.bfloat16)
  gate = _rows().reshape(DIM)

  assert torch.equal(flux2_klein_gated_residual(hidden, gate, update),
                     fused_ops._eager_gated_residual(hidden, gate.unsqueeze(0), update))


def test_silu_mul_matches_the_eager_chain():
  from cache_dit.kernels import flux2_klein_silu_mul

  torch.manual_seed(0)
  packed = torch.randn(1, 32, 2 * 3 * DIM, device=DEVICE, dtype=torch.bfloat16)

  assert fused_ops.relative_l2(flux2_klein_silu_mul(packed),
                               fused_ops._eager_silu_mul(packed)) <= fused_ops.REL_L2_TOL


def test_qk_norm_rope_matches_the_eager_chain():
  from cache_dit.kernels import flux2_klein_qk_norm_rope

  torch.manual_seed(0)
  seq = 48
  x = torch.randn(1, seq, HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
  weight = (1 + 0.1 * torch.randn(HEAD_DIM, device=DEVICE)).to(torch.bfloat16)
  cos = torch.randn(seq, HEAD_DIM, device=DEVICE, dtype=torch.float32)
  sin = torch.randn(seq, HEAD_DIM, device=DEVICE, dtype=torch.float32)

  kernel = flux2_klein_qk_norm_rope(x, weight, cos, sin, 0, 1e-6).unsqueeze(0)
  reference = fused_ops._eager_qk_norm_rope(x, weight, cos, sin, 0, 1e-6)
  # The rotation is exact; only the 128-element reduction tree can differ, by one ulp.
  assert fused_ops.relative_l2(kernel, reference) <= fused_ops.NORM_REL_L2_TOL


def test_qk_norm_rope_reads_a_fused_projection_chunk_in_place():
  """The real caller hands over `to_qkv(...).chunk(3, -1)[0]`, which is not contiguous."""

  from cache_dit.kernels import flux2_klein_qk_norm_rope

  torch.manual_seed(0)
  seq = 16
  fused_width = 3 * HEADS * HEAD_DIM
  fused = torch.randn(1, seq, fused_width, device=DEVICE, dtype=torch.bfloat16)
  chunk = fused.chunk(3, dim=-1)[0].unflatten(-1, (HEADS, HEAD_DIM))
  assert not chunk.is_contiguous()

  weight = (1 + 0.1 * torch.randn(HEAD_DIM, device=DEVICE)).to(torch.bfloat16)
  cos = torch.randn(seq, HEAD_DIM, device=DEVICE, dtype=torch.float32)
  sin = torch.randn(seq, HEAD_DIM, device=DEVICE, dtype=torch.float32)

  kernel = flux2_klein_qk_norm_rope(chunk, weight, cos, sin, 0, 1e-6)
  reference = fused_ops._eager_qk_norm_rope(chunk, weight, cos, sin, 0, 1e-6)
  assert fused_ops.relative_l2(kernel, reference.squeeze(0)) <= fused_ops.NORM_REL_L2_TOL


def test_qk_norm_rope_join_matches_the_concat():
  from cache_dit.kernels import flux2_klein_qk_norm_rope_join

  torch.manual_seed(0)
  image_seq, text_seq = 32, 8
  image = torch.randn(1, image_seq, HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
  text = torch.randn(1, text_seq, HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
  image_weight = (1 + 0.1 * torch.randn(HEAD_DIM, device=DEVICE)).to(torch.bfloat16)
  text_weight = (1 + 0.1 * torch.randn(HEAD_DIM, device=DEVICE)).to(torch.bfloat16)
  cos = torch.randn(image_seq + text_seq, HEAD_DIM, device=DEVICE, dtype=torch.float32)
  sin = torch.randn(image_seq + text_seq, HEAD_DIM, device=DEVICE, dtype=torch.float32)

  joined = flux2_klein_qk_norm_rope_join(image, image_weight, text, text_weight, cos, sin,
                                         1e-6).unsqueeze(0)
  expected = torch.cat([
    fused_ops._eager_qk_norm_rope(text, text_weight, cos, sin, 0, 1e-6),
    fused_ops._eager_qk_norm_rope(image, image_weight, cos, sin, text_seq, 1e-6),
  ],
                       dim=1)
  assert joined.shape == expected.shape
  assert fused_ops.relative_l2(joined, expected) <= fused_ops.NORM_REL_L2_TOL


def test_preconditions_fall_back_to_the_eager_expression(monkeypatch):
  """A layout the kernels cannot read must produce the eager values, not an error."""

  torch.manual_seed(0)
  x = torch.randn(1, 32, DIM, device=DEVICE, dtype=torch.bfloat16)
  one_plus = _rows()
  shift = _rows()
  # Transposed rows are not dense, so the fused branch must be skipped.
  transposed = x.transpose(1, 2).transpose(1, 2)
  assert transposed.data_ptr() == x.data_ptr()

  monkeypatch.setattr(fused_ops, "FUSED_OPS_ENABLED", False)
  fused_values = fused_ops.apply_ln_modulate(x, one_plus, shift, 1e-6)
  monkeypatch.setattr(fused_ops, "FUSED_OPS_ENABLED", True)
  eager_values = fused_ops._eager_ln_modulate(x, one_plus, shift, 1e-6)
  assert fused_ops.relative_l2(fused_values, eager_values) <= fused_ops.NORM_REL_L2_TOL

  # fp16 is outside the kernels' envelope (they only read bf16), so the eager branch must run.
  half = x.to(torch.float16)
  result = fused_ops.apply_qk_norm_rope(
    half.unsqueeze(1).unsqueeze(1).reshape(1, 32, 1, DIM),
    torch.ones(DIM, device=DEVICE, dtype=torch.float16),
    torch.zeros(32, DIM, device=DEVICE, dtype=torch.float32),
    torch.zeros(32, DIM, device=DEVICE, dtype=torch.float32), 0, 1e-6)
  assert result.dtype == torch.float16


def test_operator_switches_parse_from_the_environment(monkeypatch):
  monkeypatch.delenv("CACHE_DIT_FLUX2_KLEIN_OPS", raising=False)
  assert fused_ops._ops_from_env() == frozenset(fused_ops.DEFAULT_OPS)

  monkeypatch.setenv("CACHE_DIT_FLUX2_KLEIN_OPS", "all")
  assert fused_ops._ops_from_env() == frozenset(fused_ops.ALL_OPS)

  monkeypatch.setenv("CACHE_DIT_FLUX2_KLEIN_OPS", "0")
  assert fused_ops._ops_from_env() == frozenset()

  monkeypatch.setenv("CACHE_DIT_FLUX2_KLEIN_OPS", "qk_norm_rope, gated_residual")
  assert fused_ops._ops_from_env() == frozenset({"qk_norm_rope", "gated_residual"})

  # The measured default is a subset, never "everything".
  assert set(fused_ops.DEFAULT_OPS) < set(fused_ops.ALL_OPS)

  monkeypatch.delenv("CACHE_DIT_FLUX2_KLEIN_ROPE_CACHE", raising=False)
  assert fused_ops._rope_cache_from_env() is False
  monkeypatch.setenv("CACHE_DIT_FLUX2_KLEIN_ROPE_CACHE", "1")
  assert fused_ops._rope_cache_from_env() is True


def test_fullgraph_compile_runs_the_operators():
  from cache_dit.kernels import flux2_klein_ln_modulate

  torch.manual_seed(0)
  x = torch.randn(1, 16, DIM, device=DEVICE, dtype=torch.bfloat16)
  one_plus = _rows()
  shift = _rows()

  @torch.compile(fullgraph=True)
  def step(value: torch.Tensor) -> torch.Tensor:
    return flux2_klein_ln_modulate(value, one_plus, shift, 1e-6)

  compiled = step(x)
  eager = flux2_klein_ln_modulate(x, one_plus, shift, 1e-6)
  assert torch.equal(compiled, eager)


class _FakeAttention(torch.nn.Module):

  def __init__(self):
    super().__init__()
    from diffusers.models.transformers.transformer_flux2 import Flux2AttnProcessor

    self.processor = Flux2AttnProcessor()


class _FakeBlock(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.attn = _FakeAttention()


class _FakeTransformer(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.transformer_blocks = torch.nn.ModuleList([_FakeBlock()])
    self.single_transformer_blocks = torch.nn.ModuleList([_FakeBlock()])


def test_attention_patch_installs_and_restores():
  from cache_dit.models.pipelines.flux2_klein import attention as fused_attention

  transformer = _FakeTransformer()
  originals = [block.attn.processor for block in transformer.transformer_blocks]

  fused_attention.install_attention_patch(transformer)
  assert fused_attention.attention_patch_is_installed(transformer)
  assert isinstance(transformer.transformer_blocks[0].attn.processor,
                    fused_attention._FusedFlux2AttnProcessor)

  fused_attention.uninstall_attention_patch(transformer)
  assert not fused_attention.attention_patch_is_installed(transformer)
  assert transformer.transformer_blocks[0].attn.processor is originals[0]


def test_install_is_skipped_under_async_ulysses(monkeypatch):
  """The async path patches the stock class, so a fused subclass would bypass it silently."""
  from diffusers.models.transformers.transformer_flux2 import Flux2AttnProcessor

  from cache_dit.distributed.transformers.async_ulysses import AsyncUlyssesRegistry
  from cache_dit.models.pipelines.flux2_klein import attention as fused_attention

  transformer = _FakeTransformer()
  assert not fused_attention.async_ulysses_is_active()

  monkeypatch.setitem(AsyncUlyssesRegistry._original_methods, (Flux2AttnProcessor, "__call__"),
                      Flux2AttnProcessor.__call__)
  assert fused_attention.async_ulysses_is_active()

  fused_attention.install_attention_patch(transformer)
  assert not fused_attention.attention_patch_is_installed(transformer)
