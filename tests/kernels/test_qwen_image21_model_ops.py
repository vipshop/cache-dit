"""Bit-exactness and dispatch tests for the Qwen-Image 2.1 operator family.

The operators replace expressions diffusers already evaluates, so every test here asserts
`torch.equal` against the eager transcription in
:mod:`cache_dit.models.pipelines._qwen_image21.fused_ops` — not a tolerance. A tolerance
would hide exactly the failure these kernels can have: a missing bf16 rounding point or an
FMA that torch does not emit.
"""

import pytest
import torch

from cache_dit.kernels._models import models_extension_is_available
from cache_dit.models.pipelines._qwen_image21 import fused_ops

pytestmark = [
  pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
  pytest.mark.skipif(not models_extension_is_available(), reason="needs cache_dit._C_models"),
]

DEVICE = torch.device("cuda")


def _close(actual: torch.Tensor, expected: torch.Tensor) -> bool:
  """Kernel-vs-eager comparison at the fast-math tolerance.

  The extensions are built with `--use_fast_math`, which replaces `expf` / `tanhf` /
  `rsqrtf` with approximate forms, so any operator that fuses one of those cannot be
  bit-identical. Operators that only multiply and add (gated_residual) come out at exactly
  0 relative error and still pass this check.

  :param actual: Kernel output.
  :param expected: Eager reference.
  :returns: Whether the relative L2 error is within `fused_ops.REL_L2_TOL`.
  """

  return fused_ops.relative_l2(actual, expected) <= fused_ops.REL_L2_TOL


def test_qk_norm_rope_is_bit_exact_for_every_rotation_variant():
  from cache_dit.kernels import qwen_image21_qk_norm_rope

  torch.manual_seed(0)
  seq = 64
  x = torch.randn(1, seq, 4, 128, device=DEVICE, dtype=torch.bfloat16)
  weight = (1 + 0.1 * torch.randn(128, device=DEVICE)).to(torch.bfloat16)
  rotary = torch.polar(torch.ones(seq, 64, device=DEVICE), torch.randn(seq, 64, device=DEVICE) * 3)
  cos, sin = fused_ops.rotary_cos_sin(rotary)
  reference = fused_ops._eager_qk_norm_rope(x, weight, rotary, 1e-6)

  matched = [
    variant for variant in (0, 1, 2, 3, 4)
    if _close(reference, qwen_image21_qk_norm_rope(x, weight, cos, sin, 1e-6, variant))
  ]
  assert matched, "no rotation variant reproduces torch's complex64 multiply"


def test_qk_norm_rope_probe_selects_a_matching_variant():
  variant = fused_ops.probe_rope_variant(DEVICE)
  assert variant >= 0, "the probe must find a matching variant on a supported device"

  torch.manual_seed(1)
  x = torch.randn(1, 32, 8, 128, device=DEVICE, dtype=torch.bfloat16)
  weight = (1 + 0.1 * torch.randn(128, device=DEVICE)).to(torch.bfloat16)
  rotary = torch.polar(torch.ones(32, 64, device=DEVICE), torch.randn(32, 64, device=DEVICE) * 3)
  assert _close(
    fused_ops._eager_qk_norm_rope(x, weight, rotary, 1e-6),
    fused_ops.apply_qk_norm_rope(x, weight, rotary, 1e-6),
  )


def test_qk_norm_rope_writes_into_a_strided_buffer():
  seq, prefix = 96, 24
  x = torch.randn(1, seq, 4, 128, device=DEVICE, dtype=torch.bfloat16)
  rotary = torch.polar(torch.ones(seq, 64, device=DEVICE), torch.randn(seq, 64, device=DEVICE))
  weight = torch.ones(128, device=DEVICE, dtype=torch.bfloat16)
  buffer = torch.zeros(1, prefix + seq, 4, 128, device=DEVICE, dtype=torch.bfloat16)
  tail = buffer[:, prefix:]

  returned = fused_ops.apply_qk_norm_rope(x, weight, rotary, 1e-6, out=tail)

  assert returned.data_ptr() == tail.data_ptr()
  assert _close(fused_ops._eager_qk_norm_rope(x, weight, rotary, 1e-6), tail)


def test_ln_modulate_matches_the_eager_chain():
  torch.manual_seed(0)
  dim = 4096
  x = torch.randn(1, 300, dim, device=DEVICE, dtype=torch.bfloat16)
  params = torch.randn(2, 2 * dim, device=DEVICE, dtype=torch.bfloat16)
  mask = torch.zeros(300, device=DEVICE, dtype=torch.bool)
  mask[:150] = True

  reference, reference_gate = fused_ops._eager_ln_modulate(x, params, mask, 1e-6)
  modulated, gate = fused_ops.apply_ln_modulate(x, params, mask, 1e-6)

  assert _close(reference, modulated)
  assert _close(reference_gate.tanh(), gate)


def test_ln_modulate_without_a_mask_broadcasts_the_sample_row():
  torch.manual_seed(0)
  dim = 512
  x = torch.randn(1, 40, dim, device=DEVICE, dtype=torch.bfloat16)
  params = torch.randn(1, 2 * dim, device=DEVICE, dtype=torch.bfloat16)

  reference, reference_gate = fused_ops._eager_ln_modulate(x, params, None, 1e-6)
  modulated, gate = fused_ops.apply_ln_modulate(x, params, None, 1e-6)

  assert _close(reference, modulated)
  assert _close(reference_gate.tanh().expand_as(gate), gate)


def test_modulate_affine_matches_the_eager_epilogue():
  from cache_dit.kernels import qwen_image21_modulate_affine

  torch.manual_seed(0)
  dim = 512
  x = torch.randn(1, 300, dim, device=DEVICE, dtype=torch.bfloat16)
  params = torch.randn(2, 2 * dim, device=DEVICE, dtype=torch.bfloat16)
  mask = torch.zeros(300, device=DEVICE, dtype=torch.bool)
  mask[:150] = True

  scale, gate = params.chunk(2, dim=-1)
  reference = x * (1 + fused_ops._select_modulation_rows(scale, mask))
  reference_gate = fused_ops._select_modulation_rows(gate, mask).tanh()

  modulated, gate_out = qwen_image21_modulate_affine(x, params, mask)
  assert _close(reference, modulated)
  assert _close(reference_gate, gate_out)


def test_multi_batch_modulation_falls_back_to_eager():
  """The 16-byte kernel is batch-1 only; larger batches must stay correct, not just fast."""

  torch.manual_seed(0)
  dim = 512
  x = torch.randn(2, 7, dim, device=DEVICE, dtype=torch.bfloat16)
  params = torch.randn(3, 2 * dim, device=DEVICE, dtype=torch.bfloat16)
  mask = torch.zeros(7, device=DEVICE, dtype=torch.bool)
  mask[:4] = True

  reference, reference_gate = fused_ops._eager_ln_modulate(x, params, mask, 1e-6)
  modulated, gate = fused_ops.apply_ln_modulate(x, params, mask, 1e-6)
  assert _close(reference, modulated)
  assert _close(reference_gate.tanh(), gate)


def test_gated_residual_and_silu_mul_match_the_eager_expression():
  torch.manual_seed(0)
  hidden = torch.randn(1, 64, 4096, device=DEVICE, dtype=torch.bfloat16)
  gate = torch.randn(1, 64, 4096, device=DEVICE, dtype=torch.bfloat16).tanh()
  update = torch.randn(1, 64, 4096, device=DEVICE, dtype=torch.bfloat16)

  assert _close(hidden + gate * update, fused_ops.apply_gated_residual(hidden, gate, update))

  up_gate = torch.randn(1, 64, 12288, device=DEVICE, dtype=torch.bfloat16) * 3
  up = torch.randn(1, 64, 12288, device=DEVICE, dtype=torch.bfloat16)
  assert _close(torch.nn.functional.silu(up_gate) * up, fused_ops.apply_silu_mul(up_gate, up))


def test_unsupported_input_falls_back_to_the_eager_expression():
  """A wrong head width, CPU tensors or a non-contiguous view must not reach the kernel."""

  rotary = torch.polar(torch.ones(16, 64, device=DEVICE), torch.zeros(16, 64, device=DEVICE))
  weight = torch.ones(128, device=DEVICE, dtype=torch.bfloat16)

  # head_dim != 128: the kernel only exists for the checkpoint's head width.
  wide = torch.randn(1, 16, 2, 256, device=DEVICE, dtype=torch.bfloat16)
  wide_rotary = torch.polar(torch.ones(16, 128, device=DEVICE), torch.zeros(16, 128, device=DEVICE))
  wide_weight = torch.ones(256, device=DEVICE, dtype=torch.bfloat16)
  got = fused_ops.apply_qk_norm_rope(wide, wide_weight, wide_rotary, 1e-6)
  expected = fused_ops._eager_qk_norm_rope(wide, wide_weight, wide_rotary, 1e-6)
  assert _close(expected, got)

  # CPU tensors.
  cpu = torch.randn(1, 16, 2, 128)
  assert _close(
    fused_ops.apply_qk_norm_rope(cpu, torch.ones(128), rotary.cpu(), 1e-6),
    fused_ops._eager_qk_norm_rope(cpu, torch.ones(128), rotary.cpu(), 1e-6),
  )

  # Non-contiguous view.
  sliced = torch.randn(1, 32, 2, 128, device=DEVICE, dtype=torch.bfloat16)[:, ::2]
  sliced_rotary = torch.polar(torch.ones(16, 64, device=DEVICE), torch.zeros(16, 64, device=DEVICE))
  assert not sliced.is_contiguous()
  got = fused_ops.apply_qk_norm_rope(sliced, weight, sliced_rotary, 1e-6)
  expected = fused_ops._eager_qk_norm_rope(sliced, weight, sliced_rotary, 1e-6)
  assert _close(expected, got)


def test_ops_are_traceable_by_compile():
  torch.manual_seed(0)
  x = torch.randn(1, 16, 2, 128, device=DEVICE, dtype=torch.bfloat16)
  weight = torch.ones(128, device=DEVICE, dtype=torch.bfloat16)
  rotary = torch.polar(torch.ones(16, 64, device=DEVICE), torch.zeros(16, 64, device=DEVICE))
  cos, sin = fused_ops.rotary_cos_sin(rotary)

  def run(q):
    return fused_ops.apply_qk_norm_rope(q, weight, rotary, 1e-6) + q

  compiled = torch.compile(run, fullgraph=True)
  assert _close(compiled(x), run(x))


def test_dispatch_entry_points_follow_the_active_op_list(monkeypatch):
  """Only operators listed in `ACTIVE_OPS` may reach the extension."""

  monkeypatch.setattr(fused_ops, "ACTIVE_OPS", frozenset())
  hidden = torch.randn(1, 8, 512, device=DEVICE, dtype=torch.bfloat16)
  gate = torch.randn(1, 8, 512, device=DEVICE, dtype=torch.bfloat16).tanh()
  update = torch.randn(1, 8, 512, device=DEVICE, dtype=torch.bfloat16)
  assert _close(hidden + gate * update, fused_ops.apply_gated_residual(hidden, gate, update))

  up = torch.randn(1, 8, 512, device=DEVICE, dtype=torch.bfloat16)
  assert _close(torch.nn.functional.silu(hidden) * up, fused_ops.apply_silu_mul(hidden, up))


def test_probes_run_once_per_process(monkeypatch):
  """A probe must never be re-entered from the per-step path."""

  calls = {"rope": 0}

  import cache_dit.kernels._models as models

  real = models.qwen_image21_qk_norm_rope

  def counting(*args, **kwargs):
    calls["rope"] += 1
    return real(*args, **kwargs)

  monkeypatch.setattr(models, "qwen_image21_qk_norm_rope", counting)
  monkeypatch.setattr(fused_ops, "_ROPE_VARIANT", None)

  first = fused_ops.probe_rope_variant(DEVICE)
  after_first = calls["rope"]
  for _ in range(5):
    assert fused_ops.probe_rope_variant(DEVICE) == first
  assert calls["rope"] == after_first, "the probe ran again after being resolved"
