"""Kernel tests for FastH3's video sparse attention.

Three properties decide whether FastH3 support is faithful; each has a test here:

1. With the sparsity switched off (and a zero gate) both kernels must reproduce plain dense
   attention, which is what proves the tiling, the padding mask and the un-tiling are lossless.
2. The compression branch must be the only difference between a zero gate and no gate, and the
   production kernel must agree with the loop-based reference.
3. `meta=None` must fall back to dense, which is the path a dense FastH3 checkpoint takes.
"""
import math

import pytest
import torch
from torch.nn import functional as F

from cache_dit.attention.vsa_h3 import (
  compute_vsa_h3_topk,
  vsa_h3_attention,
  vsa_h3_attention_reference,
  vsa_h3_impl,
)
from cache_dit.attention.vsa_h3_geometry import build_vsa_h3_meta

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _meta(prefix=(9, 5), video=(4, 8, 8), sparsity=0.0):
  return build_vsa_h3_meta(prefix, video, DEVICE, sparsity=sparsity)


def _inputs(meta, heads=4, dim=32, dtype=torch.float32, seed=0):
  generator = torch.Generator(device="cpu").manual_seed(seed)
  shape = (1, meta.total_seq_length, heads, dim)
  q, k, v = (torch.randn(shape, generator=generator).to(DEVICE, dtype) for _ in range(3))
  gate = torch.randn(shape, generator=generator).to(DEVICE, dtype) * 0.1
  return q, k, v, gate


FLEX_SUPPORTED = vsa_h3_impl("auto") == "flex" and torch.cuda.is_available()


def _impls():
  if FLEX_SUPPORTED:
    return ["gather", "flex"]
  return ["gather"]


requires_flex = pytest.mark.skipif(
  not FLEX_SUPPORTED,
  reason="torch < 2.12: FlexAttention's block-sparse template compiles but returns wrong values, "
  "so only the gather baseline is exercised",
)


@requires_flex
def test_flex_kernel_is_available():
  """Keeps the missing flex coverage visible instead of silently dropping it."""
  assert vsa_h3_impl("auto") == "flex"


def test_topk_budget_matches_the_reference_rule():
  assert compute_vsa_h3_topk(0.9, 660) == 66
  assert compute_vsa_h3_topk(0.8, 660) == 132
  assert compute_vsa_h3_topk(1.0, 660) == 1
  assert compute_vsa_h3_topk(0.0, 660) == 660
  with pytest.raises(ValueError):
    compute_vsa_h3_topk(1.5, 10)


# Other suites in the same process enable TF32 for matmuls, which moves a float32 attention by
# ~1e-3. Structural mistakes in the tiling are O(1), so a 5e-3 gate still catches them.
TOL = 5e-3


@pytest.mark.parametrize("impl", _impls())
def test_zero_sparsity_matches_dense_attention(impl):
  meta = _meta()
  q, k, v, gate = _inputs(meta)
  dense = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2)).transpose(1, 2)
  out = vsa_h3_attention(q, k, v, meta, gate=torch.zeros_like(gate), sparsity=0.0, impl=impl)
  assert (out - dense).abs().max().item() < TOL


@pytest.mark.parametrize("impl", _impls())
def test_matches_the_loop_reference(impl):
  meta = _meta(video=(16, 16, 16), sparsity=0.9)
  q, k, v, gate = _inputs(meta)
  ref = vsa_h3_attention_reference(q, k, v, meta, gate=gate, sparsity=0.9)
  out = vsa_h3_attention(q, k, v, meta, gate=gate, sparsity=0.9, impl=impl)
  assert (out - ref).abs().max().item() < TOL


def test_gate_branch_is_the_only_difference():
  meta = _meta()
  q, k, v, gate = _inputs(meta)
  without = vsa_h3_attention(q, k, v, meta, gate=None, sparsity=0.5)
  zeroed = vsa_h3_attention(q, k, v, meta, gate=torch.zeros_like(gate), sparsity=0.5)
  assert (without - zeroed).abs().max().item() < 1e-5
  with_gate = vsa_h3_attention(q, k, v, meta, gate=gate, sparsity=0.5)
  assert (with_gate - without).abs().max().item() > 1e-4


def test_missing_meta_falls_back_to_dense():
  meta = _meta()
  q, k, v, _ = _inputs(meta)
  dense = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2)).transpose(1, 2)
  out = vsa_h3_attention(q, k, v, None)
  assert torch.equal(out, dense)


def test_sparse_selection_reduces_the_attended_key_set():
  meta = build_vsa_h3_meta((9, 5), (16, 16, 16), DEVICE, sparsity=0.9)
  q, k, v, _ = _inputs(meta)
  dense = vsa_h3_attention(q, k, v, meta, sparsity=0.0)
  sparse = vsa_h3_attention(q, k, v, meta, sparsity=0.9)
  video_rows = slice(meta.num_prefix_tokens, meta.total_seq_length)
  assert (dense[:, video_rows] - sparse[:, video_rows]).abs().max().item() > 1e-3
  assert math.prod(meta.video_shape) == meta.num_video_tiles * meta.tile_elements
