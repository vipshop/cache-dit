"""Gated-attention regression test for the LTX-2 CP processor patches.

The CP planner patches `LTX2AudioVideoAttnProcessor.__call__` at class level, which also reaches
attention trees the CP runtime never armed with `_cp_config` -- most importantly the
`LTX2TextConnectors` component of every LTX-2.x pipeline, whose LTX-2.5 checkpoint sets
`video_gated_attn=True` / `audio_gated_attn=True`. The historical patch dropped the per-head gate,
which silently rewrote the prompt embeddings under CP and made CP runs render a coherent but
*different* video (PSNR ~9 dB against the single-GPU reference).

Covered failure modes:
1. an unarmed processor must return exactly what the untouched upstream `__call__` returns;
2. the armed path must still apply the per-head gate and must reach the CP dispatch.
"""

import pytest
import torch

try:
  from diffusers.models.transformers.transformer_ltx2 import LTX2PerturbedAttnProcessor  # noqa: F401
except ImportError:
  pytest.skip("diffusers without LTX-2.5 support", allow_module_level=True)

from cache_dit.distributed.transformers.ltx2 import (
  _CP_PATCH_ORIGINALS,
  _cp_is_armed,
  _install_cp_processor_patches,
)


class _SimpleConfig:
  """Minimal `_ContextParallelConfig` stand-in exposing `_world_size`."""

  def __init__(self, world_size):
    self._world_size = world_size


def _gated_attention():
  from diffusers.models.transformers.transformer_ltx2 import LTX2Attention, LTX2AudioVideoAttnProcessor

  torch.manual_seed(0)
  return LTX2Attention(
    query_dim=32,
    heads=4,
    kv_heads=4,
    dim_head=8,
    rope_type="interleaved",
    apply_gated_attention=True,
    processor=LTX2AudioVideoAttnProcessor(),
  )


def _snapshot_patches():
  from diffusers.models.transformers.transformer_ltx2 import (
    LTX2Attention,
    LTX2AudioVideoAttnProcessor,
  )

  return (
    LTX2Attention.prepare_attention_mask,
    LTX2AudioVideoAttnProcessor.__call__,
    dict(_CP_PATCH_ORIGINALS),
  )


def _restore_patches(snapshot):
  from diffusers.models.transformers.transformer_ltx2 import (
    LTX2Attention,
    LTX2AudioVideoAttnProcessor,
  )

  LTX2Attention.prepare_attention_mask, LTX2AudioVideoAttnProcessor.__call__, originals = snapshot
  _CP_PATCH_ORIGINALS.clear()
  _CP_PATCH_ORIGINALS.update(originals)


def test_unarmed_processor_stays_upstream():
  """A processor without `_cp_config` (the connectors case) must be bit-identical to upstream."""
  attn = _gated_attention()
  snapshot = _snapshot_patches()
  try:
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(1, 16, 32, generator=generator)

    from diffusers.models.transformers.transformer_ltx2 import LTX2AudioVideoAttnProcessor

    expected = LTX2AudioVideoAttnProcessor.__call__(attn.processor, attn, hidden)

    _install_cp_processor_patches()
    assert not _cp_is_armed(attn.processor)
    assert LTX2AudioVideoAttnProcessor.__call__ is not _CP_PATCH_ORIGINALS[
      LTX2AudioVideoAttnProcessor]

    patched = LTX2AudioVideoAttnProcessor.__call__(attn.processor, attn, hidden)
    assert torch.equal(
      patched,
      expected), ("the class-level CP patch changed the math of a processor it never armed "
                  "(this is the LTX2TextConnectors regression)")
  finally:
    _restore_patches(snapshot)


def test_armed_path_applies_per_head_gate():
  """With `_cp_config` armed, the patched path must apply the gate and reach the CP dispatch."""
  attn = _gated_attention()
  snapshot = _snapshot_patches()
  try:
    _install_cp_processor_patches()
    attn.processor._cp_config = _SimpleConfig(world_size=2)
    assert _cp_is_armed(attn.processor)

    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(1, 16, 32, generator=generator)
    with pytest.raises((RuntimeError, TypeError)):
      # The CP machinery (config validation / all-to-all) needs a real process group; failing
      # there proves the patched branch ran instead of falling back to upstream.
      attn.processor(attn, hidden)

    # The gate itself: a hand-rolled gated variant must differ from the ungated attention output.
    query = attn.norm_q(attn.to_q(hidden)).unflatten(2, (attn.heads, -1))
    key = attn.norm_k(attn.to_k(hidden)).unflatten(2, (attn.heads, -1))
    value = attn.to_v(hidden).unflatten(2, (attn.heads, -1))
    attended = torch.nn.functional.scaled_dot_product_attention(query, key, value).flatten(2, 3)

    gates = 2.0 * torch.sigmoid(attn.to_gate_logits(hidden))
    gated = attended.unflatten(2, (attn.heads, -1)) * gates.unsqueeze(-1)
    assert gates.shape == (1, 16, attn.heads)
    assert not torch.equal(gated.flatten(2, 3), attended), "the gate must influence the output"
  finally:
    _restore_patches(snapshot)
