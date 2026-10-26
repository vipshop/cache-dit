"""Regression tests for the Motif-Video integration (cache, CP plan, processor patches).

CPU-only coverage of the silent failure modes specific to this model:

1. `MotifVideoPatchFunctor` must split the sliced single-stream loops into dedicated
   sub-module attributes and stay bit-exact with the upstream forward (cache disabled).
2. After `enable_cache`, the cached wrapper must actually take over the three loops: the
   functor's forward walks `self._motif_single_*` attributes so `mock.patch.object` swaps
   them; a closure capture would silently bypass the cache. Counting block calls catches
   the sliced-loop collision (decoder blocks executed twice).
3. The CP plan must be resolvable on the patched transformer: the DDT decoder query stream
   is split at `_motif_single_decoder_blocks.0`, which only exists after the functor ran.
4. `enable_cache` must keep `transformer.forward`'s signature.
5. The class-level CP processor patches must leave unarmed processors bit-identical to
   upstream, and the armed joint/cross paths must hand cache-dit's CP dispatch the
   permuted mask (`[H_0, E_0, H_1, E_1, ...]`) and the rank-interleaved text mask.
"""
import inspect
from types import SimpleNamespace

import pytest
import torch

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.caching.patch_functors import MotifVideoPatchFunctor
from cache_dit.distributed.transformers import motif_video as motif_video_module
from cache_dit.distributed.transformers.motif_video import (
  MotifVideoContextParallelismPlanner,
  _cp_is_armed,
  _install_cp_processor_patches,
  _CP_PATCH_ORIGINALS,
)


def _tiny_motif_transformer():
  from diffusers import MotifVideoTransformer3DModel

  return MotifVideoTransformer3DModel(
    in_channels=33,
    out_channels=16,
    num_attention_heads=2,
    attention_head_dim=12,
    num_layers=2,
    num_single_layers=4,
    num_decoder_layers=2,
    mlp_ratio=2.0,
    patch_size=1,
    patch_size_t=1,
    qk_norm="rms_norm",
    text_embed_dim=8,
    image_embed_dim=4,
    rope_axes_dim=(4, 4, 4),
    enable_text_cross_attention_dual=False,
    enable_text_cross_attention_single=True,
  )


def _tiny_inputs(with_image_embeds=False):
  inputs = dict(
    hidden_states=torch.randn(1, 33, 4, 8, 8),
    timestep=torch.tensor([500]),
    encoder_hidden_states=torch.randn(1, 6, 8),
    encoder_attention_mask=torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.long),
    return_dict=False,
  )
  if with_image_embeds:
    inputs["image_embeds"] = torch.randn(1, 2, 4)
  return inputs


def test_functor_split_and_bit_exact():
  torch.manual_seed(0)
  transformer = _tiny_motif_transformer().eval()
  inputs = _tiny_inputs()

  with torch.no_grad():
    out_orig = transformer(**inputs)[0]

  MotifVideoPatchFunctor().apply(transformer)
  assert getattr(transformer, "_is_patched", False)
  assert len(transformer._motif_single_encoder_blocks) == 2
  assert len(transformer._motif_single_decoder_blocks) == 2
  # block objects are shared by reference, not copied
  assert (transformer._motif_single_encoder_blocks[0] is transformer.single_transformer_blocks[0])

  with torch.no_grad():
    out_patched = transformer(**inputs)[0]
  assert torch.equal(out_orig, out_patched), "functor is not bit-exact (t2v inputs)"

  # i2v path: image_embeds prepend + image_embed_seq_len threading
  torch.manual_seed(0)
  transformer_i2v = _tiny_motif_transformer().eval()
  i2v_inputs = _tiny_inputs(with_image_embeds=True)
  with torch.no_grad():
    out_orig_i2v = transformer_i2v(**i2v_inputs)[0]
  MotifVideoPatchFunctor().apply(transformer_i2v)
  with torch.no_grad():
    out_patched_i2v = transformer_i2v(**i2v_inputs)[0]
  assert torch.equal(out_orig_i2v, out_patched_i2v), "functor is not bit-exact (i2v inputs)"


def test_cached_wrapper_takes_over_all_three_loops():
  torch.manual_seed(0)
  transformer = _tiny_motif_transformer().eval()
  inputs = _tiny_inputs()

  with torch.no_grad():
    out_ref = transformer(**inputs)[0]

  # Count per-block executions: with the functor applied every block must run exactly
  # once per forward. If the sliced loops ever collapse onto one wrapper (the failure
  # this integration exists to fix), the encoder pass runs all single blocks and the
  # decoder pass runs them again — decoder blocks would fire twice.
  calls: dict[int, int] = {}

  def _count_hook(module, args, kwargs):
    calls[id(module)] = calls.get(id(module), 0) + 1
    return None

  all_blocks = list(transformer.transformer_blocks) + list(transformer.single_transformer_blocks)
  hooks = [block.register_forward_pre_hook(_count_hook, with_kwargs=True) for block in all_blocks]

  cache_dit.enable_cache(
    transformer,
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      residual_diff_threshold=0.05,
    ),
  )

  # The functor must have been applied by the adapter path (enable_cache -> BlockAdapter):
  # otherwise the sliced loops would run the single wrapper twice and the cache would
  # silently corrupt the bookkeeping.
  assert getattr(transformer, "_is_patched", False)
  with torch.no_grad():
    out_cached = transformer(**inputs)[0]
  assert out_cached.shape == out_ref.shape
  assert torch.isfinite(out_cached).all()

  for hook in hooks:
    hook.remove()
  assert calls, "no block was executed"
  assert all(count == 1 for count in calls.values()), (
    "some blocks ran more than once in a single forward — the sliced-loop collision "
    "is back (decoder blocks executed by both the encoder and the decoder pass)")

  cache_dit.disable_cache(transformer)


def test_cp_plan_resolves_on_patched_transformer():
  transformer = _tiny_motif_transformer()
  planner = MotifVideoContextParallelismPlanner()
  cp_plan = planner._apply(transformer)

  # Every hook key must resolve on the (functor-patched) module tree: the DDT decoder
  # query-stream split anchors on `_motif_single_decoder_blocks.0`, which only exists
  # after the functor ran.
  for key in cp_plan:
    if key == "":
      continue
    module = transformer
    for atom in key.split("."):
      module = getattr(module, atom)

  assert "transformer_blocks.0" in cp_plan
  assert "hidden_states" in cp_plan["transformer_blocks.0"]
  assert "encoder_hidden_states" in cp_plan["transformer_blocks.0"]
  # the attention mask is NOT hooked: the patched forward permutes it (full length)
  assert "attention_mask" not in cp_plan["transformer_blocks.0"]
  assert "_motif_single_decoder_blocks.0" not in cp_plan, (
    "the decoder query stream is chunked inside the patched forward, not hooked")
  # rope returns (cos, sin): both outputs are split
  for index in (0, 1):
    assert cp_plan["rope"][index].split_output is True
  assert "proj_out" in cp_plan


def test_forward_signature_preserved_after_enable_cache():
  transformer = _tiny_motif_transformer()
  signature_before = inspect.signature(transformer.forward)

  cache_dit.enable_cache(
    transformer,
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      residual_diff_threshold=0.05,
    ),
  )

  signature_after = inspect.signature(transformer.forward)
  assert tuple(signature_after.parameters) == tuple(signature_before.parameters)
  for name in ("hidden_states", "timestep", "encoder_hidden_states", "encoder_attention_mask",
               "image_embeds"):
    assert name in signature_after.parameters, (
      f"'{name}' disappeared from the patched transformer.forward signature")

  cache_dit.disable_cache(transformer)


def _tiny_joint_attn():
  from diffusers.models.transformers.transformer_motif_video import MotifVideoAttention

  torch.manual_seed(0)
  return MotifVideoAttention(
    query_dim=32,
    added_kv_proj_dim=32,
    heads=2,
    dim_head=8,
    out_dim=32,
    bias=True,
    context_pre_only=False,
    qk_norm="rms_norm",
    eps=1e-6,
  )


def _tiny_cross_attn():
  from diffusers.models.transformers.transformer_motif_video import MotifVideoCrossAttention

  torch.manual_seed(0)
  return MotifVideoCrossAttention(
    query_dim=32,
    heads=2,
    dim_head=8,
    bias=True,
    qk_norm="rms_norm",
    eps=1e-6,
  )


def _snapshot_processor_patches():
  from diffusers.models.transformers.transformer_motif_video import (
    MotifVideoAttnProcessor2_0,
    MotifVideoCrossAttnProcessor2_0,
  )

  return (
    MotifVideoAttnProcessor2_0.__call__,
    MotifVideoCrossAttnProcessor2_0.__call__,
    dict(_CP_PATCH_ORIGINALS),
  )


def _restore_processor_patches(snapshot):
  from diffusers.models.transformers.transformer_motif_video import (
    MotifVideoAttnProcessor2_0,
    MotifVideoCrossAttnProcessor2_0,
  )

  (MotifVideoAttnProcessor2_0.__call__, MotifVideoCrossAttnProcessor2_0.__call__,
   originals) = snapshot
  _CP_PATCH_ORIGINALS.clear()
  _CP_PATCH_ORIGINALS.update(originals)


def test_unarmed_processors_stay_upstream():
  """Unarmed processors must be bit-identical to upstream after the class-level patch."""
  from diffusers.models.transformers.transformer_motif_video import (
    MotifVideoAttnProcessor2_0,
    MotifVideoCrossAttnProcessor2_0,
  )

  joint_attn, cross_attn = _tiny_joint_attn(), _tiny_cross_attn()
  snapshot = _snapshot_processor_patches()
  try:
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(1, 8, 32, generator=generator)
    encoder = torch.randn(1, 4, 32, generator=generator)
    mask = torch.ones(1, 1, 1, 12, dtype=torch.bool)
    rope = (torch.randn(8, 8), torch.randn(8, 8))

    joint_out = joint_attn.processor(joint_attn, hidden, encoder, mask, rope)
    cross_out = cross_attn.processor(cross_attn, hidden, encoder, mask, rope, 0)

    _install_cp_processor_patches()
    assert not _cp_is_armed(joint_attn.processor)
    joint_patched = MotifVideoAttnProcessor2_0.__call__(joint_attn.processor, joint_attn, hidden,
                                                        encoder, mask, rope)
    cross_patched = MotifVideoCrossAttnProcessor2_0.__call__(cross_attn.processor, cross_attn,
                                                             hidden, encoder, mask, rope, 0)
    assert torch.equal(joint_patched[0], joint_out[0]) and torch.equal(
      joint_patched[1], joint_out[1]), ("the joint-processor CP patch changed the math of "
                                        "a processor it never armed")
    assert torch.equal(
      cross_patched,
      cross_out), ("the cross-processor CP patch changed the math of a processor it never armed")
  finally:
    _restore_processor_patches(snapshot)


def test_armed_processors_feed_cp_dispatch_permuted_masks():
  """Armed processors must route through cache-dit's CP dispatch with correct masks.

  The joint path passes the (functor-permuted, full-length) mask through unchanged; the
  cross path re-extracts this rank's text segments from it in rank-interleaved order
  `[E_0, E_1, ...]` to match the K/V all-to-all layout.
  """
  from diffusers.models.transformers.transformer_motif_video import (
    MotifVideoAttnProcessor2_0,
    MotifVideoCrossAttnProcessor2_0,
  )

  joint_attn, cross_attn = _tiny_joint_attn(), _tiny_cross_attn()
  snapshot = _snapshot_processor_patches()
  captured = []
  real_dispatch = motif_video_module._dispatch_attention_fn
  try:
    _install_cp_processor_patches()

    def _fake_dispatch(query, key, value, attn_mask=None, backend=None, cp_config=None, **kwargs):
      assert backend is None, "the CP path must not force a specific attention backend"
      captured.append(attn_mask)
      return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    motif_video_module._dispatch_attention_fn = _fake_dispatch

    # Joint: hidden 8 img + 4 txt tokens; the functor's forward has already permuted
    # the mask into all-to-all order, so the processor must hand it to the CP dispatch
    # unchanged (full length). Hand-craft the permuted layout directly:
    # [H_0(4T), E_0(1T,1F), H_1(4T), E_1(1T,1F)].
    joint_attn.processor._cp_config = SimpleNamespace(_world_size=2, _rank=0)
    cross_attn.processor._cp_config = joint_attn.processor._cp_config
    assert _cp_is_armed(joint_attn.processor)

    hidden = torch.randn(1, 8, 32)
    encoder = torch.randn(1, 4, 32)
    permuted_mask = torch.cat([
      torch.ones(1, 1, 1, 4, dtype=torch.bool),
      torch.tensor([[[[1, 0]]]], dtype=torch.bool),
      torch.ones(1, 1, 1, 4, dtype=torch.bool),
      torch.tensor([[[[1, 0]]]], dtype=torch.bool),
    ],
                              dim=-1)

    MotifVideoAttnProcessor2_0.__call__(joint_attn.processor, joint_attn, hidden, encoder,
                                        permuted_mask, None)
    got = captured[-1]
    assert torch.equal(got, permuted_mask), (
      "the armed joint processor must pass the functor-permuted mask through unchanged")

    # Cross: Q = the local image stream (4 tokens = the img segment inside one rank's
    # chunk), K/V = this rank's text segment (2 tokens, t2v: image_embed_seq_len=0).
    # The permuted mask's text segments sit at [i*chunk + q_len, (i+1)*chunk);
    # concatenated they form the K/V all-to-all order [E_0, E_1].
    captured.clear()
    attn_output = torch.randn(1, 4, 32)
    MotifVideoCrossAttnProcessor2_0.__call__(cross_attn.processor, cross_attn, attn_output, encoder,
                                             permuted_mask, None, 0)
    got = captured[-1]
    # chunk = 6 (4 img + 2 txt per rank); E_i = mask[..., i*6+4 : (i+1)*6]
    expected_cross = torch.cat([permuted_mask[..., 4:6], permuted_mask[..., 10:12]], dim=-1)
    assert got is not None and got.shape[-1] == 4, (
      "the cross mask must cover the two local text segments (4 tokens)")
    assert torch.equal(
      got,
      expected_cross), ("the armed cross processor must extract the rank-interleaved text segments "
                        "([E_0, E_1, ...]) from the permuted mask")
  finally:
    motif_video_module._dispatch_attention_fn = real_dispatch
    _restore_processor_patches(snapshot)


def test_cp_planner_rejects_ulysses_anything():
  transformer = _tiny_motif_transformer()
  planner = MotifVideoContextParallelismPlanner()
  parallelism_config = SimpleNamespace(
    ulysses_size=2,
    ring_size=None,
    tp_size=1,
    ulysses_anything=True,
    ulysses_async=False,
  )
  with pytest.raises(NotImplementedError, match="ulysses-anything"):
    planner._apply(transformer, parallelism_config)
