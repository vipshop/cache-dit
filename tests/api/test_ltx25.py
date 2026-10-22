"""Regression tests for the LTX-2.5 integration (cache, CP, TP).

LTX-2.5 reuses the `LTX2VideoTransformer3DModel` class, but the surrounding contract changed in
ways that fail silently rather than loudly. The modes covered here, all CPU-only:

1. `LTX2PatchFunctor` must stay a faithful copy of the upstream forward. The forward is duplicated
   because cache-dit's block wrapper needs the four required tensors positionally while upstream
   passes everything by keyword, so any upstream addition (`sigma`, `video_keyframes_mask`,
   `isolate_modalities`, STG) that is not mirrored changes the output without raising.
2. The four positional arguments must be exactly the block's first four parameters, otherwise the
   wrapper binds the audio stream into the wrong slot.
3. LTX-2.5 builds blocks with `perturbed_attn=True`; `LTX2PerturbedAttnProcessor` is a standalone
   class, so the CP planner has to patch it as well or CP silently runs on local shards.
4. `video_keyframes_mask` is applied inside the forward, after `hidden_states` was already sharded,
   so the CP plan must split it on the same axis.
5. TP must shard `to_gate_logits` together with `attn.heads`, and `ShardRotaryEmbProcessor` must
   forward `perturbation_mask` / `all_perturbed`, which `LTX2Attention.forward` would otherwise
   filter out by signature.
6. DFR tiles are equal-length token subsets, so the cache has to identify a forward by its geometry
   origin (not by the token count) before deciding it may reuse a residual.

The GPU parity checks for CP and TP live in `tests/parallelism/test_ltx25_cp_tp_tiny.py`.
"""
import inspect
import sys

import pytest
import torch

try:
  from diffusers.models.transformers.transformer_ltx2 import (
    LTX2PerturbedAttnProcessor as _LTX2PerturbedAttnProcessor, )
except ImportError:
  # LTX-2.5 needs a diffusers build that ships the perturbed attention processor.
  pytest.skip("diffusers without LTX-2.5 support", allow_module_level=True)

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.caching.block_adapters import BlockAdapterRegister
from cache_dit.caching.forward_pattern import ForwardPattern
from cache_dit.caching.patch_functors import LTX2PatchFunctor
from cache_dit.distributed.transformers.ltx2 import LTX2ContextParallelismPlanner
from cache_dit.distributed.transformers.ltx2_video import (
  ShardRotaryEmbProcessor,
  _ltx2_block_layer_plan,
)

_VIDEO_TOKENS = 8
_AUDIO_TOKENS = 8
_TEXT_TOKENS = 4
_INNER_DIM = 32


def _tiny_ltx25_transformer(
  use_prompt_embeddings: bool = False,
  perturbed_attn: bool = True,
  gated_attn: bool = True,
):
  from diffusers import LTX2VideoTransformer3DModel

  torch.manual_seed(0)
  transformer = LTX2VideoTransformer3DModel(
    in_channels=8,
    out_channels=8,
    patch_size=1,
    patch_size_t=1,
    num_attention_heads=4,
    attention_head_dim=8,
    cross_attention_dim=_INNER_DIM,
    vae_scale_factors=(1, 1, 1),
    pos_embed_max_pos=8,
    base_height=8,
    base_width=8,
    gated_attn=gated_attn,
    cross_attn_mod=True,
    audio_in_channels=8,
    audio_out_channels=8,
    audio_patch_size=1,
    audio_patch_size_t=1,
    audio_num_attention_heads=4,
    audio_attention_head_dim=8,
    audio_cross_attention_dim=_INNER_DIM,
    audio_scale_factor=1,
    audio_pos_embed_max_pos=8,
    audio_gated_attn=gated_attn,
    audio_cross_attn_mod=True,
    num_layers=2,
    caption_channels=_INNER_DIM,
    rope_type="split",
    use_prompt_embeddings=use_prompt_embeddings,
    perturbed_attn=perturbed_attn,
    ff_bias=False,
    audio_ff_bias=False,
    use_prompt_adaln_single=True,
    use_keyframes_abs_pos_embedding=True,
  )
  transformer.eval()
  # `keyframes_abs_pos_embedding` ships zero-initialized, so a mask test could not tell the two
  # branches apart without perturbing it.
  torch.nn.init.normal_(transformer.keyframes_abs_pos_embedding, std=0.02)
  return transformer


def _inputs(batch_size=1, **overrides):
  generator = torch.Generator(device="cpu").manual_seed(1)
  inputs = {
    "hidden_states":
    torch.randn(batch_size, _VIDEO_TOKENS, 8, generator=generator),
    "audio_hidden_states":
    torch.randn(batch_size, _AUDIO_TOKENS, 8, generator=generator),
    "encoder_hidden_states":
    torch.randn(batch_size, _TEXT_TOKENS, _INNER_DIM, generator=generator),
    "audio_encoder_hidden_states":
    torch.randn(batch_size, _TEXT_TOKENS, _INNER_DIM, generator=generator),
    "timestep":
    torch.full((batch_size, _VIDEO_TOKENS), 750.0),
    # Keep every timestep-like input distinct: `video_ca_timestep` / `audio_ca_timestep` take the
    # *cross* modality value, which identical values would hide.
    "audio_timestep":
    torch.full((batch_size, ), 500.0),
    "sigma":
    torch.full((batch_size, ), 0.75),
    "audio_sigma":
    torch.full((batch_size, ), 0.25),
    "num_frames":
    2,
    "height":
    2,
    "width":
    2,
    "fps":
    24.0,
    "audio_num_frames":
    _AUDIO_TOKENS,
    "use_cross_timestep":
    True,
  }
  inputs.update(overrides)
  return inputs


def _patched_pair(**transformer_kwargs):
  reference = _tiny_ltx25_transformer(**transformer_kwargs)
  patched = _tiny_ltx25_transformer(**transformer_kwargs)
  patched.load_state_dict(reference.state_dict())
  LTX2PatchFunctor().apply(patched)
  return reference, patched


def _forward(transformer, inputs):
  video, audio = transformer(**inputs, return_dict=False)
  return video, audio


def _assert_parity(reference, patched, inputs):
  video_ref, audio_ref = _forward(reference, inputs)
  video_new, audio_new = _forward(patched, inputs)
  torch.testing.assert_close(video_new, video_ref, rtol=1e-5, atol=1e-6)
  torch.testing.assert_close(audio_new, audio_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("use_cross_timestep", [True, False])
def test_patch_functor_matches_upstream_forward(use_cross_timestep):
  reference, patched = _patched_pair()
  _assert_parity(reference, patched, _inputs(use_cross_timestep=use_cross_timestep))


@pytest.mark.parametrize("perturbed_attn,use_prompt_embeddings", [(False, True), (True, True)])
def test_patch_functor_matches_ltx2_legacy_variants(perturbed_attn, use_prompt_embeddings):
  """LTX-2.0/2.3 checkpoints share this forward: plain processor + `caption_projection`."""
  reference, patched = _patched_pair(
    perturbed_attn=perturbed_attn,
    use_prompt_embeddings=use_prompt_embeddings,
  )
  _assert_parity(reference, patched, _inputs())


def test_patch_functor_matches_upstream_with_keyframes_and_modality_guidance():
  reference, patched = _patched_pair()
  inputs = _inputs()
  keyframes_mask = torch.zeros(1, _VIDEO_TOKENS, 1)
  keyframes_mask[:, 0, :] = 1.0
  inputs["video_keyframes_mask"] = keyframes_mask
  _assert_parity(reference, patched, inputs)

  # Modality guidance takes the off-cache branch (block semantics change).
  inputs["isolate_modalities"] = True
  _assert_parity(reference, patched, inputs)


def test_patch_functor_matches_upstream_with_spatio_temporal_guidance():
  reference, patched = _patched_pair()
  # STG needs a per-block perturbation mask, which a single collapsed wrapper call cannot express,
  # so the patched forward replays the original blocks.
  _assert_parity(reference, patched, _inputs(spatio_temporal_guidance_blocks=[1]))


def test_block_call_uses_the_four_required_positional_arguments():
  from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformerBlock

  # `forward` is read off the class, so `self` is the first parameter.
  parameters = list(inspect.signature(LTX2VideoTransformerBlock.forward).parameters)[1:]
  assert parameters[:4] == [
    "hidden_states",
    "audio_hidden_states",
    "encoder_hidden_states",
    "audio_encoder_hidden_states",
  ], "The cache wrapper binds the first two positionally and the rest through *args."


class LTX2DFRPipeline:
  """Minimal stand-in; only the class name matters for adapter lookup."""

  def __init__(self, transformer):
    self.transformer = transformer


def test_adapter_registers_ltx2_5_patch_functor():
  from cache_dit.caching.block_adapters.adapters import ltxvideo_adapter

  # `LTX2DFRPipeline` is not a `DiffusionPipeline`, so the registry cannot resolve it directly;
  # the adapter itself only relies on the `LTX` class prefix, which the registry applies.
  adapter = ltxvideo_adapter(LTX2DFRPipeline(_tiny_ltx25_transformer()))

  assert adapter.forward_pattern is ForwardPattern.Pattern_0
  assert isinstance(adapter.patch_functor, LTX2PatchFunctor)
  assert adapter.check_forward_pattern is False

  # Transformer-only lookup goes through the same adapter.
  by_transformer = BlockAdapterRegister.get_adapter(_tiny_ltx25_transformer())
  assert isinstance(by_transformer.patch_functor, LTX2PatchFunctor)


def _enable_cache(transformer, **cache_kwargs):
  adapter = BlockAdapterRegister.get_adapter(transformer)
  config_kwargs = dict(
    Fn_compute_blocks=1,
    Bn_compute_blocks=0,
    max_warmup_steps=0,
    max_cached_steps=-1,
    max_continuous_cached_steps=3,
    enable_separate_cfg=False,
    num_inference_steps=8,
  )
  config_kwargs.update(cache_kwargs)
  # The transformer is mutated in place: `transformer_blocks` is swapped for the cache wrapper.
  cache_dit.enable_cache(adapter, cache_config=DBCacheConfig(**config_kwargs))
  return transformer


def test_cache_never_corrupts_the_audio_stream():
  """Even with a forced hit, the audio stream must track the full-compute result closely."""
  reference = _tiny_ltx25_transformer()
  LTX2PatchFunctor().apply(reference)
  cached = _enable_cache(_tiny_ltx25_transformer(), residual_diff_threshold=1e9)

  cached_outputs = []
  for step in range(2):
    cached_outputs.append(
      _forward(cached, _inputs(timestep=torch.full((1, _VIDEO_TOKENS), 1000.0 - step * 250.0))))

  assert transformer_cached_a_step(cached), "No step was cached; the guard under test is idle."

  # Step 1 reuses step 0's residual, so compare against a full compute of the same step.
  video_ref, audio_ref = _forward(reference,
                                  _inputs(timestep=torch.full((1, _VIDEO_TOKENS), 750.0)))
  for got, ref in zip(cached_outputs[1], (video_ref, audio_ref)):
    relative = (got - ref).abs().mean() / ref.abs().mean().clamp_min(1e-6)
    assert relative < 0.1, f"cached output diverged (rel={relative.item():.4f})"


def transformer_cached_a_step(transformer):
  return len(transformer._context_manager.get_cached_steps()) > 0


def test_cache_threshold_zero_never_reuses_steps():
  transformer = _enable_cache(_tiny_ltx25_transformer(), residual_diff_threshold=0.0)

  for step in range(3):
    _forward(transformer, _inputs(timestep=torch.full((1, _VIDEO_TOKENS), 1000.0 - step * 250.0)))

  assert transformer._context_manager.get_cached_steps() == []


def test_stg_forward_bypasses_the_cache():
  """STG needs a per-block mask, so that forward must replay the original blocks verbatim."""
  transformer = _tiny_ltx25_transformer()
  reference = _tiny_ltx25_transformer()
  reference.load_state_dict(transformer.state_dict())
  cached = _enable_cache(transformer, residual_diff_threshold=1e9)

  inputs = _inputs(spatio_temporal_guidance_blocks=[1])
  video_cached, audio_cached = _forward(cached, inputs)
  video_ref, audio_ref = _forward(reference, inputs)

  torch.testing.assert_close(video_cached, video_ref, rtol=1e-5, atol=1e-6)
  torch.testing.assert_close(audio_cached, audio_ref, rtol=1e-5, atol=1e-6)


def test_equal_length_tile_geometry_bypasses_the_cache():
  """DFR tiles are equal length, so a token-count check alone would let them share residuals."""
  transformer = _enable_cache(_tiny_ltx25_transformer(), residual_diff_threshold=1e9)

  canvas_a = torch.zeros(1, 3, _VIDEO_TOKENS, 2)
  canvas_b = canvas_a.clone()
  canvas_b[:, 0, :, :] += 1.0  # same length, different origin: a different tile

  _forward(transformer, _inputs(video_coords=canvas_a))
  _forward(transformer, _inputs(video_coords=canvas_b))
  assert transformer._context_manager.get_cached_steps() == [], (
    "A tile with the same token count but a different geometry must not reuse a residual.")

  # A repeated geometry is cacheable again.
  _forward(transformer, _inputs(video_coords=canvas_b))
  assert transformer_cached_a_step(transformer)


def test_cp_plan_splits_the_keyframe_mask():
  transformer = _tiny_ltx25_transformer()
  processors = _snapshot_attention_patches()
  try:
    plan = LTX2ContextParallelismPlanner()._apply(transformer=transformer)
  finally:
    _restore_attention_patches(processors)

  root = plan[""]
  assert "video_keyframes_mask" in root, (
    "LTX-2.5 adds the keyframe embedding onto the already sharded hidden states, so the mask has "
    "to be split on the same axis.")
  assert root["video_keyframes_mask"].split_dim == 1
  assert root["video_keyframes_mask"].expected_dims == 3
  assert root["timestep"].split_dim == 1


def _snapshot_attention_patches():
  from diffusers.models.transformers.transformer_ltx2 import (
    LTX2Attention,
    LTX2AudioVideoAttnProcessor,
  )

  return (
    LTX2Attention,
    LTX2Attention.prepare_attention_mask,
    LTX2AudioVideoAttnProcessor,
    LTX2AudioVideoAttnProcessor.__call__,
    _LTX2PerturbedAttnProcessor,
    _LTX2PerturbedAttnProcessor.__call__,
  )


def _restore_attention_patches(snapshot):
  # The planner patches processor classes at class level; leaving that in place would change every
  # later test in the same session.
  attention_cls, prepare_mask, av_cls, av_call, pert_cls, pert_call = snapshot
  attention_cls.prepare_attention_mask = prepare_mask
  av_cls.__call__ = av_call
  pert_cls.__call__ = pert_call


def test_tp_plan_shards_per_head_gates():
  transformer = _tiny_ltx25_transformer()
  plan = _ltx2_block_layer_plan(transformer.transformer_blocks[0])

  for attn_name in ("attn1", "attn2", "audio_attn1", "audio_attn2", "audio_to_video_attn",
                    "video_to_audio_attn"):
    assert f"{attn_name}.to_gate_logits" in plan, (
      f"{attn_name} has per-head gates; without a colwise plan the gate width stops matching the "
      f"sharded head count.")
  # Modulation tables are replicated on purpose: TP shards Linear weights only.
  assert not any("prompt_adaln" in key for key in plan)


def test_tp_plan_skips_gates_when_disabled():
  transformer = _tiny_ltx25_transformer(gated_attn=False)
  plan = _ltx2_block_layer_plan(transformer.transformer_blocks[0])

  assert not any(key.endswith("to_gate_logits") for key in plan)


def test_shard_rotary_emb_processor_forwards_perturbation_arguments():
  LTX2PerturbedAttnProcessor = _LTX2PerturbedAttnProcessor
  captured = {}

  def _fake_call(self, attn, hidden_states, **kwargs):
    captured.update(kwargs)
    return hidden_states

  processor = LTX2PerturbedAttnProcessor()
  wrapper = ShardRotaryEmbProcessor.from_attn_processor(processor, tp_size=2, tp_rank=0)

  original = LTX2PerturbedAttnProcessor.__call__
  LTX2PerturbedAttnProcessor.__call__ = _fake_call
  try:
    cos = torch.randn(1, 4, _VIDEO_TOKENS, 4)
    sin = torch.randn(1, 4, _VIDEO_TOKENS, 4)
    wrapper(
      attn=None,
      hidden_states=torch.randn(1, _VIDEO_TOKENS, _INNER_DIM),
      perturbation_mask=torch.zeros(1, 1, 1),
      all_perturbed=False,
      query_rotary_emb=(cos, sin),
    )
  finally:
    LTX2PerturbedAttnProcessor.__call__ = original

  assert "perturbation_mask" in captured and "all_perturbed" in captured
  # Split RoPE is sharded on the head axis, `(B, H, T, D/2)` -> `(B, H/2, T, D/2)`.
  assert captured["query_rotary_emb"][0].shape[1] == 2


def _cli_args(example: str):
  from cache_dit._utils import generate

  argv = sys.argv
  sys.argv = ["cache_dit.generate", example]
  try:
    return generate.get_example_args()
  finally:
    sys.argv = argv


@pytest.mark.parametrize("example", ["ltx25_t2v", "ltx25_dfr"])
def test_ltx25_examples_construct_and_build_inputs(example):
  """A registered example that cannot even build its inputs never runs end-to-end."""
  from cache_dit._utils.registers import ExampleRegister

  try:
    # The example factories import their pipeline, which needs a diffusers build that also ships
    # the LTX-2.5 text encoder wiring. diffusers' lazy module re-raises the underlying import
    # failure as `RuntimeError`.
    import diffusers  # noqa: F401

    diffusers.LTX2Pipeline  # noqa: B018
    diffusers.LTX2DFRPipeline  # noqa: B018
  except (ImportError, AttributeError, RuntimeError):
    pytest.skip("diffusers build cannot import the LTX-2.5 pipelines")

  args = _cli_args(example)
  built = ExampleRegister.get_example(args, example)
  inputs = built.input_data.data(args)

  assert "prompt" in inputs
  if example == "ltx25_dfr":
    # DFR schedules its own sigmas and takes no `num_inference_steps`.
    assert "num_inference_steps" not in inputs
    assert "sigmas" in inputs
  else:
    assert inputs["num_inference_steps"] == 30


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
