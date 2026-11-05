"""Regression tests for the vendored HY-Nomas transformer and its segment KV cache.

HY-Nomas ships its own DiT (``cache_dit.models.pipelines.hy_nomas``) and keeps a **persistent**
per-layer KV cache across segments.  Three properties have to stay locked, because each of them can
break silently and only shows up as a wrong video many segments later:

* the block contract (``(hidden_states, encoder_hidden_states, ctx) -> (img, txt)``, Pattern_0),
* the K/V write path (``predict(collect_cache=True)`` returns one entry per layer, and the cached
  ``forward`` refuses to collect),
* the attention exemption: only ``kv_cache.py`` (plus the camera branch, which never touches the
  segment cache) may call a raw attention kernel instead of cache-dit's dispatcher.
"""
import ast
import inspect
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from cache_dit.models.pipelines.hy_nomas import (
  NomasCache,
  NomasTransformer3DModel,
)
from cache_dit.models.pipelines.hy_nomas.transformer import NomasDoubleStreamBlock

HIDDEN = 32
HEADS = 2
LAYERS = 2
TEXT_DIM = 16
TEXT_TOKENS = 6
FRAMES = 2
SIDE = 2
# Release constants, used by the state-dict layout test.
RELEASE_TENSORS_PER_BLOCK = 32
RELEASE_TENSORS_PER_ACTION_MODULE = 9


def _tiny_transformer(backend: str = "sdpa") -> NomasTransformer3DModel:
  torch.manual_seed(0)
  model = NomasTransformer3DModel(
    hidden_size=HIDDEN,
    num_attention_heads=HEADS,
    num_layers=LAYERS,
    in_channels=32,
    out_channels=32,
    text_states_dim=TEXT_DIM,
    # Tiny head_dim = 16, so the three RoPE axes must sum to it instead of the release's 128.
    rope_dim_list=(4, 6, 6),
    attention_backend=backend,
  ).eval().requires_grad_(False)
  with torch.no_grad():
    for name, param in model.named_parameters():
      if param.ndim == 1 and "norm" in name and name.endswith("weight"):
        param.fill_(1)
      else:
        param.normal_(0, 0.02)
  return model


def _inputs(model: NomasTransformer3DModel, cache: NomasCache, seed: int = 1):
  generator = torch.Generator().manual_seed(seed)
  text_states = torch.randn(1, TEXT_TOKENS, TEXT_DIM, generator=generator)
  text_mask = (torch.arange(TEXT_TOKENS)[None] < TEXT_TOKENS - 1)
  latent = torch.randn(1, 32, FRAMES, SIDE, SIDE, generator=generator)
  timestep = torch.tensor([500.0])
  return latent, text_states, timestep, text_mask


def test_block_contract_matches_pattern_0():
  model = _tiny_transformer()
  block = model.double_blocks[0]
  parameters = list(inspect.signature(NomasDoubleStreamBlock.forward).parameters)
  assert parameters == ["self", "hidden_states", "encoder_hidden_states", "ctx"]
  annotation = NomasDoubleStreamBlock.forward.__annotations__["return"]
  assert str(annotation).count("torch.Tensor") == 2
  # `cache-dit` discovers the stack from this attribute for quantization and `--compile`.
  assert model._repeated_blocks == ["NomasDoubleStreamBlock"]
  assert block.index == 0 and model.double_blocks[1].index == 1
  # The camera module is shared, not registered twice: the checkpoint names must survive.
  assert block.action is model.action_modules[0]


def test_cache_dit_pattern_checker_accepts_every_block():
  """`check_forward_pattern=True` + `check_num_outputs=True` must be usable, not disabled."""
  from types import SimpleNamespace

  from cache_dit.caching.cache_blocks.pattern_0_1_2 import CachedBlocks_Pattern_0_1_2
  from cache_dit.caching.cache_blocks.pattern_base import CachedBlocks_Pattern_Base
  from cache_dit.caching.forward_pattern import ForwardPattern

  model = _tiny_transformer()
  probe = SimpleNamespace(
    check_forward_pattern=True,
    check_num_outputs=True,
    transformer_blocks=model.double_blocks,
    forward_pattern=ForwardPattern.Pattern_0,
    _supported_patterns=CachedBlocks_Pattern_0_1_2._supported_patterns,
  )
  CachedBlocks_Pattern_Base._check_forward_pattern(probe)

  # Trap this locks down: `check_num_outputs` is *silently inert* without a return annotation
  # (`pattern_base.py` only compares when the count is above zero).  Dropping the annotation here
  # must therefore pass the checker, which is exactly why the explicit signature test above asserts
  # the annotation exists.
  block_forward = NomasDoubleStreamBlock.forward
  annotation = block_forward.__annotations__.pop("return")
  try:
    assert inspect.signature(block_forward).return_annotation is inspect.Signature.empty
    CachedBlocks_Pattern_Base._check_forward_pattern(probe)
  finally:
    block_forward.__annotations__["return"] = annotation
  CachedBlocks_Pattern_Base._check_forward_pattern(probe)


def test_state_dict_keeps_checkpoint_layout():
  model = _tiny_transformer()
  state = model.state_dict()
  assert sum(1 for key in state
             if key.startswith("double_blocks.")) == (LAYERS * RELEASE_TENSORS_PER_BLOCK)
  assert sum(1 for key in state
             if key.startswith("action_modules.")) == (LAYERS * RELEASE_TENSORS_PER_ACTION_MODULE)
  # Non-persistent camera tables must stay out of the checkpoint.
  assert not [key for key in state if "freqs" in key]
  assert "double_blocks.0.img_mlp.fc1.weight" in state
  assert "double_blocks.0.img_mod.linear.weight" in state
  assert "final_layer.adaLN_modulation.1.weight" in state


def test_forward_matches_predict_without_collect():
  model = _tiny_transformer()
  cache = model.allocate_cache(64)
  latent, text_states, timestep, text_mask = _inputs(model, cache)
  with torch.inference_mode():
    forward_out = model(latent, text_states, timestep, text_mask=text_mask, cache=cache)
    predict_out, entries = model.predict(latent, timestep, text_states, text_mask, cache=cache)
  assert entries is None
  assert forward_out.shape == (1, 32, FRAMES, SIDE, SIDE)
  assert torch.equal(forward_out, predict_out)


def test_forward_rejects_collect_cache():
  model = _tiny_transformer()
  cache = model.allocate_cache(64)
  latent, text_states, timestep, text_mask = _inputs(model, cache)
  with pytest.raises(RuntimeError, match="predict\\(collect_cache=True\\)"):
    with torch.inference_mode():
      model(latent, text_states, timestep, text_mask=text_mask, cache=cache, collect_cache=True)


def test_collect_returns_one_entry_per_layer():
  model = _tiny_transformer()
  cache = model.allocate_cache(64)
  latent, text_states, timestep, text_mask = _inputs(model, cache)
  with torch.inference_mode():
    _, entries = model.predict(latent,
                               timestep,
                               text_states,
                               text_mask,
                               cache=cache,
                               collect_cache=True)
  assert len(entries) == LAYERS
  video_tokens = FRAMES * SIDE * SIDE
  for entry in entries:
    assert set(entry) == {"k", "v"}
    assert entry["k"].shape == (1, video_tokens + TEXT_TOKENS, HEADS, HIDDEN // HEADS)


def test_commit_publishes_visibility_and_history():
  model = _tiny_transformer()
  cache = model.allocate_cache(64)
  latent, text_states, timestep, text_mask = _inputs(model, cache, seed=3)
  video_tokens = FRAMES * SIDE * SIDE
  with torch.inference_mode():
    before, entries = model.predict(latent,
                                    timestep,
                                    text_states,
                                    text_mask,
                                    cache=cache,
                                    collect_cache=True)
  cache.commit(entries, text_mask, video_tokens, FRAMES)
  assert cache.valid_length == video_tokens + TEXT_TOKENS
  assert cache.frame_offset == FRAMES
  assert int(cache.history_mask(False).sum()) == video_tokens + int(text_mask.sum())
  with torch.inference_mode():
    after = model(latent, text_states, timestep, text_mask=text_mask, cache=cache)
  # History is read, so the prediction must move.
  assert not torch.equal(before, after)
  assert torch.isfinite(after).all()
  assert cache.action_tails == []

  # A committed action tail is the only way `hide_cached_action_tails` can change the mask, and the
  # stored rows must never change with it.
  with torch.inference_mode():
    before_tail, entries_tail = model.predict(latent,
                                              timestep,
                                              text_states,
                                              text_mask,
                                              cache=cache,
                                              collect_cache=True)
  action_start = 2 if int(text_mask.sum()) > 2 else -1
  if action_start >= 0:
    cache.commit(entries_tail, text_mask, video_tokens, FRAMES, action_start=action_start)
    assert cache.action_tails
    shown = int(cache.history_mask(False).sum())
    hidden = int(cache.history_mask(True).sum())
    assert hidden < shown
    visible_rows = int(cache.visible[:, :cache.valid_length].sum())
    assert visible_rows == shown


def test_evict_oldest_segment_keeps_frame_offset():
  model = _tiny_transformer()
  cache = model.allocate_cache(64)
  latent, text_states, timestep, text_mask = _inputs(model, cache, seed=5)
  video_tokens = FRAMES * SIDE * SIDE
  with torch.inference_mode():
    _, entries = model.predict(latent,
                               timestep,
                               text_states,
                               text_mask,
                               cache=cache,
                               collect_cache=True)
  cache.commit(entries, text_mask, video_tokens, FRAMES)
  evicted = cache.evict_oldest_segment()
  assert evicted == video_tokens + TEXT_TOKENS
  assert cache.valid_length == 0
  # Absolute frame positions keep counting after an eviction.
  assert cache.frame_offset == FRAMES
  assert not cache.segment_lengths


def test_cache_head_count_must_match_rank():
  model = _tiny_transformer()
  cache = NomasCache.allocate(model.config,
                              64,
                              device=torch.device("cpu"),
                              dtype=next(model.parameters()).dtype,
                              num_heads=1)
  latent, text_states, timestep, text_mask = _inputs(model, cache)
  with pytest.raises(ValueError, match="allocate the cache after parallelism"):
    with torch.inference_mode():
      model(latent, text_states, timestep, text_mask=text_mask, cache=cache)


def test_camera_embedding_follows_latent_timeline():
  model = _tiny_transformer()
  cache = model.allocate_cache(64)
  latent, text_states, timestep, text_mask = _inputs(model, cache)
  # Whole-clip poses are `(L-1)*4+1`; four poses collapse into one camera token.
  poses = torch.randn(1, (FRAMES - 1) * 4 + 1, 4, 4)
  embedding = model.embed_camera(poses)
  assert embedding.shape == (1, FRAMES, 64)
  with torch.inference_mode():
    with_camera = model(latent,
                        text_states,
                        timestep,
                        text_mask=text_mask,
                        cache=cache,
                        camera_embedding=embedding)
    without = model(latent, text_states, timestep, text_mask=text_mask, cache=cache)
  assert torch.isfinite(with_camera).all()
  assert not torch.equal(with_camera, without)


def _raw_attention_calls(path: Path):
  tree = ast.parse(path.read_text())
  found = []
  for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
      continue
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    if name in ("scaled_dot_product_attention", "flex_attention", "flash_attn_varlen_func"):
      found.append((path.name, name))
  return found


def test_attention_kernels_stay_in_the_plan():
  """Only the segment plan (and the camera branch) may call a raw attention kernel.

  ``_dispatch_attention_fn`` cannot express this model's varlen log-sum-exp merge or its
  FlexAttention block mask, so ``kv_cache.py`` owns the exemption.  The camera branch keeps a
  chunked SDPA over at most ``camera_chunk_size`` frames; it never reads the segment cache and is
  unreachable for the released ``camera_chunk_size=1`` checkpoint.
  """
  package = (Path(__file__).resolve().parents[2] / "src" / "cache_dit" / "models" / "pipelines" /
             "hy_nomas")
  # Only the DiT files are in scope: `components.py` is the PixelUMM *decoder* network, which owns
  # its own attention and never sees the segment cache or the DiT's context-parallel plan.
  dit_files = {"__init__.py", "attention.py", "kv_cache.py", "transformer.py"}
  offenders = {
    name: [call for call in _raw_attention_calls(path) if not (name == "kv_cache.py")]
    for name, path in ((p.name, p) for p in sorted(package.glob("*.py")) if p.name in dit_files)
  }
  offenders = {name: calls for name, calls in offenders.items() if calls}
  # Exactly one tolerated second site: the chunked camera branch, which never reads the segment
  # cache.  Any additional raw kernel call (or moving this one) fails the exact-set comparison.
  expected = {"transformer.py": [("transformer.py", "scaled_dot_product_attention")]}
  assert offenders == expected, offenders


def test_kv_sink_keeps_identity_through_the_offload_argument_walk():
  """The layerwise offload hook rebuilds every dataclass/list it walks.

  ``_map_tensors`` replaces a dataclass with ``dataclasses.replace`` and a list with a fresh
  comprehension, so a plain ``[None] * num_layers`` sink would be swapped for a copy at every block
  call and every collected K/V would be dropped -- silently, until the pipeline-level
  "collection skipped N of 54 blocks" check fires after the whole segment was computed.
  """
  from cache_dit.offload.layerwise import _map_tensors
  from cache_dit.models.pipelines.hy_nomas.transformer import NomasEntrySink

  sink = NomasEntrySink(2)
  assert _map_tensors(sink, transform=lambda tensor: tensor) is sink

  @dataclass(frozen=True)
  class Holder:
    sink: NomasEntrySink
    flag: int

  holder = Holder(sink=sink, flag=1)
  rebuilt = _map_tensors(holder, transform=lambda tensor: tensor)
  assert rebuilt is not holder
  assert rebuilt.sink is sink
  rebuilt.sink[1] = {"k": torch.zeros(1)}
  assert sink.entries[1] is not None
