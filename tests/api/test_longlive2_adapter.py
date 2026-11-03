"""Regression tests for the LongLive 2.0 cache integration.

LongLive 2.0 is the second cache-dit model whose blocks write a **cross-chunk** KV cache (CausVid is
the first), so beyond the usual adapter wiring the tests here lock four things:

* the clean-latent commit pass replays the raw block stack even when the block-skip wrapper is
  active - a skipped layer would keep the noisy K/V of the previous step and poison every later
  chunk;
* the commit pass is the only writer of the clean K/V a later chunk reads;
* a chunk boundary arms `force_refresh_step_hint`, so the first denoising step of a chunk is never
  served from cache;
* the cache warmup window is capped at one step, otherwise every step of every chunk lands inside it
  and caching can never engage.

The pipeline under test is the production `_render_latents`, driven through an overridden
`__call__` so cache-dit still installs its per-inference context (`CachedAdapter` patches
`pipe.__class__.__call__`); the reference is a hand-written loop in this file.

Usage:
  python -m pytest tests/api/test_longlive2_adapter.py -q
"""

import copy

import diffusers
import pytest
import torch

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.caching.block_adapters import BlockAdapterRegister
from cache_dit.caching.forward_pattern import ForwardPattern
from cache_dit.models.pipelines.longlive2 import LongLive2Pipeline
from cache_dit.models.pipelines.longlive2.convert import validate_causal_structure
from cache_dit.models.pipelines.longlive2.transformer import LongLive2Transformer3DModel

TINY_CONFIG = dict(
  patch_size=(1, 2, 2),
  num_attention_heads=2,
  attention_head_dim=8,
  in_channels=16,
  out_channels=16,
  text_dim=32,
  freq_dim=32,
  ffn_dim=64,
  num_layers=3,
  cross_attn_norm=True,
  qk_norm="rms_norm_across_heads",
  eps=1e-6,
  rope_max_seq_len=64,
)
CHUNK_FRAMES = 2
NUM_CHUNKS = 2
LATENT_SHAPE = (1, 16, CHUNK_FRAMES * NUM_CHUNKS, 8, 8)
NUM_STEPS = 4


def _tiny_transformer(seed: int = 0) -> LongLive2Transformer3DModel:
  torch.manual_seed(seed)
  return LongLive2Transformer3DModel(**TINY_CONFIG).eval()


class LongLive2TestPipeline(LongLive2Pipeline):
  """Runs the production chunk loop without the text encoder / VAE."""

  def __init__(self, transformer):
    super().__init__(tokenizer=None, text_encoder=None, transformer=transformer, vae=None)

  def __call__(self, latents, context, **kwargs):
    transformer = self.transformer
    chunk_latent_shape = latents.shape[:2] + (CHUNK_FRAMES, ) + latents.shape[3:]
    frame_seq_len = transformer.frame_seq_len(chunk_latent_shape)
    kv_cache = transformer.init_kv_cache(
      batch_size=latents.shape[0],
      latent_shape=chunk_latent_shape,
      num_frames=latents.shape[2],
    )
    return self._render_latents(
      prompt_embeds=context,
      latents=latents,
      chunk_latent_frames=CHUNK_FRAMES,
      frame_seq_len=frame_seq_len,
      chunk_tokens=CHUNK_FRAMES * frame_seq_len,
      kv_cache=kv_cache,
      num_inference_steps=NUM_STEPS,
      context_noise=0.0,
      device=latents.device,
    ), kv_cache


def _reference_render(pipe, latents, context):
  """Hand-written equivalent of the production loop, without any cache interaction."""
  transformer = pipe.transformer
  chunk_latent_shape = latents.shape[:2] + (CHUNK_FRAMES, ) + latents.shape[3:]
  frame_seq_len = transformer.frame_seq_len(chunk_latent_shape)
  chunk_tokens = CHUNK_FRAMES * frame_seq_len
  kv_cache = transformer.init_kv_cache(
    batch_size=latents.shape[0],
    latent_shape=chunk_latent_shape,
    num_frames=latents.shape[2],
  )
  output = torch.zeros_like(latents)
  commit_timestep = torch.zeros((latents.shape[0], ), dtype=torch.float32)
  for index in range(NUM_CHUNKS):
    start_frame = index * CHUNK_FRAMES
    current_start = start_frame * frame_seq_len
    current_end = current_start + chunk_tokens
    chunk = latents[:, :, start_frame:start_frame + CHUNK_FRAMES].clone()
    pipe.scheduler.set_timesteps(NUM_STEPS)
    for timestep in pipe.scheduler.timesteps:
      model_output = transformer(
        chunk,
        timestep.expand(latents.shape[0]).float(),
        context,
        kv_cache=kv_cache,
        current_start=current_start,
        current_end=current_end,
      )
      chunk = pipe.scheduler.step(model_output, timestep, chunk, return_dict=False)[0]
    output[:, :, start_frame:start_frame + CHUNK_FRAMES] = chunk
    transformer(
      chunk,
      commit_timestep,
      context,
      kv_cache=kv_cache,
      current_start=current_start,
      current_end=current_end,
      force_full_compute=True,
    )
  return output, kv_cache


def _latents(seed: int = 7):
  return torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(seed))


def _context(seed: int = 3):
  return torch.randn(1, 5, TINY_CONFIG["text_dim"], generator=torch.Generator().manual_seed(seed))


def _cache_config(**overrides) -> DBCacheConfig:
  config = dict(
    num_inference_steps=None,
    Fn_compute_blocks=1,
    Bn_compute_blocks=0,
    max_warmup_steps=0,
    max_cached_steps=0,
    residual_diff_threshold=0.0,
  )
  config.update(overrides)
  return DBCacheConfig(**config)


def test_longlive2_prefix_is_the_unique_match():
  prefixes = BlockAdapterRegister.supported_pipelines()[1]
  assert "LongLive2" in prefixes
  matches = [name for name in prefixes if "LongLive2Transformer3DModel".startswith(name)]
  assert matches == ["LongLive2"], f"ambiguous adapter prefixes for LongLive 2.0: {matches}"


def test_adapter_wiring():
  from cache_dit.caching.block_adapters import longlive2_adapter

  transformer = _tiny_transformer()
  adapter = longlive2_adapter(LongLive2TestPipeline(transformer))

  assert adapter.blocks_name == "blocks"
  assert adapter.forward_pattern == ForwardPattern.Pattern_2
  assert adapter.check_forward_pattern is True
  # Guidance-distilled: one conditional forward per denoising step.
  assert adapter.has_separate_cfg is False
  # The commit-pass bypass lives inside the vendored forward, so no structural patch is needed.
  assert adapter.patch_functor is None


def test_registered_adapter_supports_transformer_only_usage():
  transformer = _tiny_transformer()
  adapter = BlockAdapterRegister.get_adapter(transformer)

  assert adapter is not None
  assert adapter.transformer is transformer
  assert adapter.blocks is transformer.blocks
  assert adapter.blocks_name == "blocks"


def test_builtin_transformer_declares_repeated_blocks_and_indexes():
  pipe = LongLive2TestPipeline(_tiny_transformer())
  assert cache_dit.utils.is_pipeline(pipe)
  assert isinstance(pipe.transformer, diffusers.ModelMixin)
  assert pipe.transformer._repeated_blocks == ["LongLive2TransformerBlock"]
  assert isinstance(pipe.transformer.blocks, torch.nn.ModuleList)
  assert [block.index
          for block in pipe.transformer.blocks] == list(range(TINY_CONFIG["num_layers"]))
  # The structural assertions that guard the `norm2`/`norm3` swap must pass on the real config.
  validate_causal_structure(pipe.transformer)


def test_cached_run_is_bit_exact_when_no_step_is_cached():
  reference_transformer = _tiny_transformer()
  transformer = copy.deepcopy(reference_transformer)
  pipe = LongLive2TestPipeline(transformer)
  reference_pipe = LongLive2TestPipeline(reference_transformer)

  latents = _latents()
  context = _context()
  with torch.no_grad():
    reference_output, reference_cache = _reference_render(reference_pipe, latents, context)

  cache_dit.enable_cache(BlockAdapterRegister.get_adapter(pipe),
                         cache_config=_cache_config(max_cached_steps=0))
  try:
    with torch.no_grad():
      output, kv_cache = pipe(latents, context)
    torch.testing.assert_close(output, reference_output, atol=0, rtol=0)
    for layer, (mine, ref) in enumerate(zip(kv_cache, reference_cache)):
      torch.testing.assert_close(mine.k, ref.k, atol=0, rtol=0, msg=f"K diverged at layer {layer}")
      torch.testing.assert_close(mine.v, ref.v, atol=0, rtol=0, msg=f"V diverged at layer {layer}")
  finally:
    cache_dit.disable_cache(pipe)


def test_commit_pass_runs_every_block_and_chunk_start_is_refreshed():
  transformer = _tiny_transformer()
  pipe = LongLive2TestPipeline(transformer)
  calls = [0] * TINY_CONFIG["num_layers"]
  hooks = [
    block.register_forward_hook(
      lambda *args, index=index: calls.__setitem__(index, calls[index] + 1))
    for index, block in enumerate(transformer.blocks)
  ]

  cache_dit.enable_cache(
    BlockAdapterRegister.get_adapter(pipe),
    cache_config=_cache_config(max_warmup_steps=4, max_cached_steps=99,
                               residual_diff_threshold=1e9),
  )
  try:
    with torch.no_grad():
      pipe(_latents(), _context())

    manager = pipe._context_manager
    contexts = list(manager._cached_context_manager.values())
    assert len(contexts) == 1, f"unexpected context count: {len(contexts)}"
    cached_steps = contexts[0].get_cached_steps()
    accumulated_cached = contexts[0].get_accumulated_cached_steps()
    total_forwards = NUM_CHUNKS * (NUM_STEPS + 1)

    # Caching must actually have happened for the assertions below to mean anything.
    assert accumulated_cached >= 2, "no step was cached; the test would pass trivially"
    assert contexts[0].cache_config.max_warmup_steps == 1, "the per-chunk warmup cap did not apply"
    # The refresh gives every chunk a fresh step counter, so the recorded indices of the last chunk
    # are exactly its second, third and fourth denoising steps.
    assert set(cached_steps) == set(range(
      1, NUM_STEPS)), (f"chunk-boundary refresh did not reset the step counter: {cached_steps}")

    # The first block runs on every forward: Fn block of the denoise steps + the commit passes.
    assert calls[0] == total_forwards
    for index in range(1, TINY_CONFIG["num_layers"]):
      assert calls[index] >= NUM_CHUNKS, f"block {index} missed a commit pass: {calls}"
      assert calls[index] < total_forwards, f"block {index} was never skipped: {calls}"
  finally:
    for hook in hooks:
      hook.remove()
    cache_dit.disable_cache(pipe)


def test_without_the_refresh_a_chunk_start_is_served_from_cache():
  """Negative control: without the boundary hint a chunk start becomes a cache candidate."""
  transformer = _tiny_transformer()
  pipe = LongLive2TestPipeline(transformer)
  cache_dit.enable_cache(
    BlockAdapterRegister.get_adapter(pipe),
    cache_config=_cache_config(max_warmup_steps=1, max_cached_steps=99,
                               residual_diff_threshold=1e9),
  )
  original_refresh = pipe._refresh_step_cache
  pipe._refresh_step_cache = lambda: None
  try:
    with torch.no_grad():
      pipe(_latents(), _context())
    contexts = list(pipe._context_manager._cached_context_manager.values())
    cached_steps = contexts[0].get_cached_steps()
    chunk_starts = [step for step in cached_steps if step % NUM_STEPS == 0]
    assert chunk_starts, (
      f"expected the guard-less run to cache a chunk-start step, got {cached_steps}")
  finally:
    pipe._refresh_step_cache = original_refresh
    cache_dit.disable_cache(pipe)


def test_commit_bypass_is_inert_without_cache():
  transformer = _tiny_transformer()
  calls = [0] * TINY_CONFIG["num_layers"]
  hooks = [
    block.register_forward_hook(
      lambda *args, index=index: calls.__setitem__(index, calls[index] + 1))
    for index, block in enumerate(transformer.blocks)
  ]
  try:
    with torch.no_grad():
      transformer(
        torch.randn((1, 16, CHUNK_FRAMES, 8, 8)),
        torch.tensor([500.0]),
        torch.randn(1, 5, TINY_CONFIG["text_dim"]),
        force_full_compute=True,
      )
    assert calls == [1] * TINY_CONFIG["num_layers"]
  finally:
    for hook in hooks:
      hook.remove()


def test_commit_pass_writes_the_clean_kv_of_its_chunk():
  """The first layer's committed K/V must equal what a fresh cache gets for the clean latent.

  Only layer 0 is compared: a deeper layer's K/V is a function of its own input, which already
  depends on the earlier chunks' attention, so replaying one chunk in isolation cannot reproduce it.
  """
  transformer = _tiny_transformer()
  pipe = LongLive2TestPipeline(transformer)
  latents = _latents()
  context = _context()

  with torch.no_grad():
    final, kv_cache = _reference_render(pipe, latents, context)
    fresh = transformer.init_kv_cache(
      batch_size=1,
      latent_shape=(1, 16, CHUNK_FRAMES, 8, 8),
      num_frames=latents.shape[2],
    )
    frame_seq_len = transformer.frame_seq_len((1, 16, CHUNK_FRAMES, 8, 8))
    chunk_tokens = CHUNK_FRAMES * frame_seq_len
    transformer(
      final[:, :, CHUNK_FRAMES:],
      torch.zeros((1, ), dtype=torch.float32),
      context,
      kv_cache=fresh,
      current_start=chunk_tokens,
      current_end=2 * chunk_tokens,
      force_full_compute=True,
    )
    torch.testing.assert_close(kv_cache[0].k[:, chunk_tokens:2 * chunk_tokens],
                               fresh[0].k[:, chunk_tokens:2 * chunk_tokens],
                               atol=0,
                               rtol=0,
                               msg="clean K/V of the last chunk was not committed")


def test_guidance_scale_is_rejected():
  # The guard lives in the production `__call__`, which the test subclass overrides.
  pipe = LongLive2Pipeline(tokenizer=None,
                           text_encoder=None,
                           transformer=_tiny_transformer(),
                           vae=None)
  with pytest.raises(ValueError, match="guidance-distilled"):
    pipe(prompt="x", guidance_scale=1.3)
