"""Regression tests for the CausVid-Wan cache integration.

CausVid is the first cache-dit model whose blocks write a **cross-chunk** KV cache, so the tests here
lock two things beyond the usual adapter wiring:

* the clean-latent commit pass replays the raw block stack even when the block-skip wrapper is
  active — a skipped layer would keep the noisy K/V of the previous step and poison every later
  chunk;
* a chunk boundary arms `force_refresh_step_hint`, so the first denoising step of a chunk is never
  served from cache.

Usage:
  python -m pytest tests/api/test_causvid_wan_adapter.py -q
"""

import copy

import diffusers
import torch

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.caching.block_adapters import BlockAdapterRegister
from cache_dit.caching.forward_pattern import ForwardPattern
from cache_dit.models.pipelines.causvid.convert import validate_causal_structure
from cache_dit.models.pipelines.causvid.pipeline import CausVidPipeline
from cache_dit.models.pipelines.causvid.transformer import CausVidTransformer3DModel

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
  cross_attn_norm=False,
  qk_norm="rms_norm_across_heads",
  eps=1e-6,
  rope_max_seq_len=64,
)
CHUNK_FRAMES = 2
NUM_CHUNKS = 2
LATENT_SHAPE = (1, 16, CHUNK_FRAMES, 8, 8)
DENOISING_STEPS = (1000, 757, 522)


def _tiny_transformer(seed: int = 0) -> CausVidTransformer3DModel:
  torch.manual_seed(seed)
  return CausVidTransformer3DModel(**TINY_CONFIG).eval()


class CausVidTestPipeline(CausVidPipeline):
  """Runs the real chunk loop without the text encoder / VAE."""

  def __init__(self, transformer):
    super().__init__(tokenizer=None, text_encoder=None, transformer=transformer, vae=None)
    self.steps = list(DENOISING_STEPS)

  def __call__(self, chunks, context, use_cache: bool = True):
    transformer = self.transformer
    kv_cache = transformer.init_kv_cache(
      batch_size=1,
      latent_shape=LATENT_SHAPE,
      num_frames=CHUNK_FRAMES * len(chunks),
    )
    frame_seq_len = transformer.frame_seq_len(LATENT_SHAPE)
    chunk_tokens = CHUNK_FRAMES * frame_seq_len
    outputs = []
    for index, chunk in enumerate(chunks):
      current_start = index * chunk_tokens
      current_end = current_start + chunk_tokens
      if use_cache:
        self._refresh_step_cache()
      x0 = chunk
      for step_index, timestep in enumerate(self.steps):
        timestep_batch = torch.full((1, ), timestep, dtype=torch.float32)
        model_output = transformer(
          chunk,
          timestep_batch,
          context,
          kv_cache=kv_cache,
          current_start=current_start,
          current_end=current_end,
        )
        sigma = self.scheduler.sigma(timestep_batch).view(-1, 1, 1, 1, 1).to(chunk.dtype)
        x0 = chunk - sigma * model_output
        if step_index < len(self.steps) - 1:
          chunk = x0
      outputs.append(x0)
      transformer(
        x0,
        torch.zeros((1, ), dtype=torch.float32),
        context,
        kv_cache=kv_cache,
        current_start=current_start,
        current_end=current_end,
        force_full_compute=True,
      )
    return outputs, kv_cache


def _chunks(seed: int = 7):
  generator = torch.Generator().manual_seed(seed)
  return [torch.randn(LATENT_SHAPE, generator=generator) for _ in range(NUM_CHUNKS)]


def test_causvid_wan_prefix_is_the_unique_match():
  prefixes = BlockAdapterRegister.supported_pipelines()[1]
  assert "CausVid" in prefixes
  matches = [name for name in prefixes if "CausVidTransformer3DModel".startswith(name)]
  assert matches == ["CausVid"], f"ambiguous adapter prefixes for CausVid-Wan: {matches}"


def test_adapter_wiring():
  from cache_dit.caching.block_adapters import causvid_wan_adapter

  transformer = _tiny_transformer()
  adapter = causvid_wan_adapter(CausVidTestPipeline(transformer))

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
  pipe = CausVidTestPipeline(_tiny_transformer())
  assert cache_dit.utils.is_pipeline(pipe)
  assert isinstance(pipe.transformer, diffusers.ModelMixin)
  assert pipe.transformer._repeated_blocks == ["CausVidTransformerBlock"]
  assert isinstance(pipe.transformer.blocks, torch.nn.ModuleList)
  assert [block.index
          for block in pipe.transformer.blocks] == list(range(TINY_CONFIG["num_layers"]))
  # The structural assertions that guard the `norm2`/`norm3` swap must pass on the real config.
  validate_causal_structure(pipe.transformer)


def test_cached_run_is_bit_exact_when_no_step_is_cached():
  reference_transformer = _tiny_transformer()
  transformer = copy.deepcopy(reference_transformer)
  pipe = CausVidTestPipeline(transformer)
  reference_pipe = CausVidTestPipeline(reference_transformer)

  chunks = _chunks()
  context = torch.randn(1, 5, TINY_CONFIG["text_dim"], generator=torch.Generator().manual_seed(3))

  with torch.no_grad():
    reference_outputs, reference_cache = reference_pipe(chunks, context, use_cache=False)

  cache_dit.enable_cache(
    BlockAdapterRegister.get_adapter(pipe),
    cache_config=DBCacheConfig(
      num_inference_steps=None,
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=0,
      max_cached_steps=0,  # caching disabled: the run must stay bit-exact
      residual_diff_threshold=0.0,
    ),
  )
  try:
    with torch.no_grad():
      outputs, kv_cache = pipe(chunks, context)
    for index, (out, ref) in enumerate(zip(outputs, reference_outputs)):
      torch.testing.assert_close(out, ref, atol=0, rtol=0, msg=f"chunk {index} diverged")
    for layer, (mine, ref) in enumerate(zip(kv_cache, reference_cache)):
      torch.testing.assert_close(mine.k,
                                 ref.k,
                                 atol=0,
                                 rtol=0,
                                 msg=f"K/V diverged at layer {layer}")
      torch.testing.assert_close(mine.v,
                                 ref.v,
                                 atol=0,
                                 rtol=0,
                                 msg=f"K/V diverged at layer {layer}")
  finally:
    cache_dit.disable_cache(pipe)


def test_commit_pass_runs_every_block_and_chunk_start_is_refreshed():
  transformer = _tiny_transformer()
  pipe = CausVidTestPipeline(transformer)
  calls = [0] * TINY_CONFIG["num_layers"]
  hooks = [
    block.register_forward_hook(
      lambda *args, index=index: calls.__setitem__(index, calls[index] + 1))
    for index, block in enumerate(transformer.blocks)
  ]

  cache_dit.enable_cache(
    BlockAdapterRegister.get_adapter(pipe),
    cache_config=DBCacheConfig(
      num_inference_steps=None,
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=0,
      max_cached_steps=99,
      residual_diff_threshold=1e9,  # cache every step that is allowed to be cached
    ),
  )
  try:
    chunks = _chunks()
    context = torch.randn(1, 5, TINY_CONFIG["text_dim"], generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
      pipe(chunks, context)

    manager = pipe._context_manager
    # `pipe.__call__` restores the previous (empty) current context when it returns, so read the
    # named contexts the manager keeps instead.
    contexts = list(manager._cached_context_manager.values())
    assert len(contexts) == 1, f"unexpected context count: {len(contexts)}"
    cached_steps = contexts[0].get_cached_steps()
    accumulated_cached = contexts[0].get_accumulated_cached_steps()
    total_forwards = NUM_CHUNKS * (len(DENOISING_STEPS) + 1)

    # Caching must actually have happened for the assertions below to mean anything.
    assert accumulated_cached >= 2, "no step was cached; the test would pass trivially"
    # The refresh gives every chunk a fresh step counter, so the recorded indices of the last chunk
    # are exactly its second and third denoising steps. Losing the refresh would let the indices run
    # on and serve a chunk-start step from cache (see the negative control below).
    assert set(cached_steps) == set(range(1, len(DENOISING_STEPS))), (
      f"chunk-boundary refresh did not reset the step counter: {cached_steps}")

    # The first block runs on every forward: Fn block of the denoise steps + the commit passes.
    assert calls[0] == total_forwards
    # Middle blocks are skipped on cached steps, but every commit pass must still run them.
    for index in range(1, TINY_CONFIG["num_layers"]):
      assert calls[index] >= NUM_CHUNKS, f"block {index} missed a commit pass: {calls}"
      assert calls[index] < total_forwards, f"block {index} was never skipped: {calls}"
  finally:
    for hook in hooks:
      hook.remove()
    cache_dit.disable_cache(pipe)


def test_without_the_refresh_a_chunk_start_is_served_from_cache():
  """Negative control for the chunk-boundary refresh: it is what keeps chunk starts unsafe to skip.

  With one shared context across the whole clip the step indices keep growing, so the first step of
  a later chunk is a plain cache candidate — exactly the step that must recompute after the noise is
  resampled.
  """
  transformer = _tiny_transformer()
  pipe = CausVidTestPipeline(transformer)
  cache_dit.enable_cache(
    BlockAdapterRegister.get_adapter(pipe),
    cache_config=DBCacheConfig(
      num_inference_steps=None,
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=1,
      max_cached_steps=99,
      residual_diff_threshold=1e9,
    ),
  )
  try:
    with torch.no_grad():
      pipe(_chunks(),
           torch.randn(1, 5, TINY_CONFIG["text_dim"], generator=torch.Generator().manual_seed(3)),
           use_cache=False)
    contexts = list(pipe._context_manager._cached_context_manager.values())
    cached_steps = contexts[0].get_cached_steps()
    chunk_starts = [step for step in cached_steps if step % len(DENOISING_STEPS) == 0]
    assert chunk_starts, (
      f"expected the guard-less run to cache a chunk-start step, got {cached_steps}")
  finally:
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
        torch.randn(LATENT_SHAPE),
        torch.tensor([1000.0]),
        torch.randn(1, 5, TINY_CONFIG["text_dim"]),
        force_full_compute=True,
      )
    assert calls == [1] * TINY_CONFIG["num_layers"]
  finally:
    for hook in hooks:
      hook.remove()
