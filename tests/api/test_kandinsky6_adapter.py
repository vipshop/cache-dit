"""Regression tests for the Kandinsky 6.0 cache integration.

Kandinsky 6.0 denoises a **video** and an **audio** stream through one fused block stack
(`Kandinsky6FusedTransformerDecoderBlock`), whose blocks return `(vis, aud)` — i.e. the second stream
is an evolving latent, not a static text context, which is exactly `ForwardPattern.Pattern_0`. These
CPU tests pin down the three properties the integration depends on:

1. `Kandinsky6TI2VAPipeline` dispatches to the `"Kandinsky6"` adapter (and the VSR pipeline, which
   drives a different transformer, is rejected instead of silently mis-adapted).
2. `has_separate_cfg` follows the pipeline default (`guidance_scale=5.0`), so a distilled run has to
   opt out explicitly.
3. A cached run is bit-exact against the unpatched transformer — for **both** streams.
"""
import inspect
import types

import pytest
import torch

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.caching.block_adapters import BlockAdapterRegister
from cache_dit.caching.forward_pattern import ForwardPattern

diffusers = pytest.importorskip("diffusers")

if not hasattr(diffusers, "Kandinsky6Transformer3DModel"):
  pytest.skip("Kandinsky 6.0 requires a newer diffusers", allow_module_level=True)

_TEXT_LEN = 5
_FRAMES = 2
_LATENT_H = 4
_LATENT_W = 4
_VISUAL_IN = 4
_AUDIO_IN = 4
_AUDIO_LEN = 3
_MODEL_DIM = 32
_AUDIO_DIM = 16


def _tiny_transformer(num_visual_blocks: int = 2):
  from diffusers import Kandinsky6Transformer3DModel

  torch.manual_seed(0)
  transformer = Kandinsky6Transformer3DModel(
    in_visual_dim=_VISUAL_IN,
    out_visual_dim=_VISUAL_IN,
    in_text_dim=8,
    in_text_dim2=8,
    time_dim=8,
    patch_size=(1, 2, 2),
    model_dim=_MODEL_DIM,
    ff_dim=2 * _MODEL_DIM,
    num_text_blocks=1,
    num_visual_blocks=num_visual_blocks,
    # `sum(axes_dims)` is the head dim: 32 / 8 = 4 video heads, 16 / 8 = 2 audio heads.
    axes_dims=(2, 2, 4),
    visual_cond=False,
    in_audio_dim=_AUDIO_IN,
    out_audio_dim=_AUDIO_IN,
    model_dim_a=_AUDIO_DIM,
    time_dim_a=8,
    ff_dim_a=2 * _AUDIO_DIM,
    axes_dims_a=(2, 2, 4),
    # Mirror the released config: cross-modal rope, cross-modal gates and the fixed modulation.
    ca_rope=True,
    cross_gates=True,
    fix_modulation=True,
    text_token_padding=True,
  )
  return transformer.eval()


class _Kandinsky6LikePipe:
  """Minimal stand-in for `Kandinsky6TI2VAPipeline`: a transformer plus `__call__` defaults."""

  def __init__(self, transformer, guidance_scale: float = 5.0):
    self.transformer = transformer

    def __call__(self, guidance_scale: float = guidance_scale):
      return None

    self.__call__ = __call__


def _inputs(batch_size: int = 1, seed: int = 1, with_audio: bool = True):
  generator = torch.Generator().manual_seed(seed)
  inputs = dict(
    hidden_states=torch.randn(batch_size,
                              _FRAMES,
                              _LATENT_H,
                              _LATENT_W,
                              _VISUAL_IN,
                              generator=generator),
    encoder_hidden_states=torch.randn(batch_size, _TEXT_LEN, 8, generator=generator),
    pooled_projections=torch.randn(batch_size, 8, generator=generator),
    timestep=torch.tensor([0.5] * batch_size),
  )
  if with_audio:
    inputs["audio_hidden_states"] = torch.randn(batch_size,
                                                _AUDIO_LEN,
                                                _AUDIO_IN,
                                                generator=generator)
  return inputs


def test_kandinsky6_prefix_wins_over_the_kandinsky5_adapter():
  prefixes = BlockAdapterRegister.supported_pipelines()[1]
  assert "Kandinsky6" in prefixes
  matches = [name for name in prefixes if "Kandinsky6TI2VAPipeline".startswith(name)]
  assert matches and matches[0] == "Kandinsky6", f"wrong adapter for Kandinsky 6.0: {matches}"
  # The Kandinsky 5 adapter must never swallow a 6.0 pipeline.
  assert not "Kandinsky6TI2VAPipeline".startswith("Kandinsky5")


@pytest.mark.parametrize(("guidance_scale", "expected"), [(5.0, True), (1.0, False)])
def test_has_separate_cfg_follows_pipeline_default(guidance_scale, expected):
  from cache_dit.caching.block_adapters import kandinsky6_adapter

  pipe = _Kandinsky6LikePipe(_tiny_transformer(), guidance_scale=guidance_scale)
  adapter = kandinsky6_adapter(pipe)
  assert adapter.has_separate_cfg is expected
  assert adapter.blocks is pipe.transformer.visual_transformer_blocks
  assert adapter.forward_pattern is ForwardPattern.Pattern_0


def test_vsr_pipeline_is_rejected():
  from cache_dit.caching.block_adapters import kandinsky6_adapter

  # The VSR pipeline drives `Kandinsky6SRTransformer3DModel`, a text-free SR transformer with its
  # own pipeline; adapting it with this plan would silently cache the wrong stack.
  pipe = types.SimpleNamespace(transformer=torch.nn.Linear(2, 2))
  with pytest.raises(NotImplementedError):
    kandinsky6_adapter(pipe)


def test_cache_run_is_bit_exact_when_no_step_is_cached():
  from cache_dit.caching.block_adapters import kandinsky6_adapter

  reference = _tiny_transformer()
  inputs = _inputs()
  with torch.no_grad():
    video, audio = reference(**inputs, return_dict=False)

  transformer = _tiny_transformer()
  adapter = kandinsky6_adapter(_Kandinsky6LikePipe(transformer))
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=1,
      max_warmup_steps=0,
      max_cached_steps=0,
      num_inference_steps=4,
      enable_separate_cfg=True,
      residual_diff_threshold=0.0,
    ),
  )
  signature = str(inspect.signature(transformer.forward))
  with torch.no_grad():
    cached_video, cached_audio = transformer(**inputs, return_dict=False)

  assert str(inspect.signature(transformer.forward)) == signature
  assert torch.equal(cached_video, video), "the cached run must reproduce the video velocity"
  assert torch.equal(cached_audio, audio), "the cached run must reproduce the audio velocity"


def test_cp_planner_splits_the_video_stream_and_gathers_on_the_last_block():
  from cache_dit.distributed.config import ParallelismConfig
  from cache_dit.distributed.transformers.kandinsky6 import (
    Kandinsky6ContextParallelismPlanner, )

  transformer = _tiny_transformer(num_visual_blocks=3)
  plan = Kandinsky6ContextParallelismPlanner()._apply(
    transformer=transformer,
    parallelism_config=ParallelismConfig(ulysses_size=2),
  )

  # The video stream is split once, its rope on every block, and only the last block gathers.
  assert "vis" in plan["visual_transformer_blocks.0"]
  assert "vis_rope" in plan["visual_transformer_blocks.*"]
  last = plan["visual_transformer_blocks.2"]
  assert last[1] is None, "the audio stream is replicated and must not be gathered"
  assert last[0].gather_dim == 1

  # Exactly one attention per fused block stays distributed, and exactly one rebuilds its KV.
  for block in transformer.visual_transformer_blocks:
    assert block.video_dec_block.self_attention._cp_self_distributed is True
    assert block.av_cross_attention._cp_gather_kv is True
    assert not hasattr(block.video_dec_block.cross_attention, "_cp_gather_kv")

  with pytest.raises(NotImplementedError):
    Kandinsky6ContextParallelismPlanner()._apply(
      transformer=_tiny_transformer(),
      parallelism_config=ParallelismConfig(ring_size=2),
    )


def test_cp_gather_hook_tolerates_a_none_stream():
  from cache_dit.distributed.core._context_parallel import _ContextParallelGatherHook

  # A video-only run returns `(vis, None)`; the plan carries `None` in the audio slot, so the hook
  # must pass that slot through instead of raising on the container check.
  hook = _ContextParallelGatherHook([None, None], parallel_config=None)
  video = torch.zeros(1, 4, 8)
  out = hook.post_forward(None, (video, None))
  assert out[0] is video
  assert out[1] is None


def test_cp_gather_hook_rejects_a_none_stream_the_plan_expected():
  from cache_dit.distributed.core._context_parallel import _ContextParallelGatherHook
  from cache_dit.distributed.core import _ContextParallelOutput

  # The mirror case: the plan asks for a gather but the module returned nothing. Silently skipping
  # would hand a sharded tensor to the caller, so the hook has to complain.
  hook = _ContextParallelGatherHook([None, _ContextParallelOutput(gather_dim=1)],
                                    parallel_config=None)
  with pytest.raises(ValueError):
    hook.post_forward(None, (torch.zeros(1, 4, 8), None))


def test_cp_needs_at_least_one_tail_block():
  from cache_dit.caching.cache_interface import check_min_bn_blocks
  from cache_dit.distributed.config import ParallelismConfig
  from cache_dit.distributed.transformers.kandinsky6 import (
    Kandinsky6ContextParallelismPlanner, )

  transformer = _tiny_transformer(num_visual_blocks=3)
  cp_config = ParallelismConfig(ulysses_size=2)
  Kandinsky6ContextParallelismPlanner()._apply(
    transformer=transformer,
    parallelism_config=cp_config,
  )
  # The planner advertises the tail-block requirement, `enable_cache` enforces it: the gather lives
  # on the last block, which the cached path only runs inside the Bn group.
  assert transformer._cp_requires_min_bn_blocks == 1
  with pytest.raises(ValueError):
    check_min_bn_blocks([transformer], DBCacheConfig(Bn_compute_blocks=0), cp_config)
  check_min_bn_blocks([transformer], DBCacheConfig(Bn_compute_blocks=1), cp_config)
