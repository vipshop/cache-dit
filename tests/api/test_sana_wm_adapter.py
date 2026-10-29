"""Regression tests for the SANA-WM cache integration (stage 1).

`SanaWMTransformer3DModel` runs 20 `SanaVideoMSCamCtrlBlock` layers whose `forward` takes twelve
positional arguments: `(x, y, t, mask, THW, rotary_emb, block_mask, camera_conditions,
ucpe_ray_transforms, plucker_emb, frame_valid_mask, chunk_size)`. Everything except `x` is built
once per forward *outside* the block loop, so block-level caching stays valid — but only if the
cached wrapper passes all eleven extras through unchanged. These CPU tests pin down:

1. The `"SanaWM"` adapter wins over `"Sana"` for `SanaWMTransformer3DModel` (the registry takes the
   first matching class-name prefix) and reports the right block contract.
2. A cached run is bit-exact against the unpatched transformer when nothing is cached.
3. A fully cached step reproduces the computed one — i.e. the positional extras survive the
   collapsed loop.
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

if not hasattr(diffusers, "SanaWMTransformer3DModel"):
  pytest.skip("SANA-WM requires a newer diffusers", allow_module_level=True)

_HIDDEN = 32
_HEADS = 2
_HEAD_DIM = 16
_FRAMES = 2
_LATENT_H = 4
_LATENT_W = 4
_IN_CHANNELS = 4
_TEXT_LEN = 5
_CAPTION_CHANNELS = 8


def _tiny_transformer(num_layers: int = 2):
  from diffusers import SanaWMTransformer3DModel

  torch.manual_seed(0)
  transformer = SanaWMTransformer3DModel(
    in_channels=_IN_CHANNELS,
    num_layers=num_layers,
    hidden_size=_HIDDEN,
    num_attention_heads=_HEADS,
    patch_size=(1, 1, 1),
    # Every second layer swaps the GDN recurrence for softmax attention, so a 2-layer model
    # exercises both attention variants.
    softmax_every_n=2,
    linear_head_dim=_HEAD_DIM,
    caption_channels=_CAPTION_CHANNELS,
  )
  return transformer.eval()


class _SanaWMLikePipe:
  """Minimal stand-in for `SanaWMPipeline`: a transformer plus `__call__` defaults."""

  def __init__(self, transformer, guidance_scale: float = 5.0):
    self.transformer = transformer

    def __call__(self, guidance_scale: float = guidance_scale):
      return None

    self.__call__ = __call__


def _inputs(batch_size: int = 1, seed: int = 1, with_camera: bool = True):
  generator = torch.Generator().manual_seed(seed)
  inputs = dict(
    hidden_states=torch.randn(batch_size,
                              _IN_CHANNELS,
                              _FRAMES,
                              _LATENT_H,
                              _LATENT_W,
                              generator=generator),
    timestep=torch.zeros(batch_size, 1, _FRAMES),
    encoder_hidden_states=torch.randn(batch_size,
                                      1,
                                      _TEXT_LEN,
                                      _CAPTION_CHANNELS,
                                      generator=generator),
    mask=torch.ones(batch_size, _TEXT_LEN, dtype=torch.int16),
    data_info={"img_hw": torch.tensor([[64.0, 64.0]])},
    return_dict=False,
  )
  if with_camera:
    inputs["camera_conditions"] = torch.zeros(batch_size, _FRAMES, 20)
  return inputs


def test_sana_wm_prefix_wins_over_the_sana_adapter():
  prefixes = BlockAdapterRegister.supported_pipelines()[1]
  assert "SanaWM" in prefixes
  matches = [name for name in prefixes if "SanaWMTransformer3DModel".startswith(name)]
  assert matches and matches[0] == "SanaWM", f"wrong adapter for SANA-WM: {matches}"
  # The 2D Sana adapter must never swallow the world model.
  assert "SanaWMTransformer3DModel".startswith("Sana")
  assert not "SanaWMTransformer3DModel".startswith("SanaTransformer")


def test_adapter_reports_the_stage1_block_contract():
  from cache_dit.caching.block_adapters import sana_wm_adapter

  transformer = _tiny_transformer()
  adapter = sana_wm_adapter(_SanaWMLikePipe(transformer))
  assert adapter.blocks is transformer.blocks
  assert adapter.blocks_name == "blocks"
  assert adapter.forward_pattern is ForwardPattern.Pattern_3
  # The block parameters are `x` / `y`, not `hidden_states` / `encoder_hidden_states`.
  assert adapter.check_forward_pattern is False
  # CFG runs inside one forward (`cat([negative, conditional])`), not as two separate passes.
  assert adapter.has_separate_cfg is False


def test_cache_run_is_bit_exact_when_no_step_is_cached():
  reference = _tiny_transformer()
  inputs = _inputs()
  with torch.no_grad():
    expected = reference(**inputs)[0]

  transformer = _tiny_transformer()
  # `get_adapter` builds the `FakeDiffusionPipeline` that owns the cache contexts; enabling the
  # cache on a hand-rolled pipe object silently skips every step instead.
  adapter = BlockAdapterRegister.get_adapter(transformer)
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=1,
      max_warmup_steps=0,
      max_cached_steps=0,
      num_inference_steps=4,
      residual_diff_threshold=0.0,
    ),
  )
  signature = str(inspect.signature(transformer.forward))
  with torch.no_grad():
    cached = transformer(**inputs)[0]

  assert str(inspect.signature(transformer.forward)) == signature
  assert torch.equal(cached, expected), "the cached run must reproduce the velocity"


def test_fully_cached_step_reproduces_the_computed_one():
  # Three layers so that `Fn_compute_blocks=1` + `Bn_compute_blocks=1` leave the middle layer as
  # the only skippable one -- block 0 always runs and cannot prove anything about the cache.
  transformer = _tiny_transformer(num_layers=3)
  adapter = BlockAdapterRegister.get_adapter(transformer)
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=1,
      max_warmup_steps=0,
      max_cached_steps=-1,
      num_inference_steps=4,
      # Identical inputs on the second call -> residual diff is 0, so the middle blocks are skipped.
      residual_diff_threshold=1e9,
    ),
  )

  calls = [0]
  transformer.blocks[1].register_forward_pre_hook(
    lambda module, args, kwargs=None: calls.__setitem__(0, calls[0] + 1))

  inputs = _inputs()
  with torch.no_grad():
    first = transformer(**inputs)[0]
    second = transformer(**inputs)[0]

  assert calls[0] == 1, (f"the middle block ran {calls[0]} times over two identical forwards; "
                         "the second forward was not served from cache")
  assert torch.allclose(first, second, atol=1e-6), (
    "a fully cached step must reproduce the computed one; the positional extras (camera "
    "conditions, UCPE transforms, RoPE) did not survive the collapsed block loop")


def test_context_parallelism_is_refused_with_an_explanation():
  from cache_dit.distributed.config import ParallelismConfig
  from cache_dit.distributed.transformers.sana_wm import SanaWMContextParallelismPlanner

  transformer = _tiny_transformer()
  for kwargs in ({"ulysses_size": 2}, {"ring_size": 2}):
    with pytest.raises(NotImplementedError, match="GDN recurrence"):
      SanaWMContextParallelismPlanner()._apply(
        transformer=transformer,
        parallelism_config=ParallelismConfig(**kwargs),
      )


def test_tp_plan_shards_weights_but_not_activations():
  from torch.distributed._tensor import Replicate
  from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

  from cache_dit.distributed.transformers.sana_wm import sana_wm_block_plan

  transformer = _tiny_transformer()
  plan = sana_wm_block_plan(transformer.blocks[0])

  # Everything the block reshapes with the global channel count keeps its activations replicated:
  # only the weights are sharded. `attn.qkv` is one fused `Linear(C, 3C)`, the Q/K RMSNorm runs
  # over all channels, and the short convolution reshapes back with `channels = x.shape[-1]`.
  for name in ("attn.qkv", "attn.output_gate", "attn.beta_proj", "attn.gate_proj",
               "attn.q_proj_cam", "attn.k_proj_cam", "attn.v_proj_cam", "cross_attn.q_linear",
               "cross_attn.kv_linear"):
    style = plan[name]
    assert isinstance(style, ColwiseParallel), name
    assert isinstance(style.output_layouts[0], Replicate), name

  for name in ("attn.proj", "attn.out_proj_cam", "cross_attn.proj"):
    style = plan[name]
    assert isinstance(style, RowwiseParallel), name
    assert isinstance(style.input_layouts[0], Replicate), name

  # The FFN is Conv2d end to end (`GLUMBConvTemp`), which the TP API cannot shard.
  assert not any(name.startswith("mlp") for name in plan)
  # A block without the post-attention Plücker projection / output gate must not carry those keys:
  # `parallelize_module` resolves every key with `get_submodule` and would fail on a missing one.
  bare = types.SimpleNamespace(attn=types.SimpleNamespace(output_gate=None), plucker_proj=None)
  assert "plucker_proj" not in sana_wm_block_plan(bare)
  assert "attn.output_gate" not in sana_wm_block_plan(bare)


def test_camera_conditioning_can_be_absent():
  transformer = _tiny_transformer()
  adapter = BlockAdapterRegister.get_adapter(transformer)
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=1,
      max_warmup_steps=0,
      max_cached_steps=0,
      num_inference_steps=4,
      residual_diff_threshold=0.0,
    ),
  )

  inputs = _inputs(with_camera=False)
  with torch.no_grad():
    out = transformer(**inputs)[0]
  assert tuple(out.shape) == (1, _IN_CHANNELS, _FRAMES, _LATENT_H, _LATENT_W)
