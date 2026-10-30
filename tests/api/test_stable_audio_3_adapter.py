"""Regression tests for the Stable Audio 3 cache integration.

Stable Audio 3 is an audio-only diffusers pipeline whose distilled checkpoints run a single velocity
forward per step (`guidance_scale` defaults to 1.0, CFG is baked into the weights). These CPU tests
lock down the properties the integration relies on:

1. `"StableAudio3"` is registered ahead of the Stable Audio 1/2 `"StableAudio"` prefix, so a
   `StableAudio3*` pipeline resolves to its own adapter instead of failing the older one's assertion.
2. The adapter points at `transformer.transformer_blocks` with `ForwardPattern.Pattern_3`, and
   `has_separate_cfg` follows the pipeline's reachable `guidance_scale` default.
3. Collapsing the block list into the cached wrapper — which replays the keyword-only block arguments —
   is bit-exact against the unpatched transformer when no step is served from cache.
4. The CP plan splits the sequence after the memory tokens are prepended, splits the RoPE table on the
   same token axis, gathers on the last block, and advertises the `Bn_compute_blocks` requirement.
5. The TP row permutation is exact (it only moves output rows between ranks) and fails fast on a
   non-divisible fused projection.

The wav writer and the sampling-rate resolution are covered here as well, since both are pure CPU
logic and the audio path had no rate source for this model.
"""
import argparse
import types

import numpy as np
import pytest
import torch

from cache_dit._utils.registers import (
  ExampleOutputData,
  ExampleType,
  _resolve_audio_sampling_rate,
  parse_audio_output,
)
from cache_dit.caching.block_adapters import (
  BlockAdapterRegister,
  stable_audio_3_adapter,
)
from cache_dit.caching.forward_pattern import ForwardPattern
from cache_dit.distributed.config import ParallelismConfig
from cache_dit.distributed.transformers.stable_audio_3 import (
  StableAudio3ContextParallelismPlanner,
  _permute_fused_rows,
)

diffusers = pytest.importorskip("diffusers")

if not hasattr(diffusers, "StableAudio3DiTModel"):
  pytest.skip("Stable Audio 3 requires a newer diffusers", allow_module_level=True)

_IO_CHANNELS = 4
_EMBED_DIM = 32
_NUM_HEADS = 2
_TEXT_DIM = 16
_GLOBAL_DIM = 16
_MEMORY_TOKENS = 4
_FRAMES = 6
_TEXT_LEN = 5


def _tiny_transformer():
  from diffusers import StableAudio3DiTModel

  torch.manual_seed(0)
  transformer = StableAudio3DiTModel(
    io_channels=_IO_CHANNELS,
    patch_size=1,
    embed_dim=_EMBED_DIM,
    depth=3,
    num_heads=_NUM_HEADS,
    cond_token_dim=_TEXT_DIM,
    global_cond_dim=_GLOBAL_DIM,
    local_add_cond_dim=8,
    timestep_features_dim=16,
    ff_mult=2.0,
    num_memory_tokens=_MEMORY_TOKENS,
    use_differential_attention=True,
  ).eval()
  # `preprocess_conv` / `postprocess_conv` and the inpaint projection are zero-initialized (they learn
  # a zero delta), which would make every cache comparison trivially exact.
  with torch.no_grad():
    transformer.preprocess_conv.weight.normal_()
    transformer.postprocess_conv.weight.normal_()
    for block in transformer.transformer_blocks:
      block.to_local_embed[-1].weight.normal_()
      block.to_local_embed[-1].bias.normal_()
  return transformer


def _inputs(seed: int = 0):
  generator = torch.Generator().manual_seed(seed)
  return dict(
    hidden_states=torch.randn(1, _IO_CHANNELS, _FRAMES, generator=generator),
    timestep=torch.tensor([0.4]),
    encoder_hidden_states=torch.randn(1, _TEXT_LEN, _TEXT_DIM, generator=generator),
    global_hidden_states=torch.randn(1, _GLOBAL_DIM, generator=generator),
    encoder_attention_mask=torch.ones(1, _TEXT_LEN, dtype=torch.bool),
    return_dict=False,
  )


class _StableAudio3LikePipe:
  """Minimal stand-in for `StableAudio3Pipeline` (one or two forwards per step)."""

  def __init__(self, transformer):
    self.transformer = transformer

  def __call__(self, hidden_states, guidance_scale: float = 1.0):
    velocity = self.transformer(hidden_states, **self.extra)[0]
    if guidance_scale > 1.0:
      velocity = velocity * guidance_scale
    return velocity


def test_adapter_is_registered_for_the_dit():
  transformer = _tiny_transformer()
  adapter = stable_audio_3_adapter(types.SimpleNamespace(transformer=transformer))

  assert adapter.blocks is transformer.transformer_blocks
  assert adapter.forward_pattern is ForwardPattern.Pattern_3
  assert adapter.check_forward_pattern is True

  resolved = BlockAdapterRegister.get_adapter(transformer)
  assert resolved.blocks is transformer.transformer_blocks


def test_stable_audio_3_prefix_is_registered_before_stable_audio():
  """Adapter lookup is a first-hit prefix scan, so `StableAudio` must not shadow Stable Audio 3."""

  names = list(BlockAdapterRegister._adapters)
  assert "StableAudio3" in names and "StableAudio" in names
  assert names.index("StableAudio3") < names.index("StableAudio")


@pytest.mark.parametrize("guidance_scale", [1.0, 2.5])
def test_has_separate_cfg_is_off_because_guidance_is_fused_into_one_forward(guidance_scale):
  """SA3 concatenates the latents and runs a single transformer forward even with guidance on."""

  pipe = _StableAudio3LikePipe(_tiny_transformer())

  def __call__(self, hidden_states, guidance_scale: float = guidance_scale):
    return None

  pipe.__call__ = __call__.__get__(pipe, type(pipe))
  assert stable_audio_3_adapter(pipe).has_separate_cfg is False


def test_has_separate_cfg_stays_off_without_a_call_default():
  pipe = types.SimpleNamespace(transformer=_tiny_transformer())
  assert stable_audio_3_adapter(pipe).has_separate_cfg is False


def test_example_rejects_parallel_vae():
  """`AutoencoderSAME` has no VAE parallel planner; the example must say so instead of crashing."""

  from cache_dit._utils.examples import stable_audio_3_example

  args = argparse.Namespace(
    parallel_vae=True,
    model_path=None,
    prompt=None,
    negative_prompt=None,
    num_inference_steps=None,
    height=None,
    width=None,
    num_frames=None,
    extra_input_kwargs=None,
    seed=None,
    skip_negative_prompt=False,
  )
  with pytest.raises(NotImplementedError, match="parallel-vae"):
    stable_audio_3_example(args)


def test_cp_requires_a_tail_block_when_caching_is_enabled():
  """The gather lives on the last block, which the cached path only runs inside the Bn group."""

  from cache_dit import DBCacheConfig
  from cache_dit.caching.cache_interface import check_min_bn_blocks

  transformer = _tiny_transformer()
  StableAudio3ContextParallelismPlanner()._apply(
    transformer=transformer,
    parallelism_config=ParallelismConfig(ulysses_size=2),
  )
  with pytest.raises(ValueError, match="Bn_compute_blocks"):
    check_min_bn_blocks([transformer], DBCacheConfig(Bn_compute_blocks=0),
                        ParallelismConfig(ulysses_size=2))
  # `Bn_compute_blocks=1` satisfies it.
  check_min_bn_blocks([transformer], DBCacheConfig(Bn_compute_blocks=1),
                      ParallelismConfig(ulysses_size=2))


def test_cached_blocks_are_bit_exact_when_no_step_is_served_from_cache():
  """The collapsed loop must reproduce the unpatched transformer output exactly (`rdt = 0`)."""

  from cache_dit import DBCacheConfig, enable_cache

  inputs = _inputs()
  reference = _tiny_transformer()
  with torch.no_grad():
    expected = reference(**inputs)[0]

  transformer = _tiny_transformer()
  adapter = stable_audio_3_adapter(_StableAudio3LikePipe(transformer))
  enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      num_inference_steps=4,
      residual_diff_threshold=0.0,
      max_warmup_steps=0,
      enable_separate_cfg=False,
    ),
  )
  with torch.no_grad():
    actual = transformer(**inputs)[0]

  assert torch.equal(expected, actual), (expected - actual).abs().max().item()


def test_cp_plan_splits_after_the_memory_tokens_and_gathers_on_the_last_block():
  transformer = _tiny_transformer()
  planner = StableAudio3ContextParallelismPlanner()
  plan = planner._apply(
    transformer=transformer,
    parallelism_config=ParallelismConfig(ulysses_size=2),
  )

  # The sequence is split on block 0's input, i.e. `[memory | audio]`, and gathered on the last block.
  split = plan["transformer_blocks.0"]["hidden_states"]
  assert (split.split_dim, split.expected_dims, split.split_output) == (1, 3, False)
  gather = plan[f"transformer_blocks.{len(transformer.transformer_blocks) - 1}"]
  assert (gather.gather_dim, gather.expected_dims) == (1, 3)
  # The full-length RoPE table is cut on the token axis by the same plan entry.
  rope = plan["transformer_blocks.*"]["rope"]
  assert (rope.split_dim, rope.expected_dims) == (0, 2)
  # The cached path only runs the tail blocks inside the Bn group.
  assert transformer._cp_requires_min_bn_blocks == 1


def test_cp_rejects_inpainting_local_conditioning():
  transformer = _tiny_transformer()
  StableAudio3ContextParallelismPlanner()._apply(
    transformer=transformer,
    parallelism_config=ParallelismConfig(ulysses_size=2),
  )
  inputs = _inputs()
  inputs["local_add_cond"] = torch.zeros(1, 8, _FRAMES)
  with pytest.raises(NotImplementedError, match="inpainting"):
    transformer(**inputs)


@pytest.mark.parametrize("ring_size", [2, 1])
def test_cp_only_supports_ulysses(ring_size):
  """Ring attention is unverified for this plan, so it is rejected instead of silently drifting."""

  if ring_size == 1:
    # A pure Ulysses mesh is accepted.
    plan = StableAudio3ContextParallelismPlanner()._apply(
      transformer=_tiny_transformer(),
      parallelism_config=ParallelismConfig(ulysses_size=2, ring_size=0),
    )
    assert "transformer_blocks.0" in plan
    return
  with pytest.raises(NotImplementedError, match="only supports Ulysses"):
    StableAudio3ContextParallelismPlanner()._apply(
      transformer=_tiny_transformer(),
      parallelism_config=ParallelismConfig(ulysses_size=2, ring_size=ring_size),
    )


def test_cp_rejects_heads_indivisible_by_the_sequence_parallel_size():
  # 2 heads, tp_size 1: a sequence parallel size of 4 cannot shard the heads with Ulysses.
  with pytest.raises(ValueError, match="not divisible"):
    StableAudio3ContextParallelismPlanner()._apply(
      transformer=_tiny_transformer(),
      parallelism_config=ParallelismConfig(ulysses_size=4),
    )


@pytest.mark.parametrize("sections", [2, 3, 5])
def test_tp_row_permutation_keeps_every_rank_s_section_slice(sections):
  """A uniform shard of the permuted weight must yield `[section_0_part_r | section_1_part_r |
  ...]`."""

  torch.manual_seed(0)
  out_features, in_features = sections * 8, 8
  module = torch.nn.Linear(in_features, out_features, bias=True)
  original_weight = module.weight.detach().clone()
  original_bias = module.bias.detach().clone()

  _permute_fused_rows(module, sections, 2, "test")

  rows_per_rank = out_features // 2
  rows_per_section = rows_per_rank // sections
  expected_sections = original_weight.chunk(sections, dim=0)
  expected_bias = original_bias.chunk(sections, dim=0)
  for rank in range(2):
    local = module.weight.detach()[rank * rows_per_rank:(rank + 1) * rows_per_rank]
    local_bias = module.bias.detach()[rank * rows_per_rank:(rank + 1) * rows_per_rank]
    for section, (expected, expected_b) in enumerate(zip(expected_sections, expected_bias)):
      start = section * rows_per_section
      stop = start + rows_per_section
      rank_start = rank * rows_per_section
      rank_stop = rank_start + rows_per_section
      assert torch.equal(local[start:stop], expected[rank_start:rank_stop])
      assert torch.equal(local_bias[start:stop], expected_b[rank_start:rank_stop])


def test_tp_row_permutation_is_a_no_op_for_a_single_rank():
  torch.manual_seed(0)
  module = torch.nn.Linear(8, 5 * 8, bias=False)
  original = module.weight.detach().clone()
  _permute_fused_rows(module, 5, 1, "test")
  assert torch.equal(module.weight.detach(), original)


def test_tp_row_permutation_fails_fast_on_a_non_divisible_projection():
  module = torch.nn.Linear(8, 5 * 8, bias=False)
  with pytest.raises(ValueError, match="not divisible"):
    _permute_fused_rows(module, 5, 3, "test")


def test_layerwise_offload_never_streams_a_forward_less_state_holder():
  """`SoftNormBottleneck.decode` is reached without `__call__`, so streaming it would strand its
  params."""

  from cache_dit.offload.layerwise import (
    _default_module_filter,
    _iter_unstreamable_state_holders,
  )

  class _Holder(torch.nn.Module):

    def __init__(self):
      super().__init__()
      self.scale = torch.nn.Parameter(torch.ones(2))
      self.register_buffer("running_std", torch.ones(1))

    def decode(self, x):
      return x * self.running_std

  class _Stack(torch.nn.Module):

    def __init__(self):
      super().__init__()
      self.holder = _Holder()
      self.proj = torch.nn.Linear(4, 4)

    def forward(self, x):
      return self.proj(x) + self.holder.decode(x)

  stack = _Stack()
  assert _default_module_filter("holder", stack.holder) is False
  assert _default_module_filter("proj", stack.proj) is True

  holders = _iter_unstreamable_state_holders(stack, resolved_targets=[("proj", stack.proj)])
  assert [name for name, _ in holders] == ["holder"]


def test_wav_writer_transposes_channels_and_samples(tmp_path):
  output = ExampleOutputData(
    model_tag="stable_audio_3",
    strify_tag="t2a",
    audio=np.zeros((2, 32), dtype=np.float32),
    audio_sampling_rate=44100,
  )
  path = tmp_path / "out.wav"
  output.save(argparse.Namespace(save_path=str(path), saved_fps=8))

  import soundfile as sf

  data, rate = sf.read(str(path))
  assert rate == 44100
  assert data.shape == (32, 2)


def test_parse_audio_output_reads_the_rate_from_the_audio_vae():
  audios = torch.zeros(1, 2, 16)

  class _Vae:
    config = types.SimpleNamespace(sampling_rate=44100)

  pipe = types.SimpleNamespace(vae=_Vae())
  parsed = parse_audio_output({"audios": audios}, ExampleType.T2A, pipe=pipe)
  assert parsed["audio"].shape == (2, 16)
  assert parsed["audio_sampling_rate"] == 44100


def test_audio_sampling_rate_ignores_image_vaes():

  class _Vae:
    config = types.SimpleNamespace(block_out_channels=(32, 64))

  assert _resolve_audio_sampling_rate(types.SimpleNamespace(vae=_Vae())) is None
  assert _resolve_audio_sampling_rate(types.SimpleNamespace()) is None
