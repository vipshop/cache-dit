"""Regression tests for the Ideogram 4 cache integration.

`Ideogram4Pipeline` denoises with an asymmetric-CFG transformer pair: the conditional `transformer`
runs the packed [text, image] sequence, the optional `unconditional_transformer` runs image-only
positions with zeroed text features, and the two velocities are blended after the fact. Both
partitions own a `layers` stack with the `Pattern_3` block contract, so the adapter caches them as
two sibling entries (the Wan 2.2 `transformer` / `transformer_2` layout). Each partition is called
exactly once per denoising step, hence `has_separate_cfg=False` and one cache context per branch.

These CPU tests lock the adapter wiring (registration, dual `layers` / `Pattern_3`, `False`
separate-CFG), the missing-unconditional fallback, and the per-branch cache bookkeeping plus
bit-exactness on tiny transformers.
"""
import copy
import inspect

import pytest
import torch

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.caching.block_adapters import BlockAdapter, BlockAdapterRegister
from cache_dit.caching.forward_pattern import ForwardPattern

diffusers = pytest.importorskip("diffusers")

if not hasattr(diffusers, "Ideogram4Transformer2DModel"):
  pytest.skip("Ideogram 4 requires a newer diffusers", allow_module_level=True)

from diffusers.models.transformers.transformer_ideogram4 import (  # noqa: E402
  LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR,
)

IN_CHANNELS = 8
LLM_FEATURES_DIM = 32
TEXT_TOKENS = 2
IMAGE_TOKENS = 4
TOTAL_TOKENS = TEXT_TOKENS + IMAGE_TOKENS


def _tiny_transformer(seed: int = 0):
  from diffusers import Ideogram4Transformer2DModel

  torch.manual_seed(seed)
  return Ideogram4Transformer2DModel(
    in_channels=IN_CHANNELS,
    num_layers=2,
    attention_head_dim=16,
    num_attention_heads=2,
    intermediate_size=32,
    adaln_dim=8,
    llm_features_dim=LLM_FEATURES_DIM,
    mrope_section=(6, 1, 1),
  ).eval()


def _step_inputs(generator, image_only: bool = False):
  """Build one denoise-step input set, mirroring the pipeline's cond / uncond passes.

  The conditional pass feeds the packed [text, image] sequence, the unconditional one the
  image-only slice (`neg_position_ids` / `neg_segment_ids` / `neg_indicator` upstream).
  """
  seq_len = IMAGE_TOKENS if image_only else TOTAL_TOKENS
  indicator = torch.full((1, seq_len), OUTPUT_IMAGE_INDICATOR, dtype=torch.long)
  segment_ids = torch.zeros(1, seq_len, dtype=torch.long)
  position_ids = torch.zeros(1, seq_len, 3, dtype=torch.long)
  if not image_only:
    indicator[:, :TEXT_TOKENS] = LLM_TOKEN_INDICATOR
  position_ids[..., 1] = torch.arange(seq_len)  # h axis
  position_ids[..., 2] = torch.arange(seq_len)  # w axis
  return dict(
    hidden_states=torch.randn(1, seq_len, IN_CHANNELS, generator=generator),
    timestep=torch.tensor([0.5]),
    encoder_hidden_states=torch.randn(1, seq_len, LLM_FEATURES_DIM, generator=generator),
    position_ids=position_ids,
    segment_ids=segment_ids,
    indicator=indicator,
    return_dict=False,
  )


class _Ideogram4LikePipe:
  """Minimal stand-in for `Ideogram4Pipeline`: one call = one denoise step.

  `__call__` runs the conditional pass and then the unconditional one, the way the upstream
  denoising loop does. The cache adapter wraps this `__call__` to enter one cache context per
  partition, so the tests must go through it instead of calling the partitions directly.
  """

  def __init__(self, transformer, unconditional_transformer=None):
    self.transformer = transformer
    self.unconditional_transformer = unconditional_transformer

  def __call__(self, cond_inputs, uncond_inputs, num_steps: int = 1):
    cond_outs, uncond_outs = [], []
    for _ in range(num_steps):
      cond_outs.append(self.transformer(**cond_inputs)[0])
      if self.unconditional_transformer is not None:
        uncond_outs.append(self.unconditional_transformer(**uncond_inputs)[0])
    return cond_outs, uncond_outs


class Ideogram4TestPipeline(diffusers.DiffusionPipeline):
  """Tiny real DiffusionPipeline used to verify pipeline-level lifecycle handling."""

  def __init__(self, transformer, unconditional_transformer):
    super().__init__()
    self.register_modules(
      transformer=transformer,
      unconditional_transformer=unconditional_transformer,
    )

  def __call__(self, cond_inputs, uncond_inputs, num_steps: int = 1):
    for _ in range(num_steps):
      self.transformer(**cond_inputs)
      self.unconditional_transformer(**uncond_inputs)


def test_ideogram4_prefix_is_the_unique_match():
  prefixes = BlockAdapterRegister.supported_pipelines()[1]
  assert "Ideogram4" in prefixes
  matches = [name for name in prefixes if "Ideogram4Pipeline".startswith(name)]
  assert matches == ["Ideogram4"], f"ambiguous adapter prefixes for Ideogram 4: {matches}"


def test_dual_partition_adapter_wiring():
  from cache_dit.caching.block_adapters import ideogram4_adapter

  transformer, unconditional_transformer = _tiny_transformer(), _tiny_transformer(seed=1)
  pipe = _Ideogram4LikePipe(transformer, unconditional_transformer)
  adapter = ideogram4_adapter(pipe)

  assert adapter.transformer == [transformer, unconditional_transformer]
  assert adapter.blocks == [transformer.layers, unconditional_transformer.layers]
  assert adapter.blocks_name == ["layers", "layers"]
  assert adapter.forward_pattern == [ForwardPattern.Pattern_3, ForwardPattern.Pattern_3]
  assert adapter.check_forward_pattern is True
  assert adapter.has_separate_cfg is False
  assert adapter.patch_functor is None
  # two distinct stacks -> two cache contexts
  assert adapter.blocks[0] is not adapter.blocks[1]


def test_missing_unconditional_transformer_degrades_to_single_partition():
  from cache_dit.caching.block_adapters import ideogram4_adapter

  transformer = _tiny_transformer()
  adapter = ideogram4_adapter(_Ideogram4LikePipe(transformer))

  assert adapter.transformer == [transformer]
  assert adapter.blocks == [transformer.layers]
  assert adapter.blocks_name == ["layers"]
  assert adapter.forward_pattern == [ForwardPattern.Pattern_3]
  assert adapter.has_separate_cfg is False


def test_registered_adapter_supports_transformer_only_usage():
  transformer = _tiny_transformer()
  adapter = BlockAdapterRegister.get_adapter(transformer)

  assert adapter is not None
  assert adapter.transformer == [transformer]
  assert adapter.blocks == [transformer.layers]
  assert adapter.has_separate_cfg is False


def test_transformer_only_cache_run_uses_one_context():
  transformer = _tiny_transformer()
  cache_dit.enable_cache(
    transformer,
    cache_config=DBCacheConfig(
      num_inference_steps=3,
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=3,
      residual_diff_threshold=0.05,
    ),
  )

  context_names = transformer._context_names
  assert len(context_names) == 1, "the transformer-only fallback must cache one `layers` stack"

  generator = torch.Generator().manual_seed(3)
  inputs = _step_inputs(generator)
  try:
    with torch.no_grad():
      for _ in range(3):
        out = transformer(**inputs)[0]
    assert out.shape == (1, TOTAL_TOKENS, IN_CHANNELS)
    assert torch.isfinite(out).all()
    context = transformer._context_manager.get_context(context_names[0])
    assert context.transformer_executed_steps == 3
  finally:
    cache_dit.disable_cache(transformer)


def test_dual_branch_cache_lifecycle_counts_each_partition_per_step():
  from cache_dit.caching.block_adapters import ideogram4_adapter

  transformer, unconditional_transformer = _tiny_transformer(), _tiny_transformer(seed=1)
  pipe = _Ideogram4LikePipe(transformer, unconditional_transformer)
  adapter = ideogram4_adapter(pipe)

  cache_config = DBCacheConfig(
    num_inference_steps=4,
    Fn_compute_blocks=1,
    Bn_compute_blocks=0,
    max_warmup_steps=4,
    residual_diff_threshold=0.05,
  )
  cache_dit.enable_cache(adapter, cache_config=cache_config)

  # The adapter's has_separate_cfg=False is what keeps the step counters 1:1 with the
  # denoising steps: separate-CFG bookkeeping would pair the two partitions' forwards.
  assert cache_config.enable_separate_cfg is False

  context_manager = pipe._context_manager
  assert len(adapter.unique_blocks_name) == 2
  cond_context = adapter.unique_blocks_name[0][0]
  uncond_context = adapter.unique_blocks_name[1][0]
  assert cond_context != uncond_context

  generator = torch.Generator().manual_seed(42)
  cond_inputs = _step_inputs(generator)
  uncond_inputs = _step_inputs(generator, image_only=True)
  try:
    with torch.no_grad():
      cond_outs, uncond_outs = pipe(cond_inputs, uncond_inputs, num_steps=4)
    for cond_out, uncond_out in zip(cond_outs, uncond_outs):
      assert cond_out.shape == (1, TOTAL_TOKENS, IN_CHANNELS)
      assert uncond_out.shape == (1, IMAGE_TOKENS, IN_CHANNELS)
      assert torch.isfinite(cond_out).all() and torch.isfinite(uncond_out).all()

    # One forward per partition per step: 4 steps -> 4 transformer forwards on each branch, and
    # (because `enable_separate_cfg` is False) 4 counted steps on each, not 8 on one / 2 on each.
    assert context_manager.get_context(cond_context).transformer_executed_steps == 4
    assert context_manager.get_context(uncond_context).transformer_executed_steps == 4
    assert context_manager.get_context(cond_context).executed_steps == 4
    assert context_manager.get_context(uncond_context).executed_steps == 4
  finally:
    cache_dit.disable_cache(adapter)

  # disable_cache hands the original block stacks back
  assert len(transformer.layers) == 2
  assert len(unconditional_transformer.layers) == 2


def test_cached_run_is_bit_exact_when_no_step_is_cached():
  from cache_dit.caching.block_adapters import ideogram4_adapter

  reference_transformer, reference_unconditional = _tiny_transformer(), _tiny_transformer(seed=1)
  transformer = copy.deepcopy(reference_transformer)
  unconditional_transformer = copy.deepcopy(reference_unconditional)

  pipe = _Ideogram4LikePipe(transformer, unconditional_transformer)
  adapter = ideogram4_adapter(pipe)
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      num_inference_steps=3,
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=0,
      max_cached_steps=0,  # caching disabled: the run must stay bit-exact
      residual_diff_threshold=0.0,
    ),
  )

  # The cached wrapper must keep both partitions' forward signatures discoverable.
  signatures = (str(inspect.signature(transformer.forward)),
                str(inspect.signature(unconditional_transformer.forward)))
  reference_signatures = (str(inspect.signature(reference_transformer.forward)),
                          str(inspect.signature(reference_unconditional.forward)))
  assert signatures == reference_signatures

  context_manager = pipe._context_manager
  generator = torch.Generator().manual_seed(7)
  cond_inputs = _step_inputs(generator)
  uncond_inputs = _step_inputs(generator, image_only=True)
  try:
    with torch.no_grad():
      cond_outs, uncond_outs = pipe(cond_inputs, uncond_inputs, num_steps=3)
      for step, (cond_out, uncond_out) in enumerate(zip(cond_outs, uncond_outs)):
        cond_ref = reference_transformer(**cond_inputs)[0]
        uncond_ref = reference_unconditional(**uncond_inputs)[0]
        torch.testing.assert_close(cond_out,
                                   cond_ref,
                                   atol=0,
                                   rtol=0,
                                   msg=f"conditional branch diverged at step {step}")
        torch.testing.assert_close(uncond_out,
                                   uncond_ref,
                                   atol=0,
                                   rtol=0,
                                   msg=f"unconditional branch diverged at step {step}")

    # The cached wrapper really ran (a missing context would silently bypass it and make the
    # bit-exactness above vacuous).
    for unique_names in adapter.unique_blocks_name:
      assert context_manager.get_context(unique_names[0]).transformer_executed_steps == 3
  finally:
    cache_dit.disable_cache(adapter)


def test_pipeline_disable_and_summary_cover_both_partitions():
  pipe = Ideogram4TestPipeline(_tiny_transformer(), _tiny_transformer(seed=1))
  cache_dit.enable_cache(
    pipe,
    cache_config=DBCacheConfig(
      num_inference_steps=2,
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=2,
      residual_diff_threshold=0.05,
    ),
  )
  generator = torch.Generator().manual_seed(11)
  cond_inputs = _step_inputs(generator)
  uncond_inputs = _step_inputs(generator, image_only=True)
  with torch.no_grad():
    pipe(cond_inputs, uncond_inputs, num_steps=2)

  stats = cache_dit.summary(pipe, logging=False)
  assert len(stats) == 4
  assert all(stat.cache_options for stat in stats)

  cache_dit.disable_cache(pipe)
  assert not BlockAdapter.is_cached(pipe.transformer)
  assert not BlockAdapter.is_cached(pipe.unconditional_transformer)
  assert not hasattr(pipe.transformer, "_context_manager")
  assert not hasattr(pipe.unconditional_transformer, "_context_manager")


def test_dual_branch_cache_actually_hits_after_warmup():
  pipe = Ideogram4TestPipeline(_tiny_transformer(), _tiny_transformer(seed=1))
  cache_dit.enable_cache(
    pipe,
    cache_config=DBCacheConfig(
      num_inference_steps=6,
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=1,
      max_cached_steps=6,
      residual_diff_threshold=0.05,
    ),
  )
  generator = torch.Generator().manual_seed(13)
  cond_inputs = _step_inputs(generator)
  uncond_inputs = _step_inputs(generator, image_only=True)
  try:
    with torch.no_grad():
      pipe(cond_inputs, uncond_inputs, num_steps=6)
    contexts = [
      module._context_manager.get_context(module._context_names[0])
      for module in (pipe.transformer, pipe.unconditional_transformer)
    ]
    assert all(context.get_cached_steps() for context in contexts)
    assert all(context.transformer_executed_steps == 6 for context in contexts)
  finally:
    cache_dit.disable_cache(pipe)


def test_qwen3_vl_model_text_parallel_planner_is_registered():
  from cache_dit.distributed.text_encoders.planners import _activate_text_encoder_tp_planners
  from cache_dit.distributed.text_encoders.qwen3 import Qwen3VLModelTensorParallelismPlanner
  from cache_dit.distributed.text_encoders.register import (
    TextEncoderTensorParallelismPlannerRegister, )

  _activate_text_encoder_tp_planners()
  planner = TextEncoderTensorParallelismPlannerRegister.get_planner("Qwen3VLModel")
  assert planner is Qwen3VLModelTensorParallelismPlanner
