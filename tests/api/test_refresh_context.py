import gc
import pytest
import torch
import cache_dit
from cache_dit import ForwardPattern, BlockAdapter, DBCacheConfig, DBPruneConfig
from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager
from cache_dit.caching.cache_contexts.prune_manager import PrunedContextManager
from cache_dit.platforms import current_platform
from utils import RandPipeline

DEVICES = (["cpu"] if not current_platform.is_accelerator_available() else
           ["cpu", current_platform.device_type])
PATTERNS = [
  ForwardPattern.Pattern_0,
  ForwardPattern.Pattern_1,
  ForwardPattern.Pattern_2,
  ForwardPattern.Pattern_3,
  ForwardPattern.Pattern_4,
  ForwardPattern.Pattern_5,
]

DTYPES = ([torch.float32]
          if not current_platform.is_accelerator_available() else [torch.float32, torch.bfloat16])


class CountingPruneBlock(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.calls = 0

  def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
    self.calls += 1
    return hidden_states + 1, encoder_hidden_states


class CountingPruneTransformer(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.transformer_blocks = torch.nn.ModuleList([CountingPruneBlock() for _ in range(4)])

  def forward(self, hidden_states, encoder_hidden_states=None):
    for block in self.transformer_blocks:
      hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states)
    return hidden_states


@pytest.mark.parametrize("enable_separate_cfg,cfg_compute_first", [
  (False, False),
  (True, False),
  (True, True),
])
@pytest.mark.parametrize("limit,expected", [
  (-1, [0, 3, 3, 3, 3, 3, 3]),
  (0, [0, 0, 0, 0, 0, 0, 0]),
  (1, [0, 3, 0, 3, 0, 3, 0]),
  (2, [0, 3, 3, 0, 3, 3, 0]),
])
def test_prune_cache_limit_counts_timesteps_not_blocks(
  enable_separate_cfg,
  cfg_compute_first,
  limit,
  expected,
):
  transformer = CountingPruneTransformer()
  blocks = list(transformer.transformer_blocks)
  cache_dit.enable_cache(
    BlockAdapter(
      transformer=transformer,
      blocks=transformer.transformer_blocks,
      forward_pattern=ForwardPattern.Pattern_0,
    ),
    cache_config=DBPruneConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=1,
      max_continuous_cached_steps=limit,
      residual_diff_threshold=0.06,
      enable_separate_cfg=enable_separate_cfg,
      cfg_compute_first=cfg_compute_first,
      cfg_diff_compute_separate=not cfg_compute_first,
    ),
  )
  if enable_separate_cfg:
    expected = [count for count in expected for _ in range(2)]
  pruned_blocks = []
  for _ in expected:
    previous_calls = [block.calls for block in blocks]
    output = transformer(torch.ones(1, 2, 2))
    assert torch.equal(output, torch.full((1, 2, 2), 5.0))
    pruned_blocks.append(sum(block.calls == calls for block, calls in zip(blocks, previous_calls)))
  assert pruned_blocks == expected


@pytest.mark.parametrize("enable_separate_cfg,cfg_compute_first", [
  (False, False),
  (True, False),
  (True, True),
])
def test_prune_rechecks_block_diff_after_recording_timestep(
  enable_separate_cfg,
  cfg_compute_first,
):
  manager = PrunedContextManager()
  manager.set_context(
    manager.new_context(cache_config=DBPruneConfig(
      max_warmup_steps=1,
      max_continuous_cached_steps=1,
      residual_diff_threshold=0.06,
      enable_separate_cfg=enable_separate_cfg,
      cfg_compute_first=cfg_compute_first,
      cfg_diff_compute_separate=not cfg_compute_first,
    )))
  forwards_per_step = 2 if enable_separate_cfg else 1
  states = torch.ones(1)
  for step in range(3):
    for _ in range(forwards_per_step):
      manager.mark_step_begin()
      if step == 0:
        manager.set_Fn_buffer(states, prefix="first_Fn")
        manager.set_Fn_buffer(states, prefix="second_Fn")
      elif step == 1:
        assert manager.can_prune(states, prefix="first_Fn")
        manager.add_pruned_step()
        assert not manager.can_prune(states * 2, prefix="second_Fn")
        assert manager.can_prune(states, prefix="second_Fn")
        manager.add_pruned_step()
      else:
        assert not manager.can_prune(states, prefix="first_Fn")
        assert not manager.can_prune(states, prefix="second_Fn")
  assert manager.get_cached_steps() == [1]
  assert manager.get_cfg_cached_steps() == ([1] if enable_separate_cfg else [])


@pytest.mark.parametrize(
  "enable_separate_cfg,cfg_compute_first",
  [
    (False, False),
    (True, False),
    (True, True),
  ],
)
@pytest.mark.parametrize(
  "cache_kwargs,expected",
  [
    pytest.param({}, list(range(4, 19, 2)), id="cap"),
    pytest.param(
      {
        "max_warmup_steps": 8,
        "warmup_interval": 2
      },
      list(range(1, 19, 2)),
      id="interleaved-warmup",
    ),
    pytest.param(
      {
        "max_warmup_steps": 0,
        "steps_computation_mask": [1, 0] * 9 + [1]
      },
      list(range(1, 19, 2)),
      id="dynamic-mask",
    ),
  ],
)
def test_continuous_cache_limit_restarts_after_compute_step(
  enable_separate_cfg,
  cfg_compute_first,
  cache_kwargs,
  expected,
):
  config = DBCacheConfig(
    max_warmup_steps=4,
    max_continuous_cached_steps=1,
    residual_diff_threshold=0.06,
    enable_separate_cfg=enable_separate_cfg,
    cfg_compute_first=cfg_compute_first,
    cfg_diff_compute_separate=not cfg_compute_first,
  )
  config.update(**cache_kwargs)
  manager = CachedContextManager()
  context = manager.new_context(cache_config=config)
  manager.set_context(context)
  cached_steps = {False: [], True: []}

  num_forwards = 19 * (2 if enable_separate_cfg else 1)
  for _ in range(num_forwards):
    manager.mark_step_begin()
    residual = manager.get_Fn_buffer()
    if residual is None:
      residual = torch.ones(1)

    if manager.can_cache(residual):
      manager.add_cached_step()
      cached_steps[manager.is_separate_cfg_step()].append(manager.get_current_step())
    else:
      manager.set_Fn_buffer(residual)

  assert cached_steps[False] == expected
  assert cached_steps[True] == (expected if enable_separate_cfg else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_refresh_context(device, pattern, dtype):
  gc.collect()
  pipe = RandPipeline(pattern=pattern)  # type: RandPipeline
  transformer = pipe.transformer

  transformer = pipe.transformer
  adapter = cache_dit.enable_cache(
    BlockAdapter(
      transformer=transformer,
      blocks=transformer.transformer_blocks,
      forward_pattern=pipe.pattern,
    ),
    cache_config=DBCacheConfig(
      Fn_compute_blocks=8,
      Bn_compute_blocks=0,
      residual_diff_threshold=0.05,
    ),
  )

  # Transformer only API
  bs, seq_len, headdim = 1, 1024, 64

  hidden_states = torch.normal(
    mean=100.0,
    std=20.0,
    size=(bs, seq_len, headdim),
    dtype=dtype,
  )

  encoder_hidden_states = None
  if pattern in [
      ForwardPattern.Pattern_0,
      ForwardPattern.Pattern_1,
      ForwardPattern.Pattern_2,
  ]:
    encoder_hidden_states = torch.normal(
      mean=100.0,
      std=20.0,
      size=(bs, seq_len, headdim),
      dtype=dtype,
    )

  if device == current_platform.device_type:
    pipe.to(device)
    hidden_states = hidden_states.to(device)
    if encoder_hidden_states is not None:
      encoder_hidden_states = encoder_hidden_states.to(device)

  STEPS = [16, 28, 50]
  if pattern in [
      ForwardPattern.Pattern_0,
      ForwardPattern.Pattern_1,
      ForwardPattern.Pattern_2,
  ]:
    for i, steps in enumerate(STEPS):
      # Refresh cache context
      if i == 0:
        # Test num_inference_steps only case
        cache_dit.refresh_context(
          transformer,
          num_inference_steps=steps,
          verbose=True,
        )
      else:
        cache_dit.refresh_context(
          transformer,
          cache_config=DBCacheConfig(
            Fn_compute_blocks=1,
            Bn_compute_blocks=0,
            residual_diff_threshold=0.08,
            num_inference_steps=steps,
            steps_computation_mask=cache_dit.steps_mask(
              mask_policy="fast",
              total_steps=steps,
            ),
            steps_computation_policy="dynamic",
            enable_separate_cfg=False,
          ),
          verbose=True,
        )
      _ = pipe(
        hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        num_inference_steps=steps,
      )
  else:
    for i, steps in enumerate(STEPS):
      if i == 0:
        # Test num_inference_steps only case
        cache_dit.refresh_context(
          transformer,
          num_inference_steps=steps,
          verbose=True,
        )
      else:
        # Refresh cache context
        cache_dit.refresh_context(
          transformer,
          cache_config=DBCacheConfig(
            Fn_compute_blocks=1,
            Bn_compute_blocks=0,
            residual_diff_threshold=0.08,
            num_inference_steps=steps,
            steps_computation_mask=cache_dit.steps_mask(
              mask_policy="fast",
              total_steps=steps,
            ),
            steps_computation_policy="dynamic",
            enable_separate_cfg=False,
          ),
          verbose=True,
        )
      _ = pipe(
        hidden_states,
        num_inference_steps=steps,
      )

  cache_dit.summary(transformer)
  # We have to disable cache before deleting the pipe and adapter
  # using block adapter instance due to the fake pipe we used in
  # transformer only API.
  cache_dit.disable_cache(adapter)

  del pipe
  del adapter
  del hidden_states
  if encoder_hidden_states is not None:
    del encoder_hidden_states
  gc.collect()
