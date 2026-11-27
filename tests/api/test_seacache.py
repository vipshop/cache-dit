import gc
import math
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

import cache_dit
from cache_dit import BlockAdapter, CacheType, DBCacheConfig, ForwardPattern
from cache_dit.caching.cache_contexts import CachedContextManager
from cache_dit.caching.cache_contexts.seacache_uitls import (
  apply_sea_filter,
  build_seacache_indicator,
  resolve_seacache_ab,
)
from cache_dit.cli.utils import get_args, maybe_postprocess_args
from utils import RandPipeline


def _numpy_sea_reference(tensor, a, b, feature_shape):
  """Independent NumPy implementation of the official separable SEA equation."""

  value = tensor.numpy().astype(np.float32).reshape(tensor.shape[0], *feature_shape,
                                                    tensor.shape[-1])
  spectrum = np.fft.fftn(value, axes=tuple(range(1, value.ndim - 1)))
  gain = None
  for axis in range(1, value.ndim - 1):
    frequencies = np.fft.fftfreq(value.shape[axis]).astype(np.float32)
    clean_power = 1.0 / (np.abs(frequencies) ** 2.0 + 1e-16)
    axis_gain = a * clean_power / (a * a * clean_power + b * b + 1e-16)
    reshape = [1] * value.ndim
    reshape[axis] = len(frequencies)
    axis_gain = axis_gain.reshape(reshape)
    gain = axis_gain if gain is None else gain * axis_gain
  gain = gain / gain.mean()
  return np.fft.ifftn(spectrum * gain, axes=tuple(range(1,
                                                        value.ndim - 1))).real.reshape(tensor.shape)


def _manager(config):
  manager = CachedContextManager(persistent_context=True)
  manager.set_context(manager.new_context(name="default", cache_config=config))
  return manager


def _sea_config(**kwargs):
  values = dict(
    max_warmup_steps=0,
    enable_seacache=True,
    seacache_feature_shape=(2, 2),
    seacache_ab_schedule=((0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4)),
  )
  values.update(kwargs)
  return DBCacheConfig(**values)


def test_filter_matches_independent_numpy_for_rectangular_packed_tokens():
  """The packed 2D filter agrees with an independent NumPy implementation."""

  torch.manual_seed(0)
  tensor = torch.randn(2, 6, 3)
  actual = apply_sea_filter(tensor, 0.7, 0.3, feature_shape=(2, 3))
  expected = _numpy_sea_reference(tensor, 0.7, 0.3, (2, 3))
  np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-5, atol=2e-6)


def test_dense_and_packed_image_video_filters_are_equivalent():
  """Dense channels-first and packed token layouts describe the same N-D filter."""

  torch.manual_seed(1)
  dense = torch.randn(1, 3, 2, 3)
  packed = dense.movedim(1, -1).reshape(1, 6, 3)
  dense_filtered = apply_sea_filter(dense, 0.8, 0.2)
  packed_filtered = apply_sea_filter(packed, 0.8, 0.2, feature_shape=(2, 3))
  assert torch.allclose(dense_filtered.movedim(1, -1).reshape(1, 6, 3), packed_filtered)

  video = torch.randn(1, 2, 2, 2, 3)
  packed_video = video.movedim(1, -1).reshape(1, 12, 2)
  assert torch.allclose(
    apply_sea_filter(video, 0.6, 0.4, power_exp=3.0).movedim(1, -1).reshape(1, 12, 2),
    apply_sea_filter(packed_video, 0.6, 0.4, feature_shape=(2, 2, 3), power_exp=3.0),
  )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_filter_restores_dtype_and_shape(dtype):
  """SEA filtering uses fp32 internally and returns the original tensor contract."""

  tensor = torch.randn(1, 2, 3, 4, dtype=dtype)
  filtered = apply_sea_filter(tensor, 0.7, 0.3)
  assert filtered.shape == tensor.shape
  assert filtered.dtype == dtype
  assert torch.isfinite(filtered).all()


def test_flow_vp_and_explicit_schedule_resolution():
  """Generic flow, VP, and custom schedules resolve the expected AB pairs."""

  flow = SimpleNamespace(sigmas=torch.tensor([0.9, 0.5, 0.1]))
  assert resolve_seacache_ab(flow, 1, "flow") == pytest.approx((0.5, 0.5))
  vp = SimpleNamespace(
    alphas_cumprod=torch.tensor([0.64, 0.25, 0.09]),
    timesteps=torch.tensor([2, 0]),
  )
  assert resolve_seacache_ab(vp, 0, "vp") == pytest.approx((0.3, math.sqrt(0.91)))
  assert resolve_seacache_ab(vp, 1, "vp") == pytest.approx((0.8, 0.6))
  assert resolve_seacache_ab(index=1, ab_schedule=[(0.9, 0.1), (0.7, 0.3)]) == pytest.approx(
    (0.7, 0.3))


@pytest.mark.parametrize("call", [
  lambda: resolve_seacache_ab(index=0),
  lambda: resolve_seacache_ab(SimpleNamespace(sigmas=[]), 0, "flow"),
  lambda: resolve_seacache_ab(index=1, ab_schedule=[(0.9, 0.1)]),
  lambda: resolve_seacache_ab(index=0, ab_schedule=[(0.0, 0.1)]),
  lambda: apply_sea_filter(torch.randn(1, 4, 2), 0.9, 0.1),
  lambda: apply_sea_filter(torch.randn(1, 5, 2), 0.9, 0.1, feature_shape=(2, 2)),
  lambda: apply_sea_filter(torch.randn(1, 2, 2, 2), 0.9, 0.1, feature_shape=(3, 2)),
  lambda: build_seacache_indicator(torch.randn(1, 2, 3), 0.9, 0.1, feature_shape=(2, 2)),
])
def test_missing_or_inconsistent_metadata_fails_loudly(call):
  """Missing schedules and incomplete/ambiguous geometry never fall back to raw L1."""

  with pytest.raises(ValueError):
    call()


def test_default_dbcache_config_and_summary_are_unchanged():
  """SeaCache defaults to off and does not create a new CacheType or artifact suffix."""

  config = DBCacheConfig()
  assert config.cache_type == CacheType.DBCache
  assert config.enable_seacache is False
  assert config.seacache_feature_shape is None
  assert "_SEA" not in config.strify()
  enabled = _sea_config()
  assert enabled.cache_type == CacheType.DBCache
  assert "_SEAflow_P2.0_G2x2_ABS" in enabled.strify()


def test_default_similarity_patch_point_is_not_used_by_seacache_dispatch():
  """Default DBCache still calls the original patchable similarity, SeaCache uses its own entry."""

  default_manager = _manager(DBCacheConfig(max_warmup_steps=0))
  default_manager.similarity = mock.Mock(wraps=default_manager.similarity)
  default_manager.seacache_similarity = mock.Mock()
  default_manager.mark_step_begin()
  default_manager.set_Fn_buffer(torch.ones(1, 4, 2))
  default_manager.mark_step_begin()
  assert default_manager.can_cache(torch.ones(1, 4, 2), threshold=0.5)
  assert default_manager.similarity.call_count == 1
  assert default_manager.seacache_similarity.call_count == 0
  assert torch.equal(default_manager.get_Fn_buffer(), torch.ones(1, 4, 2))

  sea_manager = _manager(_sea_config())
  sea_manager.seacache_similarity = mock.Mock(wraps=sea_manager.seacache_similarity)
  sea_manager.mark_step_begin()
  sea_manager.set_Fn_buffer(torch.ones(1, 4, 2))
  sea_manager.mark_step_begin()
  assert sea_manager.can_cache(torch.ones(1, 4, 2), threshold=0.5)
  assert sea_manager.seacache_similarity.call_count == 1


def test_anchor_and_current_step_use_their_own_ab_coefficients():
  """The previous buffer uses its write-time AB and the current probe uses the current AB."""

  manager = _manager(_sea_config())
  with mock.patch(
      "cache_dit.caching.cache_contexts.cache_manager.build_seacache_indicator",
      wraps=build_seacache_indicator,
  ) as build:
    first = torch.randn(1, 4, 2)
    manager.mark_step_begin()
    manager.set_Fn_buffer(first)
    expected = apply_sea_filter(first, 0.9, 0.1, feature_shape=(2, 2))
    assert torch.equal(manager.get_seacache_Fn_buffer(), expected)
    assert torch.equal(manager.get_Fn_buffer(), first)
    manager.mark_step_begin()
    manager.can_cache(torch.randn(1, 4, 2), threshold=0.5)
  assert [call.args[1:3] for call in build.call_args_list] == [(0.9, 0.1), (0.8, 0.2)]


def test_guards_skip_filtering_and_hit_discards_temporary_indicator():
  """Existing early guards avoid FFT, and a hit never advances the filtered anchor."""

  torch.manual_seed(0)
  manager = _manager(_sea_config(max_cached_steps=0, seacache_ab_schedule=((0.9, 0.1), ) * 4))
  first = torch.randn(1, 4, 2)
  manager.mark_step_begin()
  manager.set_Fn_buffer(first)
  anchor = manager.get_seacache_Fn_buffer().clone()
  with mock.patch.object(manager, "_build_seacache_indicator", side_effect=AssertionError):
    manager.mark_step_begin()
    assert manager.can_cache(torch.randn(1, 4, 2), threshold=0.5) is False
  manager.current_context.cache_config.max_cached_steps = -1
  manager.mark_step_begin()
  assert manager.can_cache(first, threshold=0.5) is True
  assert torch.equal(manager.get_seacache_Fn_buffer(), anchor)
  assert torch.equal(manager.get_Fn_buffer(), first)
  assert not any(key.startswith("__seacache_current") for key in manager.current_context.buffers)


def test_fn_apply_cache_never_reads_filtered_indicator_as_value():
  """Third-party Fn-prefix apply_cache still reads the raw residual, not the SEA indicator."""

  manager = _manager(_sea_config())
  residual = torch.randn(1, 4, 2)
  manager.mark_step_begin()
  manager.set_Fn_buffer(residual)
  current = torch.randn(1, 4, 2)
  output, _ = manager.apply_cache(current, prefix="Fn")
  assert torch.equal(output, residual + current)
  assert torch.equal(manager.get_Fn_buffer(), residual)
  assert not torch.equal(manager.get_seacache_Fn_buffer(), residual)


def _run_pipeline(cache_config=None, scheduler=None, calibrator_config=None, steps=8):
  gc.collect()
  torch.manual_seed(0)
  pipe = RandPipeline(pattern=ForwardPattern.Pattern_0)
  if scheduler is not None:
    pipe.scheduler = scheduler
  transformer = pipe.transformer
  adapter = None
  if cache_config is not None:
    adapter = cache_dit.enable_cache(
      BlockAdapter(
        pipe=pipe,
        transformer=transformer,
        blocks=transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_0,
      ),
      cache_config=cache_config,
      calibrator_config=calibrator_config,
    )
  inputs = (torch.randn(1, 64, 16), torch.randn(1, 64, 16))
  output = pipe(*inputs, num_inference_steps=steps)
  cached_steps = list(getattr(transformer, "_cached_steps", []))
  manager = getattr(pipe, "_context_manager", None)
  if adapter is not None:
    cache_dit.disable_cache(adapter)
  del pipe, adapter
  gc.collect()
  return output, cached_steps, manager


def _pipeline_sea_config(**kwargs):
  values = dict(
    Fn_compute_blocks=8,
    Bn_compute_blocks=0,
    max_warmup_steps=1,
    max_continuous_cached_steps=1,
    residual_diff_threshold=0.99,
    num_inference_steps=8,
    enable_seacache=True,
    seacache_feature_shape=(8, 8),
    seacache_ab_schedule=tuple((0.9 - i * 0.01, 0.1 + i * 0.01) for i in range(8)),
  )
  values.update(kwargs)
  return DBCacheConfig(**values)


def test_real_pipeline_seacache_hits_and_calibrator_composition():
  """The real DBCache path executes SEA gating while TaylorSeer still controls cached values."""

  _, cached_steps, _ = _run_pipeline(_pipeline_sea_config())
  assert sum(cached_steps) > 0
  from cache_dit import TaylorSeerCalibratorConfig

  _, calibrated_steps, _ = _run_pipeline(
    _pipeline_sea_config(),
    calibrator_config=TaylorSeerCalibratorConfig(),
  )
  assert sum(calibrated_steps) > 0

  pipe_config = _pipeline_sea_config()
  pipe_config.seacache_ab_schedule = None
  scheduler = SimpleNamespace(sigmas=torch.linspace(0.99, 0.01, 9))
  _, scheduler_steps, manager = _run_pipeline(pipe_config, scheduler=scheduler)
  assert sum(scheduler_steps) > 0
  assert manager._seacache_scheduler is None
  assert manager._seacache_parallelism_config is None


def test_repeated_inference_resets_seacache_state():
  """Two identical requests on one pipeline produce the same output and fresh per-request state."""

  gc.collect()
  torch.manual_seed(0)
  pipe = RandPipeline(pattern=ForwardPattern.Pattern_0)
  transformer = pipe.transformer
  adapter = cache_dit.enable_cache(
    BlockAdapter(
      pipe=pipe,
      transformer=transformer,
      blocks=transformer.transformer_blocks,
      forward_pattern=ForwardPattern.Pattern_0,
    ),
    cache_config=_pipeline_sea_config(),
  )
  inputs = (torch.randn(1, 64, 16), torch.randn(1, 64, 16))
  pipe(*inputs, num_inference_steps=8)
  first_cached_steps = list(getattr(transformer, "_cached_steps", []))
  pipe(*inputs, num_inference_steps=8)
  second_cached_steps = list(getattr(transformer, "_cached_steps", []))
  assert sum(first_cached_steps) > 0
  assert second_cached_steps == first_cached_steps
  manager = pipe._context_manager
  assert all(not key.startswith("__seacache_current")
             for context in manager._cached_context_manager.values() for key in context.buffers)
  cache_dit.disable_cache(adapter)


def test_seacache_zero_cache_is_bit_exact():
  """Disabling cache steps reproduces the uncached pipeline even with SeaCache enabled."""

  base, _, _ = _run_pipeline()
  cached, cached_steps, _ = _run_pipeline(
    _pipeline_sea_config(max_cached_steps=0, residual_diff_threshold=0.0))
  assert sum(cached_steps) == 0
  assert torch.equal(base.sample, cached.sample)


def test_transformer_only_infers_request_length_from_ab_schedule():
  """Persistent transformer-only SeaCache resets per request instead of exceeding the schedule."""

  pipe = RandPipeline(pattern=ForwardPattern.Pattern_0)
  transformer = pipe.transformer
  adapter = cache_dit.enable_cache(
    BlockAdapter(
      transformer=transformer,
      blocks=transformer.transformer_blocks,
      forward_pattern=ForwardPattern.Pattern_0,
    ),
    cache_config=_pipeline_sea_config(num_inference_steps=None),
  )
  manager = transformer._context_manager
  context = next(iter(manager._cached_context_manager.values()))
  assert context.cache_config.num_inference_steps == 8
  cache_dit.disable_cache(adapter)


def test_cli_seacache_validation_and_options():
  """The generic CLI selects SeaCache and SoftCache with the --cache-params type."""

  parser = get_args(parse=False)
  args = maybe_postprocess_args(
    parser.parse_args([
      "--cache-params",
      "type=SeaCache,seacache_mode=flow,seacache_feature_shape=64x64,seacache_power_exp=2.0"
    ]))
  assert args.seacache is True
  assert args.softcache is False
  assert args.seacache_feature_shape == [64, 64]

  args = maybe_postprocess_args(parser.parse_args(["--cache-params", "type=SoftCache"]))
  assert (args.cache, args.seacache, args.softcache) == (True, True, True)
  assert args.softcache_tau == 0.5

  with pytest.raises(ValueError, match="requires a SeaCache"):
    maybe_postprocess_args(parser.parse_args(["--cache-params", "seacache_feature_shape=64x64"]))
  with pytest.raises(ValueError, match="positive"):
    maybe_postprocess_args(
      parser.parse_args(["--cache-params", "type=SeaCache,seacache_feature_shape=0x64"]))
  with pytest.raises(ValueError, match="requires SeaCache"):
    maybe_postprocess_args(parser.parse_args(["--cache-params", "softcache=true"]))
  with pytest.raises(ValueError, match="seacache_mode=flow"):
    maybe_postprocess_args(parser.parse_args(["--cache-params", "type=SoftCache,seacache_mode=vp"]))
  with pytest.raises(ValueError, match="must be positive"):
    maybe_postprocess_args(parser.parse_args(["--cache-params", "type=SoftCache,softcache_tau=0"]))
