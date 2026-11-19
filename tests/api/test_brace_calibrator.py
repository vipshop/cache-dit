import gc

import pytest
import torch

import cache_dit
from cache_dit import BlockAdapter, BRACECalibratorConfig, DBCacheConfig, ForwardPattern
from cache_dit.caching.cache_contexts.calibrators import BRACECalibrator
from cache_dit.caching.cache_contexts.calibrators.brace import BRACEState
from utils import RandPipeline


def _tensor(value):
  return torch.tensor([[[float(value)]]], dtype=torch.float64)


def _feed_state(state, values):
  for value in values:
    state.mark_step_begin()
    state.update(_tensor(value))
  state.mark_step_begin()


def test_c3_standard_weights_reproduce_quadratic():
  """C=3, gamma=0.5 has hand-derived coefficients (1, -3, 3) at the next step."""
  state = BRACEState(capacity=3, gamma=0.5)
  _feed_state(state, (1.0, 4.0, 9.0))
  assert state.approximate().item() == pytest.approx(16.0, abs=1e-12)
  assert state.num_fallbacks == 0


def test_c3_default_gamma_matches_hand_computed_value():
  """C=3, gamma=0.7 maps [1,2,3] to 38/11, not exact linear extrapolation."""
  state = BRACEState(capacity=3, gamma=0.7)
  _feed_state(state, (1.0, 2.0, 3.0))
  assert state.approximate().item() == pytest.approx(38.0 / 11.0, abs=1e-12)
  assert state.num_fallbacks == 0


def test_c2_standard_weights_reproduce_linear():
  """C=2, gamma=0.5 gives 2*F_latest-F_previous at the next logical step."""
  state = BRACEState(capacity=2, gamma=0.5)
  _feed_state(state, (1.0, 2.0))
  assert state.approximate().item() == pytest.approx(3.0, abs=1e-12)


def test_nonuniform_anchors_match_hand_computed_value():
  """Steps [0,1,3], target 4 and gamma=0.5 give coefficients [3/7,-8/7,12/7]."""
  state = BRACEState(capacity=3, gamma=0.5)
  state.mark_step_begin()
  state.update(_tensor(1.0))
  state.mark_step_begin()
  state.update(_tensor(2.0))
  state.mark_step_begin()
  state.mark_step_begin()
  state.update(_tensor(4.0))
  state.mark_step_begin()
  assert [step for step, _ in state.anchors] == [0, 1, 3]
  assert state.current_step == 4
  assert state.approximate().item() == pytest.approx(5.0, abs=1e-12)
  assert state.num_fallbacks == 0


def test_partial_window_uses_effective_length():
  """A capacity-3 window with two anchors uses the C=2 terminal gamma weight."""
  partial = BRACEState(capacity=3, gamma=0.7)
  exact_c2 = BRACEState(capacity=2, gamma=0.7)
  _feed_state(partial, (1.0, 2.0))
  _feed_state(exact_c2, (1.0, 2.0))
  assert partial.approximate().item() == pytest.approx(23.0 / 9.0, abs=1e-12)
  assert torch.equal(partial.approximate(), exact_c2.approximate())


def test_capacity_one_reuses_latest_anchor():
  """C=1 is the explicit plain-reuse control."""
  state = BRACEState(capacity=1)
  _feed_state(state, (3.0, 7.0))
  assert torch.equal(state.approximate(), _tensor(7.0))
  assert state.num_fallbacks == 0


def test_zero_denominator_falls_back_and_is_counted():
  """Force an otherwise unsupported gamma to exercise exact denominator cancellation."""
  state = BRACEState(capacity=2, gamma=0.5)
  state.gamma = 0.4
  state.mark_step_begin()
  state.update(_tensor(1.0))
  state.mark_step_begin()
  state.update(_tensor(2.0))
  for _ in range(4):
    state.mark_step_begin()
  assert state.current_step == 5
  assert torch.equal(state.approximate(), _tensor(2.0))
  assert state.num_fallbacks == 1


def test_approximate_at_existing_anchor_raises():
  """Querying an already computed step signals incorrect call ordering."""
  state = BRACEState()
  state.mark_step_begin()
  state.update(_tensor(1.0))
  with pytest.raises(RuntimeError, match="must be later"):
    state.approximate()


def test_approximate_before_update_raises():
  """An empty stream fails loudly instead of inventing an anchor."""
  with pytest.raises(RuntimeError, match="before any update"):
    BRACEState().approximate()


def test_fifo_eviction_and_predictions_stay_out_of_window():
  """Only full-compute values enter the FIFO and its capacity stays bounded."""
  state = BRACEState(capacity=3)
  _feed_state(state, (1.0, 2.0, 3.0))
  state.approximate()
  assert [step for step, _ in state.anchors] == [0, 1, 2]
  state.mark_step_begin()
  state.update(_tensor(4.0))
  assert [step for step, _ in state.anchors] == [1, 2, 4]


def test_approximate_is_idempotent_within_step():
  """Repeated queries neither advance time nor mutate the anchor window."""
  state = BRACEState()
  _feed_state(state, (1.0, 2.0, 4.0))
  before = list(state.anchors)
  assert torch.equal(state.approximate(), state.approximate())
  assert [step for step, _ in state.anchors] == [step for step, _ in before]


def test_multi_stream_independent():
  """Named streams keep separate FIFO windows and forecasts."""
  cal = BRACECalibrator()
  cal.mark_step_begin()
  cal.update(torch.randn(1, 4, 8), name="Bn")
  cal.update(torch.randn(1, 4, 8) * 10.0, name="Bn_encoder")
  cal.mark_step_begin()
  assert not torch.allclose(cal.approximate(name="Bn"), cal.approximate(name="Bn_encoder"))


def test_reset_cache_clears_state():
  """reset_cache clears anchors and counters while retaining hyper-parameters."""
  cal = BRACECalibrator(capacity=2, gamma=0.6)
  cal.mark_step_begin()
  cal.update(torch.randn(1, 4, 8))
  cal.reset_cache()
  state = cal.states["default"]
  assert len(state.anchors) == 0
  assert state.current_step == -1
  assert state.num_fallbacks == 0
  assert (state.capacity, state.gamma) == (2, 0.6)


def test_shape_and_dtype_change_reset_fifo():
  """A changed tensor contract cannot leave incompatible anchors in the window."""
  state = BRACEState()
  _feed_state(state, (1.0, 2.0))
  state.mark_step_begin()
  state.update(torch.zeros(1, 2, 3, dtype=torch.bfloat16))
  assert len(state.anchors) == 1
  state.mark_step_begin()
  pred = state.approximate()
  assert pred.shape == (1, 2, 3)
  assert pred.dtype == torch.bfloat16


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_dtype_shape_and_normal_numerics(dtype):
  """Low-precision inputs use fp32 accumulation and return to the input dtype."""
  torch.manual_seed(0)
  state = BRACEState()
  for _ in range(3):
    state.mark_step_begin()
    state.update(torch.randn(2, 5, 8, dtype=dtype))
  state.mark_step_begin()
  pred = state.approximate()
  assert pred.dtype == dtype
  assert pred.shape == (2, 5, 8)
  assert torch.isfinite(pred).all()
  assert state.num_fallbacks == 0


@pytest.mark.parametrize("capacity,gamma", [(0, 0.7), (4, 0.7), (True, 0.7), (3, 0.0), (3, 0.4),
                                            (3, True), (3, -0.1), (3, float("inf")),
                                            (3, float("nan"))])
def test_invalid_params_are_rejected(capacity, gamma):
  """Invalid values fail at state and calibrator construction time."""
  with pytest.raises(ValueError):
    BRACEState(capacity=capacity, gamma=gamma)
  with pytest.raises(ValueError):
    BRACECalibrator(capacity=capacity, gamma=gamma)


def _calibrator_info(context_manager) -> dict:
  context = context_manager.get_context()
  info = {}
  for label, pair in (("", context.get_calibrators()), ("cfg_", context.get_cfg_calibrators())):
    calibrator, encoder_calibrator = pair
    states = list(calibrator.states.values()) if calibrator is not None else []
    info[f"{label}calibrator"] = type(calibrator).__name__ if calibrator is not None else None
    info[f"{label}num_states"] = len(states)
    info[f"{label}state_types"] = {type(state).__name__ for state in states}
    info[f"{label}capacities"] = {state.capacity for state in states}
    info[f"{label}fallbacks"] = calibrator.num_fallbacks if calibrator is not None else 0
    info[f"{label}encoder_states"] = (len(encoder_calibrator.states)
                                      if encoder_calibrator is not None else 0)
  return info


def _run_pipeline(cache_config, calibrator_config=None, steps: int = 4):
  gc.collect()
  torch.manual_seed(0)
  pipe = RandPipeline(pattern=ForwardPattern.Pattern_0)
  transformer = pipe.transformer
  adapter = None
  if cache_config is not None:
    adapter = cache_dit.enable_cache(
      BlockAdapter(
        transformer=transformer,
        blocks=transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_0,
      ),
      cache_config=cache_config,
      calibrator_config=calibrator_config,
    )
  output = pipe(
    torch.randn(1, 64, 16),
    encoder_hidden_states=torch.randn(1, 64, 16),
    num_inference_steps=steps,
  )
  cached_steps = list(getattr(transformer, "_cached_steps", []))
  info = {}
  if adapter is not None:
    info = _calibrator_info(transformer._context_manager)
    cache_dit.disable_cache(adapter)
  del pipe, adapter
  gc.collect()
  return output, cached_steps, info


def _cache_config():
  return DBCacheConfig(
    Fn_compute_blocks=8,
    Bn_compute_blocks=0,
    max_warmup_steps=1,
    residual_diff_threshold=0.99,
  )


def test_pipeline_caches_with_brace_calibrator():
  """Real cache plumbing executes BRACE without numerical fallback."""
  output, cached_steps, info = _run_pipeline(_cache_config(), BRACECalibratorConfig())
  sample = output.sample if hasattr(output, "sample") else output
  assert torch.isfinite(sample).all()
  assert sum(cached_steps) > 0
  assert info["calibrator"] == "BRACECalibrator"
  assert info["num_states"] >= 1
  assert info["state_types"] == {"BRACEState"}
  assert info["capacities"] == {3}
  assert info["fallbacks"] == 0
  assert info["encoder_states"] >= 1


def test_pipeline_requests_forecast_without_shifting_hits(monkeypatch):
  """A data-independent hit sequence is identical with reuse and BRACE."""
  calls = []
  original = BRACECalibrator.approximate

  def counting_approximate(self, name="default"):
    calls.append(name)
    return original(self, name)

  monkeypatch.setattr(BRACECalibrator, "approximate", counting_approximate)
  _, reuse_steps, _ = _run_pipeline(_cache_config())
  _, brace_steps, info = _run_pipeline(_cache_config(), BRACECalibratorConfig())
  assert sum(reuse_steps) > 0
  assert brace_steps == reuse_steps
  assert any("Bn_residual" in name for name in calls)
  assert info["fallbacks"] == 0


def test_pipeline_capacity_one_is_bit_exact_with_reuse():
  """C=1 must be exactly the calibrator-free residual-reuse path."""
  reuse, reuse_steps, _ = _run_pipeline(_cache_config())
  brace, brace_steps, info = _run_pipeline(_cache_config(), BRACECalibratorConfig(brace_capacity=1))
  reuse_sample = reuse.sample if hasattr(reuse, "sample") else reuse
  brace_sample = brace.sample if hasattr(brace, "sample") else brace
  assert brace_steps == reuse_steps
  assert info["fallbacks"] == 0
  assert torch.equal(reuse_sample, brace_sample)


def test_pipeline_zero_cache_is_bit_exact():
  """max_cached_steps=0 reproduces the un-cached run bit for bit."""
  base, _, _ = _run_pipeline(None)
  cached, cached_steps, info = _run_pipeline(
    DBCacheConfig(
      Fn_compute_blocks=8,
      Bn_compute_blocks=0,
      max_warmup_steps=1,
      max_cached_steps=0,
      residual_diff_threshold=0.0,
    ),
    BRACECalibratorConfig(),
  )
  assert sum(cached_steps) == 0
  assert info["calibrator"] == "BRACECalibrator"
  base_sample = base.sample if hasattr(base, "sample") else base
  cached_sample = cached.sample if hasattr(cached, "sample") else cached
  assert torch.equal(base_sample, cached_sample)
