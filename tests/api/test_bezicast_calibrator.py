import gc
import math

import numpy as np
import pytest
import torch

import cache_dit
from cache_dit import (
  BeziCastCalibratorConfig,
  BlockAdapter,
  DBCacheConfig,
  ForwardPattern,
)
from cache_dit.caching.cache_contexts.calibrators import BeziCastCalibrator
from cache_dit.caching.cache_contexts.calibrators.bezicast import (
  BeziCastState,
  bernstein_basis,
  project_simplex,
)
from utils import RandPipeline


def _tensor(value):
  return torch.tensor([[[float(value)]]], dtype=torch.float64)


def _feed_state(state, values):
  for value in values:
    state.mark_step_begin()
    state.update(_tensor(value))
  state.mark_step_begin()


def test_n2_zero_ridge_matches_linear_extrapolation():
  """K=n=2, steps 0/1, query 2 gives hand-derived weights [-1,2] and value 3."""
  state = BeziCastState(history=2, control_points=2, ridge=0.0, u_max=10.0)
  _feed_state(state, (1.0, 2.0))
  assert state.approximate().item() == pytest.approx(3.0, abs=1e-9)
  assert state.num_forecasts == 1
  assert state.num_query_clips == 0


def test_n4_zero_ridge_reproduces_cubic():
  """Four exact cubic anchors extrapolate 0,1,8,27 to 64 by polynomial interpolation."""
  state = BeziCastState(history=4, control_points=4, ridge=0.0)
  _feed_state(state, (0.0, 1.0, 8.0, 27.0))
  assert state.approximate().item() == pytest.approx(64.0, abs=1e-9)


def test_nonuniform_ridge_weights_match_independent_numpy():
  """A direct NumPy implementation independently verifies non-uniform equivalent weights."""
  state = BeziCastState(history=3, control_points=3, ridge=0.25, u_max=10.0)
  steps = (0, 2, 3)
  values = (1.0, 4.0, 9.0)
  state.current_step = -1
  for step, value in zip(steps, values):
    state.current_step = step
    state.update(_tensor(value))
  state.current_step = 5
  span = float(steps[-1] - steps[0]) + state.epsilon
  coordinates = np.array([(step - steps[0]) / span for step in steps])
  query = (5.0 - steps[0]) / span
  n = 3
  design = np.array([[math.comb(n - 1, j) * u ** j * (1.0 - u) ** (n - 1 - j) for j in range(n)]
                     for u in coordinates])
  basis = np.array(
    [math.comb(n - 1, j) * query ** j * (1.0 - query) ** (n - 1 - j) for j in range(n)])
  weights = basis @ np.linalg.solve(design.T @ design + 0.25 * np.eye(n), design.T)
  expected = float(weights @ np.array(values))
  assert state.approximate().item() == pytest.approx(expected, abs=1e-9)


def test_ridge_zero_and_point_five_differ():
  """Zero and positive Tikhonov regularization are distinct fitting modes."""
  outputs = []
  for ridge in (0.0, 0.5):
    state = BeziCastState(history=2, control_points=2, ridge=ridge, u_max=10.0)
    _feed_state(state, (1.0, 4.0))
    outputs.append(state.approximate().item())
  assert outputs[0] == pytest.approx(7.0, abs=1e-9)
  assert outputs[1] != pytest.approx(outputs[0], abs=1e-3)


def test_bernstein_basis_is_partition_of_unity_inside_window():
  """The Bernstein basis sums to one inside the normalized historical window."""
  basis = bernstein_basis(0.37, 4)
  assert basis.dtype == torch.float64
  assert float(basis.sum()) == pytest.approx(1.0, abs=1e-12)


def test_simplex_projection_known_values():
  """Euclidean projection maps [-1,2] to [0,1] and preserves a valid simplex point."""
  projected = project_simplex(torch.tensor([-1.0, 2.0], dtype=torch.float64))
  assert torch.allclose(projected, torch.tensor([0.0, 1.0], dtype=torch.float64))
  valid = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
  assert torch.allclose(project_simplex(valid), valid)
  negative = project_simplex(torch.tensor([-1.0, -2.0], dtype=torch.float64))
  assert torch.allclose(negative, torch.tensor([1.0, 0.0], dtype=torch.float64))
  assert torch.allclose(project_simplex(torch.tensor([0.3], dtype=torch.float64)),
                        torch.tensor([1.0], dtype=torch.float64))


def test_guardrail_off_and_on_have_hand_computed_outputs():
  """For raw weights [-1,2], risk=2, safe weights=[0,1], and beta=0.5 gives 2.5."""
  raw_state = BeziCastState(history=2, control_points=2, ridge=0.0, u_max=10.0, guardrail=False)
  guarded = BeziCastState(
    history=2,
    control_points=2,
    ridge=0.0,
    u_max=10.0,
    guardrail=True,
    rho=0.0,
    gamma=1.0,
    beta_max=0.5,
  )
  _feed_state(raw_state, (1.0, 2.0))
  _feed_state(guarded, (1.0, 2.0))
  assert raw_state.approximate().item() == pytest.approx(3.0, abs=1e-9)
  assert guarded.approximate().item() == pytest.approx(2.5, abs=1e-9)
  assert guarded.last_risk == pytest.approx(2.0, abs=1e-8)
  assert guarded.last_beta == pytest.approx(0.5, abs=1e-8)
  assert guarded.num_guardrail_calls == 1


def test_auto_rho_uses_window_edge_risk():
  """Rho=None reproduces rho=0 for exact two-anchor linear interpolation at u=1."""
  automatic = BeziCastState(history=2,
                            control_points=2,
                            ridge=0.0,
                            u_max=10.0,
                            guardrail=True,
                            beta_max=0.5)
  explicit = BeziCastState(
    history=2,
    control_points=2,
    ridge=0.0,
    u_max=10.0,
    guardrail=True,
    rho=0.0,
    beta_max=0.5,
  )
  _feed_state(automatic, (1.0, 2.0))
  _feed_state(explicit, (1.0, 2.0))
  assert torch.equal(automatic.approximate(), explicit.approximate())


def test_insufficient_anchors_explicitly_reuse_latest():
  """K<n cannot trigger full compute in CalibratorBase, so it is counted as reuse."""
  state = BeziCastState()
  _feed_state(state, (1.0, 2.0, 4.0))
  assert torch.equal(state.approximate(), _tensor(4.0))
  assert state.num_warmup_reuses == 1
  assert state.num_forecasts == 0


def test_query_cap_is_applied_and_counted():
  """The next linear query u=2 is clipped to 1.5, producing 2.5 instead of 3."""
  state = BeziCastState(history=2, control_points=2, ridge=0.0)
  _feed_state(state, (1.0, 2.0))
  assert state.approximate().item() == pytest.approx(2.500000015, abs=1e-8)
  assert state.num_query_clips == 1


def test_numerical_failure_falls_back_and_is_counted(monkeypatch):
  """A real linear-algebra failure returns the latest real anchor and increments diagnostics."""

  def fail_pinv(*args, **kwargs):
    raise RuntimeError("directed singular fit")

  monkeypatch.setattr(torch.linalg, "pinv", fail_pinv)
  state = BeziCastState(history=2, control_points=2, ridge=0.0, u_max=10.0)
  _feed_state(state, (1.0, 2.0))
  assert torch.equal(state.approximate(), _tensor(2.0))
  assert state.num_numerical_fallbacks == 1
  assert state.num_forecasts == 0


def test_nonfinite_weights_fall_back_and_tensor_oom_propagates(monkeypatch):
  """Non-finite CPU weights use the guarded fallback, while tensor OOM is never swallowed."""
  state = BeziCastState(history=2, control_points=2, ridge=0.0, u_max=10.0)
  _feed_state(state, (1.0, 2.0))
  monkeypatch.setattr(state, "_weights_for_query",
                      lambda design, u: torch.tensor([float("nan"), float("nan")]))
  assert torch.equal(state.approximate(), _tensor(2.0))
  assert state.num_numerical_fallbacks == 1

  state = BeziCastState(history=2, control_points=2, ridge=0.0, u_max=10.0)
  _feed_state(state, (1.0, 2.0))

  def fail_aggregate(weights):
    raise torch.OutOfMemoryError("directed tensor allocation failure")

  monkeypatch.setattr(state, "_aggregate", fail_aggregate)
  with pytest.raises(torch.OutOfMemoryError):
    state.approximate()
  assert state.num_numerical_fallbacks == 0


def test_approximate_before_update_or_at_anchor_raises():
  """Empty and already-computed targets fail loudly instead of inventing values."""
  with pytest.raises(RuntimeError, match="before any update"):
    BeziCastState().approximate()
  state = BeziCastState()
  state.mark_step_begin()
  state.update(_tensor(1.0))
  with pytest.raises(RuntimeError, match="must be later"):
    state.approximate()


def test_fifo_and_predictions_do_not_mutate_anchors():
  """Only full-compute updates enter the FIFO, and repeated queries are tensor-idempotent."""
  state = BeziCastState(history=2, control_points=2, ridge=0.0, u_max=10.0)
  _feed_state(state, (1.0, 2.0, 4.0))
  assert [step for step, _ in state.anchors] == [1, 2]
  before = list(state.anchors)
  assert torch.equal(state.approximate(), state.approximate())
  assert [step for step, _ in state.anchors] == [step for step, _ in before]


def test_multi_stream_independent_and_reset():
  """Named streams keep independent anchors and reset without losing hyper-parameters."""
  cal = BeziCastCalibrator(history=4, control_points=2, ridge=0.25, guardrail=True)
  cal.mark_step_begin()
  cal.update(torch.randn(1, 4, 8), name="Bn")
  cal.update(torch.randn(1, 4, 8) * 10.0, name="Bn_encoder")
  cal.mark_step_begin()
  assert not torch.allclose(cal.approximate(name="Bn"), cal.approximate(name="Bn_encoder"))
  cal.reset_cache()
  for state in cal.states.values():
    assert len(state.anchors) == 0
    assert state.current_step == -1
    assert state.num_forecasts == 0
    assert (state.history, state.control_points, state.ridge) == (4, 2, 0.25)


def test_shape_and_dtype_change_clear_fifo_only():
  """A changed tensor contract clears incompatible anchors without resetting logical time."""
  state = BeziCastState()
  _feed_state(state, (1.0, 2.0, 4.0, 9.0))
  current_step = state.current_step
  state.update(torch.zeros(1, 2, 3, dtype=torch.bfloat16))
  assert len(state.anchors) == 1
  assert state.current_step == current_step
  state.mark_step_begin()
  assert state.approximate().dtype == torch.bfloat16


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_dtype_shape_and_normal_numerics(dtype):
  """Low-precision anchors use fp32 aggregation and return to the input dtype."""
  torch.manual_seed(0)
  state = BeziCastState()
  for _ in range(4):
    state.mark_step_begin()
    state.update(torch.randn(2, 5, 8, dtype=dtype))
  state.mark_step_begin()
  pred = state.approximate()
  assert pred.dtype == dtype
  assert pred.shape == (2, 5, 8)
  assert torch.isfinite(pred).all()
  assert state.num_forecasts == 1
  assert state.num_numerical_fallbacks == 0


@pytest.mark.parametrize("field,value", [
  ("history", 1),
  ("history", 7),
  ("control_points", 1),
  ("control_points", 7),
  ("ridge", -0.1),
  ("guardrail", 1),
  ("u_max", 0.5),
  ("beta_max", 1.1),
  ("rho", -0.1),
  ("gamma", 0.0),
])
def test_invalid_params_are_rejected(field, value):
  """Invalid fitting and guardrail parameters fail at construction time."""
  kwargs = {field: value}
  if field == "control_points" and value == 7:
    kwargs["history"] = 6
  with pytest.raises(ValueError):
    BeziCastState(**kwargs)


def _calibrator_info(context_manager) -> dict:
  context = context_manager.get_context()
  info = {}
  for label, pair in (("", context.get_calibrators()), ("cfg_", context.get_cfg_calibrators())):
    calibrator, encoder_calibrator = pair
    states = list(calibrator.states.values()) if calibrator is not None else []
    info[f"{label}calibrator"] = type(calibrator).__name__ if calibrator is not None else None
    info[f"{label}num_states"] = len(states)
    info[f"{label}state_types"] = {type(state).__name__ for state in states}
    info[f"{label}forecasts"] = calibrator.num_forecasts if calibrator is not None else 0
    info[f"{label}warmup_reuses"] = calibrator.num_warmup_reuses if calibrator is not None else 0
    info[f"{label}numerical_fallbacks"] = (calibrator.num_numerical_fallbacks
                                           if calibrator is not None else 0)
    info[
      f"{label}guardrail_calls"] = calibrator.num_guardrail_calls if calibrator is not None else 0
    info[f"{label}mean_beta"] = calibrator.mean_guardrail_beta if calibrator is not None else 0.0
    info[f"{label}encoder_states"] = (len(encoder_calibrator.states)
                                      if encoder_calibrator is not None else 0)
  return info


def _run_pipeline(cache_config, calibrator_config=None, steps: int = 8):
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
    max_continuous_cached_steps=1,
    residual_diff_threshold=0.99,
  )


def test_pipeline_caches_with_actual_bezicast_forecasts():
  """The real DBCache path accumulates enough anchors and executes BeziCast."""
  output, cached_steps, info = _run_pipeline(_cache_config(), BeziCastCalibratorConfig())
  sample = output.sample if hasattr(output, "sample") else output
  assert torch.isfinite(sample).all()
  assert sum(cached_steps) > 0
  assert info["calibrator"] == "BeziCastCalibrator"
  assert info["state_types"] == {"BeziCastState"}
  assert info["forecasts"] > 0
  assert info["warmup_reuses"] > 0
  assert info["numerical_fallbacks"] == 0
  assert info["encoder_states"] >= 1


def test_pipeline_guardrail_engages_on_real_stream():
  """Guardrail-on uses real anchors and has positive, bounded blending on risky forecasts."""
  _, cached_steps, info = _run_pipeline(_cache_config(),
                                        BeziCastCalibratorConfig(bezicast_guardrail=True))
  assert sum(cached_steps) > 0
  assert info["forecasts"] > 0
  assert info["guardrail_calls"] > 0
  assert 0.0 < info["mean_beta"] <= 1.0
  assert info["numerical_fallbacks"] == 0


def test_pipeline_requests_forecast_without_shifting_hits(monkeypatch):
  """BeziCast changes only what is predicted, not the reuse hit sequence."""
  calls = []
  original = BeziCastCalibrator.approximate

  def counting_approximate(self, name="default"):
    calls.append(name)
    return original(self, name)

  monkeypatch.setattr(BeziCastCalibrator, "approximate", counting_approximate)
  _, reuse_steps, _ = _run_pipeline(_cache_config())
  _, bezicast_steps, info = _run_pipeline(_cache_config(), BeziCastCalibratorConfig())
  assert sum(reuse_steps) > 0
  assert bezicast_steps == reuse_steps
  assert any("Bn_residual" in name for name in calls)
  assert info["forecasts"] > 0


def test_pipeline_zero_cache_is_bit_exact():
  """max_cached_steps=0 reproduces the un-cached output bit for bit."""
  base, _, _ = _run_pipeline(None)
  cached, cached_steps, info = _run_pipeline(
    DBCacheConfig(
      Fn_compute_blocks=8,
      Bn_compute_blocks=0,
      max_warmup_steps=1,
      max_continuous_cached_steps=1,
      max_cached_steps=0,
      residual_diff_threshold=0.0,
    ),
    BeziCastCalibratorConfig(),
  )
  assert sum(cached_steps) == 0
  assert info["calibrator"] == "BeziCastCalibrator"
  base_sample = base.sample if hasattr(base, "sample") else base
  cached_sample = cached.sample if hasattr(cached, "sample") else cached
  assert torch.equal(base_sample, cached_sample)
