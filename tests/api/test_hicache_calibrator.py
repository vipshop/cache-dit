import gc
import math

import pytest
import torch

import cache_dit
from cache_dit import (
  BlockAdapter,
  DBCacheConfig,
  ForwardPattern,
  HiCacheCalibratorConfig,
)
from cache_dit.caching.cache_contexts.calibrators import HiCacheCalibrator, TaylorSeerCalibrator
from cache_dit.caching.cache_contexts.calibrators.hicache import HiCacheState
from utils import RandPipeline

# H~_k(x) = sigma^k * H_k(sigma * x) with H the physicist Hermite polynomial:
# H_0 = 1, H_1 = 2y, H_2 = 4y^2 - 2, H_3 = 8y^3 - 12y.
# The numbers below are the closed forms, typed in by hand: comparing the
# implementation against its own recurrence would pass even if both were wrong.
_CLOSED_FORM = {
  0.4: {
    1: (0.32, 0.64, 0.96),
    2: (-0.2176, 0.0896, 0.6016),
    3: (-0.274432, -0.352256, -0.036864),
    4: (0.12107776, -0.31145984, -0.61292544),
  },
  0.5: {
    1: (0.5, 1.0, 1.5),
    2: (-0.25, 0.5, 1.75),
    3: (-0.625, -0.5, 1.125),
    4: (0.0625, -1.25, -0.9375),
  },
  1.0: {
    1: (2.0, 4.0, 6.0),
    2: (2.0, 14.0, 34.0),
    3: (-4.0, 40.0, 180.0),
    4: (-20.0, 76.0, 876.0),
  },
}
_CLOSED_FORM_X = (1.0, 2.0, 3.0)


@pytest.mark.parametrize("sigma", sorted(_CLOSED_FORM))
def test_hermite_basis_matches_closed_form(sigma):
  """Every order must match the hand-computed sigma^k * H_k(sigma * x)."""
  state = HiCacheState(n_derivatives=4, sigma=sigma)
  for order, expected in _CLOSED_FORM[sigma].items():
    for x, want in zip(_CLOSED_FORM_X, expected):
      assert state.hermite_basis(order, x) == pytest.approx(want, abs=1e-6)


def test_order0_and_order1_basis():
  """The constant term is 1 and the first order is 2 * sigma^2 * elapsed."""
  state = HiCacheState(sigma=0.5)
  assert state.hermite_basis(0, 7.0) == 1.0
  assert state.hermite_basis(1, 3.0) == pytest.approx(1.5, abs=1e-12)


def test_sigma_one_is_plain_physicist_hermite():
  """With sigma=1 the basis must reproduce the undamped physicist-Hermite polynomial."""
  state = HiCacheState(sigma=1.0)
  for x in (1.0, 2.0, 3.0):
    assert state.hermite_basis(2, x) == pytest.approx(4 * x * x - 2, abs=1e-9)


def _feed(cal, values):
  """Feed anchors and leave the calibrator on the next (cached) step.

  The first `mark_step_begin` runs before any state exists (states are created lazily inside
  `update`), so the i-th anchor lands on step i-2 and the ladder guard `current_step > 1` only
  starts producing a first difference from the FOURTH anchor on (the fifth for a second
  difference). Feeding fewer anchors silently returns the anchor.
  """

  for value in values:
    cal.mark_step_begin()
    cal.update(torch.tensor([[[float(value)]]]))
  cal.mark_step_begin()


def test_first_order_extrapolation_uses_two_sigma_squared_gain():
  """After four anchors a linear ladder must extrapolate by 2 * sigma^2, not by 1."""
  hicache = HiCacheCalibrator(n_derivatives=1, sigma=0.5)
  _feed(hicache, (1.0, 2.0, 3.0, 4.0))
  assert hicache.approximate().item() == pytest.approx(4.0 + 0.5, abs=1e-5)

  taylor = TaylorSeerCalibrator(n_derivatives=1)
  _feed(taylor, (1.0, 2.0, 3.0, 4.0))
  assert taylor.approximate().item() == pytest.approx(5.0, abs=1e-5)


def test_second_order_extrapolation_uses_hermite_weights():
  """A quadratic ladder extrapolates with (H~_1, H~_2 / 2), not with (e, e^2 / 2)."""
  # 1, 4, 9, 16, 25 -> first differences 3, 5, 7, 9 -> second difference 2.
  hicache = HiCacheCalibrator(n_derivatives=2, sigma=0.5)
  _feed(hicache, (1.0, 4.0, 9.0, 16.0, 25.0))
  # `hicache_order` is the highest difference order, so the ladder holds order + 1 entries.
  assert len(hicache.states["default"].state["dY_current"]) == 3
  predicted = hicache.approximate().item()
  assert predicted == pytest.approx(25.0 + 9.0 * 0.5 + (2.0 / 2.0) * -0.25, abs=1e-4)


def test_sigma_inverse_sqrt_two_matches_taylor_at_first_order():
  """With sigma=1/sqrt(2) the order-1 weight equals Taylor's, so order-1 forecasts agree."""
  sigma = 1.0 / math.sqrt(2.0)
  hicache = HiCacheCalibrator(n_derivatives=1, sigma=sigma)
  taylor = TaylorSeerCalibrator(n_derivatives=1)
  values = (0.3, -1.2, 2.5, 0.9)
  _feed(hicache, values)
  _feed(taylor, values)
  # 2 * sigma^2 is 1.0000000000000002 in fp, so allclose rather than equal.
  assert hicache.approximate().allclose(taylor.approximate(), atol=1e-5, rtol=1e-4)
  assert not torch.allclose(hicache.approximate(), torch.tensor([[[0.9]]]), atol=1e-3)


def test_higher_orders_differ_from_taylor():
  """Anti-degradation: a wired HiCache must NOT silently return the TaylorSeer forecast."""
  torch.manual_seed(0)
  values = [torch.randn(2, 4, 8) for _ in range(4)]

  def run(cal):
    for value in values:
      cal.mark_step_begin()
      cal.update(value)
    cal.mark_step_begin()
    return cal.approximate()

  hicache = HiCacheCalibrator(n_derivatives=2, sigma=0.4)
  taylor = TaylorSeerCalibrator(n_derivatives=2)
  assert not torch.equal(run(hicache), run(taylor))


def test_order0_reproduces_the_anchor():
  """With order=0 only the anchor is kept, i.e. plain DBCache-style reuse."""
  torch.manual_seed(0)
  Y = torch.randn(1, 8, 16)
  cal = HiCacheCalibrator(n_derivatives=0, sigma=0.5)
  cal.mark_step_begin()
  cal.update(Y)
  cal.mark_step_begin()
  assert torch.allclose(cal.approximate(), Y, atol=1e-6)


def test_tiny_sigma_collapses_to_the_anchor():
  """As sigma -> 0 every extrapolation term damps away, leaving the anchor even with a live
  ladder."""
  torch.manual_seed(0)
  values = [torch.randn(1, 8, 16) for _ in range(4)]
  cal = HiCacheCalibrator(n_derivatives=2, sigma=1e-6)
  for value in values:
    cal.mark_step_begin()
    cal.update(value)
  cal.mark_step_begin()
  assert torch.allclose(cal.approximate(), values[-1], atol=1e-4)


def test_prediction_moves_along_the_ladder():
  """A non-degenerate sigma must move the forecast away from the anchor, in the ladder direction."""
  hicache = HiCacheCalibrator(n_derivatives=1, sigma=0.5)
  _feed(hicache, (1.0, 2.0, 3.0, 4.0))
  predicted = hicache.approximate()
  anchor = torch.tensor([[[4.0]]])
  assert not torch.allclose(predicted, anchor)
  assert (predicted - anchor).item() > 0


def test_multi_stream_independent():
  """Two named streams keep separate ladders and forecasts."""
  torch.manual_seed(0)
  cal = HiCacheCalibrator(n_derivatives=1, sigma=0.5)
  cal.mark_step_begin()
  cal.update(torch.randn(1, 4, 8), name="Bn")
  cal.update(torch.randn(1, 4, 8) * 10, name="Bn_encoder")
  cal.mark_step_begin()
  assert not torch.allclose(cal.approximate(name="Bn"), cal.approximate(name="Bn_encoder"))


def test_approximate_is_idempotent_within_step():
  """The Hermite forecast is pure, so a second query in the same step returns the same tensor."""
  torch.manual_seed(0)
  cal = HiCacheCalibrator(n_derivatives=2, sigma=0.5)
  cal.mark_step_begin()
  cal.update(torch.randn(1, 4, 8))
  cal.mark_step_begin()
  assert torch.equal(cal.approximate(), cal.approximate())


def test_approximate_before_update_raises():
  """Querying a stream that never ran a full compute step fails loudly."""
  cal = HiCacheCalibrator()
  cal.mark_step_begin()
  with pytest.raises(AssertionError, match="State 'default' not found"):
    cal.approximate()


def test_reset_cache_clears_state():
  """reset_cache() drops the ladder and the counters of every stream."""
  cal = HiCacheCalibrator()
  cal.mark_step_begin()
  cal.update(torch.randn(1, 4, 8), name="Bn")
  cal.reset_cache()
  state = cal.states["Bn"]
  assert state.state["dY_current"][0] is None
  assert state.current_step == -1
  assert state.sigma == 0.5


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_dtype_and_shape_preserved(dtype):
  """The forecast keeps the block tensor's dtype and shape."""
  torch.manual_seed(0)
  Y = torch.randn(2, 5, 8).to(dtype)
  cal = HiCacheCalibrator(n_derivatives=1, sigma=0.5)
  cal.mark_step_begin()
  cal.update(Y)
  cal.mark_step_begin()
  pred = cal.approximate()
  assert pred.dtype == dtype
  assert pred.shape == Y.shape


@pytest.mark.parametrize("sigma,order", [(0.0, 1), (-0.1, 1), (1.5, 1), (0.5, -1), (0.5, 5)])
def test_invalid_params_are_rejected(sigma, order):
  """Bad hyper-parameters fail at construction, not in the middle of a diffusion run."""
  with pytest.raises(ValueError):
    HiCacheCalibrator(n_derivatives=order, sigma=sigma)
  with pytest.raises(ValueError):
    HiCacheState(n_derivatives=order, sigma=sigma)


def _calibrator_info(context_manager) -> dict:
  """Summarize what the calibrators actually did, so a test can prove they ran."""

  context = context_manager.get_context()
  info = {}
  for label, pair in (("", context.get_calibrators()), ("cfg_", context.get_cfg_calibrators())):
    calibrator, encoder_calibrator = pair
    states = list(calibrator.states.values()) if calibrator is not None else []
    info[f"{label}calibrator"] = type(calibrator).__name__ if calibrator is not None else None
    info[f"{label}num_states"] = len(states)
    info[f"{label}state_types"] = {type(state).__name__ for state in states}
    info[f"{label}sigmas"] = {state.sigma for state in states}
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
  hidden_states = torch.randn(1, 64, 16)
  encoder_hidden_states = torch.randn(1, 64, 16)
  output = pipe(
    hidden_states,
    encoder_hidden_states=encoder_hidden_states,
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


def test_pipeline_caches_with_hicache_calibrator():
  """End-to-end through the real cache plumbing: caching happens and HiCache really runs."""
  output, cached_steps, info = _run_pipeline(
    DBCacheConfig(
      Fn_compute_blocks=8,
      Bn_compute_blocks=0,
      max_warmup_steps=1,
      residual_diff_threshold=0.99,
    ),
    HiCacheCalibratorConfig(hicache_order=2, hicache_sigma=0.4),
  )
  sample = output.sample if hasattr(output, "sample") else output
  assert torch.isfinite(sample).all()
  assert sum(cached_steps) > 0, cached_steps
  # Caching alone would also pass the two assertions above, so pin the calibrator itself:
  # a missing `maybe_init_state` override would silently build TaylorSeerState instead.
  assert info["calibrator"] == "HiCacheCalibrator", info
  assert info["num_states"] >= 1, info
  assert info["state_types"] == {"HiCacheState"}, info
  assert info["sigmas"] == {0.4}, info
  assert info["encoder_states"] >= 1, info


def test_pipeline_requests_a_forecast_on_cached_steps(monkeypatch):
  """The cache plumbing must really call `approximate()`, without shifting the hit sequence.

  `RandPipeline` feeds the same hidden states to every step, so the forecast provably equals the
  anchor and the output cannot tell the two runs apart. What a test CAN pin is the wiring: the
  calibrator built by the factory is the one the context manager queries, and a high threshold
  makes the hit sequence data-independent so it must match the calibrator-free run.
  """
  calls = []
  original = HiCacheCalibrator.approximate

  def counting_approximate(self, name="default"):
    calls.append(name)
    return original(self, name)

  monkeypatch.setattr(HiCacheCalibrator, "approximate", counting_approximate)
  cache_config = DBCacheConfig(
    Fn_compute_blocks=8,
    Bn_compute_blocks=0,
    max_warmup_steps=1,
    residual_diff_threshold=0.99,
  )
  _, reuse_steps, _ = _run_pipeline(cache_config)
  _, hicache_steps, info = _run_pipeline(
    cache_config, HiCacheCalibratorConfig(hicache_order=2, hicache_sigma=0.4))
  assert sum(reuse_steps) > 0, reuse_steps
  assert hicache_steps == reuse_steps
  assert info["state_types"] == {"HiCacheState"}, info
  assert calls, info


def test_pipeline_zero_cache_is_bit_exact():
  """max_cached_steps=0 must reproduce the un-cached run bit for bit."""
  base, _, _ = _run_pipeline(None)
  cached, cached_steps, info = _run_pipeline(
    DBCacheConfig(
      Fn_compute_blocks=8,
      Bn_compute_blocks=0,
      max_warmup_steps=1,
      max_cached_steps=0,
      residual_diff_threshold=0.0,
    ),
    HiCacheCalibratorConfig(hicache_order=2, hicache_sigma=0.4),
  )
  assert sum(cached_steps) == 0, cached_steps
  # The calibrator still records every full-compute step (the ladder must stay fresh), but no
  # cached step ever asks it for a forecast, so the run must be identical to the un-cached one.
  assert info["calibrator"] == "HiCacheCalibrator", info
  assert info["num_states"] >= 1, info
  base_sample = base.sample if hasattr(base, "sample") else base
  cached_sample = cached.sample if hasattr(cached, "sample") else cached
  assert torch.equal(base_sample, cached_sample)
