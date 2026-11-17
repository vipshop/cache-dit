import gc

import pytest
import torch

import cache_dit
from cache_dit import (
  BlockAdapter,
  DBCacheConfig,
  ForwardPattern,
  SVDCalibratorConfig,
)
from cache_dit.caching.cache_contexts.calibrators import SVDCalibrator
from cache_dit.caching.cache_contexts.calibrators.svd import SVDState
from utils import RandPipeline


def _rank_of(Y: torch.Tensor, tau: float, max_rank: int = 0) -> int:
  state = SVDState(tau=tau, beta=0.0, max_rank=max_rank)
  state.mark_step_begin()
  state.update(Y)
  return state.basis[0].shape[1]


def test_full_rank_reproduces_anchor():
  """Tau=1 and beta=0 must reproduce the anchor (P(Y) + R == Y)."""
  torch.manual_seed(0)
  Y = torch.randn(1, 8, 16)
  cal = SVDCalibrator(tau=1.0, beta=0.0)
  cal.mark_step_begin()
  cal.update(Y)
  cal.mark_step_begin()
  pred = cal.approximate()
  assert pred.shape == Y.shape
  assert torch.allclose(pred, Y, atol=1e-5, rtol=1e-4)


def test_rank_monotonic_in_tau_and_bounded():
  """A tighter energy target retains at least as many directions, always at least one."""
  torch.manual_seed(0)
  Y = torch.randn(2, 32, 16)
  ranks = [_rank_of(Y, tau) for tau in (0.3, 0.5, 0.9, 0.999)]
  assert all(rank >= 1 for rank in ranks)
  assert ranks == sorted(ranks), ranks
  assert ranks[0] < ranks[-1]


def test_max_rank_caps_the_basis():
  """max_rank truncates the energy-based choice."""
  torch.manual_seed(0)
  Y = torch.randn(2, 32, 16)
  assert _rank_of(Y, 0.999, max_rank=3) == 3
  assert _rank_of(Y, 0.999, max_rank=0) > 3


def test_ema_lag_and_per_step_convergence():
  """The principal stream lags the anchors and converges back to the anchor per cached step."""
  cal = SVDCalibrator(tau=1.0, beta=0.9)
  cal.mark_step_begin()
  cal.update(torch.tensor([[[1.0]]]))
  cal.mark_step_begin()
  cal.update(torch.tensor([[[3.0]]]))
  assert cal.states["default"].ema.item() == pytest.approx(1.2, abs=1e-5)

  cal.mark_step_begin()
  assert cal.approximate().item() == pytest.approx(0.9 * 1.2 + 0.1 * 3.0, abs=1e-5)
  cal.mark_step_begin()
  assert cal.approximate().item() == pytest.approx(0.9 * 1.38 + 0.1 * 3.0, abs=1e-5)


def test_beta_zero_is_plain_reuse():
  """Beta=0 collapses SVD-Cache onto DBCache's plain reuse of the anchor."""
  torch.manual_seed(0)
  Y0 = torch.randn(1, 8, 16)
  Y1 = torch.randn(1, 8, 16)
  cal = SVDCalibrator(tau=0.9, beta=0.0)
  cal.mark_step_begin()
  cal.update(Y0)
  cal.mark_step_begin()
  cal.update(Y1)
  cal.mark_step_begin()
  assert torch.allclose(cal.approximate(), Y1, atol=1e-5, rtol=1e-4)


def test_long_skip_converges_to_anchor():
  """Repeated cached steps stay bounded and converge to the anchor value."""
  torch.manual_seed(0)
  Y0 = torch.randn(1, 8, 16)
  Y1 = torch.randn(1, 8, 16)
  cal = SVDCalibrator(tau=0.9, beta=0.5)
  cal.mark_step_begin()
  cal.update(Y0)
  cal.mark_step_begin()
  cal.update(Y1)
  predictions = []
  for _ in range(8):
    cal.mark_step_begin()
    predictions.append(cal.approximate())
  errors = [(pred - Y1).abs().max().item() for pred in predictions]
  assert errors == sorted(errors, reverse=True), errors
  assert errors[-1] < errors[0]
  assert all(torch.isfinite(pred).all() for pred in predictions)


def test_shape_change_resets_state():
  """A new shape refits the basis instead of reusing a mismatched subspace."""
  cal = SVDCalibrator()
  cal.mark_step_begin()
  cal.update(torch.randn(2, 4, 8), name="Bn")
  cal.mark_step_begin()
  cal.update(torch.randn(2, 6, 8), name="Bn")
  cal.mark_step_begin()
  pred = cal.approximate(name="Bn")
  assert pred.shape == (2, 6, 8)
  V, _ = cal.states["Bn"].basis
  assert V.shape[0] == 8


def test_multi_stream_independent():
  """Two named streams keep separate bases and forecasts."""
  torch.manual_seed(0)
  cal = SVDCalibrator()
  cal.mark_step_begin()
  cal.update(torch.randn(1, 4, 8), name="Bn")
  cal.update(torch.randn(1, 4, 8) * 10, name="Bn_encoder")
  cal.mark_step_begin()
  pred_bn = cal.approximate(name="Bn")
  pred_enc = cal.approximate(name="Bn_encoder")
  assert not torch.allclose(pred_bn, pred_enc)


def test_approximate_is_idempotent_within_step():
  """A second query in the same step returns the same tensor (no extra EMA tick)."""
  torch.manual_seed(0)
  cal = SVDCalibrator(tau=0.9, beta=0.5)
  cal.mark_step_begin()
  cal.update(torch.randn(1, 4, 8))
  cal.mark_step_begin()
  first = cal.approximate()
  second = cal.approximate()
  assert torch.equal(first, second)


def test_approximate_before_update_raises():
  """Querying a stream that never ran a full compute step fails loudly (same as TaylorSeer)."""
  cal = SVDCalibrator()
  cal.mark_step_begin()
  with pytest.raises(AssertionError, match="State 'default' not found"):
    cal.approximate()


def test_reset_cache_clears_state():
  """reset_cache() drops the basis and the counters of every stream."""
  cal = SVDCalibrator()
  cal.mark_step_begin()
  cal.update(torch.randn(1, 4, 8), name="Bn")
  cal.reset_cache()
  state = cal.states["Bn"]
  assert state.basis is None
  assert state.residual is None
  assert state.ema is None
  assert state.current_step == -1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_dtype_and_shape_preserved(dtype):
  """The forecast keeps the block tensor's dtype and shape."""
  torch.manual_seed(0)
  Y = torch.randn(2, 5, 8).to(dtype)
  cal = SVDCalibrator()
  cal.mark_step_begin()
  cal.update(Y)
  cal.mark_step_begin()
  pred = cal.approximate()
  assert pred.dtype == dtype
  assert pred.shape == Y.shape


def test_high_rank_input_is_rejected():
  """4D/5D block tensors would be projected along the wrong axis."""
  cal = SVDCalibrator()
  cal.mark_step_begin()
  with pytest.raises(AssertionError, match="2D or 3D block tensor"):
    cal.update(torch.randn(1, 2, 4, 4, 8))


def test_zero_tensor_stays_finite():
  """An all-zero block tensor is degenerate but must not produce nan."""
  cal = SVDCalibrator()
  cal.mark_step_begin()
  cal.update(torch.zeros(1, 4, 8))
  cal.mark_step_begin()
  pred = cal.approximate()
  assert torch.isfinite(pred).all()
  assert torch.allclose(pred, torch.zeros(1, 4, 8))


def test_taylor_order1_extrapolates_linearly():
  """SVD+Taylor: an order-1 ladder extrapolates a linear coefficient sequence exactly."""
  cal = SVDCalibrator(tau=1.0, beta=0.0, predictor="taylor", n_derivatives=1)
  for value in (1.0, 2.0):
    cal.mark_step_begin()
    cal.update(torch.tensor([[[value]]]))
  cal.mark_step_begin()
  assert cal.approximate().item() == pytest.approx(3.0, abs=1e-4)
  cal.mark_step_begin()
  assert cal.approximate().item() == pytest.approx(4.0, abs=1e-4)


def test_taylor_order0_reproduces_anchor():
  """n_derivatives=0 keeps only the anchor coefficient, i.e. plain reuse."""
  torch.manual_seed(0)
  Y0 = torch.randn(1, 8, 16)
  Y1 = torch.randn(1, 8, 16)
  cal = SVDCalibrator(tau=0.9, predictor="taylor", n_derivatives=0)
  cal.mark_step_begin()
  cal.update(Y0)
  cal.mark_step_begin()
  cal.update(Y1)
  cal.mark_step_begin()
  assert torch.allclose(cal.approximate(), Y1, atol=1e-5, rtol=1e-4)


def test_taylor_order2_matches_quadratic_extrapolation():
  """An order-2 ladder reproduces the quadratic through the last three anchors."""
  cal = SVDCalibrator(tau=1.0, predictor="taylor", n_derivatives=2)
  for value in (1.0, 4.0, 9.0):
    cal.mark_step_begin()
    cal.update(torch.tensor([[[value]]]))
  cal.mark_step_begin()
  assert cal.approximate().item() == pytest.approx(15.0, abs=1e-4)


def test_taylor_ignores_beta():
  """Beta only steers the EMA predictor, so the taylor forecast must not read it."""
  torch.manual_seed(0)
  anchors = [torch.randn(1, 8, 16) for _ in range(3)]
  predictions = []
  for beta in (0.0, 0.9):
    cal = SVDCalibrator(tau=0.9, beta=beta, predictor="taylor")
    for Y in anchors:
      cal.mark_step_begin()
      cal.update(Y)
    cal.mark_step_begin()
    predictions.append(cal.approximate())
  assert torch.equal(predictions[0], predictions[1])


def test_predictor_is_validated():
  """An unknown predictor fails at construction, not at the first cached step."""
  with pytest.raises(ValueError, match="predictor must be one of"):
    SVDCalibrator(predictor="cubic")


@pytest.mark.parametrize("bad_kwargs", [
  {
    "tau": 0.0
  },
  {
    "tau": 1.5
  },
  {
    "beta": 1.0
  },
  {
    "max_rank": -1
  },
  {
    "n_derivatives": -1
  },
])
def test_hyper_parameters_are_validated_eagerly(bad_kwargs):
  """Range checks fire at construction (ValueError), not in the middle of a run."""
  with pytest.raises(ValueError):
    SVDCalibrator(**bad_kwargs)


@pytest.mark.parametrize("predictor", ["ema", "taylor"])
def test_taylor_and_ema_handle_batched_blocks(predictor):
  """CFG pipelines hand in [2, S, D]; the forecast must keep that shape, not [2S, D]."""
  torch.manual_seed(0)
  cal = SVDCalibrator(tau=0.9, predictor=predictor)
  cal.mark_step_begin()
  cal.update(torch.randn(2, 8, 16))
  cal.mark_step_begin()
  pred = cal.approximate()
  assert pred.shape == (2, 8, 16)
  assert torch.isfinite(pred).all()


def test_taylor_approximate_is_idempotent_within_step():
  """The Taylor forecast is a pure function of the ladder, so a second query cannot drift."""
  torch.manual_seed(0)
  cal = SVDCalibrator(tau=0.9, predictor="taylor")
  cal.mark_step_begin()
  cal.update(torch.randn(1, 8, 16))
  cal.mark_step_begin()
  assert torch.equal(cal.approximate(), cal.approximate())


def test_taylor_divergence_falls_back_to_the_anchor(caplog):
  """A blown-up ladder must return the anchor value and warn exactly once."""
  torch.manual_seed(0)
  Y = torch.randn(1, 8, 16)
  cal = SVDCalibrator(tau=0.9, predictor="taylor")
  cal.mark_step_begin()
  cal.update(Y)
  state = cal.states["default"]
  state._dc_cur[0] = torch.full_like(state._dc_cur[0], float("inf"))
  cal.mark_step_begin()
  with caplog.at_level("WARNING"):
    pred = cal.approximate()
    second = cal.approximate()
  assert torch.isfinite(pred).all()
  assert torch.allclose(pred, Y, atol=1e-5, rtol=1e-4)
  assert torch.equal(pred, second)
  assert sum("not finite" in record.message for record in caplog.records) == 1


def _calibrator_info(context_manager) -> dict:
  """Summarize what the calibrators actually did, so a test can prove they ran."""

  context = context_manager.get_context()
  info = {}
  for label, pair in (("", context.get_calibrators()), ("cfg_", context.get_cfg_calibrators())):
    calibrator, encoder_calibrator = pair
    states = list(calibrator.states.values()) if calibrator is not None else []
    info[f"{label}calibrator"] = type(calibrator).__name__ if calibrator is not None else None
    info[f"{label}num_states"] = len(states)
    info[f"{label}basis_fitted"] = bool(states) and all(s.basis is not None for s in states)
    info[f"{label}residual_set"] = bool(states) and all(s.residual is not None for s in states)
    info[f"{label}encoder_states"] = (len(encoder_calibrator.states)
                                      if encoder_calibrator is not None else 0)
  return info


def _run_pipeline(cache_config, calibrator_config, steps: int = 4):
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


@pytest.mark.parametrize("predictor", ["ema", "taylor"])
def test_pipeline_caches_with_svd_calibrator(predictor):
  """End-to-end through the real cache plumbing: caching happens and SVD-Cache really runs."""
  output, cached_steps, info = _run_pipeline(
    DBCacheConfig(
      Fn_compute_blocks=8,
      Bn_compute_blocks=0,
      max_warmup_steps=1,
      residual_diff_threshold=0.99,
    ),
    SVDCalibratorConfig(svd_tau=0.9, svd_beta=0.9, svd_predictor=predictor),
  )
  sample = output.sample if hasattr(output, "sample") else output
  assert torch.isfinite(sample).all()
  assert sum(cached_steps) > 0, cached_steps
  # Caching alone would also pass the two assertions above, so pin the calibrator itself:
  assert info["calibrator"] == "SVDCalibrator", info
  assert info["num_states"] >= 1, info
  assert info["basis_fitted"] and info["residual_set"], info
  assert info["encoder_states"] >= 1, info


def test_pipeline_separate_cfg_keeps_its_own_calibrator():
  """With enable_separate_cfg the CFG branch owns an independent calibrator instance."""
  output, cached_steps, info = _run_pipeline(
    DBCacheConfig(
      Fn_compute_blocks=8,
      Bn_compute_blocks=0,
      max_warmup_steps=1,
      residual_diff_threshold=0.99,
      enable_separate_cfg=True,
    ),
    SVDCalibratorConfig(),
  )
  sample = output.sample if hasattr(output, "sample") else output
  assert torch.isfinite(sample).all()
  assert sum(cached_steps) > 0, cached_steps
  assert info["cfg_num_states"] >= 1, info
  assert info["cfg_basis_fitted"] and info["cfg_residual_set"], info


def test_max_cached_steps_zero_is_bit_exact():
  """With caching disabled the calibrator must not perturb the compute path at all."""
  reference, ref_cached, _ = _run_pipeline(None, None)
  cached, cached_steps, _ = _run_pipeline(
    DBCacheConfig(
      Fn_compute_blocks=8,
      Bn_compute_blocks=0,
      max_warmup_steps=1,
      max_cached_steps=0,
      residual_diff_threshold=0.99,
    ),
    SVDCalibratorConfig(),
  )
  assert ref_cached == [] and cached_steps == []
  ref_sample = reference.sample if hasattr(reference, "sample") else reference
  cached_sample = cached.sample if hasattr(cached, "sample") else cached
  assert torch.equal(ref_sample, cached_sample)
