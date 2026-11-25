import gc

import pytest
import torch
from diffusers import (
  FlowMatchEulerDiscreteScheduler,
  FluxPipeline,
  FluxTransformer2DModel,
)

import cache_dit
from cache_dit import DBCacheConfig, ResidualMotionCalibratorConfig, TaylorSeerCalibratorConfig
from cache_dit.caching.cache_contexts.calibrators import ResidualMotionCalibrator
from cache_dit.caching.cache_contexts.softcache_utils import (
  regrid_tail,
  softcache_decide,
  softcache_rewrite_tail,
)

STEPS = 20


def _tiny_flux(scheduler=None):
  torch.manual_seed(0)
  transformer = FluxTransformer2DModel(
    patch_size=1,
    in_channels=16,
    num_layers=2,
    num_single_layers=4,
    attention_head_dim=8,
    num_attention_heads=2,
    joint_attention_dim=32,
    pooled_projection_dim=16,
    axes_dims_rope=[2, 2, 4],
    guidance_embeds=True,
  )
  return FluxPipeline(
    scheduler=scheduler or FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
    vae=None,
    text_encoder=None,
    tokenizer=None,
    text_encoder_2=None,
    tokenizer_2=None,
    transformer=transformer,
  )


def _soft_config(**kwargs):
  values = dict(
    Fn_compute_blocks=1,
    Bn_compute_blocks=0,
    max_warmup_steps=1,
    max_continuous_cached_steps=-1,
    residual_diff_threshold=0.5,
    enable_seacache=True,
    seacache_feature_shape=(4, 4),
    enable_softcache=True,
    softcache_tau=1.5,
    # The tiny random transformer moves far more per step than a trained one; open the gate so
    # the jump path is exercised.
    softcache_sigma_hi=1.0,
    softcache_raw_rel_l1_max=1.0,
  )
  values.update(kwargs)
  return DBCacheConfig(**values)


def _call(pipe, steps=STEPS):
  generator = torch.Generator().manual_seed(0)
  return pipe(
    prompt_embeds=torch.randn(1, 8, 32, generator=generator),
    pooled_prompt_embeds=torch.randn(1, 16, generator=generator),
    height=64,
    width=64,
    num_inference_steps=steps,
    output_type="latent",
    generator=generator,
  ).images


def _run(cache_config=None, calibrator_config=None, steps=STEPS):
  """Run tiny FLUX once; return output, stats and the number of real transformer calls."""

  gc.collect()
  pipe = _tiny_flux()
  calls = []
  pipe.transformer.x_embedder.register_forward_hook(lambda *_: calls.append(1))
  if cache_config is not None:
    cache_dit.enable_cache(pipe, cache_config=cache_config, calibrator_config=calibrator_config)
  output = _call(pipe, steps)
  manager = getattr(pipe, "_context_manager", None)
  stats = dict(manager.softcache_stats) if manager is not None else {}
  cached_steps = list(getattr(pipe.transformer, "_cached_steps", []) or [])
  sigmas = pipe.scheduler.sigmas.clone()
  if cache_config is not None:
    cache_dit.disable_cache(pipe)
  return output, stats, len(calls), cached_steps, sigmas


def test_decide_is_the_reference_rule():
  """Fresh at step 0 and once the accumulated change reaches tau; otherwise cache, and jump with
  jf = 1 + (jf_max - 1) * (1 - accumulated / tau) when the gate allows. tau > 1 is valid."""

  kwargs = dict(jf_max=8.0, sigma_hi=0.9, raw_rel_l1_max=0.18)
  assert softcache_decide(0, 1.0, 0.0, 0.0, tau=0.3, **kwargs) == ("fresh", None)
  assert softcache_decide(5, 0.5, 0.01, 0.3, tau=0.3, **kwargs) == ("fresh", None)
  action, jf = softcache_decide(5, 0.5, 0.01, 0.075, tau=0.3, **kwargs)
  assert action == "jump" and jf == pytest.approx(1 + 7 * 0.75)
  action, jf = softcache_decide(5, 0.5, 0.01, 0.55, tau=1.1, **kwargs)  # tau above 1
  assert action == "jump" and jf == pytest.approx(1 + 7 * 0.5)
  assert softcache_decide(5, 0.5, 0.01, 1.1, tau=1.1, **kwargs) == ("fresh", None)
  assert softcache_decide(5, 0.95, 0.01, 0.1, tau=0.3, **kwargs) == ("cache", None)  # early sigma
  assert softcache_decide(5, 0.5, 0.2, 0.2, tau=0.3, **kwargs) == ("cache", None)  # sharp change
  assert softcache_decide(5, 0.5, 0.01, 0.2999, tau=0.3,
                          **kwargs) == ("cache", None)  # stride < 1.01


def _list_schedule(sigmas, jumps):
  """Reference loop on a shrinking list: the sigmas each step actually integrates over."""

  sigmas, visited, i = list(sigmas), [], 0
  while sigmas[i] > 0:
    if i in jumps:
      target = max(0.0, sigmas[i] + jumps[i] * (sigmas[i + 1] - sigmas[i]))
      sigmas = regrid_tail(sigmas, i, target)
    visited.append((sigmas[i], sigmas[i + 1]))
    i += 1
  return visited


def test_in_place_padded_regrid_matches_shrinking_list():
  """Fixed-length in-place rewrites integrate exactly the sigmas the reference loop does."""

  for jumps in ({3: 4.0}, {2: 3.0, 6: 8.0}, {1: 2.0, 3: 2.0, 5: 2.0, 7: 2.0}, {9: 8.0}):
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    scheduler.set_timesteps(12)
    reference = _list_schedule([float(s) for s in scheduler.sigmas], jumps)
    length = len(scheduler.sigmas)
    visited, real = [], None
    for i in range(len(scheduler.timesteps)):
      if float(scheduler.sigmas[i]) <= 0.0:  # dt = 0 padding
        assert float(scheduler.sigmas[i + 1]) == 0.0
        continue
      if i in jumps:
        _, real = softcache_rewrite_tail(scheduler, i, jumps[i], real)
      visited.append((float(scheduler.sigmas[i]), float(scheduler.sigmas[i + 1])))
    assert len(scheduler.sigmas) == length
    assert torch.allclose(scheduler.timesteps, scheduler.sigmas[:-1] * 1000)
    assert len(visited) == len(reference)
    assert sum(visited, ()) == pytest.approx(sum(reference, ()))


def test_pipeline_jumps_skip_compute_and_cache_counters():
  """Jumps shorten the run; padded steps run no transformer and are not DBCache cache steps."""

  _, stats, calls, cached_steps, sigmas = _run(_soft_config())
  assert stats["jumps"] > 0 and stats["skipped_steps"] > 0
  assert stats["padded_calls"] == stats["skipped_steps"]
  assert calls == STEPS - stats["skipped_steps"]
  assert len(sigmas) == STEPS + 1 and float(sigmas[-1]) == 0.0
  assert max(cached_steps) < STEPS - stats["skipped_steps"]


def test_softcache_with_taylorseer_and_repeated_requests():
  """An existing calibrator composes with SoftCache, and a second request reproduces the first."""

  pipe = _tiny_flux()
  cache_dit.enable_cache(
    pipe,
    cache_config=_soft_config(),
    calibrator_config=TaylorSeerCalibratorConfig(taylorseer_order=1),
  )
  first = _call(pipe)
  first_stats = dict(pipe._context_manager.softcache_stats)
  second = _call(pipe)
  assert first_stats["jumps"] > 0
  assert pipe._context_manager.softcache_stats == first_stats
  assert torch.equal(first, second)
  assert pipe._context_manager._softcache_last_outputs is None
  cache_dit.disable_cache(pipe)


def test_residual_motion_reads_the_rewritten_schedule():
  """Under jumps Residual Motion gets each real step's sigma from the regridded schedule."""

  seen, original = [], ResidualMotionCalibrator.set_sigma
  ResidualMotionCalibrator.set_sigma = lambda self, sigma: (seen.append(sigma), original(
    self, sigma))[1]
  try:
    output, stats, calls, cached_steps, sigmas = _run(_soft_config(),
                                                      ResidualMotionCalibratorConfig())
  finally:
    ResidualMotionCalibrator.set_sigma = original
  assert stats["jumps"] > 0 and sum(cached_steps) > 0
  assert torch.isfinite(output).all()
  per_step = seen[::2]  # hidden and encoder calibrators see the same sigma
  assert seen[1::2] == per_step and len(per_step) == calls
  assert all(a > b for a, b in zip(per_step, per_step[1:]))
  # The real steps are the non-padded nodes of the final schedule, in order.
  real = [float(x) for x in sigmas[:STEPS] if float(x) > 0.0]
  assert per_step == pytest.approx(real, abs=1e-6)


def test_zero_cache_is_bit_exact():
  """With caching disabled, SoftCache (with or without a calibrator) is the uncached pipeline."""

  base, _, _, _, _ = _run()
  for calibrator in (None, TaylorSeerCalibratorConfig(taylorseer_order=1)):
    cached, stats, calls, cached_steps, _ = _run(_soft_config(max_cached_steps=0), calibrator)
    assert stats["jumps"] == 0 and calls == STEPS and not cached_steps
    assert torch.equal(base, cached)


def test_tau_sets_how_often_the_step_is_computed():
  """A tiny tau computes every step (bit-exact to no cache); a larger tau caches more."""

  base, _, _, _, _ = _run()
  tiny, stats, calls, cached, _ = _run(_soft_config(softcache_tau=1e-9))
  assert not cached and stats["jumps"] == 0 and calls == STEPS
  assert torch.equal(base, tiny)
  counts = [
    len(_run(_soft_config(softcache_tau=tau, softcache_jf_max=1.0))[3]) for tau in (0.3, 1.5, 50.0)
  ]
  assert counts[0] <= counts[1] <= counts[2] and counts[2] > counts[0]
  assert counts[2] == STEPS - 1  # only step 0 is computed in full


def test_probe_hook_is_removed_on_disable():
  pipe = _tiny_flux()
  norm1 = pipe.transformer.transformer_blocks[0].norm1
  cache_dit.enable_cache(pipe, cache_config=_soft_config())
  assert len(norm1._forward_hooks) == 1
  cache_dit.disable_cache(pipe)
  assert len(norm1._forward_hooks) == 0


def test_missing_probe_warns_once():
  """A probe hook that never fires (e.g. `enable_cache` on an already-compiled transformer) makes
  every step fresh; that must not be silent."""

  from unittest import mock
  from cache_dit.caching.cache_contexts import cache_manager

  pipe = _tiny_flux()
  cache_dit.enable_cache(pipe, cache_config=_soft_config())
  pipe._context_manager._remove_softcache_hook(
  )  # what a compiled graph traced without the hook does
  with mock.patch.object(cache_manager.logger, "warning") as warning:
    _call(pipe)
    _call(pipe)
  assert pipe._context_manager.softcache_stats["jumps"] == 0
  assert warning.call_count == 1
  assert "probe" in warning.call_args[0][0]
  cache_dit.disable_cache(pipe)


def test_transformer_receives_the_rewritten_timesteps():
  """Each real call sees its step's post-jump sigma (FLUX passes t / 1000); padding sees 0."""

  pipe = _tiny_flux()
  seen = []
  pipe.transformer.register_forward_pre_hook(
    lambda _, args, kwargs: seen.append(float(kwargs["timestep"][0])), with_kwargs=True)
  cache_dit.enable_cache(pipe, cache_config=_soft_config())
  _call(pipe)
  stats = pipe._context_manager.softcache_stats
  sigmas = [float(s) for s in pipe.scheduler.sigmas[:-1]]
  assert stats["jumps"] > 0 and len(seen) == STEPS
  assert seen == pytest.approx(sigmas, abs=1e-6)
  assert sigmas[-stats["skipped_steps"]:] == [0.0] * stats["skipped_steps"]
  cache_dit.disable_cache(pipe)


@pytest.mark.parametrize(
  "kwargs, match",
  [
    (dict(enable_seacache=False), "enable_seacache"),
    (dict(seacache_mode="vp"), "enable_seacache"),
    (dict(seacache_ab_schedule=((0.5, 0.5), )), "seacache_ab_schedule"),
    (dict(enable_separate_cfg=True), "enable_separate_cfg"),
  ],
)
def test_unsupported_configs_are_rejected(kwargs, match):
  pipe = _tiny_flux()
  with pytest.raises((ValueError, NotImplementedError), match=match):
    cache_dit.enable_cache(pipe, cache_config=_soft_config(**kwargs))


def test_non_flowmatch_scheduler_is_rejected():
  from diffusers import FlowMatchHeunDiscreteScheduler

  pipe = _tiny_flux(scheduler=FlowMatchHeunDiscreteScheduler())
  with pytest.raises(ValueError, match="FlowMatchEulerDiscreteScheduler"):
    cache_dit.enable_cache(pipe, cache_config=_soft_config())
