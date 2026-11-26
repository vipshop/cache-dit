import gc
from types import SimpleNamespace

import pytest
import torch

import cache_dit
from cache_dit import BlockAdapter, DBCacheConfig, ForwardPattern, ResidualMotionCalibratorConfig
from cache_dit.caching.cache_contexts.calibrators import (
  Calibrator,
  ResidualMotionCalibrator,
  normalize_calibrator_type,
)
from cache_dit.cli.utils import build_calibrator_config, get_args, maybe_postprocess_args
from utils import RandPipeline

STEPS = 8
# A non-uniform, shifted-looking flow-matching schedule (N + 1 entries, last is 0).
SIGMAS = torch.tensor([1.0, 0.97, 0.93, 0.87, 0.78, 0.65, 0.48, 0.27, 0.0])


def _config(**kwargs):
  values = dict(
    Fn_compute_blocks=8,
    Bn_compute_blocks=0,
    max_warmup_steps=1,
    max_continuous_cached_steps=1,
    residual_diff_threshold=0.99,
    num_inference_steps=STEPS,
  )
  values.update(kwargs)
  return DBCacheConfig(**values)


def _adapter(pipe, transformer, with_pipe=True):
  return BlockAdapter(
    pipe=pipe if with_pipe else None,
    transformer=transformer,
    blocks=transformer.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_0,
  )


def _run(cache_config=None, calibrator_config=None, scheduler=SIGMAS, spy=False):
  gc.collect()
  torch.manual_seed(0)
  pipe = RandPipeline(pattern=ForwardPattern.Pattern_0)
  if scheduler is not None:
    pipe.scheduler = SimpleNamespace(sigmas=scheduler)
  transformer = pipe.transformer
  adapter = None
  if cache_config is not None:
    adapter = cache_dit.enable_cache(_adapter(pipe, transformer),
                                     cache_config=cache_config,
                                     calibrator_config=calibrator_config)
  seen = []
  if spy:
    original = ResidualMotionCalibrator.set_sigma
    ResidualMotionCalibrator.set_sigma = lambda self, sigma: (seen.append(sigma),
                                                              original(self, sigma))[1]
  try:
    inputs = (torch.randn(1, 64, 16), torch.randn(1, 64, 16))
    output = pipe(*inputs, num_inference_steps=STEPS)
  finally:
    if spy:
      ResidualMotionCalibrator.set_sigma = original
  cached_steps = list(getattr(transformer, "_cached_steps", []))
  if adapter is not None:
    cache_dit.disable_cache(adapter)
  del pipe, adapter
  gc.collect()
  return output, cached_steps, seen


def test_extrapolates_along_the_sigma_secant():
  """Two anchors at non-uniform sigmas give the sigma secant; one anchor reuses its residual."""

  calibrator = Calibrator(ResidualMotionCalibratorConfig(residual_motion_beta=0.5))
  r0, r1 = torch.randn(2, 3), torch.randn(2, 3)
  calibrator.set_sigma(0.9)
  calibrator.update(r0)
  calibrator.set_sigma(0.8)
  assert torch.equal(calibrator.approximate(), r0)
  calibrator.set_sigma(0.6)
  calibrator.update(r1)
  calibrator.set_sigma(0.25)
  lam = (0.25 - 0.6) / (0.6 - 0.9 - 1e-8)
  assert torch.allclose(calibrator.approximate(), r1 + (0.5 * lam) * (r1 - r0))
  calibrator.reset_cache()
  with pytest.raises(RuntimeError, match="sigma"):
    calibrator.update(r0)


@pytest.mark.parametrize("beta", [-0.1, 1.5, True, "1"])
def test_beta_is_validated(beta):
  with pytest.raises(ValueError, match="beta"):
    ResidualMotionCalibratorConfig(residual_motion_beta=beta)


def test_aliases_resolve():
  for name in ("residual_motion", "residual-motion", "rm"):
    assert normalize_calibrator_type(name) == "residual_motion"
    assert isinstance(Calibrator(ResidualMotionCalibratorConfig(calibrator_type=name)),
                      ResidualMotionCalibrator)


def test_plain_dbcache_hits_and_reads_the_scheduler_sigmas():
  """On plain DBCache, Residual Motion caches and sees the scheduler's sigma at every step."""

  output, cached_steps, seen = _run(_config(), ResidualMotionCalibratorConfig(), spy=True)
  assert sum(cached_steps) > 0
  assert torch.isfinite(output.sample).all()
  # Two calibrators (hidden and encoder states) per step, in schedule order.
  assert seen[::2] == pytest.approx(SIGMAS[:STEPS].tolist())
  assert seen[1::2] == seen[::2]


def test_zero_cache_is_bit_exact():
  """With no cached steps, Residual Motion reproduces the uncached pipeline bit for bit."""

  base, _, _ = _run()
  cached, cached_steps, _ = _run(_config(max_cached_steps=0, residual_diff_threshold=0.0),
                                 ResidualMotionCalibratorConfig())
  assert sum(cached_steps) == 0
  assert torch.equal(base.sample, cached.sample)


def test_repeated_requests_are_identical():
  """State resets per request: two identical calls on one pipeline give the same output."""

  torch.manual_seed(0)
  pipe = RandPipeline(pattern=ForwardPattern.Pattern_0)
  pipe.scheduler = SimpleNamespace(sigmas=SIGMAS)
  transformer = pipe.transformer
  adapter = cache_dit.enable_cache(_adapter(pipe, transformer),
                                   cache_config=_config(),
                                   calibrator_config=ResidualMotionCalibratorConfig())
  inputs = (torch.randn(1, 64, 16), torch.randn(1, 64, 16))
  torch.manual_seed(1)
  first = pipe(*inputs, num_inference_steps=STEPS).sample.clone()
  first_steps = list(transformer._cached_steps)
  torch.manual_seed(1)
  second = pipe(*inputs, num_inference_steps=STEPS).sample
  assert list(transformer._cached_steps) == first_steps and sum(first_steps) > 0
  assert torch.equal(first, second)
  manager = pipe._context_manager
  cache_dit.disable_cache(adapter)
  assert manager._residual_motion_scheduler is None


def test_missing_scheduler_fails_loudly():
  pipe = RandPipeline(pattern=ForwardPattern.Pattern_0)
  with pytest.raises(ValueError, match="scheduler"):
    cache_dit.enable_cache(_adapter(pipe, pipe.transformer),
                           cache_config=_config(),
                           calibrator_config=ResidualMotionCalibratorConfig())


def test_transformer_only_takes_an_explicit_scheduler():
  """Transformer-only caching binds `residual_motion_scheduler` and needs num_inference_steps."""

  scheduler = SimpleNamespace(sigmas=SIGMAS)
  pipe = RandPipeline(pattern=ForwardPattern.Pattern_0)
  transformer = pipe.transformer
  with pytest.raises(ValueError, match="scheduler"):
    cache_dit.enable_cache(_adapter(None, transformer, with_pipe=False),
                           cache_config=_config(),
                           calibrator_config=ResidualMotionCalibratorConfig())
  with pytest.raises(ValueError, match="num_inference_steps"):
    cache_dit.enable_cache(_adapter(None, transformer, with_pipe=False),
                           cache_config=_config(num_inference_steps=None),
                           calibrator_config=ResidualMotionCalibratorConfig(),
                           residual_motion_scheduler=scheduler)
  adapter = cache_dit.enable_cache(_adapter(None, transformer, with_pipe=False),
                                   cache_config=_config(),
                                   calibrator_config=ResidualMotionCalibratorConfig(),
                                   residual_motion_scheduler=scheduler)
  assert transformer._context_manager._residual_motion_scheduler is scheduler
  cache_dit.disable_cache(adapter)


def test_residual_motion_scheduler_requires_its_calibrator():
  """Passing the scheduler without the Residual Motion calibrator is an error, not a silent drop."""

  from cache_dit import TaylorSeerCalibratorConfig

  pipe = RandPipeline(pattern=ForwardPattern.Pattern_0)
  pipe.scheduler = SimpleNamespace(sigmas=SIGMAS)
  with pytest.raises(ValueError, match="residual_motion"):
    cache_dit.enable_cache(_adapter(pipe, pipe.transformer),
                           cache_config=_config(),
                           calibrator_config=TaylorSeerCalibratorConfig(),
                           residual_motion_scheduler=pipe.scheduler)


def test_other_calibrators_do_not_bind_a_scheduler():
  """Only Residual Motion binds a scheduler; the default path is untouched."""

  from cache_dit import TaylorSeerCalibratorConfig

  for calibrator_config in (None, TaylorSeerCalibratorConfig()):
    pipe = RandPipeline(pattern=ForwardPattern.Pattern_0)
    pipe.scheduler = SimpleNamespace(sigmas=SIGMAS)
    adapter = cache_dit.enable_cache(_adapter(pipe, pipe.transformer),
                                     cache_config=_config(),
                                     calibrator_config=calibrator_config)
    assert pipe._context_manager._residual_motion_scheduler is None
    cache_dit.disable_cache(adapter)


def test_cli_and_yaml_options():
  parser = get_args(parse=False)
  args = maybe_postprocess_args(
    parser.parse_args(["--cache", "--calibrator", "rm", "--calibrator-params", "beta=0.5"]))
  config = build_calibrator_config(args)
  assert isinstance(config, ResidualMotionCalibratorConfig)
  assert config.residual_motion_beta == 0.5
  loaded = cache_dit.load_configs({
    "cache_config": {
      "enable_residual_motion": True,
      "residual_motion_beta": 0.5,
    },
  })
  assert isinstance(loaded["calibrator_config"], ResidualMotionCalibratorConfig)
  assert loaded["calibrator_config"].residual_motion_beta == 0.5
