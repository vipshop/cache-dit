"""Unit checks for the CausVid sigma grid.

The grid is not a diffusers scheduler: CausVid looks the sigma of a fixed DMD timestep table up in a
shifted linear ramp over the 1000 training steps. Getting `extra_one_step` or the shift wrong moves
every sigma slightly and only shows up as a quality drift after 7 autoregressive chunks, so the grid
is pinned here against an independently written formula.

Usage:
  python -m pytest tests/utils/test_causvid_wan_scheduler.py -q
"""

import pytest
import torch

from cache_dit.models.pipelines.causvid.scheduler import CausVidFlowMatchScheduler


def _expected_sigmas(num_steps: int = 1000, shift: float = 8.0):
  ramps = [1.0 - index / num_steps for index in range(num_steps)]
  return [shift * value / (1 + (shift - 1) * value) for value in ramps]


def test_grid_matches_the_released_formula():
  scheduler = CausVidFlowMatchScheduler()
  expected = torch.tensor(_expected_sigmas())

  assert scheduler.sigmas.shape == (1000, )
  assert scheduler.sigmas[0].item() == 1.0
  torch.testing.assert_close(scheduler.sigmas, expected, rtol=0, atol=1e-6)
  # The tail is the `extra_one_step` fingerprint: dropping it would end the ramp at 1/1000 sigma
  # instead of 1/1000 of the shifted ramp.
  assert scheduler.sigmas[-1].item() == pytest.approx(8 * 0.001 / 1.007, rel=0, abs=1e-9)
  assert scheduler.timesteps[0].item() == pytest.approx(1000.0)
  assert torch.all(scheduler.sigmas[1:] <= scheduler.sigmas[:-1])


def test_sigma_lookup_picks_the_nearest_grid_point():
  scheduler = CausVidFlowMatchScheduler()
  timesteps = torch.tensor([1000.0, 757.0, 522.0])

  sigma = scheduler.sigma(timesteps)
  assert sigma.shape == (3, )
  assert sigma[0].item() == 1.0

  # Nearest-grid semantics: the value must equal the grid sigma of the argmin index, which is how the
  # reference implementation converts a DMD timestep to its flow-matching sigma.
  grid = scheduler.timesteps
  index = torch.argmin((grid.unsqueeze(0) - timesteps.unsqueeze(1)).abs(), dim=1)
  torch.testing.assert_close(sigma, scheduler.sigmas[index], rtol=0, atol=0)
  assert sigma[1].item() > sigma[2].item()


def test_add_noise_endpoints():
  scheduler = CausVidFlowMatchScheduler()
  generator = torch.Generator().manual_seed(0)
  clean = torch.randn(2, 4, generator=generator)
  noise = torch.randn(2, 4, generator=generator)

  at_full_noise = scheduler.add_noise(clean, noise, torch.tensor([1000.0, 1000.0]))
  torch.testing.assert_close(at_full_noise, noise, rtol=0, atol=0)

  at_zero = scheduler.add_noise(clean, noise, torch.tensor([0.0, 0.0]))
  sigma = scheduler.sigmas[-1]
  torch.testing.assert_close(at_zero, (1 - sigma) * clean + sigma * noise, rtol=0, atol=1e-6)
  assert torch.all((at_zero - clean).abs() < 0.05)


def test_add_noise_keeps_the_latent_rank():
  scheduler = CausVidFlowMatchScheduler()
  clean = torch.zeros(1, 16, 3, 8, 8)
  noise = torch.ones_like(clean)
  out = scheduler.add_noise(clean, noise, torch.tensor([1000.0]))
  assert out.shape == clean.shape
  torch.testing.assert_close(out, noise, rtol=0, atol=0)
