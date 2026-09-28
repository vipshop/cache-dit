import pytest
import torch

from cache_dit.caching.cache_contexts.calibrators.dmd import DMDState


def _trajectory(t: int) -> torch.Tensor:
  gen = torch.Generator().manual_seed(0)
  basis = torch.linalg.qr(torch.randn(32, 32, dtype=torch.float64, generator=gen))[0][:, :3]
  coeffs = torch.tensor([0.95 ** t, 0.9 ** t, 0.8 ** t], dtype=torch.float64)
  return (basis @ coeffs).float().reshape(1, 4, 8)


@pytest.mark.parametrize("svd_precision", ["low", "medium", "high"])
def test_dmd_forecast_on_cpu_for_every_svd_precision(svd_precision):
  # Rank-3 linear dynamics are exactly representable by DMD, so every precision
  # level must extrapolate instead of silently falling back to the last snapshot.
  state = DMDState(history=6, svd_precision=svd_precision)
  for t in range(6):
    state.mark_step_begin()
    state.update(_trajectory(t))
  state.mark_step_begin()
  pred = state.approximate()
  target = _trajectory(6)
  assert not torch.equal(pred, _trajectory(5))
  assert torch.allclose(pred, target, rtol=1e-4, atol=1e-5)
