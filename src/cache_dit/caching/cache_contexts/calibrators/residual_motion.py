"""Residual Motion calibrator from SoftCache (arXiv 2610.22369).

A cached residual is moved along the secant of the last two real anchors, parameterized by the
flow-matching sigma instead of the step count:

    r(sigma) = r_a + beta * lam * (r_a - r_{a-1}),  lam = (sigma - sigma_a) / (sigma_a - sigma_{a-1})

Sigma coordinates keep the extrapolation right on non-uniform grids (shifted flow-matching
schedules, or the grid a SoftCache jump leaves behind). The cache manager reads the current sigma
from the bound scheduler, as SeaCache does, and supplies it through `set_sigma` before each step.
"""

from typing import Dict, List, Optional, Tuple

import torch

from .base import CalibratorBase


def validate_residual_motion_params(beta: float):
  """Validate the Residual Motion step size.

  :param beta: Fraction of the anchor secant to apply, in [0, 1].
  :raises ValueError: When `beta` is outside [0, 1].
  """

  if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not 0.0 <= beta <= 1.0:
    raise ValueError(f"Residual Motion beta must be in [0, 1], got {beta!r}.")


class ResidualMotionState:
  """The last two real anchors of one tensor stream, keyed by their sigma."""

  def __init__(self, beta: float):
    self.beta = float(beta)
    self.reset()

  def reset(self):
    self.anchors: List[Tuple[float, torch.Tensor]] = []

  def update(self, Y: torch.Tensor, sigma: float):
    if self.anchors and Y.shape != self.anchors[-1][1].shape:
      self.reset()
    self.anchors = (self.anchors + [(sigma, Y.detach().clone())])[-2:]

  def approximate(self, sigma: float) -> torch.Tensor:
    assert self.anchors, "Residual Motion needs at least one real anchor."
    sigma_a, r_a = self.anchors[-1]
    if len(self.anchors) == 1:  # a single anchor so far: reuse its residual as is
      return r_a
    sigma_p, r_p = self.anchors[0]
    # sigma decreases along the schedule, so the denominator is negative; the 1e-8 guard and the
    # (beta * lam) grouping match the reference implementation bit for bit.
    lam = (sigma - sigma_a) / (sigma_a - sigma_p - 1e-8)
    return r_a + (self.beta * lam) * (r_a - r_p)


class ResidualMotionCalibrator(CalibratorBase):
  """Calibrator that extrapolates cached residuals along the sigma secant of two anchors."""

  def __init__(self, beta: float = 1.0, **kwargs):
    """Create a Residual Motion calibrator whose states are keyed by logical stream names.

    :param beta: Fraction of the anchor secant to apply, in [0, 1].
    :param kwargs: Additional keyword arguments accepted for config compatibility.
    """

    validate_residual_motion_params(beta)
    self.beta = float(beta)
    self.sigma: Optional[float] = None
    self.states: Dict[str, ResidualMotionState] = {}

  def set_sigma(self, sigma: float):
    """Set the flow-matching sigma of the step about to run."""

    self.sigma = float(sigma)

  def _require_sigma(self) -> float:
    if self.sigma is None:
      raise RuntimeError("Residual Motion needs the step sigma; enable it on a pipeline with a "
                         "flow-matching scheduler, or pass `residual_motion_scheduler`.")
    return self.sigma

  def reset_cache(self):
    self.sigma = None
    for state in self.states.values():
      state.reset()

  def mark_step_begin(self, *args, **kwargs):
    """Step position comes from `set_sigma`, so there is no counter to advance."""

  def approximate(self, name: str = "default") -> torch.Tensor:
    assert name in self.states, f"State '{name}' not found."
    return self.states[name].approximate(self._require_sigma())

  def update(self, Y: torch.Tensor, name: str = "default"):
    if name not in self.states:
      self.states[name] = ResidualMotionState(self.beta)
    self.states[name].update(Y, self._require_sigma())

  def __repr__(self):
    return f"ResidualMotionCalibrator(beta={self.beta:g})"
