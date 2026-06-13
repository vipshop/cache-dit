# Dynamic Mode Decomposition (Prony) calibrator — an EXPONENTIAL-basis alternative
# to the TaylorSeer polynomial forecaster. Note: "DMD" here is Dynamic Mode
# Decomposition (Schmid 2010; the SVD-regularised, multivariate generalisation of
# Prony's method, 1795), NOT Distribution Matching Distillation.
#
# Why an exponential basis: across denoising steps the cached feature/residual
# evolves under a (slowly varying) near-linear operator, so locally it is a sum of
# damped / oscillatory EXPONENTIALS — the exact solution class of a linear ODE.
# A polynomial (Taylor) forecast is only a local truncation of that class and
# diverges as the skip horizon grows; the exponential basis is exact on the class,
# so it keeps quality at larger cache intervals.
import math
from typing import Dict, List, Tuple

import torch

from .base import CalibratorBase

from ....logger import init_logger

logger = init_logger(__name__)


def _dmd_forecast(
  snapshots: List[torch.Tensor],
  k: float,
  rank: int = 0,
  ridge: float = 1e-8,
) -> torch.Tensor:
  """Forecast a feature ``k`` snapshot-spacings past the newest snapshot via
  Dynamic Mode Decomposition (Prony).

  Identify the linear propagator ``A`` from the snapshot pairs
  (``Y_{t+1} ~= A Y_t``) through one economy SVD, eigendecompose it once, and
  advance by (possibly fractional) eigenvalue powers::

      Y_{t+k} ~= Phi @ (lambda**k * b),   b = pinv(Phi) @ Y_t

  :param snapshots: >= 3 same-shape tensors, OLDEST..NEWEST, the fully computed
      features at recent compute steps (uniformly spaced).
  :param k: Forecast horizon in snapshot-spacing units (fractional allowed).
  :param rank: SVD truncation rank; 0 selects it from the spectrum (drop modes
      below 1e-4 of the leading singular value — this is what rejects noise).
  :param ridge: Tikhonov term added to the inverted singular values.
  :returns: Forecast tensor of the snapshot shape; falls back to last-value
      reuse when the history is too short or the fit is degenerate.
  """

  shp, dt = snapshots[-1].shape, snapshots[-1].dtype
  if len(snapshots) < 3:
    return snapshots[-1].clone()
  V = torch.stack([s.reshape(-1) for s in snapshots], dim=1).to(torch.float64)
  X, Xp = V[:, :-1], V[:, 1:]
  try:
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
  except Exception:  # noqa: BLE001 — degenerate fit: fall back to last-value reuse
    return snapshots[-1].clone()
  if rank <= 0:
    rank = int((S > S[0] * 1e-4).sum().clamp(min=1).item())
  rank = max(1, min(rank, S.numel()))
  Ur, Sr, Vr = U[:, :rank], S[:rank], Vh[:rank].mH
  Sinv = (1.0 / (Sr + ridge)).to(torch.complex128)
  Atil = (Ur.mH @ Xp @ Vr).to(torch.complex128) * Sinv.unsqueeze(0)
  try:
    evals, W = torch.linalg.eig(Atil)
    Phi = ((Xp @ Vr).to(torch.complex128) * Sinv.unsqueeze(0)) @ W
    b = torch.linalg.lstsq(Phi, V[:, -1].to(torch.complex128).unsqueeze(1)).solution.squeeze(1)
  except Exception:  # noqa: BLE001 — degenerate fit: fall back to last-value reuse
    return snapshots[-1].clone()
  pred = (Phi @ (evals.pow(float(k)) * b)).real
  if not torch.isfinite(pred).all():
    return snapshots[-1].clone()
  return pred.to(dt).reshape(shp)


class DMDState:
  """Per-stream snapshot history + Taylor fallback for one calibration stream."""

  def __init__(
    self,
    history: int = 6,
    rank: int = 0,
    ridge: float = 1e-8,
    n_derivatives: int = 1,
  ):
    """Initialize snapshot buffers and the polynomial-fallback ladder.

    :param history: Number of recent compute-step snapshots retained. Kept
        short on purpose: the feature dynamics are non-autonomous (the
        propagator drifts across timesteps), so a long window would average
        over changing dynamics.
    :param rank: SVD truncation rank for the DMD fit (0 = automatic).
    :param ridge: Tikhonov regulariser for the DMD fit.
    :param n_derivatives: Taylor orders kept for the warm-up fallback.
    """

    self.history = history
    self.rank = rank
    self.ridge = ridge
    self.n_derivatives = n_derivatives
    self.order = n_derivatives + 1
    self.current_step = -1
    self.last_non_approximated_step = -1
    self.snapshots: List[Tuple[int, torch.Tensor]] = []
    self.state: Dict[str, List[torch.Tensor]] = {
      "dY_prev": [None] * self.order,
      "dY_current": [None] * self.order,
    }

  def reset(self):
    """Reset snapshot history, fallback buffers, and step counters."""

    self.current_step = -1
    self.last_non_approximated_step = -1
    self.snapshots = []
    self.state = {
      "dY_prev": [None] * self.order,
      "dY_current": [None] * self.order,
    }

  def mark_step_begin(self):
    """Advance the logical step counter by one."""

    self.current_step += 1

  def _update_taylor(self, Y: torch.Tensor):
    """Maintain the TaylorSeer divided-difference ladder (warm-up fallback)."""

    dY_current: List[torch.Tensor] = [None] * self.order
    dY_current[0] = Y
    window = self.current_step - self.last_non_approximated_step
    if self.state["dY_prev"][0] is not None:
      if dY_current[0].shape != self.state["dY_prev"][0].shape:
        self.reset()
    for i in range(self.n_derivatives):
      if self.state["dY_prev"][i] is not None and self.current_step > 1 and window > 0:
        dY_current[i + 1] = (dY_current[i] - self.state["dY_prev"][i]) / window
      else:
        break
    self.state["dY_prev"] = self.state["dY_current"]
    self.state["dY_current"] = dY_current

  def _approximate_taylor(self) -> torch.Tensor:
    """Taylor-expansion fallback, identical to `TaylorSeerState.approximate`."""

    elapsed = self.current_step - self.last_non_approximated_step
    output = 0
    for i, derivative in enumerate(self.state["dY_current"]):
      if derivative is not None:
        output += (1 / math.factorial(i)) * derivative * (elapsed ** i)
      else:
        break
    return output

  def update(self, Y: torch.Tensor):
    """Commit a fully computed tensor as the newest snapshot / anchor.

    :param Y: Fully computed tensor to record at the current step.
    """

    if self.snapshots and self.snapshots[-1][1].shape != Y.shape:
      self.reset()
    self._update_taylor(Y)
    # detach().clone(): a bare detached view shares storage with the pipeline
    # tensor, so buffer-reusing inference (torch.compile / CUDA graphs) could
    # silently overwrite the snapshot history in place.
    self.snapshots.append((self.current_step, Y.detach().clone()))
    if len(self.snapshots) > self.history:
      del self.snapshots[: len(self.snapshots) - self.history]
    self.last_non_approximated_step = self.current_step

  def _uniform_tail(self) -> Tuple[List[torch.Tensor], int]:
    """Longest uniformly spaced suffix of the snapshot history.

    The DMD propagator advances exactly one snapshot-spacing per application,
    so a mixed-spacing window (e.g. across the warm-up boundary, or when the
    dynamic cache changes its compute cadence) would corrupt the fit.

    :returns: (velocities OLDEST..NEWEST, spacing) — empty list when degenerate.
    """

    if len(self.snapshots) < 2:
      return [], 0
    steps = [s for s, _ in self.snapshots]
    spacing = steps[-1] - steps[-2]
    if spacing <= 0:
      return [], 0
    tail = [self.snapshots[-1], self.snapshots[-2]]
    j = len(self.snapshots) - 2
    while j - 1 >= 0 and steps[j] - steps[j - 1] == spacing:
      tail.append(self.snapshots[j - 1])
      j -= 1
    return [v for _, v in reversed(tail)], spacing

  def approximate(self) -> torch.Tensor:
    """Forecast the tensor for the current step.

    Uses the exponential Dynamic Mode Decomposition (Prony) forecast on the
    uniformly spaced snapshot tail with a fractional horizon
    ``(current_step - last_compute_step) / spacing``. Below the 4-snapshot
    identifiability floor (a real trajectory spends two real degrees of
    freedom per complex pole, so even one oscillatory mode needs 3 snapshot
    pairs) it falls back to the Taylor expansion — DMD acts only where it is
    valid and the polynomial path covers warm-up.

    :returns: The forecast tensor for the current logical step.
    """

    vels, spacing = self._uniform_tail()
    if len(vels) >= 4:
      k = (self.current_step - self.snapshots[-1][0]) / spacing
      return _dmd_forecast(vels, k, rank=self.rank, ridge=self.ridge)
    return self._approximate_taylor()

  def step(self, Y: torch.Tensor):
    """Advance one step and return either the true tensor or its forecast.

    :param Y: Fully computed tensor when compute is required.
    :returns: The exact tensor or the DMD/Taylor forecast for this step.
    """

    self.mark_step_begin()
    if self.last_non_approximated_step < 0:
      self.update(Y)
      return Y
    return self.approximate()


class DMDCalibrator(CalibratorBase):
  """Calibrator that forecasts tensors with a Dynamic Mode Decomposition
  (Prony) exponential basis — drop-in alternative to `TaylorSeerCalibrator`."""

  def __init__(
    self,
    history: int = 6,
    rank: int = 0,
    ridge: float = 1e-8,
    n_derivatives: int = 1,
    **kwargs,
  ):
    """Create a calibrator whose states are keyed by logical tensor names.

    :param history: Snapshot window length retained per stream (5–6 typical).
    :param rank: SVD truncation rank for the DMD fit (0 = automatic).
    :param ridge: Tikhonov regulariser for the DMD fit.
    :param n_derivatives: Taylor orders for the warm-up fallback ladder.
    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    """

    self.history = history
    self.rank = rank
    self.ridge = ridge
    self.n_derivatives = n_derivatives
    self.states: Dict[str, DMDState] = {}
    self.reset_cache()

  def reset_cache(self):
    """Reset every tracked `DMDState` without dropping the key mapping."""

    if self.states:
      for state in self.states.values():
        state.reset()

  def maybe_init_state(
    self,
    name: str = "default",
  ):
    """Lazily create one DMD state for a named tensor stream.

    :param name: Logical tensor-stream name.
    """

    if name not in self.states:
      self.states[name] = DMDState(
        history=self.history,
        rank=self.rank,
        ridge=self.ridge,
        n_derivatives=self.n_derivatives,
      )

  def mark_step_begin(self, *args, **kwargs):
    """Advance every tracked state's step counter.

    :param args: Additional positional arguments forwarded to the underlying implementation.
    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    """

    if self.states:
      for state in self.states.values():
        state.mark_step_begin()

  def approximate(
    self,
    name: str = "default",
  ) -> torch.Tensor:
    """Forecast the next tensor for one named stream.

    :param name: Logical tensor-stream name.
    :returns: The forecast tensor for the named stream.
    """

    assert name in self.states, f"State '{name}' not found."
    state = self.states[name]
    return state.approximate()

  def update(
    self,
    Y: torch.Tensor,
    name: str = "default",
  ):
    """Feed a fully computed tensor into one named DMD state.

    :param Y: Fully computed tensor for the named stream.
    :param name: Logical tensor-stream name.
    """

    self.maybe_init_state(name)
    state = self.states[name]
    state.update(Y)

  def step(
    self,
    Y: torch.Tensor,
    name: str = "default",
  ):
    """Advance one named stream and return either computed or forecast output.

    :param Y: Fully computed tensor for the named stream when computation is required.
    :param name: Logical tensor-stream name.
    :returns: The exact tensor or DMD forecast for the named stream.
    """

    self.maybe_init_state(name)
    state = self.states[name]
    return state.step(Y)

  def __repr__(self):
    return f"DMDCalibrator_H({self.history})"
