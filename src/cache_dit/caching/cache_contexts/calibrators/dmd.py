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


def _dmd_fit_one(
  traj: torch.Tensor,
  rank: int = 0,
  ridge: float = 1e-8,
):
  """Fit the DMD eigendecomposition for ONE ``[d, n]`` trajectory (a single batch item's snapshot
  history, columns OLDEST..NEWEST).

  Identify the linear propagator ``A`` from the snapshot pairs
  (``Y_{t+1} ~= A Y_t``) through one economy SVD and eigendecompose it once. The
  fit is horizon-free, so the caller caches it and advances it cheaply (see
  :func:`_dmd_eval`).

  :param rank: SVD truncation rank; 0 selects it from the spectrum (drop modes
      below 1e-4 of the leading singular value — this is what rejects noise).
  :param ridge: Tikhonov term added to the inverted singular values.
  :returns: ``(Phi, evals, b)`` or ``None`` on a degenerate fit (caller reuses
      the last value).
  """

  X, Xp = traj[:, :-1], traj[:, 1:]
  try:
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
  except Exception:  # noqa: BLE001 — degenerate fit: caller falls back to reuse
    return None
  r = rank
  if r <= 0:
    r = int((S > S[0] * 1e-4).sum().clamp(min=1).item())
  r = max(1, min(r, S.numel()))
  Ur, Sr, Vr = U[:, :r], S[:r], Vh[:r].mH
  Sinv = (1.0 / (Sr + ridge)).to(torch.complex128)
  Atil = (Ur.mH @ Xp @ Vr).to(torch.complex128) * Sinv.unsqueeze(0)
  try:
    evals, W = torch.linalg.eig(Atil)
    Phi = ((Xp @ Vr).to(torch.complex128) * Sinv.unsqueeze(0)) @ W
    b = torch.linalg.lstsq(Phi, traj[:, -1].to(torch.complex128).unsqueeze(1)).solution.squeeze(1)
  except Exception:  # noqa: BLE001 — degenerate fit: caller falls back to reuse
    return None
  return (Phi, evals, b)


def _dmd_fit(
  snapshots: List[torch.Tensor],
  rank: int = 0,
  ridge: float = 1e-8,
):
  """Fit DMD once for a window of >= 4 same-shape snapshots, INDEPENDENTLY per batch item (axis 0).

  Flattening the whole tensor into a single state (the pre-fix behaviour) folds
  the batch dimension into one DMD fit, so one prompt's forecast would depend on
  the other prompts in the same batch — unlike the elementwise Taylor path.
  Fitting per batch item keeps them independent. The fit is horizon-free, so
  :class:`DMDState` caches the returned object and reuses it for every skip step
  until a new snapshot arrives (one SVD/eig per window, not per skipped step).

  :returns: ``(per_item_fits, shape, dtype)``, or ``None`` when the window is
      too short. ``per_item_fits[i]`` is ``None`` for a degenerate batch item.
  """

  if len(snapshots) < 4:
    return None
  newest = snapshots[-1]
  shp, dt = newest.shape, newest.dtype
  bsz = shp[0] if newest.dim() > 1 else 1
  # (B, d, n): per-item trajectories; B == 1 reproduces the un-batched fit.
  V = torch.stack([s.reshape(bsz, -1) for s in snapshots], dim=-1).to(torch.float64)
  fits = [_dmd_fit_one(V[i], rank=rank, ridge=ridge) for i in range(bsz)]
  return (fits, shp, dt)


def _dmd_eval(fit, k: float):
  """Advance a cached :func:`_dmd_fit` to (fractional) horizon ``k`` by eigenvalue powers —
  ``Y_{t+k} ~= Phi @ (lambda**k * b)`` — one cheap evaluation per batch item, no re-decomposition.

  :returns: The forecast tensor of the original snapshot shape, or ``None`` when
      any batch item is degenerate, or when the result is non-finite AFTER the
      output-dtype cast. The finite check is deliberately post-cast: a finite
      float64 forecast can still overflow to ``inf`` in fp16, which the caller
      must catch and fall back from rather than feed downstream.
  """

  fits, shp, dt = fit
  rows = []
  for f in fits:
    if f is None:
      return None
    Phi, evals, b = f
    rows.append((Phi @ (evals.pow(float(k)) * b)).real)
  pred = torch.stack(rows, dim=0) if len(shp) > 1 else rows[0]
  out = pred.to(dt).reshape(shp)
  if not torch.isfinite(out).all():
    return None
  return out


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
    # Cached horizon-free DMD fit + the window key it was fitted on, so skip
    # steps reuse one SVD/eig instead of recomputing it every step.
    self._fit = None
    self._fit_key = None
    self.state: Dict[str, List[torch.Tensor]] = {
      "dY_prev": [None] * self.order,
      "dY_current": [None] * self.order,
    }

  def reset(self):
    """Reset snapshot history, fallback buffers, and step counters."""

    self.current_step = -1
    self.last_non_approximated_step = -1
    self.snapshots = []
    self._fit = None
    self._fit_key = None
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
      del self.snapshots[:len(self.snapshots) - self.history]
    self.last_non_approximated_step = self.current_step
    # A new snapshot changes the window, so the cached fit is stale.
    self._fit_key = None

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

    The eigendecomposition depends only on the snapshot window, which cannot
    change between two skip steps, so it is fitted once per window and cached;
    each skip step only re-advances the cheap ``lambda**k`` horizon. A
    degenerate fit or non-finite forecast reuses the newest snapshot.

    :returns: The forecast tensor for the current logical step.
    """

    vels, spacing = self._uniform_tail()
    if len(vels) >= 4:
      k = (self.current_step - self.snapshots[-1][0]) / spacing
      key = (self.snapshots[-1][0], len(vels), spacing)
      if self._fit_key != key:
        self._fit = _dmd_fit(vels, rank=self.rank, ridge=self.ridge)
        self._fit_key = key
      if self._fit is not None:
        pred = _dmd_eval(self._fit, k)
        if pred is not None:
          return pred
      return self.snapshots[-1][1].clone()
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
  """Calibrator that forecasts tensors with a Dynamic Mode Decomposition (Prony) exponential basis —
  drop-in alternative to `TaylorSeerCalibrator`."""

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
