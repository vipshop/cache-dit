# FoCa (Forecast then Calibrate) calibrator — an ODE-based predictor-corrector
# alternative to the TaylorSeer polynomial forecaster.
#
# FoCa: Forecast then Calibrate: Feature Caching as ODE for Efficient Diffusion
# Transformers (Zheng et al., 2025, https://arxiv.org/abs/2508.16211).
#
# Why an ODE solver: across denoising steps, the hidden feature evolution follows
# a near-linear ODE dF/dt = g_θ(F, t). While g_θ is not directly solvable,
# classical linear multi-step integration methods — which only depend on cached
# historical values — can integrate this ODE stably. FoCa pairs a BDF2 predictor
# (2nd-order backward differentiation formula) with a Heun corrector (explicit
# trapezoidal rule) to achieve stable long-skip prediction.
#
# ============================================================
# KEY DIFFERENCE from TaylorSeer — recursive vs. anchored prediction
# ============================================================
#
#   TaylorSeer: always predicts from the MOST RECENT FULL-COMPUTE anchor.
#     F_pred(t+k) = F(t) + dF/dt * k + d²F/dt² * k²/2! + ...
#     Polynomial extrapolation from a single anchor → diverges at large k.
#
#   FoCA: predicts RECURSIVELY using the TWO MOST RECENT values (full-compute
#     OR cached predictions), then CORRECTS with the most recent full-compute
#     anchor to prevent drift.
#
#     Step 0(FC) → F0          Step 1(FC) → F1
#     Step 2(cache): BDF2(F1, F0) → P2, Heun(P2, anchor=F1)
#     Step 3(cache): BDF2(P2, F1) → P3, Heun(P3, anchor=F1)  ← recursive!
#     Step 4(FC)   → F4 (new anchor)
#     Step 5(cache): BDF2(F4, P3) → P5, Heun(P5, anchor=F4)
#
#   This recursive approach better preserves local smoothness across adjacent
#   cached steps. The Heun corrector prevents error accumulation by anchoring
#   the prediction to the most recent full-compute step's derivative.
#
# ============================================================
# Relationship with DBCache — a drop-in forecast layer
# ============================================================
#
#   DBCache's residual fusion formula is fixed (apply_cache):
#       hidden_states = Bn_residual(t) + Fn_output(t+1)
#   The calibrator only forecasts the LEFT operand (Bn_residual from step t).
#   The RIGHT operand, Fn_output(t+1), is ALWAYS freshly computed by the Fn
#   blocks — this is how current-step information is injected.
#   Without a calibrator, Bn_residual(t) is the stale value from the last
#   full-compute step. With FoCaCalibrator, it is replaced by the FoCA
#   forecast of what the Bn residual should be at the CURRENT step.
#   The calibrator does not touch DBCache's threshold logic, diff tracking,
#   or accumulation guards; it only provides a better `hidden_states_prev`
#   via get_Bn_buffer() → calibrator.approximate().
#
# ============================================================
# Core algorithm (simplified for cache-dit's uniform step size h=1)
# ============================================================
#
#   State per stream (4 tensors + 3 ints):
#     F_k, F_km1      — rolling window of the two most recent values
#                        (updated on EVERY step: full-compute or cache)
#     F_full, F_full_prev — anchors from the two most recent FULL-COMPUTE steps
#                        (updated only on full-compute steps via update())
#
#   BDF2 predictor (h=1 simplified):
#     F_bdf2 = 2 * F_k - F_km1
#     (linear extrapolation from the last two values)
#
#   Heun corrector (drift-suppression term):
#     deriv_full = F_full - F_full_prev   # reliable derivative at full-compute anchor
#     deriv_curr = F_k - F_km1           # local derivative (may contain cache drift)
#     F_heun = F_bdf2 + 0.5 * (deriv_full - deriv_curr)
#
#   When the cached predictions follow the same trend as the full-compute
#   anchors (deriv_curr ≈ deriv_full), the correction term is zero and
#   Heun ≡ BDF2. When cache drift causes deriv_curr to deviate, the Heun
#   term non-zero pulls the prediction back toward the anchor's trajectory.
#
#   After approximate() returns, F_k/F_km1 are advanced with the predicted
#   value, enabling recursive prediction on the next cache step.
#
# ============================================================
# Calling chain
# ============================================================
#
#   cache_manager.get_Bn_buffer()          # cache hit → need forecast
#     → FoCaCalibrator.approximate()
#       → FoCaState.approximate()
#         → BDF2: 2 * F_k - F_km1
#         → Heun: + 0.5 * (deriv_full - deriv_curr)
#         → advance F_k, F_km1 with prediction
#
#   cache_manager.set_Bn_buffer()          # full-compute step → store truth
#     → FoCaCalibrator.update()
#       → FoCaState.update()
#         → advance F_full, F_full_prev anchors
#         → advance F_k, F_km1 rolling window

from typing import Dict

import torch

from .base import CalibratorBase

from ....logger import init_logger

logger = init_logger(__name__)


class FoCaState:
  """Per-stream BDF2+Heun forecast state for one named calibration stream.

  Stores the two most recent feature values (full-compute or cached) for the recursive BDF2
  predictor and the two most recent full-compute anchors for the Heun drift corrector.
  """

  def __init__(self):
    self.current_step = -1
    self.last_full_step = -1
    self.prev_full_step = -1
    self.F_k: torch.Tensor | None = None
    self.F_km1: torch.Tensor | None = None
    self.F_full: torch.Tensor | None = None
    self.F_full_prev: torch.Tensor | None = None

  def reset(self):
    """Clear all stored tensors and step counters."""

    self.current_step = -1
    self.last_full_step = -1
    self.prev_full_step = -1
    self.F_k = None
    self.F_km1 = None
    self.F_full = None
    self.F_full_prev = None

  def mark_step_begin(self):
    """Advance the logical step counter by one."""

    self.current_step += 1

  def _check_shape(self, Y: torch.Tensor):
    """Reset state if the incoming tensor shape differs from the stored one.

    :param Y: Incoming tensor to compare against stored F_k.
    """

    if self.F_k is not None and Y.shape != self.F_k.shape:
      self.reset()

  def update(self, Y: torch.Tensor):
    """Commit a fully computed tensor as the newest full-compute anchor.

    Advances both the full-compute anchor pair (F_full, F_full_prev) and the rolling window (F_k,
    F_km1).  The rolling window is replaced entirely with the fresh full-compute value so that the
    next BDF2 step starts from a clean anchor.

    :param Y: Fully computed tensor to record at the current step.
    """

    self._check_shape(Y)
    self.prev_full_step = self.last_full_step
    self.last_full_step = self.current_step
    self.F_full_prev = self.F_full
    self.F_full = Y.detach().clone()
    self.F_km1 = self.F_k
    self.F_k = Y.detach().clone()

  def approximate(self) -> torch.Tensor:
    """Forecast the tensor for the current step using BDF2 + Heun.

    When fewer than two rolling-window values are available the method falls back to reusing the
    most recent value.  After the forecast the rolling window is advanced so that the next cache
    step can recurse.

    :returns: The forecast tensor for the current logical step.
    """

    if self.F_k is None:
      raise RuntimeError("FoCaState.approximate() called before any update().")
    if self.F_km1 is None:
      return self.F_k.clone()

    F_bdf2 = 2.0 * self.F_k - self.F_km1
    if self.F_full_prev is not None:
      deriv_full = self.F_full - self.F_full_prev
      deriv_curr = self.F_k - self.F_km1
      F_heun = F_bdf2 + 0.5 * (deriv_full - deriv_curr)
    else:
      F_heun = F_bdf2

    self.F_km1 = self.F_k
    self.F_k = F_heun.detach().clone()
    return F_heun

  def step(self, Y: torch.Tensor) -> torch.Tensor:
    """Advance one step and return either the true tensor or its forecast.

    The first step (cold start) always records a full-compute anchor;
    every subsequent step returns a forecast via :meth:`approximate`.

    :param Y: Fully computed tensor when compute is required.
    :returns: The exact tensor (first step) or the FoCA forecast.
    """

    self.mark_step_begin()
    if self.last_full_step < 0:
      self.update(Y)
      return Y
    return self.approximate()


class FoCaCalibrator(CalibratorBase):
  """Calibrator that forecasts tensors with a BDF2 predictor + Heun corrector — drop-in alternative
  to `TaylorSeerCalibrator` and `DMDCalibrator`."""

  def __init__(self, **kwargs):
    """Create a calibrator whose states are keyed by logical tensor names.

    FoCA has no hyper-parameters — the step intervals are derived automatically from the step
    counters.  Any extra keyword arguments are silently accepted for forward compatibility.

    :param kwargs: Ignored; accepted for config-to-kwargs compatibility.
    """

    self.states: Dict[str, FoCaState] = {}
    self.reset_cache()

  def reset_cache(self):
    """Reset every tracked `FoCaState` without dropping the key mapping."""

    if self.states:
      for state in self.states.values():
        state.reset()

  def maybe_init_state(self, name: str = "default"):
    """Lazily create one FoCa state for a named tensor stream.

    :param name: Logical tensor-stream name.
    """

    if name not in self.states:
      self.states[name] = FoCaState()

  def mark_step_begin(self, *args, **kwargs):
    """Advance every tracked state's step counter.

    :param args: Additional positional arguments forwarded to the underlying implementation.
    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    """

    if self.states:
      for state in self.states.values():
        state.mark_step_begin()

  def approximate(self, name: str = "default") -> torch.Tensor:
    """Forecast the next tensor for one named stream.

    :param name: Logical tensor-stream name.
    :returns: The forecast tensor for the named stream.
    """

    assert name in self.states, f"State '{name}' not found."
    state = self.states[name]
    return state.approximate()

  def update(self, Y: torch.Tensor, name: str = "default"):
    """Feed a fully computed tensor into one named FoCa state.

    :param Y: Fully computed tensor for the named stream.
    :param name: Logical tensor-stream name.
    """

    self.maybe_init_state(name)
    state = self.states[name]
    state.update(Y)

  def step(self, Y: torch.Tensor, name: str = "default") -> torch.Tensor:
    """Advance one named stream and return either computed or forecast output.

    :param Y: Fully computed tensor for the named stream when computation is required.
    :param name: Logical tensor-stream name.
    :returns: The exact tensor or FoCA forecast for the named stream.
    """

    self.maybe_init_state(name)
    state = self.states[name]
    return state.step(Y)

  def __repr__(self):
    return "FoCaCalibrator"
