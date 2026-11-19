# BeziCast (arXiv 2610.16448): Tikhonov-stabilized Bernstein/Bezier forecasting.
# Only the predictor is ported. DBCache retains its adaptive schedule and Fn/Bn
# granularity, and the cached object is Bn_residual rather than the paper's
# output-proximal representation.
#
# With real anchors Z_m at normalized coordinates u_m:
#   A[m,j] = C(n-1,j) * u_m**j * (1-u_m)**(n-1-j)
#   w(u) = b(u)^T @ (A^T A + ridge I)^-1 @ A^T
#   raw(u) = w(u)^T @ Z
# The guardrail computes risk = |sum(w)-1| + (sum(abs(w))-abs(sum(w))),
# projects w onto the probability simplex, and blends raw with that safe estimate.
# Predictions never become anchors. When fewer than n real anchors exist, the
# paper forces a full compute; CalibratorBase cannot change that decision, so the
# cache-dit adaptation explicitly reuses the latest real anchor and counts it.
import math
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import torch

from .base import CalibratorBase

from ....logger import init_logger

logger = init_logger(__name__)

MAX_BEZICAST_HISTORY = 6


def validate_bezicast_params(
  history: int,
  control_points: int,
  ridge: float,
  guardrail: bool,
  u_max: float,
  beta_max: float,
  rho: Optional[float],
  gamma: float,
):
  """Validate BeziCast history, fitting, clipping, and guardrail parameters.

  :param history: Maximum number of real anchors, in [2, 6].
  :param control_points: Number of Bernstein control points, in [2, history].
  :param ridge: Non-negative finite Tikhonov coefficient.
  :param guardrail: Whether to blend toward simplex-projected weights.
  :param u_max: Finite normalized query cap, at least 1.
  :param beta_max: Maximum guardrail blending strength, in [0, 1].
  :param rho: Risk activation threshold, or None for the measured risk at u=1.
  :param gamma: Positive finite guardrail transition width.
  """

  if type(history) is not int or not 2 <= history <= MAX_BEZICAST_HISTORY:
    raise ValueError(f"history must be an integer in [2, {MAX_BEZICAST_HISTORY}], got {history}.")
  if type(control_points) is not int or not 2 <= control_points <= history:
    raise ValueError(f"control_points must be an integer in [2, history], got {control_points}.")
  if type(ridge) not in (int, float) or not math.isfinite(ridge) or ridge < 0.0:
    raise ValueError(f"ridge must be finite and non-negative, got {ridge}.")
  if type(guardrail) is not bool:
    raise ValueError(f"guardrail must be bool, got {type(guardrail).__name__}.")
  if type(u_max) not in (int, float) or not math.isfinite(u_max) or u_max < 1.0:
    raise ValueError(f"u_max must be finite and at least 1, got {u_max}.")
  if type(beta_max) not in (int,
                            float) or not math.isfinite(beta_max) or not 0.0 <= beta_max <= 1.0:
    raise ValueError(f"beta_max must be finite and in [0, 1], got {beta_max}.")
  if rho is not None and (type(rho) not in (int, float) or not math.isfinite(rho) or rho < 0.0):
    raise ValueError(f"rho must be None or finite and non-negative, got {rho}.")
  if type(gamma) not in (int, float) or not math.isfinite(gamma) or gamma <= 0.0:
    raise ValueError(f"gamma must be finite and positive, got {gamma}.")


def bernstein_basis(u: float, control_points: int) -> torch.Tensor:
  """Evaluate the degree-(n-1) Bernstein basis at a normalized coordinate.

  :param u: Query coordinate; values outside [0, 1] perform extrapolation.
  :param control_points: Number of basis functions.
  :returns: A CPU float64 vector of length `control_points`.
  """

  degree = control_points - 1
  values = [
    math.comb(degree, j) * (u ** j) * ((1.0 - u) ** (degree - j)) for j in range(control_points)
  ]
  return torch.tensor(values, dtype=torch.float64)


def equivalent_weight_risk(weights: torch.Tensor) -> float:
  """Compute BeziCast Eq.15 risk from equivalent historical weights.

  :param weights: One-dimensional equivalent-weight vector.
  :returns: Non-negative affine/sign risk score.
  """

  weight_sum = float(weights.sum())
  return abs(weight_sum - 1.0) + (float(weights.abs().sum()) - abs(weight_sum))


def project_simplex(weights: torch.Tensor) -> torch.Tensor:
  """Project a vector onto the probability simplex by the sorted Euclidean algorithm.

  :param weights: One-dimensional CPU float64 vector.
  :returns: Non-negative weights summing to one.
  """

  ordered = torch.sort(weights, descending=True).values
  cumulative = torch.cumsum(ordered, dim=0) - 1.0
  indices = torch.arange(1, len(weights) + 1, dtype=torch.float64)
  valid = ordered - cumulative / indices > 0.0
  last = int(torch.nonzero(valid)[-1, 0])
  theta = cumulative[last] / float(last + 1)
  return torch.clamp(weights - theta, min=0.0)


class BeziCastState:
  """Per-stream real-anchor FIFO and BeziCast forecasting state."""

  def __init__(
    self,
    history: int = 6,
    control_points: int = 4,
    ridge: float = 0.5,
    guardrail: bool = False,
    u_max: float = 1.5,
    beta_max: float = 1.0,
    rho: Optional[float] = None,
    gamma: float = 1.0,
    epsilon: float = 1e-8,
  ):
    """Initialize an empty BeziCast stream.

    :param history: Maximum number of real anchors, in [2, 6].
    :param control_points: Number of Bernstein control points, in [2, history].
    :param ridge: Non-negative Tikhonov coefficient; 0 selects interpolation by pinv.
    :param guardrail: Whether to apply the equivalent-weight simplex guardrail.
    :param u_max: Normalized extrapolation cap; 1.5 is a cache-dit conservative default.
    :param beta_max: Maximum guardrail blending strength, in [0, 1].
    :param rho: Risk threshold; None computes the current design's risk at u=1.
    :param gamma: Positive guardrail transition width.
    :param epsilon: Small denominator offset from Eq.6/7 for logical-step coordinates.
    """

    validate_bezicast_params(history, control_points, ridge, guardrail, u_max, beta_max, rho, gamma)
    self.history = history
    self.control_points = control_points
    self.ridge = float(ridge)
    self.guardrail = guardrail
    self.u_max = float(u_max)
    self.beta_max = float(beta_max)
    self.rho = rho
    self.gamma = float(gamma)
    self.epsilon = float(epsilon)
    self.anchors: Deque[Tuple[int, torch.Tensor]] = deque(maxlen=history)
    self.current_step = -1
    self.num_forecasts = 0
    self.num_warmup_reuses = 0
    self.num_numerical_fallbacks = 0
    self.num_guardrail_calls = 0
    self.num_query_clips = 0
    self.guardrail_beta_sum = 0.0
    self.last_risk = 0.0
    self.last_beta = 0.0
    self._warned = False

  def reset(self):
    """Clear anchors, logical time, and all diagnostics for a new inference session."""

    self.anchors.clear()
    self.current_step = -1
    self.num_forecasts = 0
    self.num_warmup_reuses = 0
    self.num_numerical_fallbacks = 0
    self.num_guardrail_calls = 0
    self.num_query_clips = 0
    self.guardrail_beta_sum = 0.0
    self.last_risk = 0.0
    self.last_beta = 0.0
    self._warned = False

  def mark_step_begin(self):
    """Advance the logical step counter by one."""

    self.current_step += 1

  def _numerical_fallback(self) -> torch.Tensor:
    """Reuse the latest real anchor and expose a numerical failure through state and logs."""

    self.num_numerical_fallbacks += 1
    if not self._warned:
      logger.warning("BeziCast encountered a numerical failure while fitting equivalent weights; "
                     "falling back to the latest real anchor. Subsequent fallbacks are counted "
                     "without additional warnings.")
      self._warned = True
    return self.anchors[-1][1]

  def _weights_for_query(self, design: torch.Tensor, u: float) -> torch.Tensor:
    """Evaluate Eq.13 equivalent weights for one normalized query."""

    basis = bernstein_basis(u, self.control_points)
    if self.ridge == 0.0:
      return basis @ torch.linalg.pinv(design)
    system = design.T @ design + self.ridge * torch.eye(self.control_points, dtype=torch.float64)
    return basis @ torch.linalg.solve(system, design.T)

  def _aggregate(self, weights: torch.Tensor) -> torch.Tensor:
    """Apply scalar historical weights with fp32 accumulation for low-precision anchors."""

    latest = self.anchors[-1][1]
    accumulation_dtype = (torch.float32 if latest.dtype in (torch.float16,
                                                            torch.bfloat16) else latest.dtype)
    output = None
    for weight, (_, anchor) in zip(weights, self.anchors):
      value = anchor.to(accumulation_dtype) * float(weight)
      output = value if output is None else output + value
    return output

  def approximate(self) -> torch.Tensor:
    """Forecast the current step using BeziCast Eq.6-21 without mutating the FIFO.

    :returns: A tensor with the latest real anchor's shape and dtype.
    """

    if not self.anchors:
      raise RuntimeError("BeziCastState.approximate() called before any update().")
    latest_step, latest = self.anchors[-1]
    if self.current_step <= latest_step:
      raise RuntimeError("BeziCast approximation target must be later than the latest real anchor.")
    if len(self.anchors) < self.control_points:
      self.num_warmup_reuses += 1
      return latest

    steps = [step for step, _ in self.anchors]
    step_first, step_last = float(steps[0]), float(steps[-1])
    span = step_last - step_first + self.epsilon
    coordinates = [(float(step) - step_first) / span for step in steps]
    design = torch.stack([bernstein_basis(u, self.control_points) for u in coordinates])
    raw_query = (float(self.current_step) - step_first) / span
    query = min(raw_query, self.u_max)
    if raw_query > self.u_max:
      self.num_query_clips += 1

    try:
      weights = self._weights_for_query(design, query)
      if not bool(torch.isfinite(weights).all()):
        raise RuntimeError("Non-finite BeziCast equivalent weights.")
      rho = self.rho
      if self.guardrail and rho is None:
        edge_weights = self._weights_for_query(design, 1.0)
        if not bool(torch.isfinite(edge_weights).all()):
          raise RuntimeError("Non-finite BeziCast window-edge weights.")
        rho = equivalent_weight_risk(edge_weights)
    except RuntimeError:
      return self._numerical_fallback()

    # Only scalar CPU fitting is guarded. Tensor OOM/device errors must propagate.
    risk = equivalent_weight_risk(weights)
    beta = (self.beta_max * min(max(
      (risk - rho) / self.gamma, 0.0), 1.0) if self.guardrail else 0.0)
    raw = self._aggregate(weights)
    if beta > 0.0:
      safe_weights = project_simplex(weights)
      safe = self._aggregate(safe_weights)
      prediction = (1.0 - beta) * raw + beta * safe
      self.num_guardrail_calls += 1
      self.guardrail_beta_sum += beta
    else:
      prediction = raw

    self.num_forecasts += 1
    self.last_risk = risk
    self.last_beta = beta
    return prediction.to(latest.dtype)

  def update(self, Y: torch.Tensor):
    """Append one full-compute tensor as a real anchor, evicting the oldest if full.

    :param Y: Fully computed residual or hidden-state tensor for the current stream.
    """

    if self.anchors:
      latest = self.anchors[-1][1]
      if Y.shape != latest.shape or Y.dtype != latest.dtype or Y.device != latest.device:
        self.anchors.clear()
    self.anchors.append((self.current_step, Y.detach().clone()))

  def step(self, Y: torch.Tensor) -> torch.Tensor:
    """Record a cold-start anchor or return the current BeziCast forecast.

    :param Y: Fully computed tensor used to initialize an empty stream.
    :returns: The exact cold-start tensor or the current forecast.
    """

    self.mark_step_begin()
    if not self.anchors:
      self.update(Y)
      return Y
    return self.approximate()


class BeziCastCalibrator(CalibratorBase):
  """Calibrator that forecasts residuals with ridge-stabilized Bezier trajectories."""

  def __init__(
    self,
    history: int = 6,
    control_points: int = 4,
    ridge: float = 0.5,
    guardrail: bool = False,
    u_max: float = 1.5,
    beta_max: float = 1.0,
    rho: Optional[float] = None,
    gamma: float = 1.0,
    **kwargs,
  ):
    """Create a BeziCast calibrator whose states are keyed by logical stream names.

    :param history: Maximum number of real anchors, in [2, 6].
    :param control_points: Number of Bernstein control points, in [2, history].
    :param ridge: Non-negative Tikhonov coefficient.
    :param guardrail: Whether to apply the equivalent-weight simplex guardrail.
    :param u_max: Normalized extrapolation cap.
    :param beta_max: Maximum guardrail blending strength, in [0, 1].
    :param rho: Risk threshold, or None for automatic window-edge calibration.
    :param gamma: Positive guardrail transition width.
    :param kwargs: Additional keyword arguments accepted for config compatibility.
    """

    validate_bezicast_params(history, control_points, ridge, guardrail, u_max, beta_max, rho, gamma)
    self.history = history
    self.control_points = control_points
    self.ridge = float(ridge)
    self.guardrail = guardrail
    self.u_max = float(u_max)
    self.beta_max = float(beta_max)
    self.rho = rho
    self.gamma = float(gamma)
    self.states: Dict[str, BeziCastState] = {}
    self.reset_cache()

  def _total(self, attribute: str) -> int:
    return sum(getattr(state, attribute) for state in self.states.values())

  @property
  def num_forecasts(self) -> int:
    """Return the number of actual Bezier forecasts across all streams."""

    return self._total("num_forecasts")

  @property
  def num_warmup_reuses(self) -> int:
    """Return forecasts replaced by reuse because fewer than n anchors were available."""

    return self._total("num_warmup_reuses")

  @property
  def num_numerical_fallbacks(self) -> int:
    """Return numerical-fitting failures replaced by the latest real anchor."""

    return self._total("num_numerical_fallbacks")

  @property
  def num_guardrail_calls(self) -> int:
    """Return forecasts with a positive simplex-guardrail blending coefficient."""

    return self._total("num_guardrail_calls")

  @property
  def num_query_clips(self) -> int:
    """Return forecasts whose normalized query was clipped to `u_max`."""

    return self._total("num_query_clips")

  @property
  def mean_guardrail_beta(self) -> float:
    """Return mean beta over forecasts that actually used the guardrail."""

    total = self._total("guardrail_beta_sum")
    return total / self.num_guardrail_calls if self.num_guardrail_calls else 0.0

  @property
  def max_last_risk(self) -> float:
    """Return the maximum latest risk among active streams."""

    return max((state.last_risk for state in self.states.values()), default=0.0)

  def reset_cache(self):
    """Reset every tracked BeziCast state without dropping the stream mapping."""

    for state in self.states.values():
      state.reset()

  def maybe_init_state(self, name: str = "default"):
    """Lazily create one BeziCast state for a named tensor stream.

    :param name: Logical tensor-stream name.
    """

    if name not in self.states:
      self.states[name] = BeziCastState(
        history=self.history,
        control_points=self.control_points,
        ridge=self.ridge,
        guardrail=self.guardrail,
        u_max=self.u_max,
        beta_max=self.beta_max,
        rho=self.rho,
        gamma=self.gamma,
      )

  def mark_step_begin(self, *args, **kwargs):
    """Advance every tracked state's logical step counter.

    :param args: Additional positional arguments accepted for interface compatibility.
    :param kwargs: Additional keyword arguments accepted for interface compatibility.
    """

    for state in self.states.values():
      state.mark_step_begin()

  def approximate(self, name: str = "default") -> torch.Tensor:
    """Forecast the current tensor for one named stream without mutating its FIFO.

    :param name: Logical tensor-stream name.
    :returns: The BeziCast forecast for the current logical step.
    """

    assert name in self.states, f"State '{name}' not found."
    return self.states[name].approximate()

  def update(self, Y: torch.Tensor, name: str = "default"):
    """Feed one fully computed tensor into a named stream's real-anchor FIFO.

    :param Y: Fully computed residual or hidden-state tensor.
    :param name: Logical tensor-stream name.
    """

    self.maybe_init_state(name)
    self.states[name].update(Y)

  def step(self, Y: torch.Tensor, name: str = "default") -> torch.Tensor:
    """Advance one named stream and return its exact cold start or BeziCast forecast.

    :param Y: Fully computed tensor used to initialize an empty stream.
    :param name: Logical tensor-stream name.
    :returns: The exact cold-start tensor or current forecast.
    """

    self.maybe_init_state(name)
    return self.states[name].step(Y)

  def __repr__(self):
    return (f"BeziCastCalibrator_K({self.history}, n={self.control_points}, "
            f"ridge={self.ridge}, guardrail={self.guardrail})")
