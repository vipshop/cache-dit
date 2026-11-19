# BRACE (arXiv 2608.07572): Barycentric Rational Forecasting with Chebyshev
# Enhancement. Only the predictor is ported to cache-dit. DBCache retains its
# adaptive schedule and Fn/Bn granularity, and the cached object is Bn_residual
# rather than the paper's per-layer full feature.
#
# For real full-compute anchors (F_j, tau_j), ordered from oldest to newest:
#   x(tau) = 2 * (tau - tau_min) / (tau_max - tau_min) - 1
#   F_pred = sum_j(w_j * F_j / (x_pred - x_j)) / sum_j(w_j / (x_pred - x_j))
#   w_j = (-1)^(j + 1) * delta_j
# where delta_1 = 1/2, delta_C = gamma, and the other delta_j = 1. The weights
# use the effective number of available anchors while the FIFO is filling.
# With gamma != 0.5 these adapted weights intentionally do not reproduce exact
# linear extrapolation; gamma controls the boundary sensitivity instead.
#
# Logical step indices replace scheduler timesteps because CalibratorBase does
# not receive the latter. This preserves the canonical extrapolation direction,
# including non-uniform DBCache anchor spacing. Predictions are never appended
# to the FIFO: unlike FoCa, BRACE is non-recursive and approximate() is pure.
import math
import sys
from collections import deque
from typing import Deque, Dict, List, Tuple

import torch

from .base import CalibratorBase

from ....logger import init_logger

logger = init_logger(__name__)

MAX_BRACE_CAPACITY = 3


def validate_brace_params(capacity: int, gamma: float):
  """Validate BRACE window and boundary-sensitivity parameters.

  :param capacity: Maximum number of real anchors retained per stream, in [1, 3].
  :param gamma: Finite terminal weight in [0.5, infinity); smaller values admit extrapolation poles.
  """

  if type(capacity) is not int or not 1 <= capacity <= MAX_BRACE_CAPACITY:
    raise ValueError(f"capacity must be an integer in [1, {MAX_BRACE_CAPACITY}], got {capacity}.")
  if type(gamma) not in (int, float) or not math.isfinite(gamma) or gamma < 0.5:
    raise ValueError(f"gamma must be finite and in [0.5, infinity), got {gamma}.")


class BRACEState:
  """Per-stream FIFO of real anchors for barycentric rational forecasting."""

  def __init__(self, capacity: int = 3, gamma: float = 0.7):
    """Initialize an empty BRACE anchor window.

    :param capacity: Maximum number of real anchors retained per stream, in [1, 3].
    :param gamma: Finite terminal weight in [0.5, infinity); smaller values admit extrapolation
        poles.
    """

    validate_brace_params(capacity, gamma)
    self.capacity = capacity
    self.gamma = float(gamma)
    self.anchors: Deque[Tuple[int, torch.Tensor]] = deque(maxlen=capacity)
    self.current_step = -1
    self.num_fallbacks = 0
    self._warned = False

  def reset(self):
    """Clear all anchors, counters, and numerical-fallback diagnostics."""

    self.anchors.clear()
    self.current_step = -1
    self.num_fallbacks = 0
    self._warned = False

  def mark_step_begin(self):
    """Advance the logical step counter by one."""

    self.current_step += 1

  def _weights(self) -> List[float]:
    """Return adapted Chebyshev weights for the currently available anchors.

    :returns: Alternating-sign weights whose terminal magnitude is `gamma`.
    """

    count = len(self.anchors)
    weights = []
    for j in range(count):
      if j == 0:
        delta = 0.5
      elif j == count - 1:
        delta = self.gamma
      else:
        delta = 1.0
      weights.append(-delta if j % 2 == 0 else delta)
    return weights

  def _fallback(self) -> torch.Tensor:
    """Return the latest real anchor after recording a numerical failure.

    :returns: The newest real anchor without applying a rational forecast.
    """

    self.num_fallbacks += 1
    if not self._warned:
      logger.warning("BRACE encountered a near-zero or non-finite barycentric denominator; "
                     "falling back to the latest real anchor. Subsequent fallbacks are counted "
                     "without additional warnings.")
      self._warned = True
    return self.anchors[-1][1]

  def approximate(self) -> torch.Tensor:
    """Forecast the current-step tensor from real anchors using BRACE Eq.6-8.

    :returns: A tensor with the latest anchor's shape and dtype.
    """

    if not self.anchors:
      raise RuntimeError("BRACEState.approximate() called before any update().")
    latest_step, latest = self.anchors[-1]
    if self.current_step <= latest_step:
      raise RuntimeError("BRACE approximation target must be later than the latest real anchor.")
    if len(self.anchors) == 1:
      return latest

    steps = [step for step, _ in self.anchors]
    step_min, step_max = float(steps[0]), float(steps[-1])
    if step_max == step_min:
      raise RuntimeError("BRACE anchors must have distinct logical steps.")
    x_nodes = [2.0 * (float(step) - step_min) / (step_max - step_min) - 1.0 for step in steps]
    x_pred = 2.0 * (float(self.current_step) - step_min) / (step_max - step_min) - 1.0
    if any(x_pred == x_node for x_node in x_nodes):
      raise RuntimeError("BRACE approximation target coincides with a real anchor.")

    terms = [weight / (x_pred - x_node) for weight, x_node in zip(self._weights(), x_nodes)]
    denominator = math.fsum(terms)
    scale = math.fsum(abs(term) for term in terms)
    if not math.isfinite(denominator) or abs(denominator) <= sys.float_info.epsilon * scale:
      return self._fallback()

    accumulation_dtype = (torch.float32 if latest.dtype in (torch.float16,
                                                            torch.bfloat16) else latest.dtype)
    numerator = None
    for term, (_, anchor) in zip(terms, self.anchors):
      value = anchor.to(accumulation_dtype) * term
      numerator = value if numerator is None else numerator + value
    return (numerator / denominator).to(latest.dtype)

  def update(self, Y: torch.Tensor):
    """Append one full-compute tensor as a real anchor, evicting the oldest if full.

    :param Y: Fully computed residual or hidden-state tensor for the current stream.
    """

    if self.anchors:
      latest = self.anchors[-1][1]
      if Y.shape != latest.shape or Y.dtype != latest.dtype:
        self.reset()
    self.anchors.append((self.current_step, Y.detach().clone()))

  def step(self, Y: torch.Tensor) -> torch.Tensor:
    """Record a cold-start anchor or return the current BRACE forecast.

    :param Y: Fully computed tensor used to initialize an empty stream.
    :returns: The exact cold-start tensor or the current rational forecast.
    """

    self.mark_step_begin()
    if not self.anchors:
      self.update(Y)
      return Y
    return self.approximate()


class BRACECalibrator(CalibratorBase):
  """Calibrator that forecasts residuals with BRACE rational aggregation."""

  def __init__(self, capacity: int = 3, gamma: float = 0.7, **kwargs):
    """Create a BRACE calibrator whose states are keyed by logical stream names.

    :param capacity: Maximum number of real anchors retained per stream, in [1, 3].
    :param gamma: Finite terminal weight in [0.5, infinity); smaller values admit extrapolation
        poles.
    :param kwargs: Additional keyword arguments accepted for config compatibility.
    """

    validate_brace_params(capacity, gamma)
    self.capacity = capacity
    self.gamma = float(gamma)
    self.states: Dict[str, BRACEState] = {}
    self.reset_cache()

  @property
  def num_fallbacks(self) -> int:
    """Return the numerical-fallback count aggregated over all streams.

    :returns: Total number of forecasts replaced by the latest real anchor.
    """

    return sum(state.num_fallbacks for state in self.states.values())

  def reset_cache(self):
    """Reset every tracked BRACE state without dropping the stream mapping."""

    if self.states:
      for state in self.states.values():
        state.reset()

  def maybe_init_state(self, name: str = "default"):
    """Lazily create one BRACE state for a named tensor stream.

    :param name: Logical tensor-stream name.
    """

    if name not in self.states:
      self.states[name] = BRACEState(capacity=self.capacity, gamma=self.gamma)

  def mark_step_begin(self, *args, **kwargs):
    """Advance every tracked state's logical step counter.

    :param args: Additional positional arguments accepted for interface compatibility.
    :param kwargs: Additional keyword arguments accepted for interface compatibility.
    """

    if self.states:
      for state in self.states.values():
        state.mark_step_begin()

  def approximate(self, name: str = "default") -> torch.Tensor:
    """Forecast the current tensor for one named stream without mutating its FIFO.

    :param name: Logical tensor-stream name.
    :returns: The BRACE forecast for the current logical step.
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
    """Advance one named stream and return its exact cold start or BRACE forecast.

    :param Y: Fully computed tensor used to initialize an empty stream.
    :param name: Logical tensor-stream name.
    :returns: The exact cold-start tensor or current rational forecast.
    """

    self.maybe_init_state(name)
    return self.states[name].step(Y)

  def __repr__(self):
    return f"BRACECalibrator_C({self.capacity}, gamma={self.gamma})"
