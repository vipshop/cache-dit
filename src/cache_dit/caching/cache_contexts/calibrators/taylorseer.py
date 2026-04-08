# The TaylorSeerState codebase is adapted from FBCache. Over time its codebase
# diverged a lot, and TaylorSeerState API is no longer compatible with FBCache.
import math
import torch
from typing import List, Dict
from .base import CalibratorBase

from ....logger import init_logger

logger = init_logger(__name__)


class TaylorSeerState:
  """Per-feature Taylor expansion state for one named calibration stream."""

  def __init__(
    self,
    n_derivatives=1,
    max_warmup_steps=1,
    skip_interval_steps=1,
  ):
    """Initialize derivative buffers and warmup/skip scheduling state.

    :param n_derivatives: Number of Taylor-derivative orders to retain.
    :param max_warmup_steps: Number of initial full-compute warmup steps.
    :param skip_interval_steps: Interval between later full-compute refresh steps.
    """

    self.n_derivatives = n_derivatives
    self.order = n_derivatives + 1
    self.max_warmup_steps = max_warmup_steps
    self.skip_interval_steps = skip_interval_steps
    self.current_step = -1
    self.last_non_approximated_step = -1
    self.state: Dict[str, List[torch.Tensor]] = {
      "dY_prev": [None] * self.order,
      "dY_current": [None] * self.order,
    }

  def reset(self):
    """Reset all derivative buffers and step counters."""

    self.state: Dict[str, List[torch.Tensor]] = {
      "dY_prev": [None] * self.order,
      "dY_current": [None] * self.order,
    }
    self.current_step = -1
    self.last_non_approximated_step = -1

  def mark_step_begin(self):  # NEED
    """Advance the logical step counter by one."""

    self.current_step += 1

  def should_compute(self, step=None):
    """Return whether this step should run full compute instead of approximation.

    :param step: Optional explicit step index. When omitted, `current_step` is used.
    :returns: `True` when the state should run a real compute step instead of approximation.
    """

    step = self.current_step if step is None else step
    if (step < self.max_warmup_steps
        or (step - self.max_warmup_steps + 1) % self.skip_interval_steps == 0):
      return True
    return False

  def derivative(self, Y: torch.Tensor) -> List[torch.Tensor]:
    """Update the current Taylor derivative ladder from a fully computed tensor.

    :param Y: Fully computed tensor for the current step.
    :returns: The updated derivative ladder for the current step.
    """

    # Y(t) = Y(0) + dY(0)/dt * t + d^2Y(0)/dt^2 * t^2 / 2!
    #        + ... + d^nY(0)/dt^n * t^n / n!
    dY_current: List[torch.Tensor] = [None] * self.order
    dY_current[0] = Y
    window = self.current_step - self.last_non_approximated_step
    if self.state["dY_prev"][0] is not None:
      if dY_current[0].shape != self.state["dY_prev"][0].shape:
        self.reset()

    for i in range(self.n_derivatives):
      if self.state["dY_prev"][i] is not None and self.current_step > 1:
        dY_current[i + 1] = (dY_current[i] - self.state["dY_prev"][i]) / window
      else:
        break
    return dY_current

  def approximate(self) -> torch.Tensor:  # NEED
    """Evaluate the Taylor expansion at the current elapsed step.

    :returns: The approximated tensor value for the current logical step.
    """

    elapsed = self.current_step - self.last_non_approximated_step
    output = 0
    for i, derivative in enumerate(self.state["dY_current"]):
      if derivative is not None:
        output += (1 / math.factorial(i)) * derivative * (elapsed ** i)
      else:
        break
    return output

  def update(self, Y: torch.Tensor):  # NEED
    """Commit a full-compute tensor as the latest non-approximated state.

    :param Y: Fully computed tensor to record as the new Taylor expansion anchor.
    """

    # Directly call this method will ingnore the warmup
    # policy and force full computation.
    # Assume warmup steps is 3, and n_derivatives is 3.
    # step 0: dY_prev    = [None, None,   None,    None   ]
    #         dY_current = [Y0,   None,   None,    None   ]
    # step 1: dY_prev    = [Y0,   None,   None,    None   ]
    #         dY_current = [Y1,   dY1,    None,    None   ]
    # step 2: dY_prev    = [Y1,   dY1,    None,    None   ]
    #         dY_current = [Y2,   dY2/Y1, dY2/dY1, None   ]
    # step 3: dY_prev    = [Y2,   dY2/Y1, dY2/dY1, None   ],
    #         dY_current = [Y3,   dY3/Y2, dY3/dY2, dY3/dY1]
    # step 4: dY_prev    = [Y3,   dY3/Y2, dY3/dY2, dY3/dY1]
    #         dY_current = [Y4,   dY4/Y3, dY4/dY3, dY4/dY2]
    self.state["dY_prev"] = self.state["dY_current"]
    self.state["dY_current"] = self.derivative(Y)
    self.last_non_approximated_step = self.current_step

  def step(self, Y: torch.Tensor):
    """Advance one step and return either the true tensor or its approximation.

    :param Y: Fully computed tensor for the current step when compute is required.
    :returns: The exact tensor or the Taylor-approximated tensor for this step.
    """

    self.mark_step_begin()
    if self.should_compute():
      self.update(Y)
      return Y
    else:
      return self.approximate()


class TaylorSeerCalibrator(CalibratorBase):
  """Calibrator that forecasts tensors with a Taylor-series approximation."""

  def __init__(
    self,
    n_derivatives=1,
    max_warmup_steps=1,
    skip_interval_steps=1,
    **kwargs,
  ):
    """Create a calibrator whose states are keyed by logical tensor names.

    :param n_derivatives: Number of Taylor-derivative orders to retain per stream.
    :param max_warmup_steps: Number of initial full-compute warmup steps.
    :param skip_interval_steps: Interval between later full-compute refresh steps.
    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    """

    self.n_derivatives = n_derivatives
    self.max_warmup_steps = max_warmup_steps
    self.skip_interval_steps = skip_interval_steps
    self.states: Dict[str, TaylorSeerState] = {}
    self.reset_cache()

  def reset_cache(self):  # NEED
    """Reset every tracked `TaylorSeerState` without dropping the key mapping."""

    if self.states:
      for state in self.states.values():
        state.reset()

  def maybe_init_state(
    self,
    name: str = "default",
  ):
    """Lazily create one Taylor state for a named tensor stream.

    :param name: Logical tensor-stream name.
    """

    if name not in self.states:
      self.states[name] = TaylorSeerState(
        n_derivatives=self.n_derivatives,
        max_warmup_steps=self.max_warmup_steps,
        skip_interval_steps=self.skip_interval_steps,
      )

  def mark_step_begin(self, *args, **kwargs):
    """Advance every tracked state's step counter.

    :param args: Additional positional arguments forwarded to the underlying implementation.
    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    """

    if self.states:
      for state in self.states.values():
        state.mark_step_begin()

  def derivative(
    self,
    Y: torch.Tensor,
    name: str = "default",
  ) -> List[torch.Tensor]:
    """Return the current derivative ladder for one named tensor stream.

    :param Y: Fully computed tensor for the named stream.
    :param name: Logical tensor-stream name.
    :returns: The derivative ladder tracked for the named stream.
    """

    self.maybe_init_state(name)
    state = self.states[name]
    state.derivative(Y)
    return state.state["dY_current"]

  def approximate(
    self,
    name: str = "default",
  ) -> torch.Tensor:  # NEED
    """Approximate the next tensor for one named stream.

    :param name: Logical tensor-stream name.
    :returns: The approximated tensor for the named stream.
    """

    assert name in self.states, f"State '{name}' not found."
    state = self.states[name]
    return state.approximate()

  def update(
    self,
    Y: torch.Tensor,
    name: str = "default",
  ):  # NEED
    """Feed a fully computed tensor into one named Taylor state.

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
    """Advance one named stream and return either computed or approximated output.

    :param Y: Fully computed tensor for the named stream when computation is required.
    :param name: Logical tensor-stream name.
    :returns: The exact tensor or Taylor approximation for the named stream.
    """

    self.maybe_init_state(name)
    state = self.states[name]
    return state.step(Y)

  def __repr__(self):
    return f"TaylorSeerCalibrator_O({self.n_derivatives})"
