# The TaylorSeerState codebase is adapted from FBCache. Over time its codebase
# diverged a lot, and TaylorSeerState API is no longer compatible with FBCache.
import math
import torch
from typing import List, Dict
from cache_dit.caching.cache_contexts.calibrators.base import (
    CalibratorBase,
)

from cache_dit.logger import init_logger

logger = init_logger(__name__)


class TaylorSeerState:
    def __init__(
        self,
        n_derivatives=1,
        max_warmup_steps=1,
        skip_interval_steps=1,
    ):
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
        self.state: Dict[str, List[torch.Tensor]] = {
            "dY_prev": [None] * self.order,
            "dY_current": [None] * self.order,
        }
        self.current_step = -1
        self.last_non_approximated_step = -1

    def mark_step_begin(self):  # NEED
        self.current_step += 1

    def should_compute(self, step=None):
        step = self.current_step if step is None else step
        if (
            step < self.max_warmup_steps
            or (step - self.max_warmup_steps + 1) % self.skip_interval_steps == 0
        ):
            return True
        return False

    def derivative(self, Y: torch.Tensor) -> List[torch.Tensor]:
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
        elapsed = self.current_step - self.last_non_approximated_step
        output = 0
        for i, derivative in enumerate(self.state["dY_current"]):
            if derivative is not None:
                output += (1 / math.factorial(i)) * derivative * (elapsed**i)
            else:
                break
        return output

    def update(self, Y: torch.Tensor):  # NEED
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
        self.mark_step_begin()
        if self.should_compute():
            self.update(Y)
            return Y
        else:
            return self.approximate()


class TaylorSeerCalibrator(CalibratorBase):
    def __init__(
        self,
        n_derivatives=1,
        max_warmup_steps=1,
        skip_interval_steps=1,
        **kwargs,
    ):
        self.n_derivatives = n_derivatives
        self.max_warmup_steps = max_warmup_steps
        self.skip_interval_steps = skip_interval_steps
        self.states: Dict[str, TaylorSeerState] = {}
        self.reset_cache()

    def reset_cache(self):  # NEED
        if self.states:
            for state in self.states.values():
                state.reset()

    def maybe_init_state(
        self,
        name: str = "default",
    ):
        if name not in self.states:
            self.states[name] = TaylorSeerState(
                n_derivatives=self.n_derivatives,
                max_warmup_steps=self.max_warmup_steps,
                skip_interval_steps=self.skip_interval_steps,
            )

    def mark_step_begin(self, *args, **kwargs):
        if self.states:
            for state in self.states.values():
                state.mark_step_begin()

    def derivative(
        self,
        Y: torch.Tensor,
        name: str = "default",
    ) -> List[torch.Tensor]:
        self.maybe_init_state(name)
        state = self.states[name]
        state.derivative(Y)
        return state.state["dY_current"]

    def approximate(
        self,
        name: str = "default",
    ) -> torch.Tensor:  # NEED
        assert name in self.states, f"State '{name}' not found."
        state = self.states[name]
        return state.approximate()

    def update(
        self,
        Y: torch.Tensor,
        name: str = "default",
    ):  # NEED
        self.maybe_init_state(name)
        state = self.states[name]
        state.update(Y)

    def step(
        self,
        Y: torch.Tensor,
        name: str = "default",
    ):
        self.maybe_init_state(name)
        state = self.states[name]
        return state.step(Y)

    def __repr__(self):
        return f"TaylorSeerCalibrator_O({self.n_derivatives})"
