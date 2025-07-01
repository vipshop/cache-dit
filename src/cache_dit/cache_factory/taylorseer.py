# Adapted from: https://github.com/chengzeyi/ParaAttention/blob/main/src/para_attn/first_block_cache/taylorseer.py
# Reference: https://github.com/Shenyi-Z/TaylorSeer/TaylorSeer-FLUX/src/flux/taylor_utils/__init__.py
import math
import torch


class TaylorSeer:
    def __init__(
        self,
        n_derivatives=2,
        warmup_steps=1,
        skip_interval_steps=1,
        compute_step_map=None,
    ):
        self.n_derivatives = n_derivatives
        self.ORDER = n_derivatives + 1
        self.warmup_steps = warmup_steps
        self.skip_interval_steps = skip_interval_steps
        self.compute_step_map = compute_step_map
        self.reset_cache()

    @torch.compiler.disable
    def reset_cache(self):
        self.state = {
            "dY_prev": [None] * self.ORDER,
            "dY_current": [None] * self.ORDER,
        }
        self.current_step = -1
        self.last_non_approximated_step = -1

    @torch.compiler.disable
    def should_compute_full(self, step=None):
        step = self.current_step if step is None else step
        if self.compute_step_map is not None:
            return self.compute_step_map[step]
        if (
            step < self.warmup_steps
            or (step - self.warmup_steps + 1) % self.skip_interval_steps == 0
        ):
            return True
        return False

    @torch.compiler.disable
    def approximate_derivative(self, Y):
        # n-th order Taylor expansion:
        # Y(t) = Y(0) + dY(0)/dt * t + d^2Y(0)/dt^2 * t^2 / 2!
        #        + ... + d^nY(0)/dt^n * t^n / n!
        # TODO: Custom Triton/CUDA kernel for better performance,
        # especially for large n_derivatives.
        dY_current = [None] * self.ORDER
        dY_current[0] = Y
        window = self.current_step - self.last_non_approximated_step
        for i in range(self.n_derivatives):
            if self.state["dY_prev"][i] is not None and self.current_step > 1:
                dY_current[i + 1] = (
                    dY_current[i] - self.state["dY_prev"][i]
                ) / window
            else:
                break
        return dY_current

    @torch.compiler.disable
    def approximate_value(self):
        # TODO: Custom Triton/CUDA kernel for better performance,
        # especially for large n_derivatives.
        elapsed = self.current_step - self.last_non_approximated_step
        output = 0
        for i, derivative in enumerate(self.state["dY_current"]):
            if derivative is not None:
                output += (1 / math.factorial(i)) * derivative * (elapsed**i)
            else:
                break
        return output

    @torch.compiler.disable
    def mark_step_begin(self):
        self.current_step += 1

    @torch.compiler.disable
    def update(self, Y):
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
        self.state["dY_current"] = self.approximate_derivative(Y)
        self.last_non_approximated_step = self.current_step

    @torch.compiler.disable
    def step(self, Y):
        self.mark_step_begin()
        if self.should_compute_full():
            self.update(Y)
            return Y
        else:
            return self.approximate_value()
