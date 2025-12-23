import pytest
import numpy as np
from cache_dit.caching.cache_contexts.calibrators import (
    TaylorSeerCalibrator,
)

N_DERIVATIVES = [1, 2, 3]
MAX_WARMUP_STEPS = [2, 5]
SKIP_INTERVAL_STEPS = [1, 2]


@pytest.mark.parametrize("n_derivatives", N_DERIVATIVES)
@pytest.mark.parametrize("max_warmup_steps", MAX_WARMUP_STEPS)
@pytest.mark.parametrize("skip_interval_steps", SKIP_INTERVAL_STEPS)
def test_taylor_seer_calibrator(
    n_derivatives,
    max_warmup_steps,
    skip_interval_steps,
):
    taylor_seer = TaylorSeerCalibrator(
        n_derivatives=n_derivatives,
        max_warmup_steps=max_warmup_steps,
        skip_interval_steps=skip_interval_steps,
    )

    x_values = np.arange(0, 10, 0.1)

    y_pred = []
    errors = []
    for x in x_values:
        y = x**2
        y_approx = taylor_seer.step(y)
        y_pred.append(y_approx)
        errors.append(abs(y - y_approx))

    mean_error = np.mean(errors)
    print(
        f"Mean approximation error: {mean_error:.5f}, n_derivatives: {n_derivatives}, "
        f"max_warmup_steps: {max_warmup_steps}, skip_interval_steps: {skip_interval_steps}",
        flush=True,
    )
