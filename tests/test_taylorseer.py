import argparse
import numpy as np
import matplotlib.pyplot as plt
from cache_dit.cache_factory.taylorseer import TaylorSeer


def get_args():
    parser = argparse.ArgumentParser(
        description="Test TaylorSeer approximation."
    )
    parser.add_argument(
        "--n_derivatives",
        "--order",
        type=int,
        default=2,
        help="Number of derivatives to approximate.",
    )
    parser.add_argument(
        "--warmup_steps",
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup steps before approximation starts.",
    )
    parser.add_argument(
        "--skip_interval_steps",
        type=int,
        default=2,
        help="Interval of steps to skip for approximation.",
    )
    return parser.parse_args()


args = get_args()


taylor_seer = TaylorSeer(
    n_derivatives=args.n_derivatives,
    warmup_steps=args.warmup_steps,
    skip_interval_steps=args.skip_interval_steps,
)

x_values = np.arange(0, 10, 0.1)
y_true = x_values**2

y_pred = []
errors = []
for x in x_values:
    y = x**2
    y_approx = taylor_seer.step(y)
    y_pred.append(y_approx)
    errors.append(abs(y - y_approx))


save_path = f"taylorseer_approximation_order_{args.n_derivatives}.png"
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x_values, y_true, label="True $y=x^2$")
plt.plot(
    x_values,
    y_pred,
    "--",
    label=f"TaylorSeer Approximation, Order={args.n_derivatives}",
)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("TaylorSeer Approximation Test")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(x_values, errors, color="red", label="Absolute Error")
plt.legend()
plt.xlabel("x")
plt.ylabel("Error")
plt.title("Approximation Error")
plt.grid()

plt.tight_layout()
plt.savefig(save_path)
print(f"Test completed and saved as {save_path}.")
