import numpy as np
import matplotlib.pyplot as plt
from cache_dit.cache_factory.taylorseer import TaylorSeer

taylor_seer = TaylorSeer(n_derivatives=2, warmup_steps=2, skip_interval_steps=2)

x_values = np.arange(0, 10, 0.1)
y_true = x_values**2

y_pred = []
errors = []
for x in x_values:
    y = x**2
    y_approx = taylor_seer.step(y)
    y_pred.append(y_approx)
    errors.append(abs(y - y_approx))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x_values, y_true, label="True $y=x^2$")
plt.plot(x_values, y_pred, "--", label="TaylorSeer Approximation")
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
plt.savefig("taylorseer_approximation_test.png")
print("Test completed and saved as 'taylorseer_approximation_test.png'.")