import matplotlib.pyplot as plt
import numpy as np


methods = [
    "FLUX.1-dev: 60% steps",
    "Δ-DiT(N=2)",
    "Δ-DiT(N=3)",
    "DBCache(F8B0)",
    "FLUX.1-dev: 34% steps",
    "Chipmunk",
    "FORA(N=3)",
    "DBCache(F4B0)",  # DBCache(F=4,B=0,W=4,MC=4)
    "DBCache+TaylorSeer(F1B0O1)",  # DBCache+TaylorSeer(F=1,B=0,O=1)
    "DuCa(N=5)",
    "TaylorSeer(N=4,O=2)",
    "DBCache(F1B0)",  # DBCache(F=1,B=0,W=4,MC=6)
    "DBCache+TaylorSeer(F1B0O1,R=0.32)",
    "FoCa(N=5)",
]
speedup = [
    1.67,
    1.50,
    2.21,
    1.80,
    3.13,
    2.47,
    2.82,
    2.66,
    3.23,
    3.80,
    3.57,
    3.94,
    3.94,
    4.16,
]
image_reward = [
    0.9663,
    0.9444,
    0.8721,
    1.0370,
    0.9453,
    0.9936,
    0.9776,
    1.0065,
    1.0221,
    0.9955,
    0.9857,
    0.9997,
    1.0107,
    1.0029,
]

plt.figure(figsize=(11, 5), dpi=800)

dbc_speedup = []
dbc_reward = []
dbc_labels = []
other_speedup = []
other_reward = []
other_labels = []

for i in range(len(methods)):
    if "DBCache" in methods[i]:
        dbc_speedup.append(speedup[i])
        dbc_reward.append(image_reward[i])
        dbc_labels.append(methods[i])
    else:
        other_speedup.append(speedup[i])
        other_reward.append(image_reward[i])
        other_labels.append(methods[i])


for i in range(len(other_speedup)):
    plt.scatter(
        other_speedup[i],
        other_reward[i],
        s=other_speedup[i] * other_reward[i] * 70,
        marker="o",
        linestyle="",
        alpha=0.6,
        label=other_labels[i],
    )

sorted_indices = np.argsort(dbc_speedup)
sorted_dbc_speedup = [dbc_speedup[i] for i in sorted_indices]
sorted_dbc_reward = [dbc_reward[i] for i in sorted_indices]
sorted_dbc_labels = [dbc_labels[i] for i in sorted_indices]
sorted_dbc_sizes = [
    speedup * reward * 70
    for speedup, reward in zip(sorted_dbc_speedup, sorted_dbc_reward)
]

plt.scatter(
    sorted_dbc_speedup,
    sorted_dbc_reward,
    s=sorted_dbc_sizes,
    marker="o",
    linestyle="-",
    color="skyblue",
    linewidth=2,
    alpha=0.6,
    label="DBCache variants",
)


plt.plot(
    sorted_dbc_speedup,
    sorted_dbc_reward,
    linestyle="-",
    color="skyblue",
    linewidth=3,
    alpha=0.8,
)

for i, label in enumerate(sorted_dbc_labels):
    plt.annotate(
        label,
        (sorted_dbc_speedup[i], sorted_dbc_reward[i]),
        fontsize=10,
        ha="right",
    )

plt.title("FLUX.1-dev Performance Comparison")
plt.xlabel("Speedup Ratio")
plt.ylabel("ImageReward")

plt.legend(loc="lower right")

plt.grid(True, linestyle="--", alpha=0.8)

plt.tight_layout()

plt.savefig("image-reward-bench.png")

print("Saved image-reward-bench.png")
