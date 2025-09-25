import matplotlib.pyplot as plt
import numpy as np
import argparse

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

clip_score = [
    32.312,
    32.273,
    32.102,
    32.987,
    32.114,
    32.776,
    32.266,
    32.838,
    32.819,
    32.241,
    32.413,
    32.849,
    32.865,
    32.948,
]


dbc_speedup = []
dbc_reward = []
dbc_clip = []
dbc_labels = []
other_speedup = []
other_reward = []
other_clip = []
other_labels = []

for i in range(len(methods)):
    if "DBCache" in methods[i]:
        dbc_speedup.append(speedup[i])
        dbc_reward.append(image_reward[i])
        dbc_clip.append(clip_score[i])
        dbc_labels.append(methods[i])
    else:
        other_speedup.append(speedup[i])
        other_reward.append(image_reward[i])
        other_clip.append(clip_score[i])
        other_labels.append(methods[i])


def plot_metric(
    dbc_metric,
    other_metric,
    scatter_ratio=70,
    y_label="ImageReward",
    save_path="image-reward-bench.png",
    figsize=(11, 5),
):
    global other_speedup, other_labels, dbc_speedup
    plt.figure(figsize=figsize, dpi=800)

    for i in range(len(other_speedup)):
        plt.scatter(
            other_speedup[i],
            other_metric[i],
            s=other_speedup[i] * other_metric[i] * scatter_ratio,
            marker="o",
            linestyle="",
            alpha=0.6,
            label=other_labels[i],
        )

    sorted_indices = np.argsort(dbc_speedup)
    sorted_dbc_speedup = [dbc_speedup[i] for i in sorted_indices]
    sorted_dbc_metric = [dbc_metric[i] for i in sorted_indices]
    sorted_dbc_labels = [dbc_labels[i] for i in sorted_indices]
    sorted_dbc_sizes = [
        speedup * metric * scatter_ratio
        for speedup, metric in zip(sorted_dbc_speedup, sorted_dbc_metric)
    ]

    plt.scatter(
        sorted_dbc_speedup,
        sorted_dbc_metric,
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
        sorted_dbc_metric,
        linestyle="-",
        color="skyblue",
        linewidth=3,
        alpha=0.8,
    )

    for i, label in enumerate(sorted_dbc_labels):
        ha = "right" if i % 2 else "left"
        if "TaylorSeer" in label:
            label = label.replace("+TaylorSeer", "")
        if "R=" in label:
            label = label.replace("DBCache", "DBCache\n")
        plt.annotate(
            label,
            (sorted_dbc_speedup[i], sorted_dbc_metric[i]),
            fontsize=10,
            ha=ha,
        )

    plt.title(f"FLUX.1-dev Performance Comparison: {y_label}")
    plt.xlabel("Speedup Ratio")
    plt.ylabel(y_label)

    plt.legend(loc="lower right")

    plt.grid(True, linestyle="--", alpha=0.8)

    plt.tight_layout()

    plt.savefig(save_path)

    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        "-m",
        choices=["image-reward", "clip-score"],
        default="image-reward",
    )
    args = parser.parse_args()
    print(args)

    if args.metric == "image-reward":
        plot_metric(
            dbc_metric=dbc_reward,
            other_metric=other_reward,
            scatter_ratio=70,
            y_label="ImageReward",
            save_path="image-reward-bench.png",
            figsize=(11, 5),
        )
    elif args.metric == "clip-score":
        plot_metric(
            dbc_metric=dbc_clip,
            other_metric=other_clip,
            scatter_ratio=4,
            y_label="ClipScore",
            save_path="clip-score-bench.png",
            figsize=(12, 5),
        )


if __name__ == "__main__":
    main()
