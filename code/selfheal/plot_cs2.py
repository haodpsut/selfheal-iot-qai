"""Plot CS2 results into the paper figure (Fig 3).

Reads results/cs2/cs2_results.csv and renders served-demand fraction per method
(mean +/- std). CS2's honest message: a congestion-aware greedy is strong on this
capacity-bound reallocation, while the framework still restores most served demand and
the AI warm-start helps the quantum-inspired optimizer.

    python plot_cs2.py
"""

from __future__ import annotations
import os
import csv
from collections import defaultdict
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, "..", "..", "results", "cs2", "cs2_results.csv")
OUT = os.path.join(HERE, "..", "..", "results", "cs2", "fig_cs2.png")

ORDER = ["greedy", "GA", "QIEA-noAI", "QIEA+AI"]
COLORS = {"greedy": "#b0b0b0", "GA": "#6fae6f", "QIEA-noAI": "#e0a05a", "QIEA+AI": "#c0392b"}


def main():
    data = defaultdict(list)
    with open(CSV) as f:
        for row in csv.DictReader(f):
            data[row["method"]].append(float(row["served"]))

    methods = [m for m in ORDER if m in data]
    mean = [np.mean(data[m]) for m in methods]
    std = [np.std(data[m]) for m in methods]
    cols = [COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars = ax.bar(methods, mean, yerr=std, color=cols, capsize=3,
                  edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, mean):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("served-demand fraction")
    ax.set_ylim(0, 0.85)
    ax.set_title("CS2: reallocation under a jamming surge")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
