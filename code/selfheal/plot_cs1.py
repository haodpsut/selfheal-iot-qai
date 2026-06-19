"""Plot CS1 results into the paper figure (Fig 2).

Reads results/cs1/cs1_results.csv and renders a two-panel figure:
  (a) worst-case survivable fraction per method (mean +/- std bars),
  (b) energy actually spent vs the budget, exposing that greedy under-heals.

Run after run_cs1.py:  python plot_cs1.py
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
CSV = os.path.join(HERE, "..", "..", "results", "cs1", "cs1_results.csv")
OUT = os.path.join(HERE, "..", "..", "results", "cs1", "fig_cs1.png")

ORDER = ["greedy", "greedy-conn", "GA", "SA", "QIEA-noAI", "QIEA+AI"]
COLORS = {"greedy": "#b0b0b0", "greedy-conn": "#8aa0c0", "GA": "#6fae6f",
          "SA": "#9b72c7", "QIEA-noAI": "#e0a05a", "QIEA+AI": "#c0392b"}


def load():
    data = defaultdict(lambda: defaultdict(list))
    with open(CSV) as f:
        for row in csv.DictReader(f):
            m = row["method"]
            data[m]["survivable"].append(float(row["survivable"]))
            data[m]["energy"].append(float(row["energy"]))
    return data


def main():
    data = load()
    methods = [m for m in ORDER if m in data]
    cols = [COLORS[m] for m in methods]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))

    def annotate(ax, bars, vals, fmt="{:.2f}"):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                    ha="center", va="bottom", fontsize=8)

    sv_mean = [np.mean(data[m]["survivable"]) for m in methods]
    sv_std = [np.std(data[m]["survivable"]) for m in methods]
    b1 = ax1.bar(methods, sv_mean, yerr=sv_std, color=cols, capsize=3,
                 edgecolor="black", linewidth=0.5)
    annotate(ax1, b1, sv_mean)
    ax1.set_ylabel("worst-case survivable fraction")
    ax1.set_ylim(0, 1.12)
    ax1.set_title("(a) Resilience restored")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(axis="y", alpha=0.25, linestyle=":")
    ax1.set_axisbelow(True)

    en_mean = [np.mean(data[m]["energy"]) for m in methods]
    b2 = ax2.bar(methods, en_mean, color=cols, edgecolor="black", linewidth=0.5)
    annotate(ax2, b2, en_mean)
    ax2.set_ylabel("energy spent (of budget)")
    ax2.set_ylim(0, max(en_mean) * 1.25)
    ax2.set_title("(b) Budget utilisation")
    ax2.tick_params(axis="x", rotation=20)
    ax2.grid(axis="y", alpha=0.25, linestyle=":")
    ax2.set_axisbelow(True)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
