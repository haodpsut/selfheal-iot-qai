"""CS2 figure (Fig 5): device-to-gateway survivable fraction after healing the jammed region.

Publication-quality bar chart with a significance bracket on the AI-synergy comparison.
Run after run_cs2.py:  python plot_cs2.py
"""

from __future__ import annotations
import os
import csv
from collections import defaultdict
import numpy as np

import plotstyle as S
S.apply()
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, "..", "..", "results", "cs2", "cs2_results.csv")
OUT = os.path.join(HERE, "..", "..", "results", "cs2", "fig_cs2.png")
ORDER = ["greedy", "greedy-conn", "GA", "SA", "QIEA-noAI", "QIEA+AI"]


def main():
    data = defaultdict(list)
    with open(CSV) as f:
        for row in csv.DictReader(f):
            data[row["method"]].append(float(row["gw_survivable"]))
    methods = [m for m in ORDER if m in data]
    means = [np.mean(data[m]) for m in methods]
    stds = [np.std(data[m]) for m in methods]

    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    S.bars(ax, methods, means, stds)
    ax.set_ylabel("device-to-gateway survivable fraction")
    ax.set_ylim(0, 1.18)
    ax.set_title("CS2: restoring gateway reachability under link jamming")
    S.clean(ax)

    # Significance bracket on the synergy comparison (QIEA no-AI vs QIEA+AI).
    try:
        from scipy.stats import wilcoxon
        i, j = methods.index("QIEA-noAI"), methods.index("QIEA+AI")
        p = wilcoxon(data["QIEA-noAI"], data["QIEA+AI"]).pvalue
        y = max(means[i] + stds[i], means[j] + stds[j]) + 0.09
        S.sig_bracket(ax, i, j, y, p)
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
