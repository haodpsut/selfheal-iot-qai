"""CS1 figure (Fig 3): worst-case survivable fraction and budget utilisation after healing.

Publication-quality two-panel bar chart. Run after run_cs1.py:  python plot_cs1.py
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
CSV = os.path.join(HERE, "..", "..", "results", "cs1", "cs1_results.csv")
OUT = os.path.join(HERE, "..", "..", "results", "cs1", "fig_cs1.png")
ORDER = ["greedy", "greedy-conn", "GA", "SA", "QIEA-noAI", "QIEA+AI"]


def main():
    data = defaultdict(lambda: defaultdict(list))
    with open(CSV) as f:
        for row in csv.DictReader(f):
            data[row["method"]]["survivable"].append(float(row["survivable"]))
            data[row["method"]]["energy"].append(float(row["energy"]))
    methods = [m for m in ORDER if m in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6))

    sv_m = [np.mean(data[m]["survivable"]) for m in methods]
    sv_s = [np.std(data[m]["survivable"]) for m in methods]
    S.bars(ax1, methods, sv_m, sv_s)
    ax1.set_ylabel("worst-case survivable fraction")
    ax1.set_ylim(0, 1.18)
    ax1.set_title("(a) Resilience restored")
    S.clean(ax1)

    en_m = [np.mean(data[m]["energy"]) for m in methods]
    en_s = [np.std(data[m]["energy"]) for m in methods]
    S.bars(ax2, methods, en_m, en_s)
    ax2.set_ylabel("energy spent (of budget)")
    ax2.set_ylim(0, max(m + s for m, s in zip(en_m, en_s)) * 1.18)
    ax2.set_title("(b) Budget utilisation")
    S.clean(ax2)

    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
