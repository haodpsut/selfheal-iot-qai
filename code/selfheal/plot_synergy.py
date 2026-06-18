"""Synergy figure (Fig 4): isolating the AI warm-start contribution.

Two panels, both QIEA+AI vs QIEA-noAI (mean +/- std over seeds):
  (a) healing latency  (generations to reach 95% of the final score): AI heals faster,
  (b) final quality    (survivable fraction for CS1, served-demand fraction for CS2).

Reads the main results CSV, so it is robust to the per-seed noise that muddies raw
convergence curves.

    python plot_synergy.py cs1
    python plot_synergy.py cs2
"""

from __future__ import annotations
import sys
import os
import csv
from collections import defaultdict
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
QUALITY = {"cs1": ("survivable", "survivable fraction"),
           "cs2": ("served", "served-demand fraction")}
COLORS = {"QIEA-noAI": "#e0a05a", "QIEA+AI": "#c0392b"}


def main():
    cs = sys.argv[1] if len(sys.argv) > 1 else "cs1"
    qkey, qlabel = QUALITY[cs]
    path = os.path.join(HERE, "..", "..", "results", cs, f"{cs}_results.csv")

    data = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for row in csv.DictReader(f):
            data[row["method"]]["latency"].append(float(row["latency"]))
            data[row["method"]][qkey].append(float(row[qkey]))

    methods = ["QIEA-noAI", "QIEA+AI"]
    cols = [COLORS[m] for m in methods]
    labels = ["no AI", "AI warm-start"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))

    lat_m = [np.mean(data[m]["latency"]) for m in methods]
    lat_s = [np.std(data[m]["latency"]) for m in methods]
    ax1.bar(labels, lat_m, yerr=lat_s, color=cols, capsize=4)
    ax1.set_ylabel("healing latency (generations to 95%)")
    ax1.set_title("(a) Faster healing")

    q_m = [np.mean(data[m][qkey]) for m in methods]
    q_s = [np.std(data[m][qkey]) for m in methods]
    ax2.bar(labels, q_m, yerr=q_s, color=cols, capsize=4)
    ax2.set_ylabel(qlabel)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("(b) Equal or better quality")

    fig.suptitle(f"Synergy: AI warm-start into the quantum-inspired optimizer ({cs.upper()})")
    fig.tight_layout()
    out = os.path.join(HERE, "..", "..", "results", cs, f"fig_synergy_{cs}.png")
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
