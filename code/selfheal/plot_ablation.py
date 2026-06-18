"""Ablation figures from results/ablation/ablation.csv.

Produces two panels:
  (a) budget sensitivity: survivable fraction vs energy budget fraction (resilience-energy
      trade-off, with the saturation knee),
  (b) robustness weight W: survivable and lambda2 vs W (the redundancy trade-off).

    python plot_ablation.py
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
CSV = os.path.join(HERE, "..", "..", "results", "ablation", "ablation.csv")
OUT = os.path.join(HERE, "..", "..", "results", "ablation", "fig_ablation.png")


def load():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _agg(rows, ablation, key, metric):
    d = defaultdict(list)
    for r in rows:
        if r["ablation"] == ablation and r.get(metric, "") != "":
            d[r["setting"]].append(float(r[metric]))
    settings = sorted(d, key=lambda s: float(s.split("=")[1]))
    return settings, [np.mean(d[s]) for s in settings], [np.std(d[s]) for s in settings]


def main():
    rows = load()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # (a) budget sensitivity
    s, surv_m, surv_s = _agg(rows, "C_budget", "setting", "survivable")
    x = [float(k.split("=")[1]) for k in s]
    ax1.errorbar(x, surv_m, yerr=surv_s, marker="o", color="#c0392b", capsize=3)
    ax1.set_xlabel("repair budget (fraction of total)")
    ax1.set_ylabel("survivable fraction")
    ax1.set_title("(a) Resilience vs energy")
    ax1.set_ylim(0.85, 1.02)

    # (b) robustness weight
    sw, surv_wm, _ = _agg(rows, "B_weight", "setting", "survivable")
    _, lam_wm, _ = _agg(rows, "B_weight", "setting", "lam2")
    xw = [float(k.split("=")[1]) for k in sw]
    ax2.plot(xw, surv_wm, marker="o", color="#c0392b", label="survivable")
    ax2.set_xlabel("robustness weight $w$")
    ax2.set_ylabel("survivable fraction", color="#c0392b")
    ax2.set_title("(b) Survivability vs robustness")
    ax2b = ax2.twinx()
    ax2b.plot(xw, lam_wm, marker="s", color="#2c6fbb", label="$\\lambda_2$")
    ax2b.set_ylabel("algebraic connectivity $\\lambda_2$", color="#2c6fbb")

    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
