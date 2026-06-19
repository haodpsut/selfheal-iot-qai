"""QAOA bridge-to-quantum figure: AI-prior vs uniform initialization across QAOA depth p.

Two panels: (a) approximation ratio, (b) probability of sampling the optimum, both vs p,
prior vs uniform init. Run after qaoa_demo.py:  python plot_qaoa.py
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
CSV = os.path.join(HERE, "..", "..", "results", "qaoa", "qaoa_results.csv")
OUT = os.path.join(HERE, "..", "..", "results", "qaoa", "fig_qaoa.png")


def main():
    by = defaultdict(lambda: defaultdict(list))
    with open(CSV) as f:
        for r in csv.DictReader(f):
            p = int(r["p"])
            by[p]["ar_u"].append(float(r["ar_uniform"]))
            by[p]["ar_a"].append(float(r["ar_prior"]))
            by[p]["po_u"].append(float(r["popt_uniform"]))
            by[p]["po_a"].append(float(r["popt_prior"]))
    ps = sorted(by)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.3))

    def series(ax, ku, ka, ylabel, title):
        mu = [np.mean(by[p][ku]) for p in ps]
        su = [np.std(by[p][ku]) for p in ps]
        ma = [np.mean(by[p][ka]) for p in ps]
        sa = [np.std(by[p][ka]) for p in ps]
        ax.errorbar(ps, mu, yerr=su, marker="s", color="#e8a13a", capsize=3, lw=2,
                    label="uniform init")
        ax.errorbar(ps, ma, yerr=sa, marker="o", color="#c0392b", capsize=3, lw=2,
                    label="AI-prior init")
        ax.set_xlabel("QAOA depth $p$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(ps)
        S.clean(ax)
        ax.legend()

    series(ax1, "ar_u", "ar_a", "approximation ratio", "(a) Solution quality")
    series(ax2, "po_u", "po_a", "P(optimal sample)", "(b) Hitting the optimum")

    fig.suptitle("Same AI prior warm-starts a simulated quantum optimizer (QAOA)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
