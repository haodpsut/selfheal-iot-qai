"""Synergy figure (Fig 4): isolating the AI warm-start (QIEA+AI vs QIEA-noAI).

Two panels: (a) healing latency (generations to 95% of final), (b) final quality, with a
significance bracket on the quality comparison.

    python plot_synergy.py cs1
    python plot_synergy.py cs2
"""

from __future__ import annotations
import sys
import os
import csv
from collections import defaultdict
import numpy as np

import plotstyle as S
S.apply()
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
QUALITY = {"cs1": ("survivable", "survivable fraction"),
           "cs2": ("gw_survivable", "device-to-gateway survivable")}


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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.5))

    lat_m = [np.mean(data[m]["latency"]) for m in methods]
    lat_s = [np.std(data[m]["latency"]) for m in methods]
    S.bars(ax1, methods, lat_m, lat_s, fmt="{:.0f}")
    ax1.set_ylabel("healing latency (gens to 95%)")
    ax1.set_ylim(0, max(m + s for m, s in zip(lat_m, lat_s)) * 1.25)
    ax1.set_title("(a) Faster healing")
    S.clean(ax1)

    q_m = [np.mean(data[m][qkey]) for m in methods]
    q_s = [np.std(data[m][qkey]) for m in methods]
    S.bars(ax2, methods, q_m, q_s, fmt="{:.3f}")
    ax2.set_ylabel(qlabel)
    ax2.set_ylim(0, 1.18)
    ax2.set_title("(b) Final quality")
    S.clean(ax2)
    try:
        from scipy.stats import wilcoxon
        p = wilcoxon(data["QIEA-noAI"][qkey], data["QIEA+AI"][qkey]).pvalue
        y = max(q_m[0] + q_s[0], q_m[1] + q_s[1]) + 0.09
        S.sig_bracket(ax2, 0, 1, y, p)
    except Exception:
        pass

    fig.suptitle(f"Synergy of the AI warm-start ({cs.upper()})", y=1.02)
    fig.tight_layout()
    out = os.path.join(HERE, "..", "..", "results", cs, f"fig_synergy_{cs}.png")
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
