"""Multi-seed runner for CS2 (reallocation under a jamming surge).

Mirrors run_cs1.py: runs every method over N seeds, writes results/cs2/cs2_results.csv plus
QIEA convergence histories, and prints a mean +/- std summary with Wilcoxon tests. CS2's role
is to show the AI<->QIO synergy GENERALISES to a second, structurally different resilience
problem; greedy is a strong baseline here (capacity-bound reallocation is greedy-friendly), so
the headline is the synergy delta (QIEA+AI vs QIEA-noAI), not beating greedy.

    python run_cs2.py 3        # local smoke
    python run_cs2.py 30 6     # full run, 6 parallel workers
"""

from __future__ import annotations
import sys
import os
import numpy as np

from qio import optimize, QIEAConfig
from cs2_jamming import make_jam_scenario, make_fitness, ai_prior, genetic, greedy, _served

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "cs2")
GENERATIONS = 120
POP = 40


def _latency(history, frac=0.95):
    target = frac * history[-1]
    hit = np.where(history >= target)[0]
    return int(hit[0]) if hit.size else len(history)


def _metrics(sc, x):
    served, pw = _served(sc, x)
    return dict(served=served / sc.demand.sum(), power=float(pw))


def run_seed(seed: int) -> list:
    sc = make_jam_scenario(seed=seed)
    fit = make_fitness(sc)
    prior = ai_prior(sc)
    uniform = np.full(len(sc.pairs), float(prior.mean()))
    cfg = QIEAConfig(pop_size=POP, generations=GENERATIONS, seed=seed)

    rows = []
    r_ai = optimize(fit, len(sc.pairs), cfg, prior=prior)
    rows.append(dict(method="QIEA+AI", seed=seed, latency=_latency(r_ai.history),
                     history=r_ai.history.tolist(), **_metrics(sc, r_ai.best_x)))
    r_no = optimize(fit, len(sc.pairs), cfg, prior=uniform)
    rows.append(dict(method="QIEA-noAI", seed=seed, latency=_latency(r_no.history),
                     history=r_no.history.tolist(), **_metrics(sc, r_no.best_x)))
    bx, _, bh = genetic(sc, fit, pop_size=POP, generations=GENERATIONS, seed=seed)
    rows.append(dict(method="GA", seed=seed, latency=_latency(bh), **_metrics(sc, bx)))
    gx = greedy(sc)
    rows.append(dict(method="greedy", seed=seed, latency=0, **_metrics(sc, gx)))
    return rows


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    all_rows = []
    if jobs > 1:
        from multiprocessing import Pool
        with Pool(jobs) as pool:
            for rows in pool.imap_unordered(run_seed, range(n_seeds)):
                all_rows.extend(rows)
                sv = {r["method"]: r["served"] for r in rows}
                print(f"seed {rows[0]['seed']:2d}  " + "  ".join(f"{m}={sv[m]:.3f}" for m in sv))
    else:
        for s in range(n_seeds):
            rows = run_seed(s)
            all_rows.extend(rows)
            sv = {r["method"]: r["served"] for r in rows}
            print(f"seed {s:2d}  " + "  ".join(f"{m}={sv[m]:.3f}" for m in sv))

    os.makedirs(OUT_DIR, exist_ok=True)
    cols = ["method", "seed", "served", "power", "latency"]
    path = os.path.join(OUT_DIR, "cs2_results.csv")
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in all_rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nwrote {path}")

    for tag, meth in [("ai", "QIEA+AI"), ("noai", "QIEA-noAI")]:
        hs = [r["history"] for r in all_rows if r["method"] == meth]
        with open(os.path.join(OUT_DIR, f"hist_{tag}.csv"), "w") as f:
            for h in hs:
                f.write(",".join(f"{v:.6f}" for v in h) + "\n")

    methods = ["QIEA+AI", "QIEA-noAI", "GA", "greedy"]
    by = {m: [r for r in all_rows if r["method"] == m] for m in methods}
    print("\nmethod        served            power      latency")
    for m in methods:
        sv = np.array([r["served"] for r in by[m]])
        pw = np.array([r["power"] for r in by[m]])
        lt = np.array([r["latency"] for r in by[m]])
        print(f"{m:12s}  {sv.mean():.3f}+/-{sv.std():.3f}   {pw.mean():.3f}    {lt.mean():.1f}")

    try:
        from scipy.stats import wilcoxon
        ref = np.array([r["served"] for r in by["QIEA+AI"]])
        print("\nWilcoxon (served, QIEA+AI vs ...):")
        for m in methods[1:]:
            other = np.array([r["served"] for r in by[m]])
            if np.allclose(ref, other):
                print(f"  vs {m:10s}: identical")
                continue
            stat, p = wilcoxon(ref, other)
            print(f"  vs {m:10s}: p={p:.4g}")
    except Exception as e:
        print(f"\n(scipy not available for Wilcoxon: {e})")


if __name__ == "__main__":
    main()
