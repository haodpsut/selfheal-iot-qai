"""Multi-seed runner for CS1 (topology reconfiguration under cascading failures).

Runs every method over N seeds, writes a tidy per-seed CSV to results/cs1/, and prints a
mean +/- std summary plus Wilcoxon signed-rank tests against QIEA+AI. Use a few seeds for a
local smoke test and 30 on the RTX 4090:

    python run_cs1.py 3       # local smoke
    python run_cs1.py 30      # full run on the server
"""

from __future__ import annotations
import sys
import os
import numpy as np

from qio import optimize, QIEAConfig
from cs1_topology import (
    make_scenario, set_binding_budget, make_fitness, ai_prior, genetic, greedy,
    _giant_fraction, _survivable_fraction, _algebraic_connectivity,
    _resilience, _resilience_conn,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "cs1")
GENERATIONS = 120
POP = 40

# Scenario knobs, tuned so instances are consistently HARD: heavy cascading fragmentation
# plus a tight repair budget, so reaching full survivability is non-trivial and methods
# separate cleanly instead of all saturating at 1.0.
N_NODES = 200
FAIL_FRAC = 0.32
BUDGET_FRAC = 0.035   # tighter: leaves headroom so the AI prior's quality gain stays visible


def _latency(history, frac=0.95):
    target = frac * history[-1]
    hit = np.where(history >= target)[0]
    return int(hit[0]) if hit.size else len(history)


def _metrics(sc, x):
    edges = np.concatenate([sc.base_edges, sc.cand_edges[x == 1]], axis=0)
    return dict(
        survivable=_survivable_fraction(sc.n_nodes, sc.alive, edges),
        conn=_giant_fraction(sc.n_nodes, sc.alive, edges),
        lam2=_algebraic_connectivity(sc.n_nodes, sc.alive, edges),
        energy=float(sc.cand_cost[x == 1].sum()),
        edges=int(x.sum()),
    )


def run_seed(seed: int) -> list[dict]:
    sc = make_scenario(n_nodes=N_NODES, fail_frac=FAIL_FRAC, seed=seed)
    set_binding_budget(sc, frac=BUDGET_FRAC)
    fit = make_fitness(sc)
    prior = ai_prior(sc)
    uniform = np.full(len(sc.cand_edges), float(prior.mean()))
    cfg = QIEAConfig(pop_size=POP, generations=GENERATIONS, seed=seed)

    rows = []

    r_ai = optimize(fit, len(sc.cand_edges), cfg, prior=prior)
    rows.append(dict(method="QIEA+AI", seed=seed, latency=_latency(r_ai.history),
                     history=r_ai.history.tolist(), **_metrics(sc, r_ai.best_x)))

    r_no = optimize(fit, len(sc.cand_edges), cfg, prior=uniform)
    rows.append(dict(method="QIEA-noAI", seed=seed, latency=_latency(r_no.history),
                     history=r_no.history.tolist(), **_metrics(sc, r_no.best_x)))

    bx, _, bh = genetic(sc, fit, pop_size=POP, generations=GENERATIONS, seed=seed)
    rows.append(dict(method="GA", seed=seed, latency=_latency(bh), **_metrics(sc, bx)))

    gx, _, _, gsteps = greedy(sc, obj=_resilience)
    rows.append(dict(method="greedy", seed=seed, latency=gsteps, **_metrics(sc, gx)))

    gcx, _, _, gcsteps = greedy(sc, obj=_resilience_conn)
    rows.append(dict(method="greedy-conn", seed=seed, latency=gcsteps, **_metrics(sc, gcx)))

    return rows


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 1   # parallel worker processes
    all_rows = []
    if jobs > 1:
        from multiprocessing import Pool
        with Pool(jobs) as pool:
            for rows in pool.imap_unordered(run_seed, range(n_seeds)):
                all_rows.extend(rows)
                surv = {r["method"]: r["survivable"] for r in rows}
                print(f"seed {rows[0]['seed']:2d}  " + "  ".join(f"{m}={surv[m]:.3f}" for m in surv))
    else:
        for s in range(n_seeds):
            rows = run_seed(s)
            all_rows.extend(rows)
            surv = {r["method"]: r["survivable"] for r in rows}
            print(f"seed {s:2d}  " + "  ".join(f"{m}={surv[m]:.3f}" for m in surv))

    # Write CSV (no pandas dependency needed).
    os.makedirs(OUT_DIR, exist_ok=True)
    cols = ["method", "seed", "survivable", "conn", "lam2", "energy", "edges", "latency"]
    path = os.path.join(OUT_DIR, "cs1_results.csv")
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in all_rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nwrote {path}")

    # Persist QIEA convergence histories (best-so-far per generation) for the synergy figure.
    for tag, meth in [("ai", "QIEA+AI"), ("noai", "QIEA-noAI")]:
        hs = [r["history"] for r in all_rows if r["method"] == meth]
        hp = os.path.join(OUT_DIR, f"hist_{tag}.csv")
        with open(hp, "w") as f:
            for h in hs:
                f.write(",".join(f"{v:.6f}" for v in h) + "\n")

    # Summary: mean +/- std per method, plus Wilcoxon vs QIEA+AI on survivable fraction.
    methods = ["QIEA+AI", "QIEA-noAI", "GA", "greedy", "greedy-conn"]
    by = {m: [r for r in all_rows if r["method"] == m] for m in methods}
    print("\nmethod        survivable        lam2          energy     latency")
    for m in methods:
        sv = np.array([r["survivable"] for r in by[m]])
        lam = np.array([r["lam2"] for r in by[m]])
        en = np.array([r["energy"] for r in by[m]])
        lt = np.array([r["latency"] for r in by[m]])
        print(f"{m:12s}  {sv.mean():.3f}+/-{sv.std():.3f}   {lam.mean():.4f}      "
              f"{en.mean():.3f}    {lt.mean():.1f}")

    try:
        from scipy.stats import wilcoxon
        ref = np.array([r["survivable"] for r in by["QIEA+AI"]])
        print("\nWilcoxon (survivable, QIEA+AI vs ...):")
        for m in methods[1:]:
            other = np.array([r["survivable"] for r in by[m]])
            if np.allclose(ref, other):
                print(f"  vs {m:12s}: identical")
                continue
            stat, p = wilcoxon(ref, other)
            print(f"  vs {m:12s}: p={p:.4g}")
    except Exception as e:
        print(f"\n(scipy not available for Wilcoxon: {e})")


if __name__ == "__main__":
    main()
