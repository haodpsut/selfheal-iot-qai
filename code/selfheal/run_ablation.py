"""Ablation studies for CS1 (the combinatorial survivability case).

Three ablations, each over N seeds, written to results/ablation/ as CSV and printed as a
table:

  A. Synergy components: isolate the AI warm-start and the persistent search-pruning pull.
       none            : uniform prior, no pull        (no AI guidance at all)
       warm-start      : AI prior, no pull             (warm-start only)
       warm-start+pull : AI prior + persistent pull    (full, the default)

  B. Robustness term weight W in the composite objective (survivable + W*lambda2):
       W in {0, 1, 2, 5}. W=0 ignores spectral robustness.

  C. Budget sensitivity: repair budget as a fraction of total candidate-link energy,
       frac in {0.025, 0.035, 0.05, 0.07}. Traces the resilience vs energy trade-off.

    python run_ablation.py 15 6     # 15 seeds, 6 workers
"""

from __future__ import annotations
import sys
import os
import numpy as np

from qio import optimize, QIEAConfig
import cs1_topology as C

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "ablation")
GEN, POP = 120, 40
N_NODES, FAIL_FRAC = 200, 0.32
DEFAULT_BUDGET = 0.035


def _latency(h, frac=0.95):
    t = frac * h[-1]
    hit = np.where(h >= t)[0]
    return int(hit[0]) if hit.size else len(h)


def _surv(sc, x):
    e = np.concatenate([sc.base_edges, sc.cand_edges[x == 1]], axis=0)
    return C._survivable_fraction(sc.n_nodes, sc.alive, e)


def _lam(sc, x):
    e = np.concatenate([sc.base_edges, sc.cand_edges[x == 1]], axis=0)
    return C._algebraic_connectivity(sc.n_nodes, sc.alive, e)


def run_seed(seed: int) -> list:
    rows = []
    sc = C.make_scenario(n_nodes=N_NODES, fail_frac=FAIL_FRAC, seed=seed)
    C.set_binding_budget(sc, frac=DEFAULT_BUDGET)
    fit = C.make_fitness(sc)
    prior = C.ai_prior(sc)
    uniform = np.full(len(sc.cand_edges), float(prior.mean()))

    # A. Synergy components.
    variants = [
        ("none", uniform, 0.0),
        ("warm-start", prior, 0.0),
        ("warm-start+pull", prior, 0.02),
    ]
    for name, pr, pull in variants:
        cfg = QIEAConfig(pop_size=POP, generations=GEN, seed=seed, prior_pull=pull)
        r = optimize(fit, len(sc.cand_edges), cfg, prior=pr)
        rows.append(dict(ablation="A_components", setting=name, seed=seed,
                         survivable=_surv(sc, r.best_x), latency=_latency(r.history)))

    # B. Robustness term weight W.
    for w in [0.0, 1.0, 2.0, 5.0]:
        fitw = C.make_fitness(sc, w=w)
        cfg = QIEAConfig(pop_size=POP, generations=GEN, seed=seed)
        r = optimize(fitw, len(sc.cand_edges), cfg, prior=prior)
        rows.append(dict(ablation="B_weight", setting=f"W={w:g}", seed=seed,
                         survivable=_surv(sc, r.best_x), lam2=_lam(sc, r.best_x)))

    # C. Budget sensitivity (fresh scenario per frac so budgets are comparable).
    for frac in [0.025, 0.035, 0.05, 0.07]:
        scb = C.make_scenario(n_nodes=N_NODES, fail_frac=FAIL_FRAC, seed=seed)
        C.set_binding_budget(scb, frac=frac)
        fitb = C.make_fitness(scb)
        priorb = C.ai_prior(scb)
        cfg = QIEAConfig(pop_size=POP, generations=GEN, seed=seed)
        r = optimize(fitb, len(scb.cand_edges), cfg, prior=priorb)
        e = scb.cand_cost[r.best_x == 1].sum()
        rows.append(dict(ablation="C_budget", setting=f"frac={frac:g}", seed=seed,
                         survivable=_surv(scb, r.best_x), energy=float(e), budget=float(scb.budget)))
    return rows


def _summ(rows, ablation, metrics):
    settings = []
    for r in rows:
        if r["ablation"] == ablation and r["setting"] not in settings:
            settings.append(r["setting"])
    print(f"\n[{ablation}]")
    for s in settings:
        sub = [r for r in rows if r["ablation"] == ablation and r["setting"] == s]
        cells = []
        for m in metrics:
            vals = np.array([r[m] for r in sub if m in r])
            if vals.size:
                cells.append(f"{m}={vals.mean():.3f}+/-{vals.std():.3f}")
        print(f"  {s:18s} " + "  ".join(cells))


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    all_rows = []
    if jobs > 1:
        from multiprocessing import Pool
        with Pool(jobs) as pool:
            for rows in pool.imap_unordered(run_seed, range(n_seeds)):
                all_rows.extend(rows)
                print(f"seed {rows[0]['seed']:2d} done")
    else:
        for s in range(n_seeds):
            all_rows.extend(run_seed(s))
            print(f"seed {s:2d} done")

    os.makedirs(OUT_DIR, exist_ok=True)
    cols = ["ablation", "setting", "seed", "survivable", "latency", "lam2", "energy", "budget"]
    path = os.path.join(OUT_DIR, "ablation.csv")
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in all_rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\nwrote {path}")

    _summ(all_rows, "A_components", ["survivable", "latency"])
    _summ(all_rows, "B_weight", ["survivable", "lam2"])
    _summ(all_rows, "C_budget", ["survivable", "energy"])


if __name__ == "__main__":
    main()
