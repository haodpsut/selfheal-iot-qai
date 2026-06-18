"""CS2 - Restoring device-to-gateway resilience under spatial link jamming.

An adversary jams a geographic region, knocking out every wireless link whose midpoint falls
inside the jammed disk, so a cluster of IoT links disappears at once. The self-healing
controller activates backup links from a pre surveyed set, under an energy budget, to restore
*device-to-gateway survivability*: the fraction of IoT devices that retain two edge disjoint
paths to a gateway and therefore stay reachable after any single further link failure.

This is a second, structurally distinct threat from CS1 (adversarial regional link jamming
versus random node cascades) and a different objective (sink oriented two edge connectivity
versus a global survivable core), but it is the same kind of combinatorial, non submodular
repair, so it exercises the same quantum-inspired solver and AI synergy. We reuse the CS1
machinery: the Scenario container, the bridge based two edge component finder, the algebraic
connectivity, and the QIEA / GA / SA / greedy solvers.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

from qio import optimize, QIEAConfig
from cs1_topology import (
    Scenario, set_binding_budget, genetic, greedy, simulated_annealing,
    _two_edge_components, _algebraic_connectivity,
)

W_LAMBDA = 2.0


# --------------------------------------------------------------------------------------
# Scenario: geometric IoT graph + gateways + spatial link jamming
# --------------------------------------------------------------------------------------
def make_jam_scenario(
    n_nodes: int = 220,
    radius: float = 0.115,
    backup_radius: float = 0.21,
    n_gateways: int = 2,
    jam_frac: float = 0.25,
    budget_frac: float = 0.06,
    seed: int = 0,
) -> Tuple[Scenario, np.ndarray]:
    """Random geometric IoT mesh, a few gateways, then a spatial jamming disk that removes a
    contiguous block of links. Returns (Scenario, gateways)."""
    rng = np.random.default_rng(seed)
    pos = rng.random((n_nodes, 2))

    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    base_adj = dist <= radius
    backup_adj = (dist <= backup_radius) & ~base_adj

    gateways = rng.choice(n_nodes, n_gateways, replace=False)

    bu, bv = np.where(np.triu(base_adj, 1))
    base_edges = np.stack([bu, bv], axis=1)

    # Spatial jamming: drop the base edges whose midpoints are closest to a random center,
    # i.e. a contiguous jammed region rather than scattered links.
    center = rng.random(2)
    mid = 0.5 * (pos[base_edges[:, 0]] + pos[base_edges[:, 1]])
    d2c = np.sqrt(((mid - center) ** 2).sum(1))
    order = np.argsort(d2c, kind="stable")
    n_jam = int(jam_frac * len(base_edges))
    keep = np.ones(len(base_edges), dtype=bool)
    keep[order[:n_jam]] = False
    base_edges = base_edges[keep]

    cu, cv = np.where(np.triu(backup_adj, 1))
    cand_edges = np.stack([cu, cv], axis=1)
    cand_cost = dist[cand_edges[:, 0], cand_edges[:, 1]] ** 2

    alive = np.ones(n_nodes, dtype=bool)
    budget = budget_frac * cand_cost.sum() if cand_cost.size else 0.0
    sc = Scenario(n_nodes, alive, base_edges, cand_edges, cand_cost, budget, pos)
    return sc, gateways


# --------------------------------------------------------------------------------------
# Objective: fraction of devices two-edge-connected to a gateway
# --------------------------------------------------------------------------------------
def _gateway_survivable(n_nodes, alive, edges, gateways) -> float:
    """Fraction of devices (non-gateway alive nodes) sharing a 2-edge-connected component
    with some gateway, i.e. with two edge disjoint paths to a gateway."""
    comp = _two_edge_components(n_nodes, alive, edges)
    gw_comps = set(int(comp[int(g)]) for g in gateways)
    devices = [int(i) for i in np.where(alive)[0] if i not in set(int(g) for g in gateways)]
    if not devices:
        return 0.0
    hit = sum(1 for d in devices if int(comp[d]) in gw_comps)
    return hit / len(devices)


def _gateway_connected(n_nodes, alive, edges, gateways) -> float:
    """Fraction of devices merely connected (one path) to a gateway. Used to give the greedy
    baseline a friendly connectivity-first surrogate, so it is not a straw baseline."""
    parent = np.arange(n_nodes)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    gw_comps = set(find(int(g)) for g in gateways)
    gws = set(int(g) for g in gateways)
    devices = [int(i) for i in np.where(alive)[0] if i not in gws]
    if not devices:
        return 0.0
    return sum(1 for d in devices if find(d) in gw_comps) / len(devices)


W_CONN = 0.5   # weight of the one-path connected term as a smooth gradient toward 2-edge


def gw_resilience(n, alive, edges, w, gateways):
    """Main objective: device-to-gateway two edge survivability, plus a one-path connected
    term as a cheap smooth gradient. Pure union-find and bridge finding, no eigensolve, so it
    scales. Algebraic connectivity is reported separately as a metric but not optimized here."""
    return _gateway_survivable(n, alive, edges, gateways) + w * _gateway_connected(n, alive, edges, gateways)


def gw_surv_only(n, alive, edges, w, gateways):
    """Survivability only, for the naive greedy baseline (it stalls once merely connected)."""
    return _gateway_survivable(n, alive, edges, gateways)


def gw_resilience_conn(n, alive, edges, w, gateways):
    """Connectivity-first surrogate for the stronger greedy baseline."""
    return _gateway_connected(n, alive, edges, gateways) + _gateway_survivable(n, alive, edges, gateways)


def make_fitness(sc: Scenario, gateways, w: float = W_CONN, energy_pref: float = 0.0):
    base, cand, cost = sc.base_edges, sc.cand_edges, sc.cand_cost
    budget = sc.budget if sc.budget > 0 else 1.0

    def fitness(pop: np.ndarray) -> np.ndarray:
        out = np.empty(pop.shape[0], dtype=np.float64)
        for i, x in enumerate(pop):
            sel = cand[x == 1]
            edges = np.concatenate([base, sel], axis=0) if sel.size else base
            r = gw_resilience(sc.n_nodes, sc.alive, edges, w, gateways)
            e = cost[x == 1].sum()
            over = max(0.0, e - budget) / budget
            out[i] = r - 100.0 * over - energy_pref * (e / budget)
        return out

    return fitness


# --------------------------------------------------------------------------------------
# AI warm-start prior: favour cheap backups that reconnect fragments toward a gateway
# --------------------------------------------------------------------------------------
def ai_prior(sc: Scenario, gateways) -> np.ndarray:
    parent = np.arange(sc.n_nodes)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    deg = np.zeros(sc.n_nodes)
    for u, v in sc.base_edges:
        deg[u] += 1
        deg[v] += 1
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    comp = np.array([find(i) for i in range(sc.n_nodes)])
    gw_comps = set(int(comp[int(g)]) for g in gateways)

    cu, cv = sc.cand_edges[:, 0], sc.cand_edges[:, 1]
    in_gw_u = np.array([int(comp[int(a)]) in gw_comps for a in cu])
    in_gw_v = np.array([int(comp[int(b)]) in gw_comps for b in cv])
    toward = (in_gw_u ^ in_gw_v).astype(np.float64)        # links a fragment to a gateway side
    thin = 1.0 - (deg[cu] + deg[cv]) / ((deg[cu] + deg[cv]).max() + 1e-9)
    cheap = 1.0 - (sc.cand_cost / (sc.cand_cost.max() + 1e-9))
    score = 0.5 * toward + 0.3 * thin + 0.2 * cheap
    return 0.04 + 0.30 * (score - score.min()) / (np.ptp(score) + 1e-9)


# --------------------------------------------------------------------------------------
# Smoke-test driver
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    sc, gw = make_jam_scenario(seed=0)
    set_binding_budget(sc, frac=0.06)
    pre = _gateway_survivable(sc.n_nodes, sc.alive, sc.base_edges, gw)
    print(f"nodes={sc.n_nodes} gateways={list(gw)} candidates={len(sc.cand_edges)} "
          f"budget={sc.budget:.3f} pre-heal gateway-survivable={pre:.3f}")

    fit = make_fitness(sc, gw)
    prior = ai_prior(sc, gw)
    uniform = np.full(len(sc.cand_edges), float(prior.mean()))
    cfg = QIEAConfig(pop_size=40, generations=100, seed=0)

    r_ai = optimize(fit, len(sc.cand_edges), cfg, prior=prior)
    r_no = optimize(fit, len(sc.cand_edges), cfg, prior=uniform)
    bx, _, _ = genetic(sc, fit, seed=0)
    sx, _, _ = simulated_annealing(sc, fit, iters=4000, seed=0)
    gx, _, _, _ = greedy(sc, obj=lambda n, a, e, w: gw_surv_only(n, a, e, w, gw))
    gcx, _, _, _ = greedy(sc, obj=lambda n, a, e, w: gw_resilience_conn(n, a, e, w, gw))

    def rep(name, x):
        edges = np.concatenate([sc.base_edges, sc.cand_edges[x == 1]], axis=0)
        s = _gateway_survivable(sc.n_nodes, sc.alive, edges, gw)
        e = sc.cand_cost[x == 1].sum()
        print(f"  {name:14s} gw-survivable={s:.3f}  energy={e:.3f}/{sc.budget:.3f}")

    print("post-heal (device-to-gateway survivable fraction):")
    rep("QIEA + AI", r_ai.best_x)
    rep("QIEA (no AI)", r_no.best_x)
    rep("GA", bx)
    rep("SA", sx)
    rep("greedy", gx)
    rep("greedy-conn", gcx)
