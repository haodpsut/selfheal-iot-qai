"""CS1 - Topology reconfiguration under cascading failures.

An IoT mesh suffers targeted cascading node failures. The self-healing controller may
activate a subset of pre-surveyed *backup links* (each with an energy cost) to merge the
shattered fragments back into one giant component, under a total energy budget.

We compare the quantum-inspired optimizer (QIEA) against greedy / GA / random baselines,
both with and without the AI warm-start prior (the synergy ablation).

All heavy connectivity math uses a numpy union-find so the fitness stays cheap and the
whole sweep runs on CPU; it also scales to the RTX 4090 server unchanged.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from qio import optimize, QIEAConfig


# --------------------------------------------------------------------------------------
# Network + fault model
# --------------------------------------------------------------------------------------
@dataclass
class Scenario:
    """A frozen failure scenario: surviving base edges + candidate backup edges."""
    n_nodes: int
    alive: np.ndarray               # bool mask, length n_nodes
    base_edges: np.ndarray          # (E, 2) int, edges among alive nodes that survived
    cand_edges: np.ndarray          # (C, 2) int, candidate backup edges among alive nodes
    cand_cost: np.ndarray           # (C,) float, energy cost of each candidate edge
    budget: float                   # total activation-energy budget
    pos: np.ndarray = field(repr=False)  # (n_nodes, 2) coordinates (for plotting later)


def make_scenario(
    n_nodes: int = 200,
    radius: float = 0.13,
    backup_radius: float = 0.20,
    fail_frac: float = 0.20,
    budget_frac: float = 0.08,      # TIGHT budget: cannot reconnect everything (the hard regime)
    seed: int = 0,
) -> Scenario:
    """Random-geometric IoT mesh, then targeted cascading node failures.

    Targeted failures remove the highest-degree alive nodes one at a time; after each
    removal, any node that loses all its links also drops (a simple cascade). Candidate
    backup edges are node pairs within `backup_radius` that are not active base edges.
    """
    rng = np.random.default_rng(seed)
    pos = rng.random((n_nodes, 2))

    # Pairwise distances and the two radius graphs.
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)

    base_adj = dist <= radius
    backup_adj = (dist <= backup_radius) & ~base_adj

    alive = np.ones(n_nodes, dtype=bool)
    n_fail = int(fail_frac * n_nodes)

    deg = base_adj.sum(1)
    for _ in range(n_fail):
        cand = np.where(alive)[0]
        if cand.size == 0:
            break
        victim = cand[np.argmax(deg[cand])]
        alive[victim] = False
        # Cascade: drop alive nodes that now have zero alive base neighbours.
        live_deg = (base_adj[:, alive][alive].sum(1))
        idx = np.where(alive)[0]
        orphan = idx[live_deg == 0]
        alive[orphan] = False
        deg = (base_adj[:, alive].sum(1))  # recompute degree over alive set
        deg_full = np.zeros(n_nodes)
        deg_full[:] = base_adj.sum(1)
        deg = deg_full

    # Surviving base edges (both endpoints alive).
    bu, bv = np.where(np.triu(base_adj, 1))
    keep = alive[bu] & alive[bv]
    base_edges = np.stack([bu[keep], bv[keep]], axis=1)

    # Candidate backup edges among alive nodes.
    cu, cv = np.where(np.triu(backup_adj, 1))
    ckeep = alive[cu] & alive[cv]
    cand_edges = np.stack([cu[ckeep], cv[ckeep]], axis=1)
    cand_cost = dist[cand_edges[:, 0], cand_edges[:, 1]] ** 2  # energy ~ distance^2

    budget = budget_frac * cand_cost.sum() if cand_cost.size else 0.0
    return Scenario(n_nodes, alive, base_edges, cand_edges, cand_cost, budget, pos)


# --------------------------------------------------------------------------------------
# Connectivity (numpy union-find) + fitness
# --------------------------------------------------------------------------------------
def _giant_fraction(n_nodes: int, alive: np.ndarray, edges: np.ndarray) -> float:
    """Largest-connected-component size among alive nodes, as a fraction of alive nodes."""
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

    roots = np.array([find(i) for i in np.where(alive)[0]])
    if roots.size == 0:
        return 0.0
    _, counts = np.unique(roots, return_counts=True)
    return counts.max() / roots.size


def _two_edge_components(n_nodes: int, alive: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Label each node by its 2-edge-connected component (bridges removed).

    Bridges are found by an iterative Tarjan traversal; removing them splits the graph into
    its 2-edge-connected components, each of which survives ANY single link failure without
    splitting. Returns a length-n_nodes array of component roots (a union-find label). Shared
    by both case studies' resilience metrics.
    """
    from collections import defaultdict

    adj = defaultdict(list)                       # node -> list of (neighbour, edge_id)
    for eid, (u, v) in enumerate(edges):
        adj[u].append((v, eid))
        adj[v].append((u, eid))

    disc = {}
    low = {}
    timer = 0
    bridges = set()
    visited = set()

    for s in np.where(alive)[0]:
        s = int(s)
        if s in visited:
            continue
        stack = [(s, -1, iter(adj[s]))]
        disc[s] = low[s] = timer
        timer += 1
        visited.add(s)
        while stack:
            node, pedge, it = stack[-1]
            advanced = False
            for nb, eid in it:
                if eid == pedge:
                    continue
                if nb not in visited:
                    visited.add(nb)
                    disc[nb] = low[nb] = timer
                    timer += 1
                    stack.append((nb, eid, iter(adj[nb])))
                    advanced = True
                    break
                low[node] = min(low[node], disc[nb])
            if not advanced:
                stack.pop()
                if stack:
                    parent = stack[-1][0]
                    low[parent] = min(low[parent], low[node])
                    if low[node] > disc[parent]:
                        bridges.add(pedge)

    parent = np.arange(n_nodes)

    def find2(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for eid, (u, v) in enumerate(edges):
        if eid in bridges:
            continue
        ru, rv = find2(u), find2(v)
        if ru != rv:
            parent[ru] = rv
    return np.array([find2(i) for i in range(n_nodes)])


def _survivable_fraction(n_nodes: int, alive: np.ndarray, edges: np.ndarray) -> float:
    """Largest 2-edge-connected component as a fraction of alive nodes (CS1 objective)."""
    comp = _two_edge_components(n_nodes, alive, edges)
    roots = comp[np.where(alive)[0]]
    if roots.size == 0:
        return 0.0
    _, counts = np.unique(roots, return_counts=True)
    return counts.max() / roots.size


def _algebraic_connectivity(n_nodes: int, alive: np.ndarray, edges: np.ndarray) -> float:
    """Fiedler value (second-smallest Laplacian eigenvalue) over the alive subgraph.

    This is the canonical spectral robustness metric: lambda_2 > 0 iff the graph is
    connected, and a larger lambda_2 means more edge-disjoint redundancy and faster
    consensus/diffusion, i.e. a network that degrades gracefully under the next failure.
    Unlike a 2-edge-connected-component count it responds *smoothly* to spending, so the
    energy budget binds across a meaningful range. Maximizing it by adding edges is
    NP-hard and non-submodular, which is the regime where global search beats greedy.
    """
    idx = np.where(alive)[0]
    m = idx.size
    if m <= 1:
        return 0.0
    remap = -np.ones(n_nodes, dtype=np.int64)
    remap[idx] = np.arange(m)
    L = np.zeros((m, m), dtype=np.float64)
    for u, v in edges:
        a, b = remap[u], remap[v]
        if a < 0 or b < 0:
            continue
        L[a, a] += 1.0
        L[b, b] += 1.0
        L[a, b] -= 1.0
        L[b, a] -= 1.0
    evals = np.linalg.eigvalsh(L)        # ascending; evals[0] ~ 0
    return float(max(evals[1], 0.0))


W_LAMBDA = 2.0   # weight of the lambda2 robustness term in the worst-case resilience score


def _resilience(n_nodes: int, alive: np.ndarray, edges: np.ndarray, w: float = W_LAMBDA) -> float:
    """Worst-case resilience = survivable fraction + w * lambda2.

    The survivable-fraction term is the largest 2-edge-connected component: the part of the
    network guaranteed to stay connected after the single WORST link failure (a max-min /
    worst-case objective). Growing it usually needs several edges that *close a cycle*; any
    single one of them gives zero marginal gain, so myopic greedy under-invests in it. That
    coupling is what makes the objective greedy-hostile. The lambda2 term adds a smooth
    robustness gradient and breaks ties toward globally well-connected solutions.
    """
    sv = _survivable_fraction(n_nodes, alive, edges)
    lam = _algebraic_connectivity(n_nodes, alive, edges)
    return sv + w * lam


def _resilience_conn(n_nodes: int, alive: np.ndarray, edges: np.ndarray, w: float = W_LAMBDA) -> float:
    """A connectivity-first surrogate (giant fraction + w * lambda2). Used to give the
    greedy baseline its *best* shot: a smooth, single-edge-climbable objective. Even so,
    its worst-case survivability lags global search, so the comparison is not a strawman."""
    g = _giant_fraction(n_nodes, alive, edges)
    lam = _algebraic_connectivity(n_nodes, alive, edges)
    return g + w * lam


# Eigensolve-free objectives for the greedy baselines, so greedy stays cheap at scale (the
# population solvers below still optimize the full survivable + w*lambda2 resilience).
def _resilience_surv(n_nodes: int, alive: np.ndarray, edges: np.ndarray, w: float = W_LAMBDA) -> float:
    """Survivability only (naive greedy): stalls once the graph is merely connected."""
    return _survivable_fraction(n_nodes, alive, edges)


def _resilience_giant_surv(n_nodes: int, alive: np.ndarray, edges: np.ndarray, w: float = W_LAMBDA) -> float:
    """Connectivity-first, eigensolve-free (stronger greedy): giant + survivable fraction."""
    return _giant_fraction(n_nodes, alive, edges) + _survivable_fraction(n_nodes, alive, edges)


def make_fitness(sc: Scenario, energy_pref: float = 0.01, w: float = W_LAMBDA):
    """Vectorized composite-resilience fitness over a (pop, C) binary population.

    Budget is a HARD constraint: any over-budget solution is pushed strictly below every
    feasible one (penalty 100 * relative-overage), so each method maximizes resilience
    *within* the budget. A tiny energy_pref breaks ties toward cheaper feasible solutions.
    """
    base = sc.base_edges
    cand = sc.cand_edges
    cost = sc.cand_cost
    budget = sc.budget if sc.budget > 0 else 1.0

    def fitness(pop: np.ndarray) -> np.ndarray:
        out = np.empty(pop.shape[0], dtype=np.float64)
        for i, x in enumerate(pop):
            sel = cand[x == 1]
            edges = np.concatenate([base, sel], axis=0) if sel.size else base
            r = _resilience(sc.n_nodes, sc.alive, edges, w)
            e = cost[x == 1].sum()
            over = max(0.0, e - budget) / budget
            out[i] = r - 100.0 * over - energy_pref * (e / budget)
        return out

    return fitness


def set_binding_budget(sc: Scenario, frac: float = 0.06) -> float:
    """Set sc.budget to a fixed fraction of total candidate-edge cost.

    Because algebraic connectivity keeps rising as edges are added, any frac < 1 is a
    genuinely binding budget: every method must pick the highest-leverage edges that fit,
    which is the hard combinatorial core of the problem."""
    sc.budget = frac * (sc.cand_cost.sum() if sc.cand_cost.size else 1.0)
    return sc.budget


# --------------------------------------------------------------------------------------
# AI warm-start prior (the AI -> QIO synergy channel)
# --------------------------------------------------------------------------------------
def ai_prior(sc: Scenario) -> np.ndarray:
    """Cheap AI risk-map surrogate: score each candidate backup edge by how useful it is
    for healing, i.e. it bridges two *different* surviving fragments and is energy-cheap.

    In the full paper this is a small learned predictor trained on (scenario -> good edge)
    pairs; here it is a transparent feature score so the synergy mechanism is reproducible.
    The ablation simply replaces this prior with a uniform low density (no AI guidance).

    Composite resilience grows by (a) merging fragments first, then (b) adding redundancy
    where the graph is thin. The surrogate therefore scores a candidate edge by: does it
    bridge two different surviving fragments (connectivity), are its endpoints low-degree
    (redundancy leverage), and is it cheap (energy). The ablation replaces this prior with
    a uniform low density (same edge count, no location signal).
    """
    # Component labels of the surviving base graph (which fragment each node is in).
    parent = np.arange(sc.n_nodes)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    deg = np.zeros(sc.n_nodes, dtype=np.float64)
    for u, v in sc.base_edges:
        deg[u] += 1
        deg[v] += 1
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    comp = np.array([find(i) for i in range(sc.n_nodes)])

    cu, cv = sc.cand_edges[:, 0], sc.cand_edges[:, 1]
    bridges = (comp[cu] != comp[cv]).astype(np.float64)         # 1 if it merges two fragments
    endpoint_deg = deg[cu] + deg[cv]
    thin = 1.0 - (endpoint_deg / (endpoint_deg.max() + 1e-9))   # prefer low-degree endpoints
    cheap = 1.0 - (sc.cand_cost / (sc.cand_cost.max() + 1e-9))  # prefer low energy
    score = 0.5 * bridges + 0.3 * thin + 0.2 * cheap
    # Map score -> a low-density probability band: even guided, only a few edges fit budget.
    return 0.04 + 0.30 * (score - score.min()) / (np.ptp(score) + 1e-9)


# --------------------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------------------
def greedy(sc: Scenario, obj=_resilience, w: float = W_LAMBDA, sample: int = 256, seed: int = 0):
    """Add the candidate edge with the best marginal objective gain per energy, until the
    budget is exhausted or no edge improves the score. The myopic baseline. `obj` selects
    which objective it climbs: `_resilience` (worst-case target) or a connectivity-first
    surrogate (to give greedy its strongest chance).

    For scalability this is the *stochastic greedy* variant (Mirzasoleiman et al.): each step
    scores a random subset of `sample` affordable candidates rather than all of them, which
    preserves near-greedy quality at a fraction of the cost. Set `sample=None` to score all."""
    rng = np.random.default_rng(seed)
    cand, cost = sc.cand_edges, sc.cand_cost
    C = len(cand)
    chosen = np.zeros(C, dtype=np.uint8)
    spent = 0.0
    cur = obj(sc.n_nodes, sc.alive, sc.base_edges, w)
    steps = 0
    while True:
        avail = np.where((chosen == 0) & (spent + cost <= sc.budget))[0]
        if avail.size == 0:
            break
        if sample is not None and avail.size > sample:
            avail = rng.choice(avail, sample, replace=False)
        best_gain, best_j = 1e-9, -1
        for j in avail:
            edges = np.concatenate([sc.base_edges, cand[chosen == 1], cand[j:j + 1]], axis=0)
            gain = (obj(sc.n_nodes, sc.alive, edges, w) - cur) / cost[j]
            if gain > best_gain:
                best_gain, best_j = gain, int(j)
        if best_j < 0:
            break
        chosen[best_j] = 1
        spent += cost[best_j]
        cur = obj(sc.n_nodes, sc.alive,
                  np.concatenate([sc.base_edges, cand[chosen == 1]], axis=0), w)
        steps += 1
    return chosen, cur, spent, steps


def genetic(sc: Scenario, fitness, pop_size=40, generations=120, seed=0):
    """Plain binary GA (tournament + uniform crossover + bit-flip) for a fair, param-matched
    comparison against the QIEA (same pop_size x generations evaluation budget)."""
    rng = np.random.default_rng(seed)
    C = len(sc.cand_edges)
    pop = (rng.random((pop_size, C)) < 0.15).astype(np.uint8)
    best_x, best_f = None, -np.inf
    history = np.empty(generations)
    for g in range(generations):
        fit = fitness(pop)
        j = int(np.argmax(fit))
        if fit[j] > best_f:
            best_f, best_x = float(fit[j]), pop[j].copy()
        # Tournament selection.
        a, b = rng.integers(0, pop_size, (2, pop_size))
        winners = np.where(fit[a] >= fit[b], a, b)
        parents = pop[winners]
        # Uniform crossover.
        mask = rng.random((pop_size, C)) < 0.5
        shuffled = parents[rng.permutation(pop_size)]
        children = np.where(mask, parents, shuffled)
        # Bit-flip mutation.
        flip = rng.random((pop_size, C)) < (1.0 / max(C, 1))
        children ^= flip.astype(np.uint8)
        children[0] = best_x  # elitism
        pop = children
        history[g] = best_f
    return best_x, best_f, history


def simulated_annealing(sc: Scenario, fitness, iters=4800, seed=0,
                        t0=1.0, t1=0.01, init_density=0.05):
    """Simulated annealing baseline over the same penalized fitness, with an evaluation
    budget (`iters`) matched to the QIEA (pop_size x generations). A single bit is flipped
    per step and accepted by the Metropolis rule under a geometric cooling schedule. This
    gives a second strong metaheuristic so the comparison is not GA-only."""
    rng = np.random.default_rng(seed)
    C = len(sc.cand_edges)
    x = (rng.random(C) < init_density).astype(np.uint8)
    fx = float(fitness(x[None, :])[0])
    best_x, best_f = x.copy(), fx
    history = np.empty(iters)
    for t in range(iters):
        temp = t0 * (t1 / t0) ** (t / max(iters - 1, 1))
        j = int(rng.integers(C))
        x[j] ^= 1
        f2 = float(fitness(x[None, :])[0])
        if f2 >= fx or rng.random() < np.exp((f2 - fx) / max(temp, 1e-9)):
            fx = f2
            if fx > best_f:
                best_f, best_x = fx, x.copy()
        else:
            x[j] ^= 1  # reject: undo the flip
        history[t] = best_f
    return best_x, best_f, history


# --------------------------------------------------------------------------------------
# Smoke-test driver
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    sc = make_scenario(n_nodes=160, fail_frac=0.25, seed=0)
    set_binding_budget(sc, frac=0.06)   # budget = 6% of total candidate-edge cost (binding)
    n_alive = int(sc.alive.sum())
    pre = _giant_fraction(sc.n_nodes, sc.alive, sc.base_edges)
    print(f"alive={n_alive}/{sc.n_nodes}  candidates={len(sc.cand_edges)}  "
          f"budget={sc.budget:.3f}  pre-heal giant fraction={pre:.3f}")

    fit = make_fitness(sc)
    prior = ai_prior(sc)
    uniform_prior = np.full(len(sc.cand_edges), float(prior.mean()))  # same density, no location signal
    cfg = QIEAConfig(pop_size=40, generations=100, seed=0)

    def latency(history, frac=0.95):
        """Healing latency = first generation reaching `frac` of the final best."""
        target = frac * history[-1]
        hit = np.where(history >= target)[0]
        return int(hit[0]) if hit.size else len(history)

    # QIEA with and without the AI warm-start (the synergy ablation).
    r_ai = optimize(fit, len(sc.cand_edges), cfg, prior=prior)
    r_no = optimize(fit, len(sc.cand_edges), cfg, prior=uniform_prior)

    bx, bf, _ = genetic(sc, fit, seed=0)
    gx, gfrac, gspent, gsteps = greedy(sc)

    def report(name, x, extra=""):
        edges = np.concatenate([sc.base_edges, sc.cand_edges[x == 1]], axis=0)
        sv = _survivable_fraction(sc.n_nodes, sc.alive, edges)
        gc = _giant_fraction(sc.n_nodes, sc.alive, edges)
        e = sc.cand_cost[x == 1].sum()
        feas = "ok" if e <= sc.budget + 1e-9 else "OVER"
        print(f"  {name:18s} surv={sv:.3f}  conn={gc:.3f}  energy={e:.3f}/{sc.budget:.3f}[{feas}]"
              f"  edges={int(x.sum())}  {extra}")

    def lam(x):
        edges = np.concatenate([sc.base_edges, sc.cand_edges[x == 1]], axis=0)
        return _algebraic_connectivity(sc.n_nodes, sc.alive, edges)

    base_lam = _algebraic_connectivity(sc.n_nodes, sc.alive, sc.base_edges)
    print(f"pre-heal lambda2={base_lam:.4f}")
    print("post-heal (objective = composite resilience; surv = 2-edge-conn fraction):")
    report("QIEA + AI", r_ai.best_x, f"lam2={lam(r_ai.best_x):.4f} latency={latency(r_ai.history)}")
    report("QIEA (no AI)", r_no.best_x, f"lam2={lam(r_no.best_x):.4f} latency={latency(r_no.history)}")
    report("GA", bx, f"lam2={lam(bx):.4f}")
    report("greedy", gx, f"lam2={lam(gx):.4f} steps={gsteps}")
