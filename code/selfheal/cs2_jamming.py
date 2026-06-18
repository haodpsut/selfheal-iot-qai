"""CS2 - Resource reallocation under a jamming surge.

A set of IoT devices send traffic to gateways over wireless channels. An attacker jams a
subset of channels (their usable rate drops to zero) at the same moment a traffic surge
raises some devices' demand. The self-healing controller must re-assign devices to the
surviving channels to maximise served demand under a transmit-power budget, while each
channel has finite capacity and each device uses at most one channel.

This is a multiply-constrained generalized assignment problem (NP-hard). As in CS1 the QIEA
selects device-channel pairings; the AI warm-start predicts the jammed channels and the
surged devices and biases the search. Greedy / GA / random are the baselines.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from qio import optimize, QIEAConfig


# --------------------------------------------------------------------------------------
# Scenario
# --------------------------------------------------------------------------------------
@dataclass
class JamScenario:
    n_dev: int
    n_chan: int
    demand: np.ndarray          # (n_dev,) traffic demand after the surge
    rate: np.ndarray            # (n_dev, n_chan) achievable rate (0 on jammed channels)
    power: np.ndarray           # (n_dev, n_chan) transmit power to use that channel
    cap: np.ndarray             # (n_chan,) channel capacity
    budget: float               # total transmit-power budget
    pairs: np.ndarray           # (P, 2) candidate (device, channel) pairings, rate > 0
    jammed: np.ndarray = field(repr=False)   # (n_chan,) bool
    surged: np.ndarray = field(repr=False)   # (n_dev,) bool


def make_jam_scenario(
    n_dev: int = 120,
    n_chan: int = 16,
    jam_frac: float = 0.35,
    surge_frac: float = 0.30,
    surge_mult: float = 3.0,
    budget_frac: float = 0.45,
    seed: int = 0,
) -> JamScenario:
    """Random device/channel geometry, then jam a fraction of channels and surge a fraction
    of devices. Budget is a fraction of the power needed to serve every device on its best
    channel, so it binds."""
    rng = np.random.default_rng(seed)
    dev_pos = rng.random((n_dev, 2))
    chan_pos = rng.random((n_chan, 2))

    diff = dev_pos[:, None, :] - chan_pos[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1)) + 1e-3
    rate = 1.0 / dist                      # closer channel -> higher rate
    power = dist ** 2                       # closer channel -> cheaper power

    jammed = np.zeros(n_chan, dtype=bool)
    jammed[rng.choice(n_chan, int(jam_frac * n_chan), replace=False)] = True
    rate[:, jammed] = 0.0                   # jammed channels deliver nothing

    demand = rng.uniform(0.5, 1.0, n_dev)
    surged = np.zeros(n_dev, dtype=bool)
    surged[rng.choice(n_dev, int(surge_frac * n_dev), replace=False)] = True
    demand[surged] *= surge_mult

    cap = np.full(n_chan, demand.sum() / n_chan)   # capacity tight relative to total demand

    # Candidate pairings: any device-channel with positive rate.
    du, cv = np.where(rate > 0)
    pairs = np.stack([du, cv], axis=1)

    # Budget: a fraction of the power to serve each device on its cheapest live channel.
    best_power = np.where(rate > 0, power, np.inf).min(axis=1)
    best_power[~np.isfinite(best_power)] = 0.0
    budget = budget_frac * best_power.sum()

    return JamScenario(n_dev, n_chan, demand, rate, power, cap, budget, pairs, jammed, surged)


# --------------------------------------------------------------------------------------
# Served-demand objective (with constraint penalties)
# --------------------------------------------------------------------------------------
def _served(sc: JamScenario, x: np.ndarray) -> tuple:
    """Decode a binary selection over candidate pairings into served demand + usage.

    Same feasible decoder for every method (a fair comparison): each device keeps at most one
    pairing (its highest served value), and pairings are admitted in value order while they
    respect the channel capacity and the power budget. Returns (served_demand, power_used)."""
    pairs = sc.pairs[x == 1]
    if pairs.size == 0:
        return 0.0, 0.0
    dev = pairs[:, 0]
    chan = pairs[:, 1]
    served_val = np.minimum(sc.demand[dev], sc.rate[dev, chan])
    pw = sc.power[dev, chan]

    order = np.argsort(-served_val)
    used_dev = set()
    chan_load = np.zeros(sc.n_chan)
    power_used = 0.0
    served = 0.0
    for k in order:
        i, j = int(dev[k]), int(chan[k])
        if i in used_dev or chan_load[j] + served_val[k] > sc.cap[j] or power_used + pw[k] > sc.budget:
            continue
        used_dev.add(i)
        chan_load[j] += served_val[k]
        power_used += pw[k]
        served += served_val[k]
    return served, power_used


def make_fitness(sc: JamScenario):
    total_demand = sc.demand.sum()

    def fitness(pop: np.ndarray) -> np.ndarray:
        out = np.empty(pop.shape[0], dtype=np.float64)
        for i, x in enumerate(pop):
            served, _ = _served(sc, x)
            out[i] = served / total_demand
        return out

    return fitness


# --------------------------------------------------------------------------------------
# AI warm-start: predict jammed channels + surged devices, favour cheap high-value pairings
# --------------------------------------------------------------------------------------
def ai_prior(sc: JamScenario) -> np.ndarray:
    dev = sc.pairs[:, 0]
    chan = sc.pairs[:, 1]
    served_val = np.minimum(sc.demand[dev], sc.rate[dev, chan])
    pw = sc.power[dev, chan]
    value_per_power = served_val / (pw + 1e-9)
    score = value_per_power / (value_per_power.max() + 1e-9)
    return 0.05 + 0.35 * score              # gentle bias to efficient pairings


# --------------------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------------------
def greedy(sc: JamScenario):
    """Assign devices to channels in decreasing value-per-power, respecting capacity/power."""
    dev = sc.pairs[:, 0]
    chan = sc.pairs[:, 1]
    served_val = np.minimum(sc.demand[dev], sc.rate[dev, chan])
    pw = sc.power[dev, chan]
    order = np.argsort(-(served_val / (pw + 1e-9)))
    x = np.zeros(len(sc.pairs), dtype=np.uint8)
    used_dev = set()
    chan_load = np.zeros(sc.n_chan)
    power_used = 0.0
    for k in order:
        i, j = dev[k], chan[k]
        if i in used_dev or chan_load[j] + served_val[k] > sc.cap[j] or power_used + pw[k] > sc.budget:
            continue
        x[k] = 1
        used_dev.add(int(i))
        chan_load[j] += served_val[k]
        power_used += pw[k]
    return x


def greedy_balanced(sc: JamScenario):
    """Congestion-aware greedy: assign each device (in value order) to the live channel that
    maximises its *throttled* marginal served value given the current channel loads, subject
    to the power budget. A strong baseline that defuses the 'naive greedy' strawman."""
    # Best (device, channel) raw value/power per device to set an assignment order.
    order_dev = np.argsort(-(sc.demand))
    chan_load = np.zeros(sc.n_chan)
    power_used = 0.0
    x = np.zeros(len(sc.pairs), dtype=np.uint8)
    pair_index = {(int(d), int(c)): k for k, (d, c) in enumerate(sc.pairs)}
    for i in order_dev:
        best_gain, best_j, best_k = 0.0, -1, -1
        for j in range(sc.n_chan):
            if sc.rate[i, j] <= 0:
                continue
            k = pair_index.get((int(i), int(j)))
            if k is None or power_used + sc.power[i, j] > sc.budget:
                continue
            v = min(sc.demand[i], sc.rate[i, j])
            new_load = chan_load[j] + v
            factor = min(1.0, sc.cap[j] / (new_load + 1e-9))
            gain = v * factor                # throttled value if added here
            if gain > best_gain:
                best_gain, best_j, best_k = gain, j, k
        if best_j < 0:
            continue
        x[best_k] = 1
        chan_load[best_j] += min(sc.demand[i], sc.rate[i, best_j])
        power_used += sc.power[i, best_j]
    return x


def genetic(sc: JamScenario, fitness, pop_size=40, generations=120, seed=0):
    rng = np.random.default_rng(seed)
    P = len(sc.pairs)
    pop = (rng.random((pop_size, P)) < 0.1).astype(np.uint8)
    best_x, best_f = None, -np.inf
    history = np.empty(generations)
    for g in range(generations):
        fit = fitness(pop)
        j = int(np.argmax(fit))
        if fit[j] > best_f:
            best_f, best_x = float(fit[j]), pop[j].copy()
        a, b = rng.integers(0, pop_size, (2, pop_size))
        winners = np.where(fit[a] >= fit[b], a, b)
        parents = pop[winners]
        mask = rng.random((pop_size, P)) < 0.5
        children = np.where(mask, parents, parents[rng.permutation(pop_size)])
        flip = rng.random((pop_size, P)) < (1.0 / max(P, 1))
        children ^= flip.astype(np.uint8)
        children[0] = best_x
        pop = children
        history[g] = best_f
    return best_x, best_f, history


# --------------------------------------------------------------------------------------
# Smoke-test driver
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    sc = make_jam_scenario(seed=0)
    total = sc.demand.sum()
    print(f"devices={sc.n_dev} channels={sc.n_chan} jammed={int(sc.jammed.sum())} "
          f"surged={int(sc.surged.sum())} candidates={len(sc.pairs)} "
          f"budget={sc.budget:.3f} total_demand={total:.2f}")

    fit = make_fitness(sc)
    prior = ai_prior(sc)
    uniform = np.full(len(sc.pairs), float(prior.mean()))
    cfg = QIEAConfig(pop_size=40, generations=100, seed=0)

    def latency(h, frac=0.95):
        t = frac * h[-1]
        hit = np.where(h >= t)[0]
        return int(hit[0]) if hit.size else len(h)

    r_ai = optimize(fit, len(sc.pairs), cfg, prior=prior)
    r_no = optimize(fit, len(sc.pairs), cfg, prior=uniform)
    bx, bf, _ = genetic(sc, fit, seed=0)
    gx = greedy(sc)

    def rep(name, x, extra=""):
        served, pw = _served(sc, x)
        print(f"  {name:14s} served={served/total:.3f}  power={pw:.3f}/{sc.budget:.3f}  {extra}")

    print("post-heal served-demand fraction:")
    rep("QIEA + AI", r_ai.best_x, f"latency={latency(r_ai.history)}")
    rep("QIEA (no AI)", r_no.best_x, f"latency={latency(r_no.history)}")
    rep("GA", bx)
    rep("greedy", gx)
