"""Quantum-inspired evolutionary optimizer (QIEA) for binary problems.

This is a *quantum-inspired classical* solver (no QPU). Each decision variable is a
Q-bit whose state is summarized by p_j = P(bit = 1) = sin^2(theta_j). A population of
Q-bit individuals is *observed* (sampled) each generation, the incumbent best is tracked,
and every Q-bit is rotated toward the incumbent bit value via a rotation-gate update.

The AI -> QIO synergy enters as `prior`: an AI-produced probability bias over variables.
It both warm-starts the population (initial p = prior) and softly prunes the search space
(a small persistent pull toward the prior), which is exactly the synergy the paper claims.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np


@dataclass
class QIEAConfig:
    pop_size: int = 40
    generations: int = 120
    rotation: float = 0.08          # base rotation magnitude toward incumbent (in p-space)
    prior_pull: float = 0.02        # persistent soft pull toward AI prior (search pruning)
    p_min: float = 0.02             # clamp to keep exploration alive
    p_max: float = 0.98
    elitism: bool = True
    seed: int = 0


@dataclass
class QIEAResult:
    best_x: np.ndarray              # best binary vector found
    best_fitness: float
    history: np.ndarray             # best-so-far fitness per generation (length = generations)
    evals: int                      # number of fitness evaluations consumed


def optimize(
    fitness_fn: Callable[[np.ndarray], np.ndarray],
    n_vars: int,
    cfg: QIEAConfig = QIEAConfig(),
    prior: Optional[np.ndarray] = None,
) -> QIEAResult:
    """Maximize `fitness_fn` over {0,1}^n_vars.

    fitness_fn takes a (pop, n_vars) uint8 array and returns a (pop,) float array
    (vectorized so it stays GPU/numpy-friendly). `prior` is an optional length-n_vars
    array in [0,1] giving the AI warm-start probabilities; if None, starts uniform 0.5.
    """
    rng = np.random.default_rng(cfg.seed)

    if prior is None:
        prior = np.full(n_vars, 0.5, dtype=np.float64)
    else:
        prior = np.clip(np.asarray(prior, dtype=np.float64), cfg.p_min, cfg.p_max)

    # Each individual carries its own probability vector; warm-started from the prior.
    p = np.tile(prior, (cfg.pop_size, 1)).copy()

    best_x = None
    best_fit = -np.inf
    history = np.empty(cfg.generations, dtype=np.float64)
    evals = 0

    for g in range(cfg.generations):
        # Observe (sample) the Q-bit population into concrete binary strings.
        pop = (rng.random((cfg.pop_size, n_vars)) < p).astype(np.uint8)
        if cfg.elitism and best_x is not None:
            pop[0] = best_x                      # carry the incumbent so it is re-evaluated

        fit = np.asarray(fitness_fn(pop), dtype=np.float64)
        evals += cfg.pop_size

        gbest = int(np.argmax(fit))
        if fit[gbest] > best_fit:
            best_fit = float(fit[gbest])
            best_x = pop[gbest].copy()

        # Rotation-gate update: move each Q-bit probability toward the incumbent bit,
        # then apply the small persistent pull toward the AI prior (space pruning).
        target = best_x.astype(np.float64)       # incumbent bits in {0,1}
        p += cfg.rotation * (target[None, :] - p)
        p += cfg.prior_pull * (prior[None, :] - p)
        np.clip(p, cfg.p_min, cfg.p_max, out=p)

        history[g] = best_fit

    return QIEAResult(best_x=best_x, best_fitness=best_fit, history=history, evals=evals)
