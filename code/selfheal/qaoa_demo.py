"""Bridge-to-quantum demonstration: the SAME AI risk map that warm-starts the classical QIEA
also warm-starts a (simulated) quantum optimizer.

We take small instances of the budget-constrained binary resilience-selection problem (choose
backup links to maximize a value-plus-redundancy objective under a soft budget), encode each as
a QUBO over N qubits, and solve it with QAOA simulated by exact statevector evolution in pure
numpy (no PennyLane/Qiskit dependency). The AI prior r_j in [0,1] enters QAOA exactly as it
enters the QIEA: instead of the usual uniform |+>^N initial state, we prepare a biased product
state whose per-qubit amplitude encodes r_j. We compare uniform vs AI-prior initialization.

    python qaoa_demo.py 20 3      # 20 instances, up to p=3 QAOA layers
"""

from __future__ import annotations
import sys
import os
import numpy as np

try:
    from scipy.optimize import minimize
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

N = 12                      # candidate links = qubits (2^12 = 4096 statevector, exact)
DIM = 1 << N
BITS = ((np.arange(DIM)[:, None] >> np.arange(N)) & 1).astype(np.float64)   # (DIM, N)


def make_instance(seed: int):
    """A small resilience-selection QUBO: per-link value v_j, pairwise redundancy reward W_jk
    for complementary links, and a soft budget B. f(x) = v.x + sum_{j<k} W_jk x_j x_k -
    lam*(sum x - B)^2. The AI prior favours high-value links, r_j ~ normalized v_j."""
    rng = np.random.default_rng(seed)
    v = rng.uniform(0.3, 1.0, N)
    W = np.triu(rng.uniform(0.0, 0.4, (N, N)), 1)        # redundancy reward, upper triangular
    B = 5                                                 # target number of activated links
    lam = 0.6
    lin = BITS @ v
    quad = np.einsum("ij,jk,ik->i", BITS, W, BITS)
    pen = lam * (BITS.sum(1) - B) ** 2
    f = lin + quad - pen
    f = f - f.min()                                       # shift so objective >= 0
    r = 0.08 + 0.84 * (v - v.min()) / (np.ptp(v) + 1e-9)  # AI prior in [0.08, 0.92]
    return f, r


def _mixer(state, beta):
    """Apply exp(-i beta sum_j X_j) = prod_j RX(2 beta) to the statevector."""
    c = np.cos(beta)
    s = -1j * np.sin(beta)
    st = state.reshape([2] * N)
    for j in range(N):
        st = np.moveaxis(st, j, 0)
        a = st[0].copy()
        b = st[1].copy()
        st[0] = c * a + s * b
        st[1] = s * a + c * b
        st = np.moveaxis(st, 0, j)
    return st.reshape(-1)


def _init_state(r=None):
    if r is None:
        return np.full(DIM, 1.0 / np.sqrt(DIM), dtype=np.complex128)   # uniform |+>^N
    # product state: amplitude(x) = prod_j sqrt(r_j if x_j else 1-r_j)
    amp1 = np.sqrt(r)
    amp0 = np.sqrt(1.0 - r)
    amps = BITS * amp1 + (1 - BITS) * amp0
    return np.prod(amps, axis=1).astype(np.complex128)


def _expectation(f, gammas, betas, init):
    state = init.copy()
    for g, b in zip(gammas, betas):
        state = np.exp(1j * g * f) * state    # cost phase (we maximize f)
        state = _mixer(state, b)
    probs = np.abs(state) ** 2
    probs /= probs.sum()
    return float(probs @ f), float(probs[np.argmax(f)])


def run_qaoa(f, p, r=None, restarts=6, seed=0):
    """Optimize the 2p QAOA angles to maximize the expected objective; return (approx ratio,
    probability of the optimal bitstring) for the best angles found."""
    init = _init_state(r)
    fmax = f.max()
    rng = np.random.default_rng(seed)
    best = (-np.inf, 0.0)
    for _ in range(restarts):
        x0 = rng.uniform(0, np.pi, 2 * p)

        def neg(x):
            e, _ = _expectation(f, x[:p], x[p:], init)
            return -e
        if HAVE_SCIPY:
            res = minimize(neg, x0, method="Nelder-Mead",
                           options={"maxiter": 250, "xatol": 1e-3, "fatol": 1e-4})
            x = res.x
        else:
            x = x0  # no optimizer: single shot
        e, popt = _expectation(f, x[:p], x[p:], init)
        if e > best[0]:
            best = (e, popt)
    return best[0] / fmax, best[1]


def main():
    n_inst = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    pmax = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "qaoa")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for p in range(1, pmax + 1):
        for s in range(n_inst):
            f, r = make_instance(s)
            ar_u, po_u = run_qaoa(f, p, r=None, seed=s)
            ar_a, po_a = run_qaoa(f, p, r=r, seed=s)
            rows.append((p, s, ar_u, po_u, ar_a, po_a))
        ar_u = np.mean([x[2] for x in rows if x[0] == p])
        ar_a = np.mean([x[4] for x in rows if x[0] == p])
        po_u = np.mean([x[3] for x in rows if x[0] == p])
        po_a = np.mean([x[5] for x in rows if x[0] == p])
        print(f"p={p}: approx-ratio uniform={ar_u:.3f} AI-prior={ar_a:.3f} | "
              f"P(optimal) uniform={po_u:.3f} AI-prior={po_a:.3f}", flush=True)

    path = os.path.join(out_dir, "qaoa_results.csv")
    with open(path, "w") as fh:
        fh.write("p,inst,ar_uniform,popt_uniform,ar_prior,popt_prior\n")
        for row in rows:
            fh.write(",".join(str(x) for x in row) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
