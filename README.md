# Self-Healing IoT Networks: A Quantum-AI Closed-Loop Framework for Resilience

Reproducible code and manuscript for a study on **self-healing IoT networks**, where a
lightweight **AI perception layer** and a **quantum-inspired optimization (QIO) layer** are
coupled in a closed loop that restores network resilience after faults and attacks.

> **Paper:** *Self-Healing IoT Networks: A Quantum-AI Closed-Loop Framework for Resilience*
> **Author:** Phuc Hao Do (CAIRA · Da Nang Architecture University)
> **Target venue:** IEEE Network, Special Issue *"Quantum-AI Synergies Empowering IoT Network Resilience"*
> Manuscript in [`paper/main.tex`](paper/main.tex) (IEEEtran, two-column).

---

## TL;DR

A network merely *reconnected* after a fault often remains one link from the next collapse.
We treat repair as a **closed loop**: an AI layer detects and localizes the damage and emits a
**risk map**; a quantum-inspired evolutionary optimizer (QIEA) uses that risk map to recompute
a resilient configuration under an **energy budget**; recovered configurations are fed back to
keep the detector calibrated. The risk map enters the optimizer twice, as a **warm-start** and
as a **persistent pull** on the probabilistic qubit state, which is the concrete *synergy*.

We evaluate on two distinct, combinatorial threats over **50 random seeds** each and against
**genetic algorithm (GA)**, **simulated annealing (SA)**, and **greedy** baselines. The
findings are reported honestly: myopic greedy repair fails on both threats, global optimization
restores resilience, and the AI warm-start helps significantly. The quantum-inspired optimizer
is *competitive* (it beats SA) rather than dominant (a tuned GA leads).

---

## Headline results (50 seeds)

**CS1 — reconfiguration under cascading node failures** (worst-case survivable fraction):

| Method | survivable | note |
|---|---|---|
| greedy | 0.91 ± 0.12 | **fails**, spends ~10% of budget then stalls |
| greedy (conn.) | 1.00 ± 0.01 | strong heuristic |
| GA | 1.00 ± 0.01 | best |
| SA | 0.97 ± 0.05 | |
| QIEA, no AI | 0.97 ± 0.06 | |
| **QIEA + AI** | **0.98 ± 0.03** | synergy: variance halved (0.063→0.031), latency 88→81 |

**CS2 — restoring device-to-gateway reachability under spatial link jamming** (fraction of
devices keeping two edge-disjoint paths to a gateway):

| Method | gateway-survivable | note |
|---|---|---|
| greedy | 0.74 ± 0.19 | **fails** (p < 1e-9 vs QIEA+AI) |
| greedy (conn.) | 1.00 ± 0.01 | best |
| GA | 0.92 ± 0.04 | |
| SA | 0.87 ± 0.04 | |
| QIEA, no AI | 0.84 ± 0.05 | |
| **QIEA + AI** | **0.89 ± 0.05** | **synergy p = 4e-8**; beats SA (p = 9e-4) |

**Ablation (CS1, 50 seeds):** the persistent *pull*, not the one-shot warm-start, is what
matters: no-AI 0.923 ± 0.112 → warm-start-only 0.929 ± 0.104 → **warm-start + pull 0.980 ± 0.031**
(reliability ~3.6× better). Spectral weight `w` trades survivability for algebraic connectivity;
repair survivability saturates around a budget of ~5% of total candidate-link energy.

---

## Repository layout

```
paper--04/
├── paper/
│   └── main.tex            # the manuscript (IEEEtran, compiles to 6 pp)
├── code/
│   ├── environment.yml     # conda env (numpy + scipy)
│   ├── requirements.txt    # pip fallback
│   └── selfheal/
│       ├── qio.py              # QIEA: quantum-inspired optimizer with AI warm-start
│       ├── cs1_topology.py     # CS1 scenario, survivability metrics, greedy/GA/SA
│       ├── cs2_jamming.py      # CS2 scenario (link jamming, device-to-gateway survivability)
│       ├── run_cs1.py          # CS1 multi-seed runner  ->  results/cs1/
│       ├── run_cs2.py          # CS2 multi-seed runner  ->  results/cs2/
│       ├── run_ablation.py     # ablation runner        ->  results/ablation/
│       ├── plotstyle.py        # shared publication figure style + significance brackets
│       └── plot_*.py           # figure generators (read CSVs -> PNGs)
└── results/
    ├── cs1/   cs2/   ablation/ # per-seed CSVs (figures are regenerated from these)
```

## Method in one paragraph

Repair is a binary selection of backup links `x` maximizing a worst-case survivability
objective (largest two-edge-connected component, plus a small algebraic-connectivity term)
under an energy budget. The QIEA represents each candidate link by a qubit probability
`p_j = P(x_j = 1)`, samples a population, scores it, and rotates each qubit toward the best
incumbent *and* toward the AI risk map `r` (the warm-start initializes `p = r`). Greedy is the
stochastic-greedy baseline; GA and SA share the same evaluation budget. The whole resilience
evaluation (bridge finding + a Fiedler eigenvalue) is the runtime bottleneck, not the search.

## Reproducing the results

Pure CPU; needs only **numpy** (required) and **scipy** (optional, for the Wilcoxon tests).
No torch / networkx / pandas. Runs are deterministic (fixed seeds), so any machine reproduces
the same numbers.

```bash
git clone https://github.com/haodpsut/selfheal-iot-qai.git
cd selfheal-iot-qai

# environment: reuse any env with numpy, or
conda activate base
python -c "import numpy" || pip install -r code/requirements.txt
python -c "import scipy" || pip install scipy

# run (50 seeds, 6 parallel workers) inside tmux so it survives disconnects
cd code/selfheal
tmux new -s selfheal
python run_cs1.py 50 6        # -> results/cs1/cs1_results.csv (+ histories)
python run_cs2.py 50 6        # -> results/cs2/cs2_results.csv
python run_ablation.py 50 6   # -> results/ablation/ablation.csv
```

> The runners pin BLAS to one thread per worker (`OMP_NUM_THREADS=1` etc., set before numpy
> import) so the multiprocessing eigensolves do not oversubscribe the cores. Each run prints a
> mean ± std summary and Wilcoxon tests at the end.

### Figures and the paper

```bash
cd code/selfheal
python plot_cs1.py && python plot_cs2.py
python plot_synergy.py cs1 && python plot_ablation.py    # PNGs land next to the CSVs
cd ../../paper && pdflatex main.tex && pdflatex main.tex  # second pass resolves refs
```

Significance brackets in the figures use the convention `*` p<0.05, `**` p<0.01,
`***` p<0.001, `n.s.` not significant.

## Notes and honesty

- The "quantum" layer is **quantum-inspired and classical** (no QPU); we claim no quantum
  advantage, only that the probabilistic formulation is a natural bridge to a future quantum
  optimizer.
- The AI detector is a **transparent surrogate** (it scores links by fragment-bridging and
  endpoint sparsity), which isolates the synergy mechanism cleanly; a learned predictor would
  replace it in deployment.
- Evaluation is **simulation based** on random-geometric topologies with synthetic fault and
  jamming models; absolute numbers are relative comparisons under controlled conditions.

## License

Research code released for reproducibility. Please cite the paper if you use it.
