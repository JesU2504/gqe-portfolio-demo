# Generative Quantum Portfolio Optimization

A Python research implementation that uses a Generative Quantum Eigensolver
(GQE) to search for circuits selecting a fixed number of assets. It connects a
classical return–risk objective to CUDA-Q circuit simulation and a transformer
solver, with exact-search baselines and checks on the measured portfolios.

**This public release contains synthetic examples only.** The three generated
instances, their exact optima, and their seeds are included. Collaborator inputs,
research plans, experiment results, and the original private repository history
are excluded. No investment-performance or quantum-advantage claim is made.

## Try the classical checks

Use Python 3.10 or newer. No GPU is needed for this path:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/hamiltonian.py
python -m unittest discover -s tests -v
```

`scripts/generate_synthetic.py` regenerates the 4-, 8-, and 12-asset examples
from independent fixed seeds. The covariance matrices are positive definite.
The exact baseline enumerates every binary selection; it is suitable for these
small examples, and scales exponentially.

## Objective and circuit design

For binary asset selections `x`, the objective is

```text
H(x) = -mu · x + lambda * xᵀ Sigma x + A * (sum(x) - K)²
```

The code maps this objective to an Ising Hamiltonian using `x_i=(1-Z_i)/2`.
Two configurations are available:

| Configuration | Initial state | Operator pool | Selection constraint |
| --- | --- | --- | --- |
| Constraint preserving | Exactly K selected assets | Z, ZZ, paired XX+YY exchanges | Preserved by the circuit |
| Penalty based | All-zero state | X, Z, ZZ | Encouraged by the penalty |

These configurations differ in initial state and operator pool as well as the
penalty. A comparison measures the complete configurations; it does not isolate
the effect of the penalty alone. XX and YY must retain equal coefficients and
stay together to preserve the number of selected assets.

## CUDA-Q training

Training needs a compatible CUDA-Q and `cudaq_solvers` environment, including
its GQE dependencies. The training path is retained from the research code;
this public release has been checked on the CPU path only. Dependency and
backend compatibility should be checked in your CUDA-Q environment.

```bash
python src/run_gqe_portfolio.py --problem B --constrained --check
python src/run_gqe_portfolio.py --problem B --constrained \
  --ngates 10 --max-iters 2 --num-samples 5 --shots 1000 \
  --results-out results/synthetic_smoke.json
```

Use `--help` for training settings and custom JSON input paths. A normal run
checks the pool, initial state, energy bounds, returned circuit, and measured
cardinality before saving results. Result files include source hashes, package
versions, settings, counts, and timings. Choose unique output paths to retain
multiple runs.

## Code map

| File | Purpose |
| --- | --- |
| `src/hamiltonian.py` | Objective, QUBO-to-Ising mapping, exhaustive search |
| `src/operators.py` | Snapshot borrowed iterator terms; validate exchange pairs |
| `src/quantum.py` | CUDA-Q backend, Hamiltonian, circuit, preflight checks |
| `src/run_gqe_portfolio.py` | Training CLI and reproducibility metadata |
| `src/results.py` | Measurement analysis and validated JSON output |
| `scripts/generate_synthetic.py` | Independent public example generator |
| `scripts/check_results.py` | Saved-result audit without CUDA-Q |
| `scripts/plot_results.py` | Optional plots from saved results |

For plotting, install `requirements-plot.txt`. There are no bundled training
results; measurements should be produced in the environment being evaluated.

## Background

This implementation builds on the GQE approach described by
[Nakaji et al., arXiv:2401.09253](https://arxiv.org/abs/2401.09253) and integrates
the external NVIDIA CUDA-Q solver. Those projects retain their own attribution
and licensing. Public source prepared by Jesus Urbaneja.
