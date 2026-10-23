"""Generate deterministic toy inputs independently of any collaborator data."""
import itertools
import json
from pathlib import Path

import numpy as np


def generate(n, k, seed):
    rng = np.random.default_rng(seed)
    factors = rng.normal(0, 0.07, (n, 3))
    covariance = factors @ factors.T + np.diag(rng.uniform(0.01, 0.04, n))
    p = dict(name=f"Synthetic {dict(zip([4, 8, 12], 'ABC'))[n]}",
             n_assets=n, K=k, mu=rng.uniform(0.03, 0.15, n).tolist(),
             Sigma=covariance.tolist(), A=2.0, **{"lambda": 0.8},
             provenance=dict(kind="synthetic", seed=seed,
                             generator="scripts/generate_synthetic.py"))
    def objective(x):
        x = np.array(x)
        return float(-x @ p["mu"] + p["lambda"] * x @ covariance @ x
                     + p["A"] * (x.sum() - k) ** 2)
    best = min(itertools.product((0, 1), repeat=n), key=objective)
    p["known_optimum"] = dict(x=list(best), energy=objective(best))
    return p


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for alias, n, k, seed in zip("ABC", [4, 8, 12], [2, 3, 4], [91004, 91008, 91012]):
        path = root / "data" / f"problem{alias}_{n}assets.json"
        path.write_text(json.dumps(generate(n, k, seed), indent=2) + "\n")
        print(path.name)
