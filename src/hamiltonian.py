"""
Portfolio optimization QUBO Hamiltonian with synthetic demonstration inputs.

H(x) = -mu . x + lambda * (x^T Sigma x) + A * (sum(x) - K)^2

x in {0,1}^n selects assets; K of them should be chosen (soft constraint via
penalty A). The public examples are independently generated synthetic instances.
"""
import json
import itertools
from pathlib import Path

import numpy as np


def load_problem(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def energy(x, mu, Sigma, lam, A, K) -> float:
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    return float(-mu @ x + lam * (x @ Sigma @ x) + A * (x.sum() - K) ** 2)


def energy_from_problem(x, problem: dict) -> float:
    return energy(x, problem["mu"], problem["Sigma"], problem["lambda"],
                  problem["A"], problem["K"])


def brute_force(problem: dict):
    """Exhaustive search for small benchmarks (the bundled synthetic instances are tractable)."""
    n = problem["n_assets"]
    best_x, best_E = None, float("inf")
    for bits in itertools.product([0, 1], repeat=n):
        E = energy_from_problem(bits, problem)
        if E < best_E:
            best_E, best_x = E, bits
    return list(best_x), best_E


def qubo_to_ising_coeffs(mu, Sigma, lam, A, K):
    """
    Map H(x) with x_i in {0,1} to an Ising-form Hamiltonian in z_i in {-1,+1}
    via x_i = (1 - z_i) / 2, returning the coefficients needed to build a
    weighted sum of Pauli-Z strings: H = h0 + sum_i h_i Z_i + sum_{i<j} h_ij Zi Zj.

    This is the form GQE expects (Sec. 2.1 of arXiv:2401.09253): a weighted
    sum of Pauli operators, which then defines/constrains the operator pool.
    """
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(mu)

    # Expand each term of H(x) in terms of z_i using x_i = (1-z_i)/2
    # -mu.x term:
    #   -mu_i x_i = -mu_i/2 + mu_i/2 z_i
    h0 = -0.5 * mu.sum()
    h1 = 0.5 * mu.copy()  # coefficient of z_i

    # lambda * x^T Sigma x term:
    #   x_i x_j = (1-zi)(1-zj)/4 = (1 - zi - zj + zi zj)/4
    # sum_ij Sigma_ij x_i x_j = sum_ij Sigma_ij/4 * (1 - zi - zj + zi zj)
    h0 += lam * Sigma.sum() / 4.0
    h1 += -lam * Sigma.sum(axis=1) / 4.0 - lam * Sigma.sum(axis=0) / 4.0
    h2 = lam * Sigma / 4.0  # coefficient of zi zj (i != j), symmetric; diagonal folded separately

    # diagonal Sigma_ii * x_i^2 = Sigma_ii * x_i (since x_i in {0,1}) already
    # handled correctly above via the general x_i x_j expansion when i==j
    # (zi*zi = 1, constant), so fold Sigma_ii's zi*zi term into h0 and remove
    # it from h2's diagonal (no self Pauli-Z-Z term).
    diag = np.diag(Sigma)
    h0 += lam * diag.sum() / 4.0  # from zi*zi = 1 contribution of diagonal terms
    np.fill_diagonal(h2, 0.0)

    # A * (sum(x) - K)^2 term:
    #   sum(x) = n/2 - (1/2) sum(z_i)
    #   (sum(x) - K) = (n/2 - K) - (1/2) sum(z_i)
    c = (n / 2.0 - K)
    # (c - 0.5 sum z_i)^2 = c^2 - c sum z_i + 0.25 (sum z_i)^2
    # (sum z_i)^2 = sum z_i^2 + sum_{i!=j} zi zj = n + sum_{i!=j} zi zj
    h0 += A * (c ** 2 + 0.25 * n)
    h1 += A * (-c) * np.ones(n)
    h2 += A * 0.25 * (np.ones((n, n)) - np.eye(n))

    return {"h0": h0, "h1": h1, "h2": h2}


def energy_from_ising(z, coeffs) -> float:
    z = np.asarray(z, dtype=float)
    h0, h1, h2 = coeffs["h0"], coeffs["h1"], coeffs["h2"]
    return float(h0 + h1 @ z + z @ h2 @ z)


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    for name in ["problemA_4assets.json", "problemB_8assets.json", "problemC_12assets.json"]:
        problem = load_problem(data_dir / name)
        x_opt = problem.get("optimal_portfolio") or problem["known_optimum"]["x"]
        E_target = problem.get("optimal_energy", problem.get("known_optimum", {}).get("energy"))
        E_qubo = energy_from_problem(x_opt, problem)

        z_opt = [1 - 2 * xi for xi in x_opt]  # x_i=1 -> z_i=-1, x_i=0 -> z_i=+1
        coeffs = qubo_to_ising_coeffs(problem["mu"], problem["Sigma"], problem["lambda"],
                                       problem["A"], problem["K"])
        E_ising = energy_from_ising(z_opt, coeffs)

        print(f"{problem['name']:>10}: QUBO E={E_qubo:.6f} (target {E_target}), "
              f"Ising E={E_ising:.6f}, match={abs(E_qubo - E_target) < 1e-9 and abs(E_ising - E_target) < 1e-9}")
