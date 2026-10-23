"""CUDA-Q Hamiltonian, operator pool, and state preparation."""
from itertools import combinations

import cudaq
from cudaq import spin

from hamiltonian import qubo_to_ising_coeffs
from operators import flatten_ops, validate_pool


def setup_backend(mpi, seed):
    if mpi:
        cudaq.set_target("nvidia", option="mqpu")
        cudaq.mpi.initialize()
    else:
        try:
            cudaq.set_target("nvidia", option="fp64")
        except RuntimeError:
            cudaq.set_target("qpp-cpu")
            print("NVIDIA backend unavailable; using the CPU simulator.")
    if hasattr(cudaq, "set_random_seed"):
        cudaq.set_random_seed(seed)


def build_hamiltonian(problem, constrained):
    penalty = 0.0 if constrained else problem["A"]
    c = qubo_to_ising_coeffs(problem["mu"], problem["Sigma"],
                            problem["lambda"], penalty, problem["K"])
    hamiltonian = c["h0"] * cudaq.SpinOperator(spin.i(0))
    for i, value in enumerate(c["h1"]):
        if abs(value) > 1e-12:
            hamiltonian += value * cudaq.SpinOperator(spin.z(i))
    for i, j in combinations(range(problem["n_assets"]), 2):
        value = c["h2"][i, j] + c["h2"][j, i]
        if abs(value) > 1e-12:
            hamiltonian += value * cudaq.SpinOperator(spin.z(i) * spin.z(j))
    return hamiltonian


def build_pool(n, angles, zz_angles, constrained, mixer_angles):
    pairs = list(combinations(range(n), 2))
    z_ops = [cudaq.SpinOperator(spin.z(i)) for i in range(n)]
    zz_ops = [cudaq.SpinOperator(spin.z(i) * spin.z(j)) for i, j in pairs]
    if constrained:
        # Keep XX+YY atomic: separate choices would break cardinality preservation.
        mixers = [cudaq.SpinOperator(spin.x(i) * spin.x(j)) +
                  cudaq.SpinOperator(spin.y(i) * spin.y(j)) for i, j in pairs]
    else:
        mixers = [cudaq.SpinOperator(spin.x(i)) for i in range(n)]
    families = ((mixer_angles if constrained else angles, mixers),
                (angles, z_ops), (zz_angles, zz_ops))
    return [angle * op for coefficients, ops in families for angle in coefficients for op in ops]


def preflight(problem, hamiltonian, pool, constrained):
    """Check extraction, state preparation, bit order, and mixer execution."""
    from hamiltonian import energy_from_problem
    from results import portfolio_distribution
    n, k = problem["n_assets"], problem["K"] if constrained else 0
    expected = "1" * k + "0" * (n - k)
    counts = cudaq.sample(circuit, n, k, [], [], shots_count=32)
    if dict(counts.items()) != {expected: 32}:
        raise ValueError("Initial state or measurement bit order is incorrect")
    energy = cudaq.observe(circuit, hamiltonian, n, k, [], []).expectation()
    if abs(energy - energy_from_problem([int(b) for b in expected], problem)) > 1e-8:
        raise ValueError("CUDA-Q Hamiltonian disagrees with the classical initial-state energy")
    if constrained:
        validate_pool(pool, n)
        number_error = build_hamiltonian(
            dict(problem, mu=[0.0] * n, Sigma=[[0.0] * n for _ in range(n)],
                 **{"lambda": 0.0, "A": 1.0}), constrained=False)
        # Include a pair of occupied qubits and an occupied/empty pair.
        # The former exposes the YY/YY bug that an all-empty or optimal A test missed.
        for i, j in ((0, 1), (0, k)):
            if j >= n or i == j:
                continue
            exchange = (0.2 * cudaq.SpinOperator(spin.x(i) * spin.x(j)) +
                        0.2 * cudaq.SpinOperator(spin.y(i) * spin.y(j)))
            coeffs, words = flatten_ops([exchange], n)
            leakage = cudaq.observe(circuit, number_error, n, k, coeffs, words).expectation()
            if abs(leakage) > 1e-8:
                raise ValueError(f"Mixer violates cardinality: <(N-K)^2>={leakage}")
            counts = cudaq.sample(circuit, n, k, coeffs, words, shots_count=256)
            portfolio_distribution(counts, problem, 256, constrained=True)
        print("Preflight passed: pool, initial state, Hamiltonian, and cardinality.")


@cudaq.kernel
def circuit(n: int, initial_ones: int, coeffs: list[float], words: list[cudaq.pauli_word]):
    q = cudaq.qvector(n)
    for i in range(initial_ones):
        x(q[i])
    for i in range(len(coeffs)):
        exp_pauli(coeffs[i], q, words[i])
