"""Classical measurement analysis and the result JSON format."""
import json
import math

from hamiltonian import energy_from_problem
from operators import validate_sequence


def portfolio_distribution(counts, problem, shots, constrained=False):
    if sum(count for _, count in counts.items()) != shots:
        raise ValueError("measurement counts do not match the requested shots")
    rows = []
    for bitstring, count in counts.items():
        x = [int(bit) for bit in bitstring]
        if len(x) != problem["n_assets"] or any(bit not in (0, 1) for bit in x):
            raise ValueError(f"invalid measured bitstring: {bitstring}")
        if count < 0 or (constrained and count and sum(x) != problem["K"]):
            raise ValueError(f"Invalid constrained measurement: {bitstring}, expected K={problem['K']}")
        rows.append(dict(bitstring=bitstring, x=x, count=count,
                         probability=count / shots, energy=energy_from_problem(x, problem)))
    return sorted(rows, key=lambda row: -row["probability"])


def check_energy(value, optimum):
    if not math.isfinite(value) or (optimum is not None and value < optimum - 1e-6):
        raise ValueError(f"Invalid energy {value}: expected a finite value ≥ {optimum}")
    return value


def validation_errors(data):
    """Reject physically impossible results rather than treating them as convergence."""
    errors = []
    optimum = data.get("known_optimum_energy")
    try:
        check_energy(data["best_energy"], optimum)
    except ValueError as exc:
        errors.append(str(exc))
    rows = data.get("measured_portfolios", [])
    if data.get("constrained"):
        try:
            validate_sequence(data.get("ansatz_ops", []))
        except ValueError as exc:
            errors.append(str(exc))
        k = data.get("K")
        if k is None:
            # Older files omit K; use the benchmark only when its name is recognized.
            k = {"ProblemA": 2, "ProblemB": 3, "ProblemC": 4}.get(data["problem"].replace(" ", ""))
        if k is None:
            errors.append("Constrained result has no K")
        elif any(sum(row["x"]) != k and row.get("count", row["probability"]) > 0 for row in rows):
            errors.append(f"Measured portfolios violate K={k}")
    probabilities = [row["probability"] for row in rows]
    if any(not math.isfinite(p) or p < 0 for p in probabilities) or sum(probabilities) > 1 + 1e-8:
        errors.append("Invalid probabilities")
    if rows and "shots" in data:
        if data["shots"] <= 0 or sum(row["count"] for row in rows) != data["shots"]:
            errors.append("Measurement counts do not sum to shots")
        elif any(abs(row["probability"] - row["count"] / data["shots"]) > 1e-8 for row in rows):
            errors.append("Counts and probabilities disagree")
    return errors


def save_result(path, problem, args, best_energy, coeffs, words, measured,
                training_seconds, sampling_seconds, trajectory, metadata=None):
    settings = ("ngates", "max_iters", "num_samples", "lr", "seed", "shots",
                "pool_coeffs", "zz_coeffs", "constrained")
    result = {key: getattr(args, key) for key in settings}
    result.update(
        problem=problem["name"], problem_file=str(args.problem),
        n_assets=problem["n_assets"], K=problem["K"],
        mixer_coeffs=args.mixer_coeffs if args.constrained else None,
        penalty_A_used=0.0 if args.constrained else problem["A"],
        best_energy=float(best_energy),
        known_optimum_energy=problem.get("optimal_energy", problem.get("known_optimum", {}).get("energy")),
        training_seconds=training_seconds, sampling_seconds=sampling_seconds,
        trajectory_file=str(trajectory) if args.save_trajectory else None,
        ansatz_ops=[dict(coeff=c, pauli_word=str(w)) for c, w in zip(coeffs, words)],
        measured_portfolios=measured,
        metadata=metadata or {},
    )
    errors = validation_errors(result)
    if errors:
        raise ValueError("; ".join(errors))
    # The plotter sees either a complete JSON or no final file, even during a live sweep.
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
