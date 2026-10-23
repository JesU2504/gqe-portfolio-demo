"""Run one portfolio benchmark. CUDA-Q is imported only when training starts."""
import argparse
import math
import os
from pathlib import Path
import sys
import time

from hamiltonian import load_problem
from results import check_energy, portfolio_distribution, save_result

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = dict(zip("ABC", ("problemA_4assets.json", "problemB_8assets.json",
                          "problemC_12assets.json")))
ANGLES = [0.05, -0.05, 0.1, -0.1, 0.2, -0.2, 0.4, -0.4, 0.8, -0.8]


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--problem", required=True, help="A, B, C, or a problem JSON path")
    parser.add_argument("--constrained", action="store_true", help="preserve exactly K selected assets")
    parser.add_argument("--ngates", type=positive_int, default=10, help="selected pool operations per circuit")
    parser.add_argument("--max-iters", type=positive_int, default=200, help="training epochs")
    parser.add_argument("--num-samples", type=positive_int, default=50, help="circuits per epoch")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--seed", type=int, default=3047, help="training and sampling seed")
    parser.add_argument("--shots", type=positive_int, default=2000, help="final measurement shots")
    parser.add_argument("--pool-coeffs", type=float, nargs="+", default=ANGLES, help="Z and X mixer angles")
    parser.add_argument("--zz-coeffs", type=float, nargs="+", help="ZZ angles; defaults to pool angles")
    parser.add_argument("--mixer-coeffs", type=float, nargs="+", help="XY angles; defaults to pool angles")
    parser.add_argument("--save-trajectory", action="store_true", help="save the solver's training trajectory")
    parser.add_argument("--trajectory-out", type=Path, help="trajectory path; also enables trajectory saving")
    parser.add_argument("--results-out", type=Path, help="result path; defaults to results/<problem>_gqe[_constrained].json")
    parser.add_argument("--mpi", action="store_true", help="use NVIDIA multi-QPU MPI execution")
    parser.add_argument("--check", action="store_true", help="run CUDA-Q preflight checks only; no training")
    args = parser.parse_args(argv)
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and greater than zero")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    args.problem = str(ROOT / "data" / PROBLEMS[args.problem]) if args.problem in PROBLEMS else args.problem
    if not Path(args.problem).is_file():
        parser.error(f"problem file not found: {args.problem}")
    args.zz_coeffs = args.zz_coeffs if args.zz_coeffs is not None else args.pool_coeffs
    args.mixer_coeffs = args.mixer_coeffs if args.mixer_coeffs is not None else args.pool_coeffs
    if not all(math.isfinite(c) for cs in (args.pool_coeffs, args.zz_coeffs, args.mixer_coeffs) for c in cs):
        parser.error("operator angles must be finite")
    args.save_trajectory = args.save_trajectory or args.trajectory_out is not None
    return args


def run(args):
    # Keep --help, argument validation, and classical analysis usable on a Mac.
    import cudaq
    from quantum import build_hamiltonian, build_pool, circuit, flatten_ops, preflight, setup_backend

    problem = load_problem(args.problem)
    n, k = problem["n_assets"], problem["K"]
    optimum = problem.get("optimal_energy", problem.get("known_optimum", {}).get("energy"))
    suffix = "_constrained" if args.constrained else ""
    output = args.results_out or ROOT / "results" / f"{problem['name'].replace(' ', '')}_gqe{suffix}.json"
    trajectory = args.trajectory_out or output.with_name(output.stem + "_trajectory.json")
    if args.save_trajectory and output.resolve() == trajectory.resolve():
        raise ValueError("results and trajectory must use different paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.save_trajectory:
        trajectory.parent.mkdir(parents=True, exist_ok=True)

    setup_backend(args.mpi, args.seed)
    try:
        hamiltonian = build_hamiltonian(problem, args.constrained)
        pool = build_pool(n, args.pool_coeffs, args.zz_coeffs, args.constrained, args.mixer_coeffs)
        initial_ones = k if args.constrained else 0
        preflight(problem, hamiltonian, pool, args.constrained)
        if args.check:
            return
        import cudaq_solvers as solvers
        from cudaq_solvers.gqe_algorithm.gqe import get_default_config
        import random
        import numpy as np
        import torch
        # Upstream seeds inside the training loop, after creating model weights.
        # Seed here too so model initialization is reproducible.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudaq.set_random_seed(args.seed)

        def cost(ops, **kwargs):
            coeffs, words = flatten_ops(ops, n)
            if args.mpi:
                handle = cudaq.observe_async(circuit, hamiltonian, n, initial_ones,
                                             coeffs, words, qpu_id=kwargs["qpu_id"])
                return handle, lambda result: check_energy(result.get().expectation(), optimum)
            return check_energy(
                cudaq.observe(circuit, hamiltonian, n, initial_ones, coeffs, words).expectation(), optimum)

        # The config object overrides solver keyword arguments.
        config = get_default_config()
        for key in ("max_iters", "ngates", "num_samples", "lr", "seed", "save_trajectory"):
            setattr(config, key, getattr(args, key))
        config.trajectory_file_path = str(trajectory)
        config.use_fabric_logging = False
        config.verbose = True
        # GQE supplies its own starting index and generates a fixed-length sequence.
        config.bos_token_id = None
        config.eos_token_id = None

        start = time.perf_counter()
        best_energy, best_indices = solvers.gqe(cost, pool, config=config)
        training_seconds = time.perf_counter() - start
        if args.mpi and cudaq.mpi.rank() != 0:
            return

        coeffs, words = flatten_ops([pool[i] for i in best_indices], n)
        verified = cudaq.observe(circuit, hamiltonian, n, initial_ones, coeffs, words).expectation()
        check_energy(verified, optimum)
        if abs(verified - best_energy) > 1e-5:
            raise ValueError("Returned best circuit disagrees with its training energy")
        cudaq.set_random_seed(args.seed)
        start = time.perf_counter()
        counts = cudaq.sample(circuit, n, initial_ones, coeffs, words, shots_count=args.shots)
        sampling_seconds = time.perf_counter() - start
        measured = portfolio_distribution(counts, problem, args.shots, args.constrained)
        from importlib.metadata import version, PackageNotFoundError
        import hashlib
        packages = {}
        for name in ("cudaq", "cudaq-cu12", "cudaq-solvers", "cudaq-solvers-cu12", "torch", "transformers"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                pass
        sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((ROOT / "src").glob("*.py"))}
        metadata = dict(packages=packages, python=sys.version, target=str(cudaq.get_target()),
                        sources=sources, checks="preflight, energy bound, replay, cardinality")
        save_result(output, problem, args, best_energy, coeffs, words, measured,
                    training_seconds, sampling_seconds, trajectory, metadata)
        print(f"Best expectation: {best_energy:.8f}")
        for row in measured[:5]:
            print(f"  {row['bitstring']}  p={row['probability']:.4f}  E={row['energy']:.6f}")
        print(f"Saved {output}")
    finally:
        if args.mpi:
            cudaq.mpi.finalize()


def main():
    args = parse_args()
    run(args)
    if not args.mpi:
        # The lab CUDA-Q build crashes during interpreter teardown after a successful run.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
