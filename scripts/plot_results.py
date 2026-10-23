"""Plot portfolio probabilities, convergence, and experiment comparisons."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from results import validation_errors

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, GREEN, GRAY = "#3274a1", "#27845b", "#b6bec6"


def read_history(path):
    """NVIDIA writes one JSON object per line; also accept a JSON array."""
    text = path.read_text().strip()
    if not text:
        raise ValueError(f"Empty trajectory: {path}")
    rows = json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines()]
    rows = [json.loads(row) if isinstance(row, str) else row for row in rows]
    rows.sort(key=lambda row: row["iter"])
    epochs = np.array([row["iter"] + 1 for row in rows])
    energies = np.array([min(row["energies"]) for row in rows], dtype=float)
    if not np.isfinite(energies).all() or len(set(epochs)) != len(epochs):
        raise ValueError(f"Invalid trajectory: {path}")
    return epochs, np.minimum.accumulate(energies)


def load_runs(folder, tolerance):
    runs = []
    invalid = []
    for path in sorted(folder.glob("*.json")):
        if path.stem.endswith("_trajectory"):
            continue
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or "best_energy" not in data:
            continue
        errors = validation_errors(data)
        if errors:
            invalid.append(f"{path.name}: {'; '.join(errors)}")
            continue
        rows = data.get("measured_portfolios", [])
        mass = sum(row["probability"] for row in rows)
        if mass > 1 + 1e-8 or any(row["probability"] < 0 for row in rows):
            raise ValueError(f"Invalid probabilities: {path}")
        optimum = data["known_optimum_energy"]
        n = data.get("n_assets") or len(rows[0]["x"] if rows else data["ansatz_ops"][0]["pauli_word"])
        optimal_mass = sum(row["probability"] for row in rows if abs(row["energy"] - optimum) < 1e-8)
        # Prefer the adjacent file so results copied from the server remain portable.
        trajectory = path.with_name(path.stem + "_trajectory.json")
        if not trajectory.exists() and data.get("trajectory_file"):
            named = Path(data["trajectory_file"])
            trajectory = path.parent / named.name
            if not trajectory.exists():
                trajectory = named
        history = read_history(trajectory) if trajectory.exists() else None
        hit = None
        if history is not None:
            epochs, best = history
            if min(best) < optimum - 1e-6 or abs(best[-1] - data["best_energy"]) > 1e-5:
                invalid.append(f"{path.name}: trajectory violates the energy bound or disagrees with the result")
                continue
            hits = epochs[best - optimum <= tolerance + 1e-10]
            hit = int(hits[0]) if len(hits) else None
        runs.append(dict(data=data, path=path, n=n, optimum=optimum, history=history,
                         hit=hit, optimal_mass=optimal_mass if rows else None,
                         complete=bool(rows) and abs(mass - 1) < 1e-8))
    if invalid:
        raise ValueError("Invalid results. No plots were updated:\n" + "\n".join(invalid))
    return runs


def save(fig, output, name):
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=180)
    plt.close(fig)


def plot_run(run, output, top, tolerance):
    data = run["data"]
    title = f"{data['problem'].replace('Problem', 'Problem ').replace('  ', ' ')} · {data['ngates']} operations"
    if "seed" in data:
        title += f" · seed {data['seed']}"
    rows = sorted(data.get("measured_portfolios", []), key=lambda row: -row["probability"])[:top]
    if rows:
        labels = ["{" + ", ".join(str(i + 1) for i, bit in enumerate(row["x"]) if bit) + "}" for row in rows]
        probabilities = [100 * row["probability"] for row in rows]
        colors = [GREEN if abs(row["energy"] - run["optimum"]) < 1e-8 else BLUE for row in rows]
        other = 100 - sum(probabilities)
        if other > 1e-6:
            labels.append("Other")
            probabilities.append(other)
            colors.append(GRAY)
        fig, ax = plt.subplots(figsize=(8, max(3.5, len(labels) * 0.36 + 1.5)))
        bars = ax.barh(labels, probabilities, color=colors)
        ax.bar_label(bars, labels=[f"{p:.2f}%" for p in probabilities], padding=4, fontsize=9)
        ax.invert_yaxis()
        ax.set(xlabel="Probability (%)", ylabel="Assets", xlim=(0, 112), title=title)
        ax.text(1, -0.2, "Green: optimal · Sampled probabilities", transform=ax.transAxes,
                ha="right", fontsize=9)
        save(fig, output, run["path"].stem + "_probabilities")
    if run["history"] is not None:
        epochs, best = run["history"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, best - run["optimum"], color=BLUE)
        ax.axhline(tolerance, color=GRAY, linestyle="--", label=f"Target: gap ≤ {tolerance:g}")
        ax.set(xlabel="Iteration", ylabel="Best energy − optimum", title=title)
        ax.legend(frameon=False)
        save(fig, output, run["path"].stem + "_convergence")


def comparisons(runs, output, tolerance, runtime):
    # Compare seeded constrained runs only, with the same training settings.
    eligible = [r for r in runs if r["data"].get("constrained") and "seed" in r["data"]]
    settings = {(r["data"]["max_iters"], r["data"]["num_samples"], r["data"]["lr"],
                 tuple(r["data"]["pool_coeffs"]), tuple(r["data"]["zz_coeffs"]),
                 tuple(r["data"]["mixer_coeffs"])) for r in eligible}
    if len(settings) > 1:
        print("Skipping comparisons: training settings differ. Plot each experiment folder separately.")
        return
    keys = [(r["data"]["problem"], r["data"]["ngates"], r["data"]["seed"]) for r in eligible]
    if len(keys) != len(set(keys)):
        print("Skipping comparisons: duplicate problem/length/seed runs.")
        return
    for gates in sorted({r["data"]["ngates"] for r in eligible}):
        group = [r for r in eligible if r["data"]["ngates"] == gates and r["history"] is not None]
        if len({r["n"] for r in group}) < 2:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for reached, marker, color, label in ((True, "o", BLUE, "Reached"),
                                              (False, "x", "#b54b43", "Not reached at last iteration")):
            subset = [r for r in group if (r["hit"] is not None) == reached]
            ax.scatter([r["n"] for r in subset],
                       [r["hit"] if reached else r["history"][0][-1] for r in subset],
                       marker=marker, color=color, label=label, alpha=0.7)
        ax.set(xlabel="Assets", ylabel="Iterations to target",
               title=f"Target: gap ≤ {tolerance:g} · {gates} operations")
        ax.set_xticks(sorted({r["n"] for r in group}))
        ax.legend(frameon=False)
        save(fig, output, f"iterations_g{gates}")

    for problem in sorted({r["data"]["problem"] for r in eligible}):
        group = [r for r in eligible if r["data"]["problem"] == problem]
        if len({r["data"]["ngates"] for r in group}) < 2:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric, label in (
            (axes[0], lambda r: r["data"]["best_energy"] - r["optimum"], "Energy gap"),
            (axes[1], lambda r: 100 * r["optimal_mass"] if r["complete"] else None, "Optimal probability (%)"),
        ):
            points = [(r["data"]["ngates"], metric(r)) for r in group if metric(r) is not None]
            if points:
                xs = sorted({x for x, _ in points})
                ax.scatter(*zip(*points), alpha=0.6, color=BLUE, label="Each seed")
                ax.plot(xs, [np.median([y for x, y in points if x == length]) for length in xs],
                        color=BLUE, label="Median")
                ax.set_xticks(xs)
                ax.legend(frameon=False)
            else:
                ax.text(0.5, 0.5, "Full counts unavailable", transform=ax.transAxes, ha="center")
            ax.set(xlabel="Operations", ylabel=label)
        axes[1].set_ylim(0, 105)
        fig.suptitle(problem)
        save(fig, output, problem.replace(" ", "") + "_length")

    if runtime:
        for gates in sorted({r["data"]["ngates"] for r in eligible}):
            group = [r for r in eligible if r["data"]["ngates"] == gates and "training_seconds" in r["data"]]
            if len({r["n"] for r in group}) < 2:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter([r["n"] for r in group], [r["data"]["training_seconds"] / 60 for r in group], color=BLUE)
            ax.set(xlabel="Assets", ylabel="Training time (min)", title=f"{gates} operations · Each seed")
            ax.set_xticks(sorted({r["n"] for r in group}))
            save(fig, output, f"runtime_g{gates}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path, default=Path("results/oct7_fixed"))
    parser.add_argument("--out", type=Path, help="output folder; default: <input>/plots")
    parser.add_argument("--tolerance", type=float, default=0.01, help="energy gap target (default: 0.01)")
    parser.add_argument("--top", type=int, default=10, help="portfolios per chart (default: 10); remaining mass is Other")
    parser.add_argument("--runtime", action="store_true", help="also plot total training time; use comparable GPU conditions")
    args = parser.parse_args()
    if args.top < 1 or not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("--top must be positive and --tolerance finite and nonnegative")
    if not args.folder.is_dir():
        parser.error(f"folder not found: {args.folder}")
    try:
        runs = load_runs(args.folder, args.tolerance)
    except ValueError as exc:
        parser.error(str(exc))
    if not runs:
        parser.error("no completed result files found")
    output = args.out or args.folder / "plots"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for run in runs:
        plot_run(run, output, args.top, args.tolerance)
    comparisons(runs, output, args.tolerance, args.runtime)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "assets", "operations", "seed", "energy_gap", "optimal_probability",
                         "full_counts", "trajectory", "target_gap", "iterations_to_target", "training_seconds"])
        for r in runs:
            d = r["data"]
            writer.writerow([r["path"].stem, r["n"], d["ngates"], d.get("seed", ""),
                             d["best_energy"] - r["optimum"], r["optimal_mass"], r["complete"],
                             r["history"] is not None, args.tolerance,
                             r["hit"] if r["hit"] is not None else ("not reached" if r["history"] is not None else ""),
                             d.get("training_seconds", "")])
    print(f"Saved PNG, PDF, and summary.csv to {output}")
    print("Missing trajectories are skipped; partial counts remain unnormalized (Other includes unsaved outcomes).")


if __name__ == "__main__":
    main()
