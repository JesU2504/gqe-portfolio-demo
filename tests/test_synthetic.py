"""CPU checks of the public inputs, mapping, measurement analysis, and CLI."""
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hamiltonian import energy_from_problem, energy_from_ising, qubo_to_ising_coeffs
from results import portfolio_distribution, save_result
from run_gqe_portfolio import parse_args
from generate_synthetic import generate


class SyntheticTests(unittest.TestCase):
    def test_all_bitstrings_match_ising_and_exact_minimum(self):
        for path in sorted((ROOT / "data").glob("*.json")):
            p = json.loads(path.read_text())
            self.assertEqual(p["provenance"]["kind"], "synthetic")
            self.assertEqual(p, generate(p["n_assets"], p["K"], p["provenance"]["seed"]))
            self.assertGreater(np.linalg.eigvalsh(p["Sigma"]).min(), 0)
            coeffs = qubo_to_ising_coeffs(p["mu"], p["Sigma"], p["lambda"], p["A"], p["K"])
            energies = []
            for x in itertools.product((0, 1), repeat=p["n_assets"]):
                e = energy_from_problem(x, p)
                self.assertAlmostEqual(e, energy_from_ising(1 - 2 * np.array(x), coeffs), places=10)
                energies.append(e)
            self.assertAlmostEqual(min(energies), p["known_optimum"]["energy"])
            self.assertEqual(sum(p["known_optimum"]["x"]), p["K"])

    def test_measurements_and_result_round_trip(self):
        args = parse_args(["--problem", "B", "--constrained", "--shots", "100"])
        p = json.loads(Path(args.problem).read_text())
        bits = "".join(map(str, p["known_optimum"]["x"]))
        rows = portfolio_distribution({bits: 100}, p, 100, constrained=True)
        self.assertEqual(rows[0]["probability"], 1.0)
        self.assertAlmostEqual(rows[0]["energy"], p["known_optimum"]["energy"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "synthetic.json"
            save_result(output, p, args, rows[0]["energy"], [], [], rows, 0.0, 0.0, None)
            data = json.loads(output.read_text())
        self.assertEqual(data["measured_portfolios"], rows)
        self.assertEqual(data["penalty_A_used"], 0)
        with self.assertRaises(ValueError):
            portfolio_distribution({bits: 100}, p, 99, constrained=True)

    def test_cli_alias_and_path_are_equivalent(self):
        for alias in "ABC":
            a = parse_args(["--problem", alias])
            b = parse_args(["--problem", a.problem])
            self.assertEqual(vars(a), vars(b))


if __name__ == "__main__":
    unittest.main()
