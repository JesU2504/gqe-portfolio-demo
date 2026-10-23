"""Regression for borrowed iterator terms and the number-preserving mixer."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from operators import flatten_ops, validate_pool, validate_sequence
from results import portfolio_distribution, validation_errors


class BorrowedTerm:
    def evaluate_coefficient(self):
        return complex(self.coeff)

    def get_pauli_word(self, n):
        return self.word


class BorrowedOperator:
    """Reproduce a binding that reuses one term view as its iterator advances."""
    def __init__(self, terms):
        self.terms = terms

    def __iter__(self):
        view = BorrowedTerm()
        for view.coeff, view.word in self.terms:
            yield view


class OperatorTests(unittest.TestCase):
    def test_borrowed_terms_are_snapshotted_before_advancing(self):
        op = BorrowedOperator([(0.2, "XX"), (0.2, "YY")])
        borrowed = list(op)
        self.assertIs(borrowed[0], borrowed[1])
        self.assertEqual([t.get_pauli_word(2) for t in borrowed], ["YY", "YY"])
        self.assertEqual(flatten_ops([op], 2), ([0.2, 0.2], ["XX", "YY"]))
        validate_pool([op], 2)

    def test_invalid_mixers_are_rejected(self):
        for terms in ([(0.2, "YY"), (0.2, "YY")], [(0.2, "XX"), (0.1, "YY")],
                      [(0.2, "XXI"), (0.2, "IYY")], [(0.2, "X")]):
            with self.assertRaises(ValueError):
                validate_pool([BorrowedOperator(terms)], len(terms[0][1]))
            with self.assertRaises(ValueError):
                validate_sequence([dict(coeff=c, pauli_word=w) for c, w in terms])

    def test_mixer_preserves_all_four_pair_states_at_every_default_angle(self):
        x = np.array([[0, 1], [1, 0]])
        y = np.array([[0, -1j], [1j, 0]])
        xx, yy = np.kron(x, x), np.kron(y, y)
        weights = np.array([0, 1, 1, 2])
        for angle in (0.05, -0.05, 0.1, -0.1, 0.2, -0.2, 0.4, -0.4, 0.8, -0.8):
            exp = lambda p: np.cos(angle) * np.eye(4) + 1j * np.sin(angle) * p
            unitary = exp(xx) @ exp(yy)
            for i in range(4):
                self.assertAlmostEqual(sum(abs(unitary[weights != weights[i], i])**2), 0)
            # Independent demonstration of the observed defect.
            broken = exp(yy) @ exp(yy)
            self.assertGreater(abs(broken[0, 3])**2, 0)

    def test_measurement_guard_rejects_one_asset_for_k_three(self):
        p = json.loads((ROOT / "data/problemB_8assets.json").read_text())
        with self.assertRaisesRegex(ValueError, "expected K=3"):
            portfolio_distribution({"00100000": 100}, p, 100, constrained=True)



if __name__ == "__main__":
    unittest.main()
