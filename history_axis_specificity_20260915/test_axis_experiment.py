"""Mathematical checks for O4 foreign history-axis controls."""

from __future__ import annotations

import unittest

import numpy as np

from axis_experiment import foreign_axis_surrogate, history_tangent, optional_cosine


class AxisChecks(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(601)
        self.native = rng.normal(size=(12, 12))
        self.cross = rng.normal(size=(12, 12))
        self.donor_native = rng.normal(size=(12, 12))
        self.donor_cross = rng.normal(size=(12, 12))

    def test_foreign_axis_matches_recipient_norm_and_angle(self) -> None:
        surrogate, fallback, _ = foreign_axis_surrogate(
            self.native,
            self.cross,
            self.donor_native,
            self.donor_cross,
            602,
        )
        self.assertFalse(fallback)
        self.assertAlmostEqual(
            float(np.linalg.norm(surrogate, ord="fro")),
            float(np.linalg.norm(self.native, ord="fro")),
            places=12,
        )
        self.assertAlmostEqual(
            float(optional_cosine(self.native, surrogate)),
            float(optional_cosine(self.native, self.cross)),
            places=12,
        )

    def test_own_axis_reconstructs_cross_direction(self) -> None:
        surrogate, fallback, tangent_orientation = foreign_axis_surrogate(
            self.native,
            self.cross,
            self.native,
            self.cross,
            603,
        )
        expected = self.cross * (
            np.linalg.norm(self.native, ord="fro")
            / np.linalg.norm(self.cross, ord="fro")
        )
        self.assertFalse(fallback)
        self.assertAlmostEqual(tangent_orientation, 1.0, places=12)
        self.assertTrue(np.allclose(surrogate, expected, atol=1.0e-12))

    def test_history_tangent_is_unit_and_orthogonal(self) -> None:
        tangent = history_tangent(self.native, self.cross)
        native_unit = self.native / np.linalg.norm(self.native, ord="fro")
        self.assertAlmostEqual(float(np.linalg.norm(tangent, ord="fro")), 1.0)
        self.assertAlmostEqual(float(np.sum(tangent * native_unit)), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
