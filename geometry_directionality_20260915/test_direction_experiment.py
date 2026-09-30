"""Focused mathematical checks for O3 angle matching."""

from __future__ import annotations

import unittest

import numpy as np

from direction_experiment import angle_matched_surrogate, normalized, optional_cosine


class DirectionChecks(unittest.TestCase):
    def test_surrogate_matches_native_norm_and_cross_angle(self) -> None:
        rng = np.random.default_rng(71)
        native = rng.normal(size=(12, 12))
        cross = rng.normal(size=(12, 12))
        donor = rng.normal(size=(12, 12))
        surrogate = angle_matched_surrogate(native, cross, donor, 72)
        self.assertAlmostEqual(
            float(np.linalg.norm(surrogate, ord="fro")),
            float(np.linalg.norm(native, ord="fro")),
            places=12,
        )
        self.assertAlmostEqual(
            float(optional_cosine(native, surrogate)),
            float(optional_cosine(native, cross)),
            places=12,
        )

    def test_surrogate_uses_donor_tangent_not_cross_direction(self) -> None:
        rng = np.random.default_rng(81)
        native = rng.normal(size=(12, 12))
        cross = rng.normal(size=(12, 12))
        donor_one = rng.normal(size=(12, 12))
        donor_two = rng.normal(size=(12, 12))
        first = angle_matched_surrogate(native, cross, donor_one, 82)
        second = angle_matched_surrogate(native, cross, donor_two, 83)
        self.assertFalse(np.allclose(first, second))
        self.assertAlmostEqual(
            float(optional_cosine(native, first)),
            float(optional_cosine(native, second)),
            places=12,
        )

    def test_normalization_has_unit_frobenius_norm(self) -> None:
        matrix = np.arange(1.0, 17.0).reshape(4, 4)
        self.assertAlmostEqual(
            float(np.linalg.norm(normalized(matrix), ord="fro")), 1.0
        )


if __name__ == "__main__":
    unittest.main()
