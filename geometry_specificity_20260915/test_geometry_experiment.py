"""Focused checks for O2 geometry interventions."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from geometry_experiment import (
    Config,
    assay,
    history_states,
    rescale,
    shuffled_geometry,
    state_error,
    state_signature,
    trajectory_distance,
)


class GeometryChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config(
            worlds=1,
            modes=4,
            episode_duration=0.1,
            formation_per_edge=3,
            maintenance_cycles=1,
            withdrawal_episodes=2,
            return_offers=2,
            base_seed=402609150,
        )

    def test_rescale_matches_requested_norm(self) -> None:
        matrix = np.arange(1.0, 17.0).reshape(4, 4)
        result = rescale(matrix, 2.75)
        self.assertAlmostEqual(float(np.linalg.norm(result, ord="fro")), 2.75)

    def test_row_column_shuffle_preserves_singular_values(self) -> None:
        matrix = np.arange(16.0).reshape(4, 4)
        shuffled = shuffled_geometry(matrix, 91)
        self.assertTrue(np.allclose(
            np.linalg.svd(matrix, compute_uv=False),
            np.linalg.svd(shuffled, compute_uv=False),
            atol=1.0e-12,
        ))

    def test_read_only_assay_does_not_change_recipient(self) -> None:
        recipient = history_states(self.config, 101, "alternating")["formation"]
        before = state_signature(recipient)
        assay(
            recipient,
            self.config,
            "AB",
            recipient.edges["AB"].tissue,
            102,
            expressed=True,
        )
        self.assertEqual(state_error(before, recipient), 0.0)

    def test_current_state_erasure_equivalence_is_recovered_without_history_feedback(self) -> None:
        conventional = replace(self.config, history_expression_gain=0.0)
        recipient = history_states(conventional, 201, "blocked")["formation"]
        tissue = recipient.edges["AB"].tissue
        off = assay(recipient, conventional, "AB", tissue, 202, expressed=False)
        erased = assay(
            recipient, conventional, "AB", np.zeros_like(tissue), 202, expressed=True
        )
        distance = trajectory_distance(off, erased)
        self.assertEqual(max(distance["body_phase_rms"].values()), 0.0)
        self.assertEqual(max(distance["alignment_rmse"].values()), 0.0)

    def test_history_feedback_survives_slow_tissue_erasure(self) -> None:
        resolved = replace(self.config, history_expression_gain=0.14)
        recipient = history_states(resolved, 211, "blocked")["formation"]
        tissue = recipient.edges["AB"].tissue
        off = assay(recipient, resolved, "AB", tissue, 212, expressed=False)
        erased = assay(
            recipient, resolved, "AB", np.zeros_like(tissue), 212, expressed=True
        )
        distance = trajectory_distance(off, erased)
        self.assertGreater(
            max(distance["body_phase_rms"].values())
            + max(distance["alignment_rmse"].values()),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
