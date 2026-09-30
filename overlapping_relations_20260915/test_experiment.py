"""Focused implementation checks for O1."""

from __future__ import annotations

import unittest

import numpy as np

from experiment import (
    Config,
    _checkpoint,
    _formation_schedule,
    expression_assay,
    make_world,
    run_episode,
)


class ExperimentChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config(
            worlds=1,
            modes=4,
            episode_duration=0.1,
            assay_episodes=4,
        )

    def test_histories_have_identical_formation_tokens(self) -> None:
        alternating = sorted(_formation_schedule(self.config, "alternating"))
        blocked = sorted(_formation_schedule(self.config, "blocked"))
        self.assertEqual(alternating, blocked)

    def test_first_encounter_does_not_consolidate_slow_relation_tissue(self) -> None:
        world = make_world(self.config, 11)
        run_episode(world, self.config, "AB", 12)
        self.assertEqual(float(np.linalg.norm(world.edges["AB"].tissue)), 0.0)
        self.assertGreater(float(np.linalg.norm(world.edges["AB"].trace)), 0.0)

    def test_clone_is_disposable(self) -> None:
        parent = make_world(self.config, 21)
        branch = parent.clone()
        run_episode(branch, self.config, "AB", 22)
        self.assertEqual(float(np.linalg.norm(parent.edges["AB"].trace)), 0.0)
        self.assertGreater(float(np.linalg.norm(branch.edges["AB"].trace)), 0.0)

    def test_no_ac_storage_exists(self) -> None:
        world = make_world(self.config, 31)
        self.assertEqual(set(world.edges), {"AB", "BC"})

    def test_dynamics_has_no_absolute_phase_origin(self) -> None:
        original = make_world(self.config, 35)
        rotated = original.clone()
        shift = 0.73
        for body in rotated.bodies.values():
            body.phases = (body.phases + shift + np.pi) % (2.0 * np.pi) - np.pi
        run_episode(original, self.config, "AB", 36, plastic=False)
        run_episode(rotated, self.config, "AB", 36, plastic=False)
        for name in original.bodies:
            difference = np.angle(np.exp(1j * (
                rotated.bodies[name].phases
                - original.bodies[name].phases
                - shift
            )))
            self.assertLess(float(np.max(np.abs(difference))), 1.0e-12)

    def test_zero_reference_orientation_is_explicitly_undefined(self) -> None:
        world = make_world(self.config, 41)
        zeros = {
            name: np.zeros_like(edge.tissue) for name, edge in world.edges.items()
        }
        body_zeros = {
            name: np.zeros_like(body.local_tissue)
            for name, body in world.bodies.items()
        }
        checkpoint = _checkpoint(world, zeros, body_zeros, self.config)
        self.assertIsNone(checkpoint["edge"]["AB"]["orientation_to_formation"])

    def test_disabled_expression_matches_erased_edge_in_read_only_assay(self) -> None:
        world = make_world(self.config, 51)
        run_episode(world, self.config, "AB", 52)
        run_episode(world, self.config, "AB", 53)
        assay = expression_assay(world, self.config, 54)
        equivalence = assay["ab_off_vs_erased"]
        self.assertEqual(equivalence["maximum_phase_error"], 0.0)
        self.assertEqual(equivalence["maximum_alignment_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
