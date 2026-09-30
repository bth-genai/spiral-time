import math
import unittest
from dataclasses import replace

import numpy as np

from collective_historical_return_experiment import (
    _commutative_history,
    _secondary_tokens,
    _transform_body,
    _unitary,
    run_world,
)
from collective_reverb_experiment import (
    CollectiveConfig,
    Tissue,
    make_world,
    participate,
    probe,
)
from collective_reverb_order_audit import KEY_CELLS, schedule_orders
from collective_reverb_scaling import _extend_world, _pair_rhythm


class CollectiveHistoricalReturnTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CollectiveConfig(
            seed=9182,
            sediment_episodes_ab=2,
            sediment_episodes_bc=2,
            participation_time=1.2,
            encounter_time=2.0,
        )

    def test_probe_diagnostics_do_not_change_event_definition(self) -> None:
        world = make_world(self.config, self.config.seed)
        a, b = world.bodies["A"], world.bodies["B"]
        tissue = Tissue(world.initial_tissue.copy(), self.config.finite_capacity)
        result = probe(
            self.config,
            tissue,
            a,
            b,
            self.config.seed + 701,
            _pair_rhythm(a, b),
        )
        self.assertIn(result["status"], ("enacted", "refused", "censored"))
        self.assertGreaterEqual(float(result["cumulative_support"]), 0.0)
        self.assertTrue(math.isfinite(float(result["terminal_phase"])))
        if result["status"] == "enacted":
            self.assertTrue(math.isfinite(float(result["coordinate_phase"])))
            self.assertTrue(math.isfinite(float(result["coordinate_return"])))
        else:
            self.assertTrue(math.isnan(float(result["coordinate_phase"])))
            self.assertTrue(math.isnan(float(result["coordinate_return"])))

    def test_co_transform_preserves_probe(self) -> None:
        world = make_world(self.config, self.config.seed)
        a, b = world.bodies["A"], world.bodies["B"]
        tissue = Tissue(world.initial_tissue.copy(), self.config.finite_capacity)
        for episode in range(self.config.sediment_episodes_ab):
            participate(
                self.config,
                tissue,
                a,
                b,
                self.config.seed + 1_100 + episode,
                _pair_rhythm(a, b),
            )
        unitary = _unitary(self.config.tissue_channels, self.config.seed + 80_000)
        rotated = Tissue(
            unitary @ tissue.matrix @ unitary.conj().T,
            tissue.capacity,
        )
        original = probe(
            self.config,
            tissue,
            a,
            b,
            self.config.seed + 701,
            _pair_rhythm(a, b),
        )
        transformed = probe(
            self.config,
            rotated,
            _transform_body(a, unitary),
            _transform_body(b, unitary),
            self.config.seed + 701,
            _pair_rhythm(a, b),
        )
        self.assertEqual(original["status"], transformed["status"])
        self.assertEqual(original["failure_mode"], transformed["failure_mode"])
        for key in original:
            if key in ("status", "failure_mode"):
                continue
            left = float(original[key])
            right = float(transformed[key])
            if math.isnan(left):
                self.assertTrue(math.isnan(right))
            else:
                self.assertAlmostEqual(left, right, places=12)

    def test_tissue_rotation_preserves_norm_and_spectrum(self) -> None:
        world = make_world(self.config, self.config.seed)
        unitary = _unitary(self.config.tissue_channels, self.config.seed + 12)
        rotated = unitary @ world.initial_tissue @ unitary.conj().T
        self.assertAlmostEqual(
            float(np.linalg.norm(world.initial_tissue)),
            float(np.linalg.norm(rotated)),
            places=12,
        )
        np.testing.assert_allclose(
            np.linalg.eigvalsh(world.initial_tissue),
            np.linalg.eigvalsh(rotated),
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_commutative_history_is_order_invariant(self) -> None:
        cell = KEY_CELLS[0]
        config = replace(
            self.config,
            finite_capacity=cell.finite_capacity,
            mode_count=cell.mode_count,
            tissue_channels=cell.tissue_channels,
        )
        world = _extend_world(
            config,
            make_world(config, config.seed),
            cell.body_count,
            config.seed,
        )
        history = Tissue(world.initial_tissue.copy(), cell.finite_capacity)
        tokens = _secondary_tokens(config, cell, config.seed)
        orders = schedule_orders(
            cell.body_count, config.sediment_episodes_bc, config.seed + 31_337
        )
        matrices = [
            _commutative_history(config, history, world, tokens, order).matrix
            for order in orders.values()
        ]
        for matrix in matrices[1:]:
            np.testing.assert_allclose(matrix, matrices[0], rtol=0.0, atol=1.0e-14)

    def test_identical_secondary_null_matches_repeated_body(self) -> None:
        rows, _ = run_world(self.config, KEY_CELLS[1], 0, 0)
        repeated = next(row for row in rows if row["condition"] == "repeated_C")
        identical = next(
            row for row in rows if row["condition"] == "identical_secondary"
        )
        self.assertEqual(repeated["status"], identical["status"])
        for key in (
            "tissue_norm",
            "carried",
            "support",
            "cumulative_support",
            "min_viability",
            "max_tension",
            "event_time",
            "terminal_phase",
            "coordinate_phase",
            "coordinate_return",
        ):
            left = float(repeated[key])
            right = float(identical[key])
            if math.isnan(left):
                self.assertTrue(math.isnan(right))
            else:
                self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
