"""Schedule, stream, and interaction checks for O6."""

from __future__ import annotations

import unittest

import numpy as np

from material_experiment import (
    FORMATION_SLOTS,
    LEVELS,
    TOTAL_EPISODES,
    episode_increments,
    formation_schedule,
    make_reference_stream,
    o6_config,
)


class MaterialHistoryChecks(unittest.TestCase):
    def test_factorial_schedules_have_fixed_duration_and_counts(self) -> None:
        for order, timing, participation in LEVELS:
            schedule = formation_schedule((order, timing, participation))
            expected = 12 if participation == 0 else 8
            self.assertEqual(len(schedule), FORMATION_SLOTS)
            self.assertEqual(schedule.count("AB"), expected)
            self.assertEqual(schedule.count("BC"), expected)
            self.assertEqual(schedule.count(None), FORMATION_SLOTS - 2 * expected)

    def test_factor_levels_generate_eight_distinct_schedules(self) -> None:
        schedules = {formation_schedule(levels) for levels in LEVELS}
        self.assertEqual(len(schedules), 8)

    def test_coarse_and_fine_use_same_reference_stream(self) -> None:
        coarse = o6_config(0.02, 1)
        fine = o6_config(0.01, 1)
        samples = make_reference_stream(coarse, 901)
        self.assertEqual(
            next(iter(samples.values())).shape[0],
            TOTAL_EPISODES * round(coarse.episode_duration / coarse.reference_dt),
        )
        coarse_episode = episode_increments(samples, coarse, 17)
        fine_episode = episode_increments(samples, fine, 17)
        for name in samples:
            expected = fine_episode[name].reshape(-1, 2, fine.modes).sum(axis=1)
            self.assertTrue(np.allclose(coarse_episode[name], expected, atol=1.0e-15))


if __name__ == "__main__":
    unittest.main()
