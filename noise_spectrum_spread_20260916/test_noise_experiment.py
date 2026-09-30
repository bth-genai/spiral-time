"""Mathematical and stream checks for O5b."""

from __future__ import annotations

import unittest

import numpy as np

from noise_experiment import (
    BODY_NAMES,
    TOTAL_EPISODES,
    colored_samples,
    episode_increments,
    make_reference_stream,
    o5b_config,
)


class NoiseChecks(unittest.TestCase):
    def test_all_colors_are_centered_and_standardized(self) -> None:
        for color in ("white", "blue", "pink", "brown", "ou"):
            values = colored_samples(color, 81, 4000, 6, 0.01)
            self.assertTrue(np.allclose(np.mean(values, axis=0), 0.0, atol=1.0e-12))
            self.assertTrue(np.allclose(np.std(values, axis=0), 1.0, atol=1.0e-12))

    def test_coarse_increments_sum_the_same_fine_stream(self) -> None:
        coarse = o5b_config(1, 0.025, 0.02)
        fine = o5b_config(1, 0.025, 0.01)
        samples, _ = make_reference_stream(coarse, "pink", 82)
        coarse_episode = episode_increments(samples, coarse, 7)
        fine_episode = episode_increments(samples, fine, 7)
        for name in BODY_NAMES:
            expected = fine_episode[name].reshape(-1, 2, fine.modes).sum(axis=1)
            self.assertTrue(np.allclose(coarse_episode[name], expected, atol=1.0e-15))

    def test_reference_stream_has_full_history_length(self) -> None:
        config = o5b_config(1, 0.025, 0.02)
        samples, diagnostics = make_reference_stream(config, "white", 83)
        fine_steps = round(config.episode_duration / config.reference_dt)
        for name in BODY_NAMES:
            self.assertEqual(
                samples[name].shape,
                (TOTAL_EPISODES * fine_steps, config.modes),
            )
        self.assertAlmostEqual(diagnostics["fine_step_rms"], 1.0, places=12)

    def test_displacement_normalization_matches_white_reference(self) -> None:
        config = o5b_config(1, 0.025, 0.02)
        _, white = make_reference_stream(config, "white", 84)
        for color in ("blue", "pink", "brown", "ou"):
            _, matched = make_reference_stream(
                config, color, 84, "episode_displacement"
            )
            self.assertAlmostEqual(
                matched["episode_displacement_rms"],
                white["episode_displacement_rms"],
                places=12,
            )


if __name__ == "__main__":
    unittest.main()
