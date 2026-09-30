from __future__ import annotations

import unittest

from state_resolution_experiment import run_world


class StateResolutionChecks(unittest.TestCase):
    def test_current_state_branch_is_exactly_identical(self) -> None:
        result = run_world(0, 0.14)
        baseline = next(row for row in result["rows"] if row["resolution"] == 0)
        self.assertEqual(baseline["conventional_state_error"], 0.0)
        self.assertLess(baseline["mean_phase_rms"], 1.0e-14)
        self.assertLess(baseline["mean_alignment_rmse"], 1.0e-14)

    def test_resolved_history_changes_same_present_continuation(self) -> None:
        result = run_world(1, 0.14)
        full = next(row for row in result["rows"] if row["resolution"] == -1)
        self.assertEqual(full["conventional_state_error"], 0.0)
        self.assertGreater(full["history_field_distance_AB"] + full["history_field_distance_BC"], 0.0)
        self.assertGreater(full["mean_phase_rms"] + full["mean_alignment_rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
