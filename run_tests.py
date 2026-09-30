"""Run every test group in the isolated article source snapshot."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEST_GROUPS = (
    (
        "experiment_1_historical_return",
        (
            "test_collective_reverb.py",
            "test_collective_reverb_scaling.py",
            "test_collective_reverb_order_audit.py",
            "test_collective_historical_return.py",
        ),
    ),
    ("overlapping_relations_20260915", ("test_experiment.py",)),
    ("geometry_specificity_20260915", ("test_geometry_experiment.py",)),
    ("geometry_directionality_20260915", ("test_direction_experiment.py",)),
    ("history_axis_specificity_20260915", ("test_axis_experiment.py",)),
    ("noise_spectrum_spread_20260916", ("test_noise_experiment.py",)),
    ("material_history_factorial_20260916", ("test_material_experiment.py",)),
    ("state_resolution_20260920", ("test_state_resolution.py",)),
)


def main() -> int:
    failures: list[str] = []
    for directory, tests in TEST_GROUPS:
        print(f"\n== {directory} ==", flush=True)
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", *tests],
            cwd=ROOT / directory,
            check=False,
        )
        if completed.returncode:
            failures.append(directory)

    if failures:
        print("\nFailed test groups: " + ", ".join(failures), file=sys.stderr)
        return 1

    print("\nAll isolated article-code tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
