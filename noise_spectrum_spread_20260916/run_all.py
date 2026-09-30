"""Run every frozen O5b condition with resumable per-condition output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from noise_experiment import CONDITIONS, run_condition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bodies", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("o5b_8x4"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for index, condition in enumerate(CONDITIONS, start=1):
        destination = args.output / f"{condition}.json"
        if destination.exists():
            print(f"condition {index}/{len(CONDITIONS)} {condition}: already complete", flush=True)
            continue
        print(f"condition {index}/{len(CONDITIONS)} {condition}: starting", flush=True)
        result = run_condition(condition, args.bodies, max(1, args.workers))
        destination.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"condition {index}/{len(CONDITIONS)} {condition}: complete", flush=True)


if __name__ == "__main__":
    main()
