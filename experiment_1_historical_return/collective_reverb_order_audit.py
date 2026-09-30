"""Audit whether collective carrying depends on participation order."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from collective_reverb_experiment import CollectiveConfig, _bootstrap_interval
from collective_reverb_scaling import SweepCell, run_scaled_seed


POLICIES = ("blocked", "reverse", "interleaved", "shuffled")
KEY_CELLS = (
    SweepCell("order", 4.0, 12, 4, 12, 1.44),
    SweepCell("order", 6.0, 12, 6, 12, 1.44),
    SweepCell("order", 6.0, 12, 6, 12, 2.40),
)


def schedule_orders(
    body_count: int, episodes_per_body: int, seed: int
) -> dict[str, tuple[int, ...]]:
    """Return permutations of the same body-specific encounter tokens."""

    secondary_count = body_count - 2
    total = secondary_count * episodes_per_body
    blocked = tuple(range(total))
    reverse = tuple(
        index
        for body_offset in reversed(range(secondary_count))
        for index in range(
            body_offset * episodes_per_body,
            (body_offset + 1) * episodes_per_body,
        )
    )
    interleaved = tuple(
        body_offset * episodes_per_body + episode
        for episode in range(episodes_per_body)
        for body_offset in range(secondary_count)
    )
    shuffled = tuple(
        int(index) for index in np.random.default_rng(seed).permutation(total)
    )
    return {
        "blocked": blocked,
        "reverse": reverse,
        "interleaved": interleaved,
        "shuffled": shuffled,
    }


def run_order_audit(config: CollectiveConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cell_index, cell in enumerate(KEY_CELLS):
        for seed_index in range(config.seeds):
            orders = schedule_orders(
                cell.body_count,
                config.sediment_episodes_bc,
                config.seed + cell_index * 1_000_003 + seed_index * 10_007,
            )
            for policy in POLICIES:
                row, _ = run_scaled_seed(
                    config,
                    cell,
                    seed_index,
                    secondary_order=orders[policy],
                )
                row["policy"] = policy
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_order_audit(
    trials: pd.DataFrame, bootstrap_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    cell_columns = ["body_count", "finite_capacity"]
    for cell_index, (cell_key, cell) in enumerate(trials.groupby(cell_columns)):
        body_count, capacity = cell_key
        for policy_index, (policy, group) in enumerate(cell.groupby("policy")):
            signed = _bootstrap_interval(
                group["distributed_minus_repeated"].to_numpy(float),
                bootstrap_seed + cell_index * 101 + policy_index,
            )
            absolute = _bootstrap_interval(
                group["distributed_minus_repeated"].abs().to_numpy(float),
                bootstrap_seed + 10_000 + cell_index * 101 + policy_index,
            )
            policy_rows.append(
                {
                    "body_count": int(body_count),
                    "capacity": float(capacity),
                    "policy": policy,
                    "diversity_effect": signed["estimate"],
                    "diversity_effect_low": signed["low"],
                    "diversity_effect_high": signed["high"],
                    "absolute_diversity_effect": absolute["estimate"],
                    "absolute_diversity_effect_low": absolute["low"],
                    "absolute_diversity_effect_high": absolute["high"],
                    "material_diversity_prevalence": float(
                        (group["distributed_minus_repeated"].abs() >= 0.025).mean()
                    ),
                    "secondary_repertoire": float(
                        group["distributed_secondary_enacted"].mean()
                    ),
                    "one_way_enactment_rate": float(
                        (group["one_way_status"] == "enacted").mean()
                    ),
                }
            )

        carrying = cell.pivot(
            index="seed", columns="policy", values="distributed_ab"
        )
        repertoire = cell.pivot(
            index="seed",
            columns="policy",
            values="distributed_secondary_enacted",
        )
        carrying_range = carrying.max(axis=1) - carrying.min(axis=1)
        carrying_interval = _bootstrap_interval(
            carrying_range.to_numpy(float), bootstrap_seed + 20_000 + cell_index
        )
        status_changes = cell.groupby("seed")["distributed_status"].nunique() > 1
        cell_rows.append(
            {
                "body_count": int(body_count),
                "capacity": float(capacity),
                "mean_order_carrying_range": carrying_interval["estimate"],
                "mean_order_carrying_range_low": carrying_interval["low"],
                "mean_order_carrying_range_high": carrying_interval["high"],
                "material_order_prevalence": float((carrying_range >= 0.025).mean()),
                "status_change_prevalence": float(status_changes.mean()),
                "mean_repertoire_range": float(
                    (repertoire.max(axis=1) - repertoire.min(axis=1)).mean()
                ),
            }
        )
    return pd.DataFrame(policy_rows), pd.DataFrame(cell_rows)


def render_report(policy: pd.DataFrame, cells: pd.DataFrame) -> str:
    lines = [
        "# Collective Re-Verb participation-order audit",
        "",
        "Every policy contains the same body-specific encounter tokens. Only their",
        "historical order changes. The repeated-body, partitioned, and expanded-capacity",
        "controls remain paired to the same generated world.",
        "",
        "## Diversity effect by order",
        "",
        "| bodies | capacity | order | signed effect | absolute effect | material prevalence | repertoire |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in policy.itertuples(index=False):
        lines.append(
            f"| {row.body_count} | {row.capacity:.2f} | {row.policy} | "
            f"{row.diversity_effect:+.3f} [{row.diversity_effect_low:+.3f}, {row.diversity_effect_high:+.3f}] | "
            f"{row.absolute_diversity_effect:.3f} [{row.absolute_diversity_effect_low:.3f}, {row.absolute_diversity_effect_high:.3f}] | "
            f"{100.0 * row.material_diversity_prevalence:.1f}% | {row.secondary_repertoire:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Consequence of order itself",
            "",
            "| bodies | capacity | within-world carrying range | material order | status changes | repertoire range |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in cells.itertuples(index=False):
        lines.append(
            f"| {row.body_count} | {row.capacity:.2f} | "
            f"{row.mean_order_carrying_range:.3f} [{row.mean_order_carrying_range_low:.3f}, {row.mean_order_carrying_range_high:.3f}] | "
            f"{100.0 * row.material_order_prevalence:.1f}% | "
            f"{100.0 * row.status_change_prevalence:.1f}% | {row.mean_repertoire_range:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Order sensitivity means that an identical multiset of encounters can leave",
            "different subsequent carrying conditions. It does not establish autonomous",
            "agency, metabolism, emergent organs, or quantum non-causality. Persistence of",
            "the absolute diversity effect across policies supports participant diversity",
            "beyond one privileged schedule; variation among policies is itself a bounded",
            "classical expression of historically differentiated recurrence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=18)
    parser.add_argument("--base-seed", type=int, default=20400911)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("collective_reverb_order_output")
    )
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    config = CollectiveConfig(seeds=args.seeds, seed=args.base_seed)
    if args.quick:
        config = replace(
            config,
            seeds=min(args.seeds, 2),
            sediment_episodes_ab=2,
            sediment_episodes_bc=2,
            participation_time=2.0,
            encounter_time=3.0,
        )

    trials = run_order_audit(config)
    policy, cells = summarize_order_audit(trials, args.base_seed + 909)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.output_dir / "order_trials.csv", index=False)
    policy.to_csv(args.output_dir / "order_policy_summary.csv", index=False)
    cells.to_csv(args.output_dir / "order_cell_summary.csv", index=False)
    (args.output_dir / "order_audit_report.md").write_text(
        render_report(policy, cells), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "policies": policy.to_dict(orient="records"),
                "cells": cells.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(policy.to_string(index=False))
    print(cells.to_string(index=False))


if __name__ == "__main__":
    main()
