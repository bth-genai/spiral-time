"""Structural scaling study for the finite shared Re-Verb tissue."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from collective_reverb_experiment import (
    Body,
    CollectiveConfig,
    Tissue,
    World,
    _bootstrap_interval,
    _phase_profile,
    _project_capacity,
    _unitary_interface,
    make_world,
    participate,
    passive_elapsed_decay,
    probe,
)


@dataclass(frozen=True)
class SweepCell:
    axis: str
    value: float
    mode_count: int
    body_count: int
    tissue_channels: int
    finite_capacity: float

    @property
    def key(self) -> str:
        return f"{self.axis}:{self.value:g}"


MODE_COUNTS = (8, 12, 16, 24)
BODY_COUNTS = (3, 4, 6)
CHANNEL_COUNTS = (8, 12, 18, 24)
CAPACITIES = (0.36, 0.54, 0.72, 1.08, 1.44)


def build_cells(axis: str) -> list[SweepCell]:
    cells: list[SweepCell] = []
    if axis in ("all", "modes"):
        cells.extend(
            SweepCell("modes", float(n), n, 3, n, 0.72)
            for n in MODE_COUNTS
        )
    if axis in ("all", "bodies"):
        cells.extend(
            SweepCell("bodies", float(n), 12, n, 12, 0.72)
            for n in BODY_COUNTS
        )
    if axis in ("all", "channels"):
        cells.extend(
            SweepCell("channels", float(n), 12, 3, n, 0.72)
            for n in CHANNEL_COUNTS
        )
    if axis in ("all", "capacity"):
        cells.extend(
            SweepCell("capacity", float(c), 12, 3, 12, c)
            for c in CAPACITIES
        )
    return cells


def _extend_world(
    config: CollectiveConfig, world: World, body_count: int, seed: int
) -> World:
    if body_count < 3 or body_count > 26:
        raise ValueError("body_count must be between 3 and 26")
    bodies = dict(world.bodies)
    axis = np.linspace(-1.0, 1.0, config.mode_count)
    for index in range(3, body_count):
        name = chr(ord("A") + index)
        rng = np.random.default_rng(seed + 700_000 + index * 10_003)
        center = 1.01 + 0.075 * (index - 1)
        profile = np.roll(axis if index % 2 == 0 else axis[::-1], index + 1)
        frequencies = center + config.intrinsic_spread * profile
        frequencies += rng.normal(0.0, 0.010, size=config.mode_count)
        x = np.linspace(0.0, 1.0, config.mode_count)
        gains = np.clip(
            0.67
            + 0.29 * np.cos(math.pi * (x - 0.13 * index))
            + rng.normal(0.0, 0.018, size=config.mode_count),
            0.38,
            1.0,
        )
        bodies[name] = Body(
            name=name,
            frequencies=frequencies,
            gains=gains,
            interface=_unitary_interface(
                config.tissue_channels, config.mode_count, rng
            ),
            forcing_offsets=_phase_profile(config.mode_count, rng),
            metabolic_scale=max(0.72, 0.96 - 0.035 * (index - 2)),
            noise_sd=0.013 + 0.0015 * (index - 1),
        )
    return World(bodies=bodies, initial_tissue=world.initial_tissue.copy())


def _pair_rhythm(left: Body, right: Body) -> float:
    return 0.5 * (
        float(np.mean(left.frequencies)) + float(np.mean(right.frequencies))
    )


def _continuity(before: np.ndarray, after: np.ndarray) -> float:
    denominator = max(
        1.0e-12, float(np.linalg.norm(before) * np.linalg.norm(after))
    )
    return float(np.real(np.vdot(before, after)) / denominator)


def run_scaled_seed(
    base_config: CollectiveConfig,
    cell: SweepCell,
    seed_index: int,
    secondary_order: Sequence[int] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    config = replace(
        base_config,
        mode_count=cell.mode_count,
        tissue_channels=cell.tissue_channels,
        finite_capacity=cell.finite_capacity,
    )
    world_seed = config.seed + seed_index * 10_000
    world = _extend_world(
        config, make_world(config, world_seed), cell.body_count, world_seed
    )
    a, b = world.bodies["A"], world.bodies["B"]
    secondary_names = [chr(ord("A") + i) for i in range(2, cell.body_count)]
    rhythm_ab = _pair_rhythm(a, b)
    probe_seed_ab = world_seed + 701
    started = time.perf_counter()

    initial = Tissue(world.initial_tissue.copy(), cell.finite_capacity)
    initial_probe = probe(
        config, initial, a, b, probe_seed_ab, rhythm_ab
    )
    history = initial.copy()
    for episode in range(config.sediment_episodes_ab):
        participate(
            config,
            history,
            a,
            b,
            world_seed + 1_100 + episode,
            rhythm_ab,
        )
    history_probe = probe(
        config, history, a, b, probe_seed_ab, rhythm_ab
    )
    history_matrix = history.matrix.copy()

    tissues = {
        "distributed_finite": history.copy(),
        "repeated_finite": history.copy(),
        "partitioned": history.copy(),
        "distributed_expanded": history.copy(
            capacity=config.expanded_capacity
        ),
    }
    total_secondary_episodes = (
        len(secondary_names) * config.sediment_episodes_bc
    )
    secondary_tokens = [
        (
            secondary_names[ordinal // config.sediment_episodes_bc],
            world_seed + 2_100 + ordinal,
        )
        for ordinal in range(total_secondary_episodes)
    ]
    if secondary_order is not None:
        order = tuple(int(index) for index in secondary_order)
        if sorted(order) != list(range(total_secondary_episodes)):
            raise ValueError(
                "secondary_order must be a permutation of all secondary episodes"
            )
        secondary_tokens = [secondary_tokens[index] for index in order]
    passive_elapsed_decay(
        config, tissues["partitioned"], total_secondary_episodes
    )

    for ordinal, (distributed_name, episode_seed) in enumerate(secondary_tokens):
        distributed_body = world.bodies[distributed_name]
        rhythm = _pair_rhythm(b, distributed_body)
        participate(
            config,
            tissues["distributed_finite"],
            b,
            distributed_body,
            episode_seed,
            rhythm,
        )
        participate(
            config,
            tissues["distributed_expanded"],
            b,
            distributed_body,
            episode_seed,
            rhythm,
        )

        repeated_body = world.bodies["C"]
        participate(
            config,
            tissues["repeated_finite"],
            b,
            repeated_body,
            world_seed + 2_100 + ordinal,
            _pair_rhythm(b, repeated_body),
        )

    probes: list[dict[str, object]] = []
    ab_results: dict[str, dict[str, float | str]] = {}
    secondary_means: dict[str, float] = {}
    secondary_enacted: dict[str, int] = {}
    for condition, tissue in tissues.items():
        result = probe(
            config, tissue, a, b, probe_seed_ab, rhythm_ab
        )
        ab_results[condition] = result
        probes.append(
            {
                "axis": cell.axis,
                "value": cell.value,
                "seed": seed_index,
                "condition": condition,
                "pair": "AB",
                **result,
            }
        )
        secondary_values: list[float] = []
        enacted_count = 0
        for offset, name in enumerate(secondary_names):
            body = world.bodies[name]
            secondary_result = probe(
                config,
                tissue,
                b,
                body,
                world_seed + 907 + offset * 101,
                _pair_rhythm(b, body),
            )
            secondary_values.append(float(secondary_result["carried"]))
            enacted_count += secondary_result["status"] == "enacted"
            probes.append(
                {
                    "axis": cell.axis,
                    "value": cell.value,
                    "seed": seed_index,
                    "condition": condition,
                    "pair": f"B{name}",
                    **secondary_result,
                }
            )
        secondary_means[condition] = float(np.mean(secondary_values))
        secondary_enacted[condition] = int(enacted_count)

    one_way = probe(
        config,
        tissues["distributed_finite"],
        a,
        b,
        probe_seed_ab,
        rhythm_ab,
        feedback=(1.0, 0.0),
    )
    runtime = time.perf_counter() - started
    distributed = float(ab_results["distributed_finite"]["carried"])
    repeated = float(ab_results["repeated_finite"]["carried"])
    partitioned = float(ab_results["partitioned"]["carried"])
    expanded = float(ab_results["distributed_expanded"]["carried"])
    diversity_matrix_distance = float(
        np.linalg.norm(
            tissues["distributed_finite"].matrix
            - tissues["repeated_finite"].matrix
        )
    )
    diversity_matrix_scale = max(
        1.0e-12,
        0.5
        * (
            float(np.linalg.norm(tissues["distributed_finite"].matrix))
            + float(np.linalg.norm(tissues["repeated_finite"].matrix))
        ),
    )
    persistent_bytes = 16 * (
        cell.tissue_channels * cell.tissue_channels
        + cell.body_count * cell.tissue_channels * cell.mode_count
    )
    row: dict[str, object] = {
        "axis": cell.axis,
        "value": cell.value,
        "seed": seed_index,
        "mode_count": cell.mode_count,
        "body_count": cell.body_count,
        "tissue_channels": cell.tissue_channels,
        "finite_capacity": cell.finite_capacity,
        "initial_ab": float(initial_probe["carried"]),
        "history_ab": float(history_probe["carried"]),
        "distributed_ab": distributed,
        "repeated_ab": repeated,
        "partitioned_ab": partitioned,
        "expanded_ab": expanded,
        "history_effect": float(history_probe["carried"])
        - float(initial_probe["carried"]),
        "shared_minus_partitioned": distributed - partitioned,
        "distributed_minus_repeated": distributed - repeated,
        "diversity_matrix_distance": diversity_matrix_distance,
        "normalized_diversity_matrix_distance": (
            diversity_matrix_distance / diversity_matrix_scale
        ),
        "finite_minus_expanded": distributed - expanded,
        "distributed_secondary_mean": secondary_means[
            "distributed_finite"
        ],
        "repeated_secondary_mean": secondary_means["repeated_finite"],
        "distributed_secondary_enacted": secondary_enacted[
            "distributed_finite"
        ],
        "distributed_status": ab_results["distributed_finite"]["status"],
        "partitioned_status": ab_results["partitioned"]["status"],
        "expanded_status": ab_results["distributed_expanded"]["status"],
        "one_way_status": one_way["status"],
        "finite_capacity_use": float(
            np.linalg.norm(tissues["distributed_finite"].matrix)
            / tissues["distributed_finite"].capacity
        ),
        "expanded_capacity_use": float(
            np.linalg.norm(tissues["distributed_expanded"].matrix)
            / tissues["distributed_expanded"].capacity
        ),
        "finite_continuity": _continuity(
            history_matrix, tissues["distributed_finite"].matrix
        ),
        "repeated_continuity": _continuity(
            history_matrix, tissues["repeated_finite"].matrix
        ),
        "runtime_seconds": runtime,
        "persistent_complex_bytes": persistent_bytes,
    }
    return row, probes


def _interval_columns(
    values: np.ndarray, seed: int, prefix: str
) -> dict[str, float]:
    interval = _bootstrap_interval(values, seed)
    return {
        f"{prefix}": interval["estimate"],
        f"{prefix}_low": interval["low"],
        f"{prefix}_high": interval["high"],
    }


def summarize_cell(
    config: CollectiveConfig, cell: SweepCell, trials: pd.DataFrame
) -> dict[str, object]:
    seed = config.seed + int(round(cell.value * 1000)) + len(cell.axis) * 101
    shared = trials["shared_minus_partitioned"].to_numpy(float)
    diversity = trials["distributed_minus_repeated"].to_numpy(float)
    capacity = trials["finite_minus_expanded"].to_numpy(float)
    history = trials["history_effect"].to_numpy(float)
    summary: dict[str, object] = {
        "axis": cell.axis,
        "value": cell.value,
        "mode_count": cell.mode_count,
        "body_count": cell.body_count,
        "tissue_channels": cell.tissue_channels,
        "finite_capacity": cell.finite_capacity,
        "seeds": len(trials),
        **_interval_columns(history, seed + 1, "history_effect"),
        **_interval_columns(shared, seed + 2, "shared_minus_partitioned"),
        **_interval_columns(np.abs(shared), seed + 3, "absolute_shared_effect"),
        **_interval_columns(diversity, seed + 4, "distributed_minus_repeated"),
        **_interval_columns(np.abs(diversity), seed + 5, "absolute_diversity_effect"),
        **_interval_columns(
            trials["normalized_diversity_matrix_distance"].to_numpy(float),
            seed + 51,
            "normalized_diversity_matrix_distance",
        ),
        **_interval_columns(capacity, seed + 6, "finite_minus_expanded"),
        "material_shared_prevalence": float(np.mean(np.abs(shared) >= 0.025)),
        "shared_negative_prevalence": float(np.mean(shared < -1.0e-12)),
        "diversity_material_prevalence": float(
            np.mean(np.abs(diversity) >= 0.025)
        ),
        "capacity_negative_prevalence": float(
            np.mean(capacity < -1.0e-12)
        ),
        "distributed_enactment_rate": float(
            trials["distributed_status"].eq("enacted").mean()
        ),
        "partitioned_enactment_rate": float(
            trials["partitioned_status"].eq("enacted").mean()
        ),
        "expanded_enactment_rate": float(
            trials["expanded_status"].eq("enacted").mean()
        ),
        "one_way_enactment_rate": float(
            trials["one_way_status"].eq("enacted").mean()
        ),
        "secondary_mean_carrying": float(
            trials["distributed_secondary_mean"].mean()
        ),
        "secondary_mean_repertoire": float(
            trials["distributed_secondary_enacted"].mean()
        ),
        "mean_finite_capacity_use": float(
            trials["finite_capacity_use"].mean()
        ),
        "mean_finite_continuity": float(trials["finite_continuity"].mean()),
        "mean_runtime_seconds": float(trials["runtime_seconds"].mean()),
        "persistent_complex_bytes": int(trials["persistent_complex_bytes"].iloc[0]),
    }
    return summary


def run_scaling(
    config: CollectiveConfig, cells: list[SweepCell]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trial_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for cell in cells:
        cell_rows: list[dict[str, object]] = []
        for seed_index in range(config.seeds):
            row, probes = run_scaled_seed(config, cell, seed_index)
            cell_rows.append(row)
            trial_rows.append(row)
            probe_rows.extend(probes)
        summary_rows.append(
            summarize_cell(config, cell, pd.DataFrame(cell_rows))
        )
    return (
        pd.DataFrame(trial_rows),
        pd.DataFrame(probe_rows),
        pd.DataFrame(summary_rows),
    )


def build_summary(
    config: CollectiveConfig, summaries: pd.DataFrame
) -> dict[str, object]:
    mode_rows = summaries[summaries["axis"] == "modes"]
    body_rows = summaries[summaries["axis"] == "bodies"]
    channel_rows = summaries[summaries["axis"] == "channels"]
    capacity_rows = summaries[summaries["axis"] == "capacity"]
    return {
        "question": (
            "Where does collective mediation remain operative as modes, bodies, "
            "tissue channels, and capacity vary independently?"
        ),
        "config": asdict(config),
        "cells": summaries.to_dict(orient="records"),
        "audit": {
            "one_way_never_enacts": bool(
                summaries["one_way_enactment_rate"].eq(0.0).all()
            ),
            "history_positive_all_mode_cells": bool(
                len(mode_rows) > 0 and mode_rows["history_effect"].gt(0.0).all()
            ),
            "material_shared_effect_mode_cells": int(
                mode_rows["material_shared_prevalence"].gt(0.0).sum()
            ),
            "body_diversity_detected_beyond_three": bool(
                len(body_rows) > 0
                and body_rows[body_rows["body_count"] > 3][
                    "diversity_material_prevalence"
                ].gt(0.0).any()
            ),
            "channel_and_capacity_sweeps_completed": bool(
                len(channel_rows) == len(CHANNEL_COUNTS)
                and len(capacity_rows) == len(CAPACITIES)
            ),
        },
    }


def _format_interval(row: pd.Series, field: str) -> str:
    return (
        f"{row[field]:+.3f} "
        f"[{row[field + '_low']:+.3f}, {row[field + '_high']:+.3f}]"
    )


def write_report(summary: dict[str, object], frame: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Collective Re-Verb structural scaling",
        "",
        str(summary["question"]),
        "",
    ]
    for axis in ("modes", "bodies", "channels", "capacity"):
        axis_frame = frame[frame["axis"] == axis].sort_values("value")
        if axis_frame.empty:
            continue
        lines.extend(
            [
                f"## {axis.title()} sweep",
                "",
                "| value | history | shared - partitioned | distributed - repeated | finite - expanded | material shared | one-way | runtime/seed |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in axis_frame.iterrows():
            lines.append(
                f"| {row['value']:g} | {_format_interval(row, 'history_effect')} | "
                f"{_format_interval(row, 'shared_minus_partitioned')} | "
                f"{_format_interval(row, 'distributed_minus_repeated')} | "
                f"{_format_interval(row, 'finite_minus_expanded')} | "
                f"{row['material_shared_prevalence']:.1%} | "
                f"{row['one_way_enactment_rate']:.1%} | "
                f"{row['mean_runtime_seconds']:.2f}s |"
            )
        lines.append("")
    lines.extend(["## Audit", ""])
    for key, value in summary["audit"].items():
        lines.append(f"- {key}: {value}.")
    lines.extend(
        [
            "",
            "## Reading boundary",
            "",
            "No sweep is interpreted as a performance ranking, and no monotonic "
            "effect was required. Failed or sign-changing cells locate regimes of "
            "the constructed mechanism. Bootstrap intervals describe generated-world "
            "variation under fixed parameters, not population uncertainty. The study "
            "does not establish ecological validity, strong anticipation, quantum "
            "non-causality, or agential realism.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--axis",
        choices=("all", "modes", "bodies", "channels", "capacity"),
        default="all",
    )
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--base-seed", type=int, default=20280911)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("collective_reverb_scaling_output")
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
            sediment_episodes_bc=1,
            participation_time=4.0,
            encounter_time=7.0,
        )
    cells = build_cells(args.axis)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials, probes, summaries = run_scaling(config, cells)
    result = build_summary(config, summaries)
    trials.to_csv(args.output_dir / "scaling_trials.csv", index=False)
    probes.to_csv(args.output_dir / "scaling_probes.csv", index=False)
    summaries.to_csv(args.output_dir / "scaling_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    write_report(
        result,
        summaries,
        args.output_dir / "collective_reverb_scaling_report.md",
    )
    print(summaries[
        [
            "axis",
            "value",
            "history_effect",
            "shared_minus_partitioned",
            "distributed_minus_repeated",
            "finite_minus_expanded",
            "material_shared_prevalence",
            "one_way_enactment_rate",
            "mean_runtime_seconds",
        ]
    ].to_string(index=False))
    print(json.dumps(result["audit"], indent=2))


if __name__ == "__main__":
    main()
