"""Exact historical-return experiment for the finite collective tissue.

The present A--B probe is held numerically identical while intervening tissue
history, participation order, or relational orientation is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from collective_reverb_experiment import (
    Body,
    CollectiveConfig,
    Tissue,
    _bootstrap_interval,
    _project_capacity,
    make_world,
    participate,
    passive_elapsed_decay,
    probe,
)
from collective_reverb_order_audit import KEY_CELLS, POLICIES, schedule_orders
from collective_reverb_scaling import SweepCell, _extend_world, _pair_rhythm


MATERIAL_DIFFERENCE = 0.025
GEOMETRY_TOLERANCE = 1.0e-10
PROBE_FIELDS = (
    "carried",
    "support",
    "cumulative_support",
    "min_viability",
    "max_tension",
    "event_time",
    "terminal_phase",
    "coordinate_phase",
    "coordinate_return",
)


def _cell_id(cell: SweepCell) -> str:
    return f"bodies_{cell.body_count}_capacity_{cell.finite_capacity:.2f}"


def _identical_secondary_world(world, body_count: int):
    source = world.bodies["C"]
    bodies = dict(world.bodies)
    for index in range(2, body_count):
        name = chr(ord("A") + index)
        bodies[name] = Body(
            name=name,
            frequencies=source.frequencies.copy(),
            gains=source.gains.copy(),
            interface=source.interface.copy(),
            forcing_offsets=source.forcing_offsets.copy(),
            metabolic_scale=source.metabolic_scale,
            noise_sd=source.noise_sd,
        )
    return replace(world, bodies=bodies)


def _commutative_token(left: Body, right: Body, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phase_left = left.forcing_offsets + rng.normal(0.0, 0.08, left.gains.size)
    phase_right = right.forcing_offsets + rng.normal(0.0, 0.08, right.gains.size)
    q_left = left.interface @ (left.gains * np.exp(1j * phase_left))
    q_right = right.interface @ (right.gains * np.exp(1j * phase_right))
    q_left /= math.sqrt(left.gains.size)
    q_right /= math.sqrt(right.gains.size)
    return 0.5 * (
        np.outer(q_left, q_right.conj())
        + np.outer(q_right, q_left.conj())
    )


def _unitary(channels: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(channels, channels)) + 1j * rng.normal(
        size=(channels, channels)
    )
    q, r = np.linalg.qr(raw)
    diagonal = np.diag(r)
    phases = np.ones(channels, dtype=complex)
    nonzero = np.abs(diagonal) > 1.0e-15
    phases[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])
    return q * phases.conj()


def _transform_body(body: Body, unitary: np.ndarray) -> Body:
    return replace(body, interface=unitary @ body.interface)


def _probe_row(
    *,
    cell: SweepCell,
    seed_index: int,
    world_seed: int,
    policy: str,
    condition: str,
    tissue: Tissue,
    result: dict[str, float | str],
) -> dict[str, object]:
    return {
        "cell": _cell_id(cell),
        "body_count": cell.body_count,
        "capacity": cell.finite_capacity,
        "seed": seed_index,
        "world_seed": world_seed,
        "policy": policy,
        "condition": condition,
        "tissue_norm": float(np.linalg.norm(tissue.matrix)),
        "capacity_use": float(np.linalg.norm(tissue.matrix) / tissue.capacity),
        **result,
    }


def _secondary_tokens(
    config: CollectiveConfig, cell: SweepCell, world_seed: int
) -> list[tuple[str, int]]:
    names = [chr(ord("A") + index) for index in range(2, cell.body_count)]
    return [
        (names[ordinal // config.sediment_episodes_bc], world_seed + 2_100 + ordinal)
        for ordinal in range(len(names) * config.sediment_episodes_bc)
    ]


def _apply_history(
    config: CollectiveConfig,
    tissue: Tissue,
    world,
    tokens: list[tuple[str, int]],
    order: tuple[int, ...],
    *,
    repeated: bool = False,
) -> None:
    b = world.bodies["B"]
    for token_index in order:
        name, episode_seed = tokens[token_index]
        right = world.bodies["C"] if repeated else world.bodies[name]
        participate(
            config,
            tissue,
            b,
            right,
            episode_seed,
            _pair_rhythm(b, right),
        )


def _commutative_history(
    config: CollectiveConfig,
    history: Tissue,
    world,
    tokens: list[tuple[str, int]],
    order: tuple[int, ...],
) -> Tissue:
    matrix = history.matrix.copy()
    b = world.bodies["B"]
    for token_index in order:
        name, episode_seed = tokens[token_index]
        matrix += config.sediment_rate * _commutative_token(
            b, world.bodies[name], episode_seed
        )
    return Tissue(_project_capacity(matrix, history.capacity), history.capacity)


def _circular_distance(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        return math.nan
    return float(abs(np.angle(np.exp(1j * (left - right)))))


def run_world(
    config: CollectiveConfig, cell: SweepCell, cell_index: int, seed_index: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cell_config = replace(
        config,
        mode_count=cell.mode_count,
        tissue_channels=cell.tissue_channels,
        finite_capacity=cell.finite_capacity,
    )
    world_seed = cell_config.seed + cell_index * 1_000_003 + seed_index * 10_000
    world = _extend_world(
        cell_config,
        make_world(cell_config, world_seed),
        cell.body_count,
        world_seed,
    )
    a, b = world.bodies["A"], world.bodies["B"]
    rhythm_ab = _pair_rhythm(a, b)
    probe_seed = world_seed + 701
    history = Tissue(world.initial_tissue.copy(), cell.finite_capacity)
    for episode in range(cell_config.sediment_episodes_ab):
        participate(
            cell_config,
            history,
            a,
            b,
            world_seed + 1_100 + episode,
            rhythm_ab,
        )

    rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    baseline_result = probe(
        cell_config, history, a, b, probe_seed, rhythm_ab
    )
    rows.append(
        _probe_row(
            cell=cell,
            seed_index=seed_index,
            world_seed=world_seed,
            policy="baseline",
            condition="history_baseline",
            tissue=history,
            result=baseline_result,
        )
    )

    tokens = _secondary_tokens(cell_config, cell, world_seed)
    orders = schedule_orders(
        cell.body_count,
        cell_config.sediment_episodes_bc,
        world_seed + 31_337,
    )

    passive = history.copy()
    passive_elapsed_decay(cell_config, passive, len(tokens))
    passive_result = probe(cell_config, passive, a, b, probe_seed, rhythm_ab)
    rows.append(
        _probe_row(
            cell=cell,
            seed_index=seed_index,
            world_seed=world_seed,
            policy="passive",
            condition="passive_elapsed",
            tissue=passive,
            result=passive_result,
        )
    )

    for policy_index, policy in enumerate(POLICIES):
        order = orders[policy]
        distributed = history.copy()
        _apply_history(cell_config, distributed, world, tokens, order)
        distributed_result = probe(
            cell_config, distributed, a, b, probe_seed, rhythm_ab
        )
        rows.append(
            _probe_row(
                cell=cell,
                seed_index=seed_index,
                world_seed=world_seed,
                policy=policy,
                condition="distributed",
                tissue=distributed,
                result=distributed_result,
            )
        )

        commutative = _commutative_history(
            cell_config, history, world, tokens, order
        )
        commutative_result = probe(
            cell_config, commutative, a, b, probe_seed, rhythm_ab
        )
        rows.append(
            _probe_row(
                cell=cell,
                seed_index=seed_index,
                world_seed=world_seed,
                policy=policy,
                condition="commutative",
                tissue=commutative,
                result=commutative_result,
            )
        )

        unitary = _unitary(
            cell_config.tissue_channels,
            world_seed + 80_000 + policy_index,
        )
        rotated_matrix = unitary @ distributed.matrix @ unitary.conj().T
        tissue_only = Tissue(rotated_matrix.copy(), distributed.capacity)
        tissue_only_result = probe(
            cell_config, tissue_only, a, b, probe_seed, rhythm_ab
        )
        rows.append(
            _probe_row(
                cell=cell,
                seed_index=seed_index,
                world_seed=world_seed,
                policy=policy,
                condition="tissue_only_rotation",
                tissue=tissue_only,
                result=tissue_only_result,
            )
        )

        transformed_a = _transform_body(a, unitary)
        transformed_b = _transform_body(b, unitary)
        co_transformed_result = probe(
            cell_config,
            Tissue(rotated_matrix.copy(), distributed.capacity),
            transformed_a,
            transformed_b,
            probe_seed,
            rhythm_ab,
        )
        rows.append(
            _probe_row(
                cell=cell,
                seed_index=seed_index,
                world_seed=world_seed,
                policy=policy,
                condition="co_transformed",
                tissue=tissue_only,
                result=co_transformed_result,
            )
        )
        original_eigenvalues = np.linalg.eigvalsh(distributed.matrix)
        rotated_eigenvalues = np.linalg.eigvalsh(rotated_matrix)
        geometry_rows.append(
            {
                "cell": _cell_id(cell),
                "body_count": cell.body_count,
                "capacity": cell.finite_capacity,
                "seed": seed_index,
                "policy": policy,
                "norm_difference": abs(
                    float(np.linalg.norm(distributed.matrix))
                    - float(np.linalg.norm(rotated_matrix))
                ),
                "maximum_eigenvalue_difference": float(
                    np.max(np.abs(original_eigenvalues - rotated_eigenvalues))
                ),
                "co_transform_status_match": (
                    distributed_result["status"] == co_transformed_result["status"]
                ),
                **{
                    f"co_transform_{field}_difference": (
                        _circular_distance(
                            float(distributed_result[field]),
                            float(co_transformed_result[field]),
                        )
                        if "phase" in field
                        else abs(
                            float(distributed_result[field])
                            - float(co_transformed_result[field])
                        )
                    )
                    for field in PROBE_FIELDS
                    if np.isfinite(float(distributed_result[field]))
                    and np.isfinite(float(co_transformed_result[field]))
                },
            }
        )

        if policy == "blocked":
            repeated = history.copy()
            _apply_history(
                cell_config, repeated, world, tokens, order, repeated=True
            )
            repeated_result = probe(
                cell_config, repeated, a, b, probe_seed, rhythm_ab
            )
            rows.append(
                _probe_row(
                    cell=cell,
                    seed_index=seed_index,
                    world_seed=world_seed,
                    policy=policy,
                    condition="repeated_C",
                    tissue=repeated,
                    result=repeated_result,
                )
            )

            identical_world = _identical_secondary_world(world, cell.body_count)
            identical = history.copy()
            _apply_history(
                cell_config, identical, identical_world, tokens, order
            )
            identical_result = probe(
                cell_config, identical, a, b, probe_seed, rhythm_ab
            )
            rows.append(
                _probe_row(
                    cell=cell,
                    seed_index=seed_index,
                    world_seed=world_seed,
                    policy=policy,
                    condition="identical_secondary",
                    tissue=identical,
                    result=identical_result,
                )
            )

    return rows, geometry_rows


def run_experiment(
    config: CollectiveConfig, workers: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = [
        (config, cell, cell_index, seed_index)
        for cell_index, cell in enumerate(KEY_CELLS)
        for seed_index in range(config.seeds)
    ]
    if workers == 1:
        results = [run_world(*task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_run_world_task, tasks))
    trial_rows = [row for result, _ in results for row in result]
    geometry_rows = [row for _, geometry in results for row in geometry]
    return pd.DataFrame(trial_rows), pd.DataFrame(geometry_rows)


def _run_world_task(task):
    return run_world(*task)


def _paired_comparison(
    trials: pd.DataFrame,
    left_condition: str,
    right_condition: str,
    *,
    policy: str,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    selected = trials[
        trials["condition"].isin((left_condition, right_condition))
        & trials["policy"].isin((policy, "baseline", "passive"))
    ]
    rows: list[dict[str, object]] = []
    for cell_index, (cell, group) in enumerate(selected.groupby("cell")):
        left = group[group["condition"].eq(left_condition)].set_index("seed")
        right = group[group["condition"].eq(right_condition)].set_index("seed")
        common = left.index.intersection(right.index)
        left = left.loc[common]
        right = right.loc[common]
        difference = left["carried"].to_numpy(float) - right["carried"].to_numpy(float)
        signed = _bootstrap_interval(difference, bootstrap_seed + cell_index * 101)
        absolute = _bootstrap_interval(
            np.abs(difference), bootstrap_seed + 10_000 + cell_index * 101
        )
        jointly_enacted = left["status"].eq("enacted") & right["status"].eq("enacted")
        event_difference = np.abs(
            left.loc[jointly_enacted, "event_time"].to_numpy(float)
            - right.loc[jointly_enacted, "event_time"].to_numpy(float)
        )
        phase_difference = np.asarray(
            [
                _circular_distance(l, r)
                for l, r in zip(
                    left.loc[jointly_enacted, "coordinate_phase"].to_numpy(float),
                    right.loc[jointly_enacted, "coordinate_phase"].to_numpy(float),
                )
            ],
            dtype=float,
        )
        coordinate_return_difference = np.abs(
            left.loc[jointly_enacted, "coordinate_return"].to_numpy(float)
            - right.loc[jointly_enacted, "coordinate_return"].to_numpy(float)
        )
        support_difference = (
            left["support"].to_numpy(float) - right["support"].to_numpy(float)
        )
        cumulative_support_difference = (
            left["cumulative_support"].to_numpy(float)
            - right["cumulative_support"].to_numpy(float)
        )
        rows.append(
            {
                "cell": cell,
                "body_count": int(left["body_count"].iloc[0]),
                "capacity": float(left["capacity"].iloc[0]),
                "policy": policy,
                "left_condition": left_condition,
                "right_condition": right_condition,
                "pairs": len(common),
                "mean_carrying_difference": signed["estimate"],
                "carrying_difference_low": signed["low"],
                "carrying_difference_high": signed["high"],
                "mean_absolute_carrying_difference": absolute["estimate"],
                "absolute_difference_low": absolute["low"],
                "absolute_difference_high": absolute["high"],
                "material_difference_prevalence": float(
                    (np.abs(difference) >= MATERIAL_DIFFERENCE).mean()
                ),
                "status_change_prevalence": float(
                    (left["status"].to_numpy() != right["status"].to_numpy()).mean()
                ),
                "mean_support_difference": float(np.mean(support_difference)),
                "mean_absolute_support_difference": float(
                    np.mean(np.abs(support_difference))
                ),
                "mean_cumulative_support_difference": float(
                    np.mean(cumulative_support_difference)
                ),
                "mean_min_viability_difference": float(
                    np.mean(
                        left["min_viability"].to_numpy(float)
                        - right["min_viability"].to_numpy(float)
                    )
                ),
                "mean_max_tension_difference": float(
                    np.mean(
                        left["max_tension"].to_numpy(float)
                        - right["max_tension"].to_numpy(float)
                    )
                ),
                "jointly_enacted_pairs": int(jointly_enacted.sum()),
                "mean_absolute_event_time_difference": (
                    float(np.mean(event_difference)) if len(event_difference) else math.nan
                ),
                "mean_coordinate_phase_difference": (
                    float(np.mean(phase_difference)) if len(phase_difference) else math.nan
                ),
                "mean_coordinate_return_difference": (
                    float(np.mean(coordinate_return_difference))
                    if len(coordinate_return_difference)
                    else math.nan
                ),
            }
        )
    return rows


def summarize_comparisons(trials: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy_index, policy in enumerate(POLICIES):
        rows.extend(
            _paired_comparison(
                trials,
                "distributed",
                "history_baseline",
                policy=policy,
                bootstrap_seed=seed + policy_index * 1_003,
            )
        )
        rows.extend(
            _paired_comparison(
                trials,
                "distributed",
                "passive_elapsed",
                policy=policy,
                bootstrap_seed=seed + 10_000 + policy_index * 1_003,
            )
        )
        rows.extend(
            _paired_comparison(
                trials,
                "tissue_only_rotation",
                "distributed",
                policy=policy,
                bootstrap_seed=seed + 20_000 + policy_index * 1_003,
            )
        )
    rows.extend(
        _paired_comparison(
            trials,
            "passive_elapsed",
            "history_baseline",
            policy="passive",
            bootstrap_seed=seed + 40_000,
        )
    )
    for left in ("distributed", "identical_secondary"):
        rows.extend(
            _paired_comparison(
                trials,
                left,
                "repeated_C",
                policy="blocked",
                bootstrap_seed=seed + 50_000 + len(rows) * 17,
            )
        )
    return pd.DataFrame(rows)


def summarize_order(trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition in ("distributed", "commutative"):
        selected = trials[trials["condition"].eq(condition)]
        for cell, group in selected.groupby("cell"):
            carrying = group.pivot(index="seed", columns="policy", values="carried")
            support = group.pivot(index="seed", columns="policy", values="support")
            cumulative_support = group.pivot(
                index="seed", columns="policy", values="cumulative_support"
            )
            status_counts = group.groupby("seed")["status"].nunique()
            carrying_range = carrying.max(axis=1) - carrying.min(axis=1)
            event_ranges: list[float] = []
            phase_ranges: list[float] = []
            for _, seed_group in group.groupby("seed"):
                enacted = seed_group[seed_group["status"].eq("enacted")]
                if len(enacted) >= 2:
                    event_ranges.append(
                        float(enacted["event_time"].max() - enacted["event_time"].min())
                    )
                    phases = enacted["coordinate_phase"].to_numpy(float)
                    phase_ranges.append(
                        max(
                            _circular_distance(left, right)
                            for index, left in enumerate(phases)
                            for right in phases[index + 1 :]
                        )
                    )
            rows.append(
                {
                    "cell": cell,
                    "body_count": int(group["body_count"].iloc[0]),
                    "capacity": float(group["capacity"].iloc[0]),
                    "condition": condition,
                    "mean_carrying_range": float(carrying_range.mean()),
                    "maximum_carrying_range": float(carrying_range.max()),
                    "mean_support_range": float(
                        (support.max(axis=1) - support.min(axis=1)).mean()
                    ),
                    "mean_cumulative_support_range": float(
                        (
                            cumulative_support.max(axis=1)
                            - cumulative_support.min(axis=1)
                        ).mean()
                    ),
                    "material_order_prevalence": float(
                        (carrying_range >= MATERIAL_DIFFERENCE).mean()
                    ),
                    "status_change_prevalence": float((status_counts > 1).mean()),
                    "worlds_with_two_or_more_enacted_orders": len(event_ranges),
                    "mean_event_time_range_when_defined": (
                        float(np.mean(event_ranges)) if event_ranges else math.nan
                    ),
                    "mean_coordinate_phase_range_when_defined": (
                        float(np.mean(phase_ranges)) if phase_ranges else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_status(trials: pd.DataFrame) -> pd.DataFrame:
    selected = trials[
        trials["condition"].isin(
            (
                "history_baseline",
                "passive_elapsed",
                "distributed",
                "repeated_C",
                "identical_secondary",
                "commutative",
                "tissue_only_rotation",
                "co_transformed",
            )
        )
    ].copy()
    selected["enacted"] = selected["status"].eq("enacted").astype(float)
    selected["refused"] = selected["status"].eq("refused").astype(float)
    selected["censored"] = selected["status"].eq("censored").astype(float)
    return (
        selected.groupby(["cell", "policy", "condition"], as_index=False)
        .agg(
            observations=("status", "size"),
            enactment_rate=("enacted", "mean"),
            refusal_rate=("refused", "mean"),
            censoring_rate=("censored", "mean"),
            mean_carried=("carried", "mean"),
            mean_support=("support", "mean"),
            mean_cumulative_support=("cumulative_support", "mean"),
        )
        .sort_values(["cell", "condition", "policy"])
    )


def summarize_geometry(geometry: pd.DataFrame) -> dict[str, object]:
    difference_columns = [
        column
        for column in geometry.columns
        if column.startswith("co_transform_") and column.endswith("_difference")
    ]
    maxima = {
        column: float(geometry[column].max(skipna=True))
        for column in difference_columns
    }
    finite_maxima = [value for value in maxima.values() if np.isfinite(value)]
    return {
        "maximum_norm_difference": float(geometry["norm_difference"].max()),
        "maximum_eigenvalue_difference": float(
            geometry["maximum_eigenvalue_difference"].max()
        ),
        "all_co_transform_statuses_match": bool(
            geometry["co_transform_status_match"].all()
        ),
        "co_transform_maximum_differences": maxima,
        "all_co_transform_differences_within_tolerance": bool(
            finite_maxima and max(finite_maxima) <= GEOMETRY_TOLERANCE
        ),
    }


def render_report(
    comparisons: pd.DataFrame,
    order: pd.DataFrame,
    geometry_summary: dict[str, object],
) -> str:
    lines = [
        "# Exact historical return report",
        "",
        "The A--B return probe is numerically identical within every generated",
        "world. Differences therefore arise from the intervening tissue history or",
        "from its relational orientation to A and B.",
        "",
        "## Paired historical contrasts",
        "",
        "| cell | policy | contrast | signed carrying | absolute carrying | material | status change | support difference | cumulative support difference | joint enactment | event-time difference | phase difference |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| {row.cell} | {row.policy} | {row.left_condition} - {row.right_condition} | "
            f"{row.mean_carrying_difference:+.3f} [{row.carrying_difference_low:+.3f}, {row.carrying_difference_high:+.3f}] | "
            f"{row.mean_absolute_carrying_difference:.3f} | "
            f"{row.material_difference_prevalence:.1%} | "
            f"{row.status_change_prevalence:.1%} | "
            f"{row.mean_support_difference:+.3f} | "
            f"{row.mean_cumulative_support_difference:+.3f} | "
            f"{row.jointly_enacted_pairs}/{row.pairs} | "
            f"{row.mean_absolute_event_time_difference:.3f} | "
            f"{row.mean_coordinate_phase_difference:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Order specificity",
            "",
            "| cell | accumulation | mean carrying range | maximum carrying range | support range | cumulative support range | material order | status change | enacted-order worlds | event-time range | phase range |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in order.itertuples(index=False):
        lines.append(
            f"| {row.cell} | {row.condition} | {row.mean_carrying_range:.3e} | "
            f"{row.maximum_carrying_range:.3e} | {row.mean_support_range:.3e} | "
            f"{row.mean_cumulative_support_range:.3e} | "
            f"{row.material_order_prevalence:.1%} | "
            f"{row.status_change_prevalence:.1%} | "
            f"{row.worlds_with_two_or_more_enacted_orders} | "
            f"{row.mean_event_time_range_when_defined:.3f} | "
            f"{row.mean_coordinate_phase_range_when_defined:.3f} |"
        )
    maximum_co = geometry_summary["co_transform_maximum_differences"]
    lines.extend(
        [
            "",
            "## Relational-geometry checks",
            "",
            f"- Maximum norm difference after tissue rotation: {geometry_summary['maximum_norm_difference']:.3e}.",
            f"- Maximum eigenvalue difference after tissue rotation: {geometry_summary['maximum_eigenvalue_difference']:.3e}.",
            f"- All co-transformed statuses match: {geometry_summary['all_co_transform_statuses_match']}.",
            f"- All finite co-transformed outcome differences are within {GEOMETRY_TOLERANCE:.0e}: {geometry_summary['all_co_transform_differences_within_tolerance']}.",
            f"- Largest co-transformed carrying difference: {maximum_co.get('co_transform_carried_difference', math.nan):.3e}.",
            "",
            "## Reading boundary",
            "",
            "A differentiated return is evidence that this constructed classical",
            "apparatus carries intervening history into what can subsequently be",
            "enacted. The controls distinguish elapsed time, encounter count, material",
            "participant difference, noncommutative update history, and relational",
            "orientation. They do not establish autonomous agency, ecological validity,",
            "agential realism, quantum dynamics, or indefinite causal order.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_manifest(output_dir: Path) -> None:
    files = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "MANIFEST.sha256"
    )
    lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    (output_dir / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--base-seed", type=int, default=20510911)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("collective_historical_return_output"),
    )
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
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

    trials, geometry = run_experiment(config, args.workers)
    comparisons = summarize_comparisons(trials, args.base_seed + 909)
    order = summarize_order(trials)
    status = summarize_status(trials)
    geometry_summary = summarize_geometry(geometry)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.output_dir / "historical_return_trials.csv", index=False)
    comparisons.to_csv(
        args.output_dir / "historical_return_comparisons.csv", index=False
    )
    order.to_csv(args.output_dir / "historical_return_order_summary.csv", index=False)
    status.to_csv(
        args.output_dir / "historical_return_status_summary.csv", index=False
    )
    geometry.to_csv(
        args.output_dir / "historical_return_geometry_checks.csv", index=False
    )
    (args.output_dir / "historical_return_report.md").write_text(
        render_report(comparisons, order, geometry_summary), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "material_difference": MATERIAL_DIFFERENCE,
                "geometry_tolerance": GEOMETRY_TOLERANCE,
                "comparisons": comparisons.to_dict(orient="records"),
                "order": order.to_dict(orient="records"),
                "status": status.to_dict(orient="records"),
                "geometry": geometry_summary,
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    _write_manifest(args.output_dir)
    print(comparisons.to_string(index=False))
    print(order.to_string(index=False))
    print(json.dumps(geometry_summary, indent=2))


if __name__ == "__main__":
    main()
