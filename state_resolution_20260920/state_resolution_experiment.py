"""State-equated Spiral-Time resolution experiment.

Two different formation histories are generated from exact copies using the
same encounter tokens and token-specific stochastic streams. Immediately after
formation, the conventional current state (body phases, local tissues, fast
traces, slow tissues, and episode index) is made exactly identical between the
branches while their archived encounter observations are preserved. The same
read-only continuation is then applied at increasing historical resolution.

At resolution 0 the archive is hidden (history expression gain = 0), so the
matched branches must be numerically identical. At higher resolutions the
current state remains identical but a growing portion of the ordered archive is
available to the dynamics through the history field.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
O1 = ROOT / "overlapping_relations_20260915"
if str(O1) not in sys.path:
    sys.path.insert(0, str(O1))

from experiment import (  # noqa: E402
    BODY_NAMES,
    EDGE_NAMES,
    Config,
    HistoryRecord,
    World,
    _formation_schedule,
    _history_field,
    _token_seed,
    circular_rms,
    make_world,
    run_episode,
)

BASE_SEED = 912609200
DEFAULT_WORLDS = 100
RESOLUTIONS = (0, 1, 4, -1)  # -1 denotes the complete archived history (12 records per edge here).
CONTINUATION = ("AB", "BC", "AB", "BC")


def base_config() -> Config:
    # Formation is intentionally generated with the conventional current-state
    # propagator. The history archive is recorded but does not feed back until
    # the matched continuation, where resolution is the manipulated variable.
    return Config(
        worlds=DEFAULT_WORLDS,
        phase_noise=0.025,
        history_expression_gain=0.0,
        base_seed=BASE_SEED,
    )


def build_history(config: Config, seed: int, history: str) -> World:
    world = make_world(config, seed)
    for edge, token in _formation_schedule(config, history):
        run_episode(
            world,
            config,
            edge,
            _token_seed(seed, 1, edge, token),
            plastic=True,
            record_history=True,
        )
    return world


def _copy_history(records: list[HistoryRecord]) -> list[HistoryRecord]:
    return [
        HistoryRecord(record.episode_index, record.observation.copy())
        for record in records
    ]


def equated_branch(common_state: World, history_source: World) -> World:
    branch = common_state.clone()
    for edge in EDGE_NAMES:
        branch.edges[edge].history = _copy_history(history_source.edges[edge].history)
    branch.episode_index = common_state.episode_index
    return branch


def conventional_state_error(left: World, right: World) -> float:
    errors: list[float] = [abs(float(left.episode_index - right.episode_index))]
    for body in BODY_NAMES:
        a, b = left.bodies[body], right.bodies[body]
        errors.extend(
            (
                float(np.max(np.abs(a.phases - b.phases))),
                float(np.max(np.abs(a.local_tissue - b.local_tissue))),
            )
        )
    for edge in EDGE_NAMES:
        a, b = left.edges[edge], right.edges[edge]
        errors.extend(
            (
                float(np.max(np.abs(a.trace - b.trace))),
                float(np.max(np.abs(a.tissue - b.tissue))),
            )
        )
    return max(errors)


def archive_distance(left: World, right: World, config: Config) -> dict[str, float]:
    output: dict[str, float] = {}
    for edge in EDGE_NAMES:
        lf = _history_field(left.edges[edge], config, left.episode_index)
        rf = _history_field(right.edges[edge], config, right.episode_index)
        output[edge] = float(np.linalg.norm(lf - rf, ord="fro"))
    return output


def continuation_output(world: World, config: Config, seed_base: int) -> dict[str, Any]:
    phases = {name: [] for name in BODY_NAMES}
    alignments = {name: [] for name in EDGE_NAMES}
    field_norms = {name: [] for name in EDGE_NAMES}
    for index, offered in enumerate(CONTINUATION):
        result = run_episode(
            world,
            config,
            offered,
            seed_base + index * 1009,
            plastic=False,
            record_trajectory=True,
            record_history=False,
        )
        for name in BODY_NAMES:
            phases[name].append(result["trajectory"][name])
        for name in EDGE_NAMES:
            alignments[name].append(float(result["alignment"][name]))
            field_norms[name].append(float(result["history_field_norm"][name]))
    return {
        "phase": {name: np.concatenate(values, axis=0) for name, values in phases.items()},
        "alignment": {name: np.asarray(values, dtype=float) for name, values in alignments.items()},
        "history_field_norm": {
            name: np.asarray(values, dtype=float) for name, values in field_norms.items()
        },
    }


def response_distance(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    phase = {
        body: circular_rms(left["phase"][body], right["phase"][body])
        for body in BODY_NAMES
    }
    alignment = {
        edge: float(np.sqrt(np.mean(
            (left["alignment"][edge] - right["alignment"][edge]) ** 2
        )))
        for edge in EDGE_NAMES
    }
    return {
        "phase_rms": phase,
        "alignment_rmse": alignment,
        "mean_phase_rms": float(np.mean(list(phase.values()))),
        "mean_alignment_rmse": float(np.mean(list(alignment.values()))),
        "final_ab_alignment_difference": float(
            abs(left["alignment"]["AB"][-1] - right["alignment"]["AB"][-1])
        ),
    }


def run_world(world_index: int, history_gain: float) -> dict[str, Any]:
    config = base_config()
    seed = config.base_seed + world_index * 10_000_019
    alternating = build_history(config, seed, "alternating")
    blocked = build_history(config, seed, "blocked")

    # The alternating branch supplies the shared conventional present. Only the
    # archived histories differ after this operation.
    common = alternating.clone()
    alternating_eq = equated_branch(common, alternating)
    blocked_eq = equated_branch(common, blocked)
    state_error = conventional_state_error(alternating_eq, blocked_eq)
    if state_error != 0.0:
        raise AssertionError(f"state equating failed: {state_error}")

    rows = []
    for resolution in RESOLUTIONS:
        if resolution == 0:
            probe_config = replace(
                config,
                history_expression_gain=0.0,
                history_resolution_records=0,
            )
            label = "current-state only"
        else:
            probe_config = replace(
                config,
                history_expression_gain=history_gain,
                history_resolution_records=0 if resolution == -1 else resolution,
            )
            label = "full archive" if resolution == -1 else f"last {resolution}"
        left = equated_branch(common, alternating)
        right = equated_branch(common, blocked)
        fields = archive_distance(left, right, probe_config)
        seed_base = seed + 90_000_000
        left_output = continuation_output(left, probe_config, seed_base)
        right_output = continuation_output(right, probe_config, seed_base)
        distance = response_distance(left_output, right_output)
        rows.append(
            {
                "world": world_index,
                "world_seed": seed,
                "resolution": resolution,
                "resolution_label": label,
                "history_gain": probe_config.history_expression_gain,
                "conventional_state_error": state_error,
                "history_field_distance_AB": fields["AB"],
                "history_field_distance_BC": fields["BC"],
                **distance,
            }
        )
    return {"world": world_index, "rows": rows}


def _run_world_task(task: tuple[int, float]) -> dict[str, Any]:
    return run_world(*task)


def bootstrap(values: np.ndarray, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(5000, len(values)))]
    means = np.mean(draws, axis=1)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "maximum": float(np.max(values)),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"worlds": len({row["world"] for row in rows}), "resolutions": {}}
    for r_index, resolution in enumerate(RESOLUTIONS):
        group = [row for row in rows if row["resolution"] == resolution]
        label = group[0]["resolution_label"]
        summary["resolutions"][str(resolution)] = {
            "label": label,
            "mean_phase_rms": bootstrap(
                np.asarray([row["mean_phase_rms"] for row in group]), 4010 + r_index
            ),
            "mean_alignment_rmse": bootstrap(
                np.asarray([row["mean_alignment_rmse"] for row in group]), 5010 + r_index
            ),
            "final_ab_alignment_difference": bootstrap(
                np.asarray([row["final_ab_alignment_difference"] for row in group]), 6010 + r_index
            ),
            "history_field_distance_AB": bootstrap(
                np.asarray([row["history_field_distance_AB"] for row in group]), 7010 + r_index
            ),
            "history_field_distance_BC": bootstrap(
                np.asarray([row["history_field_distance_BC"] for row in group]), 8010 + r_index
            ),
            "maximum_conventional_state_error": max(
                row["conventional_state_error"] for row in group
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=DEFAULT_WORLDS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--history-gain", type=float, default=0.14)
    parser.add_argument("--output", type=Path, default=Path("article_results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [(index, args.history_gain) for index in range(args.worlds)]
    if args.workers == 1:
        results = [_run_world_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(_run_world_task, tasks))
    rows = [row for result in results for row in result["rows"]]
    summary = summarize(rows)
    payload = {
        "config": asdict(base_config()),
        "history_gain": args.history_gain,
        "resolutions": RESOLUTIONS,
        "continuation": CONTINUATION,
        "rows": rows,
        "summary": summary,
    }
    (args.output / "state_resolution_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
