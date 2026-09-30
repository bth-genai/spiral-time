"""O2: historical relation geometry under norm-matched replacements."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
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
    World,
    _formation_schedule,
    _token_seed,
    circular_rms,
    cosine_matrix,
    make_world,
    run_episode,
)

CHECKPOINTS = ("formation", "withdrawal", "return_6")
HISTORIES = ("alternating", "blocked")
CONDITIONS = ("native", "shuffled", "cross_history", "transplanted", "off")
ASSAY_SEQUENCE = ("AB", "BC", "AB", "BC")
O2_BASE_SEED = 402609150


def o2_config(worlds: int = 24, dt: float = 0.02) -> Config:
    return Config(worlds=worlds, dt=dt, base_seed=O2_BASE_SEED)


def world_seed(config: Config, world_index: int) -> int:
    return config.base_seed + world_index * 10_000_019


def history_states(config: Config, seed: int, history: str) -> dict[str, World]:
    world = make_world(config, seed)
    states: dict[str, World] = {}
    for edge, token in _formation_schedule(config, history):
        run_episode(world, config, edge, _token_seed(seed, 1, edge, token))
    states["formation"] = world.clone()

    for cycle in range(config.maintenance_cycles):
        for offset, edge in enumerate(EDGE_NAMES):
            run_episode(
                world,
                config,
                edge,
                _token_seed(seed, 2, edge, cycle * 2 + offset),
            )
    for index in range(config.withdrawal_episodes):
        offered = "BC" if index % config.withdrawal_bc_period == 0 else None
        run_episode(world, config, offered, _token_seed(seed, 3, offered, index))
    states["withdrawal"] = world.clone()

    for index in range(1, config.return_offers + 1):
        run_episode(world, config, "AB", _token_seed(seed, 4, "AB", index))
    states["return_6"] = world.clone()
    return states


def build_world_states(config: Config, world_index: int) -> dict[str, Any]:
    seed = world_seed(config, world_index)
    return {
        "world": world_index,
        "seed": seed,
        "histories": {
            history: history_states(config, seed, history) for history in HISTORIES
        },
    }


def parallel_states(config: Config, indices: list[int], workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        return [build_world_states(config, index) for index in indices]
    output = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(build_world_states, config, index): index for index in indices
        }
        for count, future in enumerate(as_completed(futures), start=1):
            output.append(future.result())
            print(f"states {count}/{len(indices)}", flush=True)
    return sorted(output, key=lambda item: item["world"])


def rescale(matrix: np.ndarray, target_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(matrix, ord="fro"))
    if target_norm <= 1.0e-15:
        return np.zeros_like(matrix)
    if norm <= 1.0e-15:
        raise ValueError("cannot norm-match a zero donor to nonzero native tissue")
    return matrix * (target_norm / norm)


def shuffled_geometry(matrix: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = rng.permutation(matrix.shape[0])
    columns = rng.permutation(matrix.shape[1])
    return matrix[np.ix_(rows, columns)]


def cleared_trace_clone(world: World) -> World:
    clone = world.clone()
    for edge in clone.edges.values():
        edge.trace.fill(0.0)
    return clone


def assay(
    recipient: World,
    config: Config,
    target_edge: str,
    tissue: np.ndarray,
    seed_base: int,
    *,
    expressed: bool,
) -> dict[str, Any]:
    branch = cleared_trace_clone(recipient)
    branch.edges[target_edge].tissue = tissue.copy()
    masks = {edge: (True, True) for edge in EDGE_NAMES}
    if not expressed:
        masks[target_edge] = (False, False)
    phase = {body: [] for body in BODY_NAMES}
    alignment = {edge: [] for edge in EDGE_NAMES}
    for index, offered in enumerate(ASSAY_SEQUENCE):
        result = run_episode(
            branch,
            config,
            offered,
            seed_base + index * 1009,
            plastic=False,
            expression=masks,
            record_trajectory=True,
        )
        for body in BODY_NAMES:
            phase[body].append(result["trajectory"][body])
        for edge in EDGE_NAMES:
            alignment[edge].append(result["alignment"][edge])
    return {
        "phase": {body: np.concatenate(values, axis=0) for body, values in phase.items()},
        "alignment": {
            edge: np.asarray(values, dtype=float) for edge, values in alignment.items()
        },
    }


def trajectory_distance(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "body_phase_rms": {
            body: circular_rms(left["phase"][body], right["phase"][body])
            for body in BODY_NAMES
        },
        "alignment_rmse": {
            edge: float(np.sqrt(np.mean(
                (left["alignment"][edge] - right["alignment"][edge]) ** 2
            )))
            for edge in EDGE_NAMES
        },
    }


def causal_profile(output: dict[str, Any], off: dict[str, Any]) -> dict[str, Any]:
    return trajectory_distance(output, off)


def flatten_profile(profile: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [profile["body_phase_rms"][body] for body in BODY_NAMES]
        + [profile["alignment_rmse"][edge] for edge in EDGE_NAMES],
        dtype=float,
    )


def state_signature(world: World) -> list[np.ndarray]:
    arrays = [np.asarray([float(world.episode_index)])]
    for body in BODY_NAMES:
        item = world.bodies[body]
        arrays.extend((item.phases, item.local_tissue))
    for edge in EDGE_NAMES:
        item = world.edges[edge]
        arrays.extend((item.trace, item.tissue))
        arrays.append(np.asarray([float(len(item.history))]))
        for record in item.history:
            arrays.append(np.asarray([float(record.episode_index)]))
            arrays.append(record.observation)
    return [array.copy() for array in arrays]


def state_error(before: list[np.ndarray], world: World) -> float:
    after = state_signature(world)
    return float(max(np.max(np.abs(left - right)) for left, right in zip(before, after)))


def optional_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    value = cosine_matrix(left, right)
    return None if math.isnan(value) else value


def edge_assay(
    recipient: World,
    other_history: World,
    donor: World,
    config: Config,
    world_index: int,
    history_index: int,
    checkpoint_index: int,
    edge_index: int,
) -> dict[str, Any]:
    edge = EDGE_NAMES[edge_index]
    native = recipient.edges[edge].tissue.copy()
    target_norm = float(np.linalg.norm(native, ord="fro"))
    shuffled = rescale(
        shuffled_geometry(
            native,
            O2_BASE_SEED
            + world_index * 1_000_003
            + history_index * 100_003
            + checkpoint_index * 10_007
            + edge_index * 1009,
        ),
        target_norm,
    )
    cross_history = rescale(other_history.edges[edge].tissue, target_norm)
    transplanted = rescale(donor.edges[edge].tissue, target_norm)
    tissues = {
        "native": native,
        "shuffled": shuffled,
        "cross_history": cross_history,
        "transplanted": transplanted,
        "off": native,
    }
    seed_base = (
        O2_BASE_SEED
        + 100_000_000
        + world_index * 1_000_003
        + history_index * 100_003
        + checkpoint_index * 10_007
        + edge_index * 1009
    )
    outputs = {
        condition: assay(
            recipient,
            config,
            edge,
            tissue,
            seed_base,
            expressed=condition != "off",
        )
        for condition, tissue in tissues.items()
    }
    profiles = {
        condition: causal_profile(output, outputs["off"])
        for condition, output in outputs.items()
        if condition != "off"
    }
    native_profile = flatten_profile(profiles["native"])
    comparisons = {}
    for condition in ("shuffled", "cross_history", "transplanted"):
        comparison_profile = flatten_profile(profiles[condition])
        comparisons[condition] = {
            "distance_from_native": trajectory_distance(
                outputs[condition], outputs["native"]
            ),
            "causal_profile": profiles[condition],
            "causal_profile_rmse_from_native": float(np.sqrt(np.mean(
                (comparison_profile - native_profile) ** 2
            ))),
            "geometry_orientation_to_native": optional_cosine(
                tissues[condition], native
            ),
        }

    erased = np.zeros_like(native)
    erased_recipient = recipient.clone()
    erased_recipient.edges[edge].history.clear()
    erased_output = assay(
        erased_recipient, config, edge, erased, seed_base, expressed=True
    )
    off_erasure = trajectory_distance(outputs["off"], erased_output)
    norm_errors = {
        condition: abs(float(np.linalg.norm(tissue, ord="fro")) - target_norm)
        for condition, tissue in tissues.items()
        if condition != "off"
    }
    singular_error = float(np.max(np.abs(
        np.linalg.svd(native, compute_uv=False)
        - np.linalg.svd(shuffled, compute_uv=False)
    )))
    return {
        "native_norm": target_norm,
        "native_causal_profile": profiles["native"],
        "comparisons": comparisons,
        "checks": {
            "maximum_norm_matching_error": max(norm_errors.values()),
            "shuffled_singular_value_error": singular_error,
            "off_erasure_maximum_phase_error": max(
                off_erasure["body_phase_rms"].values()
            ),
            "off_erasure_maximum_alignment_error": max(
                off_erasure["alignment_rmse"].values()
            ),
        },
    }


def summarize_world(
    config: Config,
    recipient_bundle: dict[str, Any],
    donor_bundle: dict[str, Any],
) -> dict[str, Any]:
    world_index = recipient_bundle["world"]
    output: dict[str, Any] = {
        "world": world_index,
        "seed": recipient_bundle["seed"],
        "donor_world": donor_bundle["world"],
        "geometry": {},
        "assays": {},
        "checks": {
            "donor_differs": donor_bundle["world"] != world_index,
            "no_ac_edge": True,
            "maximum_recipient_state_error": 0.0,
            "maximum_norm_matching_error": 0.0,
            "maximum_shuffled_singular_value_error": 0.0,
            "maximum_off_erasure_error": 0.0,
        },
    }
    for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
        output["geometry"][checkpoint] = {}
        alternating = recipient_bundle["histories"]["alternating"][checkpoint]
        blocked = recipient_bundle["histories"]["blocked"][checkpoint]
        output["checks"]["no_ac_edge"] = output["checks"]["no_ac_edge"] and all(
            "AC" not in state.edges for state in (alternating, blocked)
        )
        for edge in EDGE_NAMES:
            output["geometry"][checkpoint][edge] = {
                "alternating_to_blocked_orientation": optional_cosine(
                    alternating.edges[edge].tissue,
                    blocked.edges[edge].tissue,
                )
            }
        output["assays"][checkpoint] = {}
        for history_index, history in enumerate(HISTORIES):
            recipient = recipient_bundle["histories"][history][checkpoint]
            other = recipient_bundle["histories"][
                "blocked" if history == "alternating" else "alternating"
            ][checkpoint]
            donor = donor_bundle["histories"][history][checkpoint]
            before = state_signature(recipient)
            output["assays"][checkpoint][history] = {}
            for edge_index, edge in enumerate(EDGE_NAMES):
                result = edge_assay(
                    recipient,
                    other,
                    donor,
                    config,
                    world_index,
                    history_index,
                    checkpoint_index,
                    edge_index,
                )
                output["assays"][checkpoint][history][edge] = result
                checks = result["checks"]
                output["checks"]["maximum_norm_matching_error"] = max(
                    output["checks"]["maximum_norm_matching_error"],
                    checks["maximum_norm_matching_error"],
                )
                output["checks"]["maximum_shuffled_singular_value_error"] = max(
                    output["checks"]["maximum_shuffled_singular_value_error"],
                    checks["shuffled_singular_value_error"],
                )
                output["checks"]["maximum_off_erasure_error"] = max(
                    output["checks"]["maximum_off_erasure_error"],
                    checks["off_erasure_maximum_phase_error"],
                    checks["off_erasure_maximum_alignment_error"],
                )
            output["checks"]["maximum_recipient_state_error"] = max(
                output["checks"]["maximum_recipient_state_error"],
                state_error(before, recipient),
            )
    return output


def _summary_job(
    config: Config,
    recipient: dict[str, Any],
    donor: dict[str, Any],
) -> dict[str, Any]:
    return summarize_world(config, recipient, donor)


def parallel_summaries(
    config: Config,
    states: list[dict[str, Any]],
    recipient_count: int,
    workers: int,
) -> list[dict[str, Any]]:
    jobs = [
        (states[index], states[(index + 1) % len(states)])
        for index in range(recipient_count)
    ]
    if workers <= 1:
        return [_summary_job(config, recipient, donor) for recipient, donor in jobs]
    output = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_summary_job, config, recipient, donor)
            for recipient, donor in jobs
        ]
        for count, future in enumerate(as_completed(futures), start=1):
            output.append(future.result())
            print(f"assays {count}/{len(jobs)}", flush=True)
    return sorted(output, key=lambda item: item["world"])


def aggregate_checks(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all_donors_differ": all(world["checks"]["donor_differs"] for world in worlds),
        "all_no_ac_edge": all(world["checks"]["no_ac_edge"] for world in worlds),
        "maximum_recipient_state_error": max(
            world["checks"]["maximum_recipient_state_error"] for world in worlds
        ),
        "maximum_norm_matching_error": max(
            world["checks"]["maximum_norm_matching_error"] for world in worlds
        ),
        "maximum_shuffled_singular_value_error": max(
            world["checks"]["maximum_shuffled_singular_value_error"] for world in worlds
        ),
        "maximum_off_erasure_error": max(
            world["checks"]["maximum_off_erasure_error"] for world in worlds
        ),
    }


def run(worlds: int, workers: int) -> dict[str, Any]:
    config = o2_config(worlds=worlds)
    states = parallel_states(config, list(range(worlds)), workers)
    summaries = parallel_summaries(config, states, worlds, workers)

    fine_count = min(4, worlds)
    fine_config = o2_config(worlds=worlds, dt=config.reference_dt)
    fine_indices = list(range(min(worlds, fine_count + 1)))
    fine_states = parallel_states(fine_config, fine_indices, min(workers, 5))
    fine_summaries = parallel_summaries(
        fine_config, fine_states, fine_count, min(workers, 4)
    )
    return {
        "protocol": asdict(config),
        "worlds": summaries,
        "checks": aggregate_checks(summaries),
        "refinement": {
            "coarse": summaries[:fine_count],
            "fine": fine_summaries,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=24)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    result = run(args.worlds, max(1, args.workers))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["checks"], indent=2))


if __name__ == "__main__":
    main()
