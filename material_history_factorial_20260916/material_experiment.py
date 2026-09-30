"""O6: material encounter-history factorial within a shared body."""

from __future__ import annotations

import argparse
import itertools
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
O2 = ROOT / "geometry_specificity_20260915"
O3 = ROOT / "geometry_directionality_20260915"
O4 = ROOT / "history_axis_specificity_20260915"
O5B = ROOT / "noise_spectrum_spread_20260916"
for dependency in (O1, O2, O3, O4, O5B):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from experiment import EDGE_NAMES, Config, World, make_world, run_episode  # noqa: E402
from geometry_experiment import (  # noqa: E402
    assay,
    rescale,
    state_error,
    state_signature,
)
from axis_experiment import profile_distance, trajectory_profile  # noqa: E402
from direction_experiment import normalized, optional_cosine  # noqa: E402
from noise_experiment import colored_samples  # noqa: E402

O6_BASE_SEED = 902609160
BODY_COUNT = 8
REALIZATIONS_PER_BODY = 4
FORMATION_SLOTS = 48
TOTAL_EPISODES = 158
CHECKPOINTS = ("formation", "withdrawal", "return_6")
FACTORS = ("order", "timing", "participation")
PROFILE_COMPONENTS = ("body_A", "body_B", "body_C", "alignment_AB", "alignment_BC")
LEVELS = tuple(itertools.product((0, 1), repeat=3))


def history_key(levels: tuple[int, int, int]) -> str:
    return "".join(str(value) for value in levels)


def o6_config(dt: float = 0.02, body_count: int = BODY_COUNT) -> Config:
    return Config(
        worlds=body_count * REALIZATIONS_PER_BODY,
        dt=dt,
        phase_noise=0.025,
        base_seed=O6_BASE_SEED,
    )


def body_seed(body_index: int) -> int:
    return O6_BASE_SEED + body_index * 10_000_019


def realization_seed(body_index: int, realization_index: int) -> int:
    return (
        O6_BASE_SEED
        + 100_000_000
        + body_index * 10_000_019
        + realization_index * 1_000_003
    )


def encounter_tokens(order: int, participation: int) -> list[str]:
    per_edge = 12 if participation == 0 else 8
    if order == 0:
        return [edge for _ in range(per_edge) for edge in ("AB", "BC")]
    return ["AB"] * per_edge + ["BC"] * per_edge


def formation_schedule(levels: tuple[int, int, int]) -> tuple[str | None, ...]:
    order, timing, participation = levels
    tokens = encounter_tokens(order, participation)
    count = len(tokens)
    if timing == 0:
        positions = np.rint(np.linspace(0, FORMATION_SLOTS - 1, count)).astype(int)
    else:
        first = count // 2
        positions = np.asarray(
            list(range(first))
            + list(range(FORMATION_SLOTS - (count - first), FORMATION_SLOTS)),
            dtype=int,
        )
    if len(set(positions.tolist())) != count:
        raise ValueError("formation positions are not unique")
    schedule: list[str | None] = [None] * FORMATION_SLOTS
    for position, token in zip(positions, tokens):
        schedule[int(position)] = token
    return tuple(schedule)


def schedule_metadata(levels: tuple[int, int, int]) -> dict[str, Any]:
    schedule = formation_schedule(levels)
    return {
        "levels": dict(zip(FACTORS, levels)),
        "slots": len(schedule),
        "AB": schedule.count("AB"),
        "BC": schedule.count("BC"),
        "none": schedule.count(None),
        "schedule": list(schedule),
    }


def make_reference_stream(config: Config, seed: int) -> dict[str, np.ndarray]:
    fine_steps = round(config.episode_duration / config.reference_dt)
    length = TOTAL_EPISODES * fine_steps
    return {
        name: colored_samples(
            "white",
            seed + index * 10_007,
            length,
            config.modes,
            config.reference_dt,
        )
        for index, name in enumerate(("A", "B", "C"))
    }


def episode_increments(
    samples: dict[str, np.ndarray],
    config: Config,
    episode_index: int,
) -> dict[str, np.ndarray]:
    fine_steps = round(config.episode_duration / config.reference_dt)
    ratio = round(config.dt / config.reference_dt)
    start = episode_index * fine_steps
    stop = start + fine_steps
    output = {}
    for name in ("A", "B", "C"):
        fine = math.sqrt(config.reference_dt) * samples[name][start:stop]
        output[name] = fine.reshape(
            fine_steps // ratio, ratio, config.modes
        ).sum(axis=1)
    return output


def history_states(
    config: Config,
    initial: World,
    samples: dict[str, np.ndarray],
    levels: tuple[int, int, int],
) -> dict[str, World]:
    world = initial.clone()
    episode_index = 0
    for offered in formation_schedule(levels):
        run_episode(
            world,
            config,
            offered,
            0,
            increments_override=episode_increments(samples, config, episode_index),
        )
        episode_index += 1
    states = {"formation": world.clone()}

    for _cycle in range(config.maintenance_cycles):
        for offered in EDGE_NAMES:
            run_episode(
                world,
                config,
                offered,
                0,
                increments_override=episode_increments(samples, config, episode_index),
            )
            episode_index += 1
    for index in range(config.withdrawal_episodes):
        offered = "BC" if index % config.withdrawal_bc_period == 0 else None
        run_episode(
            world,
            config,
            offered,
            0,
            increments_override=episode_increments(samples, config, episode_index),
        )
        episode_index += 1
    states["withdrawal"] = world.clone()

    for _index in range(config.return_offers):
        run_episode(
            world,
            config,
            "AB",
            0,
            increments_override=episode_increments(samples, config, episode_index),
        )
        episode_index += 1
    states["return_6"] = world.clone()
    if episode_index != TOTAL_EPISODES:
        raise AssertionError(f"consumed {episode_index}, expected {TOTAL_EPISODES}")
    return states


def maximum_signature_error(left: World, right: World) -> float:
    return state_error(state_signature(left), right)


def build_body_bundle(config: Config, body_index: int) -> dict[str, Any]:
    initial = make_world(config, body_seed(body_index))
    realizations = []
    for realization_index in range(REALIZATIONS_PER_BODY):
        seed = realization_seed(body_index, realization_index)
        samples = make_reference_stream(config, seed)
        starts = {history_key(levels): initial.clone() for levels in LEVELS}
        reference_start = starts[history_key(LEVELS[0])]
        start_error = max(
            maximum_signature_error(reference_start, value)
            for value in starts.values()
        )
        histories = {
            history_key(levels): {
                "levels": levels,
                "states": history_states(config, starts[history_key(levels)], samples, levels),
            }
            for levels in LEVELS
        }
        realizations.append({
            "body": body_index,
            "realization": realization_index,
            "noise_seed": seed,
            "histories": histories,
            "checks": {"maximum_initial_state_error": start_error},
        })
    return {
        "body": body_index,
        "body_seed": body_seed(body_index),
        "realizations": realizations,
    }


def parallel_bodies(config: Config, body_count: int, workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        return [build_body_bundle(config, index) for index in range(body_count)]
    output = []
    with ProcessPoolExecutor(max_workers=min(workers, body_count)) as executor:
        futures = [
            executor.submit(build_body_bundle, config, index)
            for index in range(body_count)
        ]
        for count, future in enumerate(as_completed(futures), start=1):
            output.append(future.result())
            print(f"material histories {count}/{body_count}", flush=True)
    return sorted(output, key=lambda value: value["body"])


def pairwise_orientations(vectors: list[np.ndarray | None]) -> tuple[list[float], int]:
    values = []
    undefined = 0
    for left_index in range(len(vectors)):
        for right_index in range(left_index + 1, len(vectors)):
            left, right = vectors[left_index], vectors[right_index]
            if left is None or right is None:
                undefined += 1
            else:
                values.append(float(np.sum(left * right)))
    return values, undefined


def unit_direction(values: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(values))
    return None if norm <= 1.0e-12 else values / norm


def normalized_tissue(world: World, edge: str) -> np.ndarray:
    return normalized(world.edges[edge].tissue)


def factor_contrasts(
    states: dict[str, World],
    config: Config,
    checkpoint_index: int,
    edge_index: int,
    seed_base: int,
) -> dict[str, Any]:
    edge = EDGE_NAMES[edge_index]
    cached: dict[str, dict[str, Any]] = {}

    def native_assay(levels: tuple[int, int, int]) -> dict[str, Any]:
        key = history_key(levels)
        if key not in cached:
            recipient = states[key]
            tissue = recipient.edges[edge].tissue.copy()
            off = assay(recipient, config, edge, tissue, seed_base, expressed=False)
            native = assay(recipient, config, edge, tissue, seed_base, expressed=True)
            cached[key] = {
                "off": off,
                "profile": trajectory_profile(native, off),
            }
        return cached[key]

    output: dict[str, Any] = {}
    for factor_index, factor in enumerate(FACTORS):
        contrasts = []
        tissue_axes: list[np.ndarray | None] = []
        profile_axes: list[np.ndarray | None] = []
        other_indices = [index for index in range(3) if index != factor_index]
        for context in itertools.product((0, 1), repeat=2):
            base = [0, 0, 0]
            for index, value in zip(other_indices, context):
                base[index] = value
            target = base.copy()
            target[factor_index] = 1
            base_levels = tuple(base)
            target_levels = tuple(target)
            recipient = states[history_key(base_levels)]
            target_state = states[history_key(target_levels)]
            native_tissue = recipient.edges[edge].tissue.copy()
            target_raw = target_state.edges[edge].tissue.copy()
            target_tissue = rescale(
                target_raw, float(np.linalg.norm(native_tissue, ord="fro"))
            )
            base_assay = native_assay(base_levels)
            cross_output = assay(
                recipient,
                config,
                edge,
                target_tissue,
                seed_base,
                expressed=True,
            )
            cross_profile = trajectory_profile(cross_output, base_assay["off"])
            profile_delta = cross_profile - base_assay["profile"]
            profile_axis = unit_direction(profile_delta)
            tissue_delta = normalized(target_raw) - normalized(native_tissue)
            tissue_axis = unit_direction(tissue_delta)
            orientation = optional_cosine(native_tissue, target_raw)
            if orientation is None:
                raise ValueError("factor contrast has undefined tissue orientation")
            tissue_axes.append(tissue_axis)
            profile_axes.append(profile_axis)
            contrasts.append({
                "base": history_key(base_levels),
                "target": history_key(target_levels),
                "context": {
                    FACTORS[index]: value for index, value in zip(other_indices, context)
                },
                "native_norm": float(np.linalg.norm(native_tissue, ord="fro")),
                "target_raw_norm": float(np.linalg.norm(target_raw, ord="fro")),
                "target_to_native_norm_ratio": float(
                    np.linalg.norm(target_raw, ord="fro")
                    / np.linalg.norm(native_tissue, ord="fro")
                ),
                "norm_match_error": abs(
                    float(np.linalg.norm(target_tissue, ord="fro"))
                    - float(np.linalg.norm(native_tissue, ord="fro"))
                ),
                "tissue_orientation": orientation,
                "tissue_angle_degrees": math.degrees(math.acos(np.clip(
                    orientation, -1.0, 1.0
                ))),
                "expression_effect": profile_distance(
                    cross_profile, base_assay["profile"]
                ),
                "expression_profile_delta": profile_delta.tolist(),
                "expression_profile_components": dict(zip(
                    PROFILE_COMPONENTS, profile_delta.tolist()
                )),
            })
        tissue_pairwise, tissue_undefined = pairwise_orientations(tissue_axes)
        profile_pairwise, profile_undefined = pairwise_orientations(profile_axes)
        effects = np.asarray([value["expression_effect"] for value in contrasts])
        output[factor] = {
            "contrasts": contrasts,
            "tissue_axis_pairwise_orientations": tissue_pairwise,
            "median_tissue_axis_orientation": (
                float(np.median(tissue_pairwise)) if tissue_pairwise else None
            ),
            "undefined_tissue_axis_pairs": tissue_undefined,
            "profile_axis_pairwise_orientations": profile_pairwise,
            "median_profile_axis_orientation": (
                float(np.median(profile_pairwise)) if profile_pairwise else None
            ),
            "undefined_profile_axis_pairs": profile_undefined,
            "median_expression_effect": float(np.median(effects)),
            "expression_effect_q25": float(np.quantile(effects, 0.25)),
            "expression_effect_q75": float(np.quantile(effects, 0.75)),
        }
    return output


def interaction_diagnostics(states: dict[str, World], edge: str) -> dict[str, Any]:
    output = {}
    for first, second in itertools.combinations(range(3), 2):
        remaining = next(index for index in range(3) if index not in (first, second))
        pair_name = f"{FACTORS[first]}_x_{FACTORS[second]}"
        values = []
        for remaining_level in (0, 1):
            matrices = {}
            for first_level, second_level in itertools.product((0, 1), repeat=2):
                levels = [0, 0, 0]
                levels[first] = first_level
                levels[second] = second_level
                levels[remaining] = remaining_level
                matrices[(first_level, second_level)] = normalized_tissue(
                    states[history_key(tuple(levels))], edge
                )
            residual = (
                matrices[(1, 1)] - matrices[(1, 0)]
                - matrices[(0, 1)] + matrices[(0, 0)]
            )
            adjacent = [
                matrices[(1, 0)] - matrices[(0, 0)],
                matrices[(1, 1)] - matrices[(0, 1)],
                matrices[(0, 1)] - matrices[(0, 0)],
                matrices[(1, 1)] - matrices[(1, 0)],
            ]
            scale = float(np.mean([np.linalg.norm(value) for value in adjacent]))
            values.append({
                "fixed_factor": FACTORS[remaining],
                "fixed_level": remaining_level,
                "residual_norm": float(np.linalg.norm(residual)),
                "relative_residual": (
                    float(np.linalg.norm(residual) / scale)
                    if scale > 1.0e-12 else None
                ),
            })
        output[pair_name] = values
    return output


def summarize_realization(config: Config, realization: dict[str, Any]) -> dict[str, Any]:
    body_index = realization["body"]
    realization_index = realization["realization"]
    output: dict[str, Any] = {
        "body": body_index,
        "realization": realization_index,
        "noise_seed": realization["noise_seed"],
        "cells": {},
        "checks": {
            "maximum_initial_state_error": realization["checks"]["maximum_initial_state_error"],
            "no_ac_edge": True,
            "maximum_recipient_state_error": 0.0,
            "maximum_norm_match_error": 0.0,
            "undefined_tissue_axis_pairs": 0,
            "undefined_profile_axis_pairs": 0,
        },
    }
    for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
        states = {
            key: value["states"][checkpoint]
            for key, value in realization["histories"].items()
        }
        output["checks"]["no_ac_edge"] = output["checks"]["no_ac_edge"] and all(
            "AC" not in state.edges for state in states.values()
        )
        signatures = {key: state_signature(state) for key, state in states.items()}
        output["cells"][checkpoint] = {}
        for edge_index, edge in enumerate(EDGE_NAMES):
            seed_base = (
                O6_BASE_SEED
                + 300_000_000
                + body_index * 10_000_019
                + realization_index * 1_000_003
                + checkpoint_index * 10_007
                + edge_index * 1009
            )
            factors = factor_contrasts(
                states, config, checkpoint_index, edge_index, seed_base
            )
            output["cells"][checkpoint][edge] = {
                "factors": factors,
                "interactions": interaction_diagnostics(states, edge),
            }
            output["checks"]["undefined_tissue_axis_pairs"] += sum(
                value["undefined_tissue_axis_pairs"] for value in factors.values()
            )
            output["checks"]["undefined_profile_axis_pairs"] += sum(
                value["undefined_profile_axis_pairs"] for value in factors.values()
            )
            output["checks"]["maximum_norm_match_error"] = max(
                output["checks"]["maximum_norm_match_error"],
                max(
                    contrast["norm_match_error"]
                    for factor in factors.values()
                    for contrast in factor["contrasts"]
                ),
            )
        for key, state in states.items():
            output["checks"]["maximum_recipient_state_error"] = max(
                output["checks"]["maximum_recipient_state_error"],
                state_error(signatures[key], state),
            )
    return output


def parallel_summaries(
    config: Config,
    bodies: list[dict[str, Any]],
    workers: int,
) -> list[dict[str, Any]]:
    jobs = [
        realization
        for body in bodies
        for realization in body["realizations"]
    ]
    if workers <= 1:
        return [summarize_realization(config, job) for job in jobs]
    output = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(summarize_realization, config, job) for job in jobs
        ]
        for count, future in enumerate(as_completed(futures), start=1):
            output.append(future.result())
            print(f"material assays {count}/{len(jobs)}", flush=True)
    return sorted(output, key=lambda value: (value["body"], value["realization"]))


def aggregate_checks(values: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all_initial_states_exact": all(
            value["checks"]["maximum_initial_state_error"] == 0.0 for value in values
        ),
        "all_no_ac_edge": all(value["checks"]["no_ac_edge"] for value in values),
        "maximum_recipient_state_error": max(
            value["checks"]["maximum_recipient_state_error"] for value in values
        ),
        "maximum_norm_match_error": max(
            value["checks"]["maximum_norm_match_error"] for value in values
        ),
        "undefined_tissue_axis_pairs": sum(
            value["checks"]["undefined_tissue_axis_pairs"] for value in values
        ),
        "undefined_profile_axis_pairs": sum(
            value["checks"]["undefined_profile_axis_pairs"] for value in values
        ),
    }


def run(body_count: int, workers: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "body_count": body_count,
        "realizations_per_body": REALIZATIONS_PER_BODY,
        "schedules": {
            history_key(levels): schedule_metadata(levels) for levels in LEVELS
        },
        "coarse": [],
        "fine": [],
        "checks": {},
    }
    for resolution, dt in (("coarse", 0.02), ("fine", 0.01)):
        config = o6_config(dt, body_count)
        bodies = parallel_bodies(config, body_count, workers)
        summaries = parallel_summaries(config, bodies, workers)
        result[resolution] = summaries
        result["checks"][resolution] = aggregate_checks(summaries)
        if resolution == "coarse":
            result["protocol"] = asdict(config)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bodies", type=int, default=BODY_COUNT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    result = run(args.bodies, max(1, args.workers))
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "results.json"
    destination.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(destination)
    print(json.dumps(result["checks"], indent=2))


if __name__ == "__main__":
    main()
