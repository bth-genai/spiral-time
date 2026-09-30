"""O4: own history axis versus foreign history axes at matched angle."""

from __future__ import annotations

import argparse
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
for dependency in (O1, O2, O3):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from experiment import BODY_NAMES, EDGE_NAMES, Config, World, circular_rms  # noqa: E402
from geometry_experiment import (  # noqa: E402
    CHECKPOINTS,
    HISTORIES,
    assay,
    parallel_states,
    rescale,
    state_error,
    state_signature,
)
from direction_experiment import normalized, optional_cosine  # noqa: E402

O4_BASE_SEED = 602609150
SURROGATES = 16


def o4_config(worlds: int = 24, dt: float = 0.02) -> Config:
    return Config(worlds=worlds, dt=dt, base_seed=O4_BASE_SEED)


def history_tangent(native: np.ndarray, other: np.ndarray) -> np.ndarray:
    native_unit = normalized(native)
    other_unit = normalized(other)
    tangent = other_unit - float(np.sum(native_unit * other_unit)) * native_unit
    norm = float(np.linalg.norm(tangent, ord="fro"))
    if norm <= 1.0e-12:
        raise ValueError("history axis is degenerate")
    return tangent / norm


def foreign_axis_surrogate(
    recipient_native: np.ndarray,
    recipient_cross: np.ndarray,
    donor_native: np.ndarray,
    donor_cross: np.ndarray,
    fallback_seed: int,
) -> tuple[np.ndarray, bool, float]:
    recipient_norm = float(np.linalg.norm(recipient_native, ord="fro"))
    recipient_unit = normalized(recipient_native)
    cross_unit = normalized(recipient_cross)
    target_cosine = float(np.clip(
        np.sum(recipient_unit * cross_unit), -1.0, 1.0
    ))
    own_tangent = history_tangent(recipient_native, recipient_cross)
    donor_tangent = history_tangent(donor_native, donor_cross)
    transported = donor_tangent - float(
        np.sum(recipient_unit * donor_tangent)
    ) * recipient_unit
    transported_norm = float(np.linalg.norm(transported, ord="fro"))
    used_fallback = transported_norm <= 1.0e-12
    if used_fallback:
        rng = np.random.default_rng(fallback_seed)
        transported = rng.normal(size=recipient_native.shape)
        transported -= float(
            np.sum(recipient_unit * transported)
        ) * recipient_unit
        transported_norm = float(np.linalg.norm(transported, ord="fro"))
    transported /= transported_norm
    tangent_orientation = float(np.sum(own_tangent * transported))
    sine = math.sqrt(max(0.0, 1.0 - target_cosine * target_cosine))
    surrogate = recipient_norm * (
        target_cosine * recipient_unit + sine * transported
    )
    return surrogate, used_fallback, tangent_orientation


def trajectory_profile(output: dict[str, Any], off: dict[str, Any]) -> np.ndarray:
    body = [
        circular_rms(output["phase"][name], off["phase"][name])
        for name in BODY_NAMES
    ]
    alignment = [
        float(np.sqrt(np.mean((
            output["alignment"][edge] - off["alignment"][edge]
        ) ** 2)))
        for edge in EDGE_NAMES
    ]
    return np.asarray(body + alignment, dtype=float)


def profile_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def cell_summary(
    recipient: World,
    recipient_other: World,
    donor_pairs: list[tuple[World, World]],
    config: Config,
    world_index: int,
    history_index: int,
    checkpoint_index: int,
    edge_index: int,
) -> dict[str, Any]:
    edge = EDGE_NAMES[edge_index]
    native = recipient.edges[edge].tissue.copy()
    native_norm = float(np.linalg.norm(native, ord="fro"))
    cross = rescale(recipient_other.edges[edge].tissue, native_norm)
    target_orientation = optional_cosine(native, cross)
    if target_orientation is None:
        raise ValueError("recipient history axis is undefined")

    surrogates = []
    fallback_count = 0
    tangent_orientations = []
    for donor_index, (donor_native, donor_other) in enumerate(donor_pairs):
        surrogate, used_fallback, tangent_orientation = foreign_axis_surrogate(
            native,
            cross,
            donor_native.edges[edge].tissue,
            donor_other.edges[edge].tissue,
            O4_BASE_SEED
            + world_index * 1_000_003
            + history_index * 100_003
            + checkpoint_index * 10_007
            + edge_index * 1009
            + donor_index,
        )
        surrogates.append(surrogate)
        fallback_count += int(used_fallback)
        tangent_orientations.append(tangent_orientation)

    seed_base = (
        O4_BASE_SEED
        + 100_000_000
        + world_index * 1_000_003
        + history_index * 100_003
        + checkpoint_index * 10_007
        + edge_index * 1009
    )
    off_output = assay(recipient, config, edge, native, seed_base, expressed=False)
    native_output = assay(recipient, config, edge, native, seed_base, expressed=True)
    cross_output = assay(recipient, config, edge, cross, seed_base, expressed=True)
    native_profile = trajectory_profile(native_output, off_output)
    cross_effect = profile_distance(
        trajectory_profile(cross_output, off_output), native_profile
    )
    surrogate_effects = []
    for surrogate in surrogates:
        output = assay(recipient, config, edge, surrogate, seed_base, expressed=True)
        surrogate_effects.append(profile_distance(
            trajectory_profile(output, off_output), native_profile
        ))
    effects = np.asarray(surrogate_effects, dtype=float)
    percentile = float(
        (np.count_nonzero(effects < cross_effect)
         + 0.5 * np.count_nonzero(effects == cross_effect))
        / len(effects)
    )
    norm_errors = [
        abs(float(np.linalg.norm(matrix, ord="fro")) - native_norm)
        for matrix in [cross, *surrogates]
    ]
    angle_errors = [
        abs(float(optional_cosine(native, matrix)) - target_orientation)
        for matrix in surrogates
    ]
    return {
        "native_norm": native_norm,
        "target_orientation": target_orientation,
        "target_angle_degrees": math.degrees(math.acos(np.clip(
            target_orientation, -1.0, 1.0
        ))),
        "cross_history_effect": cross_effect,
        "foreign_axis_effects": surrogate_effects,
        "foreign_axis_median": float(np.median(effects)),
        "foreign_axis_q25": float(np.quantile(effects, 0.25)),
        "foreign_axis_q75": float(np.quantile(effects, 0.75)),
        "cross_minus_foreign_median": float(cross_effect - np.median(effects)),
        "cross_empirical_percentile": percentile,
        "foreign_tangent_orientation_to_own": tangent_orientations,
        "median_foreign_tangent_orientation_to_own": float(np.median(
            tangent_orientations
        )),
        "checks": {
            "maximum_norm_error": max(norm_errors),
            "maximum_angle_error": max(angle_errors),
            "fallback_count": fallback_count,
        },
    }


def summarize_world(
    config: Config,
    recipient_bundle: dict[str, Any],
    donor_bundles: list[dict[str, Any]],
) -> dict[str, Any]:
    world_index = recipient_bundle["world"]
    output: dict[str, Any] = {
        "world": world_index,
        "seed": recipient_bundle["seed"],
        "donor_worlds": [bundle["world"] for bundle in donor_bundles],
        "cells": {},
        "checks": {
            "all_donors_differ": all(
                bundle["world"] != world_index for bundle in donor_bundles
            ),
            "no_ac_edge": True,
            "maximum_recipient_state_error": 0.0,
            "maximum_norm_error": 0.0,
            "maximum_angle_error": 0.0,
            "fallback_count": 0,
        },
    }
    for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
        output["cells"][checkpoint] = {}
        for history_index, history in enumerate(HISTORIES):
            other_history = "blocked" if history == "alternating" else "alternating"
            recipient = recipient_bundle["histories"][history][checkpoint]
            recipient_other = recipient_bundle["histories"][other_history][checkpoint]
            donor_pairs = [
                (
                    bundle["histories"][history][checkpoint],
                    bundle["histories"][other_history][checkpoint],
                )
                for bundle in donor_bundles
            ]
            output["checks"]["no_ac_edge"] = output["checks"]["no_ac_edge"] and all(
                "AC" not in state.edges
                for pair in [(recipient, recipient_other), *donor_pairs]
                for state in pair
            )
            before = state_signature(recipient)
            output["cells"][checkpoint][history] = {}
            for edge_index, edge in enumerate(EDGE_NAMES):
                item = cell_summary(
                    recipient,
                    recipient_other,
                    donor_pairs,
                    config,
                    world_index,
                    history_index,
                    checkpoint_index,
                    edge_index,
                )
                output["cells"][checkpoint][history][edge] = item
                output["checks"]["maximum_norm_error"] = max(
                    output["checks"]["maximum_norm_error"],
                    item["checks"]["maximum_norm_error"],
                )
                output["checks"]["maximum_angle_error"] = max(
                    output["checks"]["maximum_angle_error"],
                    item["checks"]["maximum_angle_error"],
                )
                output["checks"]["fallback_count"] += item["checks"]["fallback_count"]
            output["checks"]["maximum_recipient_state_error"] = max(
                output["checks"]["maximum_recipient_state_error"],
                state_error(before, recipient),
            )
    return output


def _summary_job(
    config: Config,
    recipient: dict[str, Any],
    donors: list[dict[str, Any]],
) -> dict[str, Any]:
    return summarize_world(config, recipient, donors)


def parallel_summaries(
    config: Config,
    states: list[dict[str, Any]],
    workers: int,
) -> list[dict[str, Any]]:
    jobs = []
    for recipient_index in range(len(states)):
        donors = [
            states[(recipient_index + offset) % len(states)]
            for offset in range(1, SURROGATES + 1)
        ]
        jobs.append((states[recipient_index], donors))
    if workers <= 1:
        return [_summary_job(config, recipient, donors) for recipient, donors in jobs]
    output = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_summary_job, config, recipient, donors)
            for recipient, donors in jobs
        ]
        for count, future in enumerate(as_completed(futures), start=1):
            output.append(future.result())
            print(f"axis assays {count}/{len(jobs)}", flush=True)
    return sorted(output, key=lambda item: item["world"])


def aggregate_checks(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all_donors_differ": all(world["checks"]["all_donors_differ"] for world in worlds),
        "all_no_ac_edge": all(world["checks"]["no_ac_edge"] for world in worlds),
        "maximum_recipient_state_error": max(
            world["checks"]["maximum_recipient_state_error"] for world in worlds
        ),
        "maximum_norm_error": max(world["checks"]["maximum_norm_error"] for world in worlds),
        "maximum_angle_error": max(world["checks"]["maximum_angle_error"] for world in worlds),
        "fallback_count": sum(world["checks"]["fallback_count"] for world in worlds),
    }


def run(worlds: int, workers: int) -> dict[str, Any]:
    coarse_config = o4_config(worlds, dt=0.02)
    coarse_states = parallel_states(coarse_config, list(range(worlds)), workers)
    coarse = parallel_summaries(coarse_config, coarse_states, workers)

    fine_config = o4_config(worlds, dt=0.01)
    fine_states = parallel_states(fine_config, list(range(worlds)), workers)
    fine = parallel_summaries(fine_config, fine_states, workers)
    return {
        "protocol": asdict(coarse_config),
        "foreign_axes_per_cell": SURROGATES,
        "coarse": coarse,
        "fine": fine,
        "checks": {
            "coarse": aggregate_checks(coarse),
            "fine": aggregate_checks(fine),
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
