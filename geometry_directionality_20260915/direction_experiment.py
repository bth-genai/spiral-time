"""O3: compare cross-history geometry with angle-matched surrogate directions."""

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
for dependency in (O1, O2):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from experiment import (  # noqa: E402
    BODY_NAMES,
    EDGE_NAMES,
    Config,
    World,
    circular_rms,
)
from geometry_experiment import (  # noqa: E402
    ASSAY_SEQUENCE,
    CHECKPOINTS,
    HISTORIES,
    assay,
    parallel_states,
    rescale,
    state_error,
    state_signature,
)

O3_BASE_SEED = 502609150
SURROGATES = 16


def o3_config(worlds: int = 24, dt: float = 0.02) -> Config:
    return Config(worlds=worlds, dt=dt, base_seed=O3_BASE_SEED)


def normalized(matrix: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(matrix, ord="fro"))
    if norm <= 1.0e-15:
        raise ValueError("cannot normalize zero tissue")
    return matrix / norm


def angle_matched_surrogate(
    native: np.ndarray,
    cross_history: np.ndarray,
    donor: np.ndarray,
    fallback_seed: int,
) -> np.ndarray:
    native_norm = float(np.linalg.norm(native, ord="fro"))
    native_unit = normalized(native)
    cross_unit = normalized(cross_history)
    cosine = float(np.clip(np.sum(native_unit * cross_unit), -1.0, 1.0))
    donor_unit = normalized(donor)
    tangent = donor_unit - float(np.sum(native_unit * donor_unit)) * native_unit
    tangent_norm = float(np.linalg.norm(tangent, ord="fro"))
    if tangent_norm <= 1.0e-12:
        rng = np.random.default_rng(fallback_seed)
        tangent = rng.normal(size=native.shape)
        tangent -= float(np.sum(native_unit * tangent)) * native_unit
        tangent_norm = float(np.linalg.norm(tangent, ord="fro"))
    tangent /= tangent_norm
    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    return native_norm * (cosine * native_unit + sine * tangent)


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


def optional_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left, ord="fro") * np.linalg.norm(right, ord="fro"))
    if denominator <= 1.0e-15:
        return None
    return float(np.sum(left * right) / denominator)


def cell_summary(
    recipient: World,
    cross_world: World,
    donor_worlds: list[World],
    config: Config,
    world_index: int,
    history_index: int,
    checkpoint_index: int,
    edge_index: int,
) -> dict[str, Any]:
    edge = EDGE_NAMES[edge_index]
    native = recipient.edges[edge].tissue.copy()
    native_norm = float(np.linalg.norm(native, ord="fro"))
    cross = rescale(cross_world.edges[edge].tissue, native_norm)
    target_orientation = optional_cosine(native, cross)
    if target_orientation is None:
        raise ValueError("cross-history orientation is undefined")
    surrogates = [
        angle_matched_surrogate(
            native,
            cross,
            donor.edges[edge].tissue,
            O3_BASE_SEED
            + world_index * 1_000_003
            + history_index * 100_003
            + checkpoint_index * 10_007
            + edge_index * 1009
            + donor_index,
        )
        for donor_index, donor in enumerate(donor_worlds)
    ]
    seed_base = (
        O3_BASE_SEED
        + 100_000_000
        + world_index * 1_000_003
        + history_index * 100_003
        + checkpoint_index * 10_007
        + edge_index * 1009
    )
    off_output = assay(
        recipient, config, edge, native, seed_base, expressed=False
    )
    native_output = assay(
        recipient, config, edge, native, seed_base, expressed=True
    )
    cross_output = assay(
        recipient, config, edge, cross, seed_base, expressed=True
    )
    native_profile = trajectory_profile(native_output, off_output)
    cross_profile = trajectory_profile(cross_output, off_output)
    cross_effect = profile_distance(cross_profile, native_profile)
    surrogate_effects = []
    for surrogate in surrogates:
        output = assay(
            recipient, config, edge, surrogate, seed_base, expressed=True
        )
        surrogate_effects.append(
            profile_distance(trajectory_profile(output, off_output), native_profile)
        )
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
        "target_angle_degrees": math.degrees(math.acos(np.clip(target_orientation, -1.0, 1.0))),
        "cross_history_effect": cross_effect,
        "surrogate_effects": surrogate_effects,
        "surrogate_median": float(np.median(effects)),
        "surrogate_q25": float(np.quantile(effects, 0.25)),
        "surrogate_q75": float(np.quantile(effects, 0.75)),
        "cross_minus_surrogate_median": float(cross_effect - np.median(effects)),
        "cross_empirical_percentile": percentile,
        "checks": {
            "maximum_norm_error": max(norm_errors),
            "maximum_angle_error": max(angle_errors),
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
            "all_donors_differ": all(bundle["world"] != world_index for bundle in donor_bundles),
            "no_ac_edge": True,
            "maximum_recipient_state_error": 0.0,
            "maximum_norm_error": 0.0,
            "maximum_angle_error": 0.0,
        },
    }
    for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
        output["cells"][checkpoint] = {}
        for history_index, history in enumerate(HISTORIES):
            recipient = recipient_bundle["histories"][history][checkpoint]
            cross = recipient_bundle["histories"][
                "blocked" if history == "alternating" else "alternating"
            ][checkpoint]
            donors = [bundle["histories"][history][checkpoint] for bundle in donor_bundles]
            output["checks"]["no_ac_edge"] = output["checks"]["no_ac_edge"] and all(
                "AC" not in state.edges for state in [recipient, cross, *donors]
            )
            before = state_signature(recipient)
            output["cells"][checkpoint][history] = {}
            for edge_index, edge in enumerate(EDGE_NAMES):
                item = cell_summary(
                    recipient,
                    cross,
                    donors,
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
    recipient_indices: list[int],
    workers: int,
) -> list[dict[str, Any]]:
    jobs = []
    for recipient_index in recipient_indices:
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
            print(f"direction assays {count}/{len(jobs)}", flush=True)
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
    }


def run(worlds: int, workers: int) -> dict[str, Any]:
    config = o3_config(worlds)
    states = parallel_states(config, list(range(worlds)), workers)
    summaries = parallel_summaries(config, states, list(range(worlds)), workers)

    fine_config = o3_config(worlds, dt=config.reference_dt)
    fine_states = parallel_states(fine_config, list(range(worlds)), workers)
    fine_summaries = parallel_summaries(
        fine_config, fine_states, list(range(min(4, worlds))), min(workers, 4)
    )
    return {
        "protocol": asdict(config),
        "surrogates_per_cell": SURROGATES,
        "worlds": summaries,
        "checks": aggregate_checks(summaries),
        "refinement": {
            "coarse": summaries[:min(4, worlds)],
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
