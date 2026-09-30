"""O5b: noise amplitude, temporal spectrum, and sedimented spread."""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
O1 = ROOT / "overlapping_relations_20260915"
O2 = ROOT / "geometry_specificity_20260915"
O3 = ROOT / "geometry_directionality_20260915"
O4 = ROOT / "history_axis_specificity_20260915"
O5 = ROOT / "history_body_factorial_20260915"
for dependency in (O1, O2, O3, O4, O5):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from experiment import (  # noqa: E402
    BODY_NAMES,
    EDGE_NAMES,
    Config,
    World,
    _formation_schedule,
    make_world,
    run_episode,
)
from geometry_experiment import (  # noqa: E402
    CHECKPOINTS,
    HISTORIES,
    assay,
    rescale,
    state_error,
    state_signature,
)
from axis_experiment import (  # noqa: E402
    foreign_axis_surrogate,
    history_tangent,
    profile_distance,
    trajectory_profile,
)
from direction_experiment import optional_cosine  # noqa: E402

O5B_BASE_SEED = 802609160
DEFAULT_BODIES = 8
REALIZATIONS_PER_BODY = 4
TOTAL_EPISODES = 134
BASELINE_ASSAY_NOISE = 0.025

CONDITIONS = {
    "white_0000": {"color": "white", "amplitude": 0.0, "normalization": "step_rms"},
    "white_0125": {"color": "white", "amplitude": 0.0125, "normalization": "step_rms"},
    "white_0250": {"color": "white", "amplitude": 0.025, "normalization": "step_rms"},
    "white_0500": {"color": "white", "amplitude": 0.05, "normalization": "step_rms"},
    "white_1000": {"color": "white", "amplitude": 0.10, "normalization": "step_rms"},
    "blue_0250": {"color": "blue", "amplitude": 0.025, "normalization": "step_rms"},
    "pink_0250": {"color": "pink", "amplitude": 0.025, "normalization": "step_rms"},
    "brown_0250": {"color": "brown", "amplitude": 0.025, "normalization": "step_rms"},
    "ou_0250": {"color": "ou", "amplitude": 0.025, "normalization": "step_rms"},
    "blue_displacement_0250": {"color": "blue", "amplitude": 0.025, "normalization": "episode_displacement"},
    "pink_displacement_0250": {"color": "pink", "amplitude": 0.025, "normalization": "episode_displacement"},
    "brown_displacement_0250": {"color": "brown", "amplitude": 0.025, "normalization": "episode_displacement"},
    "ou_displacement_0250": {"color": "ou", "amplitude": 0.025, "normalization": "episode_displacement"},
}


def o5b_config(body_count: int, amplitude: float, dt: float) -> Config:
    return Config(
        worlds=body_count * REALIZATIONS_PER_BODY,
        dt=dt,
        phase_noise=amplitude,
        base_seed=O5B_BASE_SEED,
    )


def body_seed(body_index: int) -> int:
    return O5B_BASE_SEED + body_index * 10_000_019


def realization_seed(body_index: int, realization_index: int) -> int:
    return (
        O5B_BASE_SEED
        + 100_000_000
        + body_index * 10_000_019
        + realization_index * 1_000_003
    )


def _standardize(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, keepdims=True)
    if np.any(scale <= 1.0e-12):
        raise ValueError("noise stream has a degenerate mode")
    return centered / scale


def colored_samples(
    color: str,
    seed: int,
    length: int,
    modes: int,
    reference_dt: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.normal(size=(length, modes))
    if color == "white":
        values = white
    elif color == "blue":
        previous = rng.normal(size=(1, modes))
        values = np.diff(np.vstack((previous, white)), axis=0)
    elif color == "pink":
        spectrum = np.fft.rfft(white, axis=0)
        frequencies = np.fft.rfftfreq(length, d=reference_dt)
        shaping = np.zeros_like(frequencies)
        shaping[1:] = 1.0 / np.sqrt(frequencies[1:])
        values = np.fft.irfft(spectrum * shaping[:, None], n=length, axis=0)
    elif color == "brown":
        values = np.cumsum(white, axis=0)
    elif color == "ou":
        correlation_time = 1.0
        rho = math.exp(-reference_dt / correlation_time)
        innovation = math.sqrt(1.0 - rho * rho)
        values = np.empty_like(white)
        values[0] = white[0]
        for index in range(1, length):
            values[index] = rho * values[index - 1] + innovation * white[index]
    else:
        raise ValueError(f"unknown noise color: {color}")
    return _standardize(values)


def effective_rank(matrix: np.ndarray) -> float:
    covariance = np.cov(matrix, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(np.sum(eigenvalues))
    if total <= 1.0e-15:
        return 0.0
    return float(total * total / np.sum(eigenvalues * eigenvalues))


def stream_diagnostics(
    samples: dict[str, np.ndarray],
    fine_steps: int,
    reference_dt: float,
) -> dict[str, float]:
    combined = np.concatenate([samples[name] for name in BODY_NAMES], axis=1)
    lag_values = []
    for column in range(combined.shape[1]):
        lag_values.append(float(np.corrcoef(
            combined[:-1, column], combined[1:, column]
        )[0, 1]))
    episode_displacements = (
        math.sqrt(reference_dt)
        * combined.reshape(TOTAL_EPISODES, fine_steps, -1).sum(axis=1)
    )
    return {
        "fine_step_rms": float(np.sqrt(np.mean(combined * combined))),
        "median_lag1_correlation": float(np.median(lag_values)),
        "spatial_effective_rank": effective_rank(combined),
        "episode_displacement_effective_rank": effective_rank(episode_displacements),
        "episode_displacement_rms": float(np.sqrt(np.mean(
            episode_displacements * episode_displacements
        ))),
    }


def make_reference_stream(
    config: Config,
    color: str,
    seed: int,
    normalization: str = "step_rms",
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    fine_steps = round(config.episode_duration / config.reference_dt)
    length = TOTAL_EPISODES * fine_steps
    samples = {
        name: colored_samples(
            color,
            seed + index * 10_007,
            length,
            config.modes,
            config.reference_dt,
        )
        for index, name in enumerate(BODY_NAMES)
    }
    if normalization == "episode_displacement":
        white_samples = {
            name: colored_samples(
                "white",
                seed + index * 10_007,
                length,
                config.modes,
                config.reference_dt,
            )
            for index, name in enumerate(BODY_NAMES)
        }
        current = stream_diagnostics(samples, fine_steps, config.reference_dt)
        target = stream_diagnostics(white_samples, fine_steps, config.reference_dt)
        scale = (
            target["episode_displacement_rms"]
            / current["episode_displacement_rms"]
        )
        samples = {name: values * scale for name, values in samples.items()}
    elif normalization != "step_rms":
        raise ValueError(f"unknown normalization: {normalization}")
    diagnostics = stream_diagnostics(samples, fine_steps, config.reference_dt)
    diagnostics["applied_fine_increment_rms"] = (
        config.phase_noise
        * math.sqrt(config.reference_dt)
        * diagnostics["fine_step_rms"]
    )
    diagnostics["applied_episode_displacement_rms"] = (
        config.phase_noise * diagnostics["episode_displacement_rms"]
    )
    return samples, diagnostics


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
    for name in BODY_NAMES:
        fine = math.sqrt(config.reference_dt) * samples[name][start:stop]
        output[name] = fine.reshape(
            fine_steps // ratio, ratio, config.modes
        ).sum(axis=1)
    return output


def history_states(
    config: Config,
    initial_world: World,
    samples: dict[str, np.ndarray],
    history: str,
) -> dict[str, World]:
    world = initial_world.clone()
    states: dict[str, World] = {}
    episode_index = 0

    for edge, _token in _formation_schedule(config, history):
        run_episode(
            world,
            config,
            edge,
            0,
            increments_override=episode_increments(samples, config, episode_index),
        )
        episode_index += 1
    states["formation"] = world.clone()

    for _cycle in range(config.maintenance_cycles):
        for edge in EDGE_NAMES:
            run_episode(
                world,
                config,
                edge,
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

    for _index in range(1, config.return_offers + 1):
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
        raise AssertionError(f"consumed {episode_index} episodes, expected {TOTAL_EPISODES}")
    return states


def maximum_signature_error(left: World, right: World) -> float:
    return state_error(state_signature(left), right)


def build_body_bundle(
    config: Config,
    color: str,
    normalization: str,
    body_index: int,
) -> dict[str, Any]:
    initial = make_world(config, body_seed(body_index))
    realizations = []
    for realization_index in range(REALIZATIONS_PER_BODY):
        seed = realization_seed(body_index, realization_index)
        samples, diagnostics = make_reference_stream(
            config, color, seed, normalization
        )
        alternating_start = initial.clone()
        blocked_start = initial.clone()
        realizations.append({
            "body": body_index,
            "realization": realization_index,
            "noise_seed": seed,
            "noise_diagnostics": diagnostics,
            "histories": {
                "alternating": history_states(
                    config, alternating_start, samples, "alternating"
                ),
                "blocked": history_states(
                    config, blocked_start, samples, "blocked"
                ),
            },
            "checks": {
                "paired_initial_error": maximum_signature_error(
                    alternating_start, blocked_start
                ),
                "template_initial_error": maximum_signature_error(
                    initial, alternating_start
                ),
            },
        })
    return {
        "body": body_index,
        "body_seed": body_seed(body_index),
        "realizations": realizations,
    }


def parallel_bodies(
    config: Config,
    color: str,
    normalization: str,
    body_count: int,
    workers: int,
) -> list[dict[str, Any]]:
    if workers <= 1:
        return [
            build_body_bundle(config, color, normalization, index)
            for index in range(body_count)
        ]
    output = []
    with ProcessPoolExecutor(max_workers=min(workers, body_count)) as executor:
        futures = [
            executor.submit(build_body_bundle, config, color, normalization, index)
            for index in range(body_count)
        ]
        for count, future in enumerate(as_completed(futures), start=1):
            output.append(future.result())
            print(f"{color} body histories {count}/{body_count}", flush=True)
    return sorted(output, key=lambda value: value["body"])


def cell_summary(
    recipient: World,
    recipient_other: World,
    donor_pairs: list[tuple[World, World]],
    formation_config: Config,
    assay_config: Config,
    body_index: int,
    realization_index: int,
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

    seed_base = (
        O5B_BASE_SEED
        + 300_000_000
        + body_index * 10_000_019
        + realization_index * 1_000_003
        + history_index * 100_003
        + checkpoint_index * 10_007
        + edge_index * 1009
    )
    surrogates = []
    tangent_orientations = []
    native_orientations = []
    fallback_count = 0
    for donor_index, (donor_native_state, donor_other_state) in enumerate(donor_pairs):
        donor_native = donor_native_state.edges[edge].tissue
        donor_cross = donor_other_state.edges[edge].tissue
        surrogate, fallback, tangent_orientation = foreign_axis_surrogate(
            native,
            cross,
            donor_native,
            donor_cross,
            seed_base + donor_index,
        )
        surrogates.append(surrogate)
        tangent_orientations.append(tangent_orientation)
        native_orientation = optional_cosine(native, donor_native)
        native_orientations.append(
            float(native_orientation) if native_orientation is not None else 0.0
        )
        fallback_count += int(fallback)

    off_output = assay(recipient, assay_config, edge, native, seed_base, expressed=False)
    native_output = assay(recipient, assay_config, edge, native, seed_base, expressed=True)
    own_output = assay(recipient, assay_config, edge, cross, seed_base, expressed=True)
    native_profile = trajectory_profile(native_output, off_output)
    own_effect = profile_distance(
        trajectory_profile(own_output, off_output), native_profile
    )
    donor_effects = []
    for surrogate in surrogates:
        output = assay(
            recipient, assay_config, edge, surrogate, seed_base, expressed=True
        )
        donor_effects.append(profile_distance(
            trajectory_profile(output, off_output), native_profile
        ))
    donor_median = float(np.median(donor_effects))
    donor_array = np.asarray(donor_effects)
    percentile = float(
        (np.count_nonzero(donor_array < own_effect)
         + 0.5 * np.count_nonzero(donor_array == own_effect))
        / len(donor_array)
    )
    matrices = [cross, *surrogates]
    norm_errors = [
        abs(float(np.linalg.norm(matrix, ord="fro")) - native_norm)
        for matrix in matrices
    ]
    angle_errors = [
        abs(float(optional_cosine(native, matrix)) - target_orientation)
        for matrix in surrogates
    ]
    return {
        "native_norm": native_norm,
        "target_orientation": target_orientation,
        "own_effect": own_effect,
        "within_body_effects": donor_effects,
        "within_body_median": donor_median,
        "own_minus_within_median": own_effect - donor_median,
        "own_percentile_within": percentile,
        "within_tangent_orientations": tangent_orientations,
        "median_within_tangent_orientation": float(np.median(tangent_orientations)),
        "within_native_tissue_orientations": native_orientations,
        "median_within_native_tissue_orientation": float(np.median(native_orientations)),
        "history_axis_tangent_norm": float(np.linalg.norm(
            history_tangent(native, cross), ord="fro"
        )),
        "checks": {
            "maximum_norm_error": max(norm_errors),
            "maximum_angle_error": max(angle_errors),
            "fallback_count": fallback_count,
            "formation_noise_amplitude": formation_config.phase_noise,
            "assay_noise_amplitude": assay_config.phase_noise,
        },
    }


def summarize_recipient(
    formation_config: Config,
    recipient: dict[str, Any],
    donors: list[dict[str, Any]],
) -> dict[str, Any]:
    assay_config = replace(formation_config, phase_noise=BASELINE_ASSAY_NOISE)
    body_index = recipient["body"]
    realization_index = recipient["realization"]
    output: dict[str, Any] = {
        "body": body_index,
        "realization": realization_index,
        "noise_seed": recipient["noise_seed"],
        "noise_diagnostics": recipient["noise_diagnostics"],
        "donors": [[item["body"], item["realization"]] for item in donors],
        "cells": {},
        "checks": {
            "paired_initial_error": recipient["checks"]["paired_initial_error"],
            "template_initial_error": recipient["checks"]["template_initial_error"],
            "donor_identity_valid": all(
                item["body"] == body_index
                and item["realization"] != realization_index
                for item in donors
            ),
            "no_ac_edge": True,
            "maximum_recipient_state_error": 0.0,
            "maximum_norm_error": 0.0,
            "maximum_angle_error": 0.0,
            "fallback_count": 0,
            "assay_noise_fixed": True,
        },
    }
    for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
        output["cells"][checkpoint] = {}
        for history_index, history in enumerate(HISTORIES):
            other_history = "blocked" if history == "alternating" else "alternating"
            recipient_state = recipient["histories"][history][checkpoint]
            recipient_other = recipient["histories"][other_history][checkpoint]
            donor_pairs = [
                (
                    donor["histories"][history][checkpoint],
                    donor["histories"][other_history][checkpoint],
                )
                for donor in donors
            ]
            output["checks"]["no_ac_edge"] = output["checks"]["no_ac_edge"] and all(
                "AC" not in state.edges
                for pair in [(recipient_state, recipient_other), *donor_pairs]
                for state in pair
            )
            before = state_signature(recipient_state)
            output["cells"][checkpoint][history] = {}
            for edge_index, edge in enumerate(EDGE_NAMES):
                result = cell_summary(
                    recipient_state,
                    recipient_other,
                    donor_pairs,
                    formation_config,
                    assay_config,
                    body_index,
                    realization_index,
                    history_index,
                    checkpoint_index,
                    edge_index,
                )
                output["cells"][checkpoint][history][edge] = result
                output["checks"]["maximum_norm_error"] = max(
                    output["checks"]["maximum_norm_error"],
                    result["checks"]["maximum_norm_error"],
                )
                output["checks"]["maximum_angle_error"] = max(
                    output["checks"]["maximum_angle_error"],
                    result["checks"]["maximum_angle_error"],
                )
                output["checks"]["fallback_count"] += result["checks"]["fallback_count"]
                output["checks"]["assay_noise_fixed"] = (
                    output["checks"]["assay_noise_fixed"]
                    and result["checks"]["assay_noise_amplitude"]
                    == BASELINE_ASSAY_NOISE
                )
            output["checks"]["maximum_recipient_state_error"] = max(
                output["checks"]["maximum_recipient_state_error"],
                state_error(before, recipient_state),
            )
    return output


def parallel_summaries(
    config: Config,
    bodies: list[dict[str, Any]],
    workers: int,
) -> list[dict[str, Any]]:
    jobs = []
    for body in bodies:
        for recipient in body["realizations"]:
            donors = [
                item for item in body["realizations"]
                if item["realization"] != recipient["realization"]
            ]
            jobs.append((recipient, donors))
    if workers <= 1:
        return [summarize_recipient(config, *job) for job in jobs]
    output = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(summarize_recipient, config, *job) for job in jobs
        ]
        for count, future in enumerate(as_completed(futures), start=1):
            output.append(future.result())
            print(f"within-body assays {count}/{len(jobs)}", flush=True)
    return sorted(output, key=lambda value: (value["body"], value["realization"]))


def aggregate_checks(recipients: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all_paired_initial_states_exact": all(
            value["checks"]["paired_initial_error"] == 0.0 for value in recipients
        ),
        "all_realizations_share_body_template": all(
            value["checks"]["template_initial_error"] == 0.0 for value in recipients
        ),
        "all_donor_identities_valid": all(
            value["checks"]["donor_identity_valid"] for value in recipients
        ),
        "all_assay_noise_fixed": all(
            value["checks"]["assay_noise_fixed"] for value in recipients
        ),
        "all_no_ac_edge": all(value["checks"]["no_ac_edge"] for value in recipients),
        "maximum_recipient_state_error": max(
            value["checks"]["maximum_recipient_state_error"] for value in recipients
        ),
        "maximum_norm_error": max(
            value["checks"]["maximum_norm_error"] for value in recipients
        ),
        "maximum_angle_error": max(
            value["checks"]["maximum_angle_error"] for value in recipients
        ),
        "fallback_count": sum(value["checks"]["fallback_count"] for value in recipients),
    }


def run_condition(
    condition_name: str,
    body_count: int,
    workers: int,
) -> dict[str, Any]:
    condition = CONDITIONS[condition_name]
    result: dict[str, Any] = {
        "condition": condition_name,
        "color": condition["color"],
        "amplitude": condition["amplitude"],
        "normalization": condition["normalization"],
        "body_count": body_count,
        "realizations_per_body": REALIZATIONS_PER_BODY,
        "coarse": [],
        "fine": [],
        "checks": {},
    }
    for resolution, dt in (("coarse", 0.02), ("fine", 0.01)):
        config = o5b_config(body_count, condition["amplitude"], dt)
        bodies = parallel_bodies(
            config,
            condition["color"],
            condition["normalization"],
            body_count,
            workers,
        )
        recipients = parallel_summaries(config, bodies, workers)
        result[resolution] = recipients
        result["checks"][resolution] = aggregate_checks(recipients)
        if resolution == "coarse":
            result["protocol"] = asdict(config)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=tuple(CONDITIONS))
    parser.add_argument("--bodies", type=int, default=DEFAULT_BODIES)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    result = run_condition(args.condition, args.bodies, max(1, args.workers))
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"{args.condition}.json"
    destination.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(destination)
    print(json.dumps(result["checks"], indent=2))


if __name__ == "__main__":
    main()
