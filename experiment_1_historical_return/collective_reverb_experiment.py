"""Collective Re-Verb through a finite shared oscillator tissue.

All bodies read and write one capacity-limited complex tissue.  No persistent
state is indexed by body pair.  The experiment asks whether participation by
one pair changes what another pair can later carry.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd


TAU = 2.0 * math.pi


def wrap(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + math.pi) % TAU - math.pi


@dataclass(frozen=True)
class CollectiveConfig:
    dt: float = 0.04
    mode_count: int = 12
    tissue_channels: int = 12
    seeds: int = 12
    sediment_episodes_ab: int = 7
    sediment_episodes_bc: int = 5
    participation_time: float = 18.0
    encounter_time: float = 30.0
    intrinsic_spread: float = 0.15
    body_coupling: float = 0.88
    drive_gain: float = 0.50
    return_gain: float = 1.42
    tissue_start: float = 0.055
    sediment_rate: float = 0.34
    sediment_decay: float = 0.0007
    finite_capacity: float = 0.72
    expanded_capacity: float = 2.40
    viability_start: float = 0.68
    viability_cut_floor: float = 0.25
    viability_failure: float = 0.09
    tension_start: float = 0.07
    tension_failure: float = 0.94
    carrying_cut: float = 0.50
    carrying_gain: float = 0.53
    carrying_decay: float = 0.22
    metabolic_income: float = 0.27
    basal_cost: float = 0.042
    mismatch_cost: float = 0.14
    tension_build: float = 0.23
    tension_release: float = 0.50
    seed: int = 20260911


@dataclass(frozen=True)
class Body:
    name: str
    frequencies: np.ndarray
    gains: np.ndarray
    interface: np.ndarray
    forcing_offsets: np.ndarray
    metabolic_scale: float
    noise_sd: float


@dataclass
class Tissue:
    matrix: np.ndarray
    capacity: float

    def copy(self, *, capacity: float | None = None) -> "Tissue":
        return Tissue(
            matrix=self.matrix.copy(),
            capacity=self.capacity if capacity is None else capacity,
        )


@dataclass(frozen=True)
class World:
    bodies: dict[str, Body]
    initial_tissue: np.ndarray


def _unitary_interface(
    channels: int, modes: int, rng: np.random.Generator
) -> np.ndarray:
    raw = rng.normal(size=(channels, modes)) + 1j * rng.normal(
        size=(channels, modes)
    )
    if channels >= modes:
        q, _ = np.linalg.qr(raw)
        return q[:, :modes]
    norms = np.linalg.norm(raw, axis=0, keepdims=True)
    return raw / np.maximum(norms, 1.0e-12)


def _phase_profile(size: int, rng: np.random.Generator) -> np.ndarray:
    offsets = rng.uniform(-math.pi, math.pi, size=size)
    center = float(np.angle(np.mean(np.exp(1j * offsets))))
    return np.asarray(wrap(offsets - center), dtype=float)


def make_world(config: CollectiveConfig, seed: int) -> World:
    rng = np.random.default_rng(seed)
    axis = np.linspace(-1.0, 1.0, config.mode_count)
    bodies: dict[str, Body] = {}
    specifications = (
        ("A", 0.93, axis, 1.00, 0.012),
        ("B", 1.01, np.roll(axis[::-1], 2), 0.97, 0.013),
        ("C", 1.09, np.roll(axis, 4), 0.90, 0.016),
    )
    for index, (name, center, profile, metabolic, noise) in enumerate(
        specifications
    ):
        frequencies = center + config.intrinsic_spread * profile
        frequencies += rng.normal(0.0, 0.010, size=config.mode_count)
        x = np.linspace(0.0, 1.0, config.mode_count)
        gains = np.clip(
            0.70
            + 0.28 * np.cos(math.pi * (x - 0.18 * index))
            + rng.normal(0.0, 0.018, size=config.mode_count),
            0.42,
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
            metabolic_scale=metabolic,
            noise_sd=noise,
        )

    raw = rng.normal(
        size=(config.tissue_channels, config.tissue_channels)
    ) + 1j * rng.normal(size=(config.tissue_channels, config.tissue_channels))
    hermitian = 0.5 * (raw + raw.conj().T)
    initial_tissue = hermitian * (
        config.tissue_start / max(1.0e-12, float(np.linalg.norm(hermitian)))
    )
    return World(bodies=bodies, initial_tissue=initial_tissue)


def _project_capacity(matrix: np.ndarray, capacity: float) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.conj().T)
    norm = float(np.linalg.norm(matrix))
    if norm > capacity:
        matrix = matrix * (capacity / norm)
    return matrix


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 1.0e-12:
        return 0.0
    return float(np.sum(values * weights) / total)


def _tissue_emission(body: Body, phases: np.ndarray) -> np.ndarray:
    return body.interface @ (body.gains * np.exp(1j * phases)) / math.sqrt(
        len(phases)
    )


def _body_return(body: Body, tissue_signal: np.ndarray) -> np.ndarray:
    return body.gains * (body.interface.conj().T @ tissue_signal)


def _reciprocal_terms(
    tissue: Tissue,
    left: Body,
    right: Body,
    theta_left: np.ndarray,
    theta_right: np.ndarray,
    feedback: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    emission_left = _tissue_emission(left, theta_left)
    emission_right = _tissue_emission(right, theta_right)
    return_left = feedback[0] * _body_return(
        left, tissue.matrix @ emission_right
    )
    return_right = feedback[1] * _body_return(
        right, tissue.matrix @ emission_left
    )
    denominator = max(
        1.0e-12,
        float(np.linalg.norm(emission_left) * np.linalg.norm(emission_right)),
    )
    contact = float(
        np.clip(abs(np.vdot(emission_left, emission_right)) / denominator, 0.0, 1.0)
    )
    return emission_left, emission_right, return_left, return_right, contact


def _support(
    left: Body,
    right: Body,
    theta_left: np.ndarray,
    theta_right: np.ndarray,
    return_left: np.ndarray,
    return_right: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    align_left = _weighted_mean(
        np.clip(
            np.cos(np.angle(return_left + 1.0e-12) - theta_left), 0.0, 1.0
        ),
        left.gains,
    )
    align_right = _weighted_mean(
        np.clip(
            np.cos(np.angle(return_right + 1.0e-12) - theta_right), 0.0, 1.0
        ),
        right.gains,
    )
    strength_left = _weighted_mean(np.abs(return_left), left.gains)
    strength_right = _weighted_mean(np.abs(return_right), right.gains)
    reciprocal_strength = math.sqrt(max(0.0, strength_left * strength_right))
    support = (
        math.sqrt(max(0.0, align_left * align_right))
        * math.tanh(2.4 * reciprocal_strength)
    )
    return (
        float(support),
        np.asarray([align_left, align_right]),
        np.asarray([strength_left, strength_right]),
    )


def _advance_phases(
    config: CollectiveConfig,
    body: Body,
    phases: np.ndarray,
    body_return: np.ndarray,
    drive_phase: float,
    rhythm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    drive = config.drive_gain * np.exp(
        1j * (drive_phase + body.forcing_offsets)
    )
    signal = body.gains * drive + config.return_gain * body_return
    velocity = body.frequencies + config.body_coupling * np.tanh(
        np.abs(signal)
    ) * np.sin(np.angle(signal + 1.0e-12) - phases)
    return np.asarray(
        wrap(
            phases
            + config.dt * velocity
            + math.sqrt(config.dt)
            * rng.normal(0.0, body.noise_sd, size=len(phases))
        )
    )


def participate(
    config: CollectiveConfig,
    tissue: Tissue,
    left: Body,
    right: Body,
    seed: int,
    rhythm: float,
    *,
    plasticity: float = 1.0,
    feedback: tuple[float, float] = (1.0, 1.0),
) -> dict[str, float]:
    """Let a pair alter the shared tissue through its reciprocal trajectory."""
    rng = np.random.default_rng(seed)
    theta_left = rng.uniform(-math.pi, math.pi, size=config.mode_count)
    theta_right = rng.uniform(-math.pi, math.pi, size=config.mode_count)
    drive_phase = 0.0
    support_sum = 0.0
    contact_sum = 0.0
    steps = int(round(config.participation_time / config.dt))

    for _ in range(steps):
        (
            emission_left,
            emission_right,
            return_left,
            return_right,
            contact,
        ) = _reciprocal_terms(
            tissue,
            left,
            right,
            theta_left,
            theta_right,
            feedback,
        )
        support, _, _ = _support(
            left,
            right,
            theta_left,
            theta_right,
            return_left,
            return_right,
        )
        drive_phase += config.dt * rhythm
        theta_left = _advance_phases(
            config,
            left,
            theta_left,
            return_left,
            drive_phase,
            rhythm,
            rng,
        )
        theta_right = _advance_phases(
            config,
            right,
            theta_right,
            return_right,
            drive_phase,
            rhythm,
            rng,
        )

        observation = 0.5 * (
            np.outer(emission_left, np.conj(emission_right))
            + np.outer(emission_right, np.conj(emission_left))
        )
        gate = contact * (0.25 + 0.75 * support)
        tissue.matrix += config.dt * (
            config.sediment_rate * plasticity * gate * observation
            - config.sediment_decay * tissue.matrix
        )
        tissue.matrix = _project_capacity(tissue.matrix, tissue.capacity)
        support_sum += support
        contact_sum += contact

    return {
        "mean_support": support_sum / steps,
        "mean_contact": contact_sum / steps,
        "tissue_norm": float(np.linalg.norm(tissue.matrix)),
        "capacity_use": float(np.linalg.norm(tissue.matrix) / tissue.capacity),
    }


def passive_elapsed_decay(
    config: CollectiveConfig, tissue: Tissue, episodes: int
) -> None:
    duration = episodes * config.participation_time
    tissue.matrix *= math.exp(-config.sediment_decay * duration)
    tissue.matrix = _project_capacity(tissue.matrix, tissue.capacity)


def probe(
    config: CollectiveConfig,
    tissue: Tissue,
    left: Body,
    right: Body,
    seed: int,
    rhythm: float,
    *,
    feedback: tuple[float, float] = (1.0, 1.0),
) -> dict[str, float | str]:
    """Read carrying without changing bodies or tissue."""
    rng = np.random.default_rng(seed)
    theta_left = rng.uniform(-math.pi, math.pi, size=config.mode_count)
    theta_right = rng.uniform(-math.pi, math.pi, size=config.mode_count)
    viability = np.full(2, config.viability_start, dtype=float)
    tension = np.full(2, config.tension_start, dtype=float)
    carried = 0.0
    drive_phase = 0.0
    status = "censored"
    failure_mode = "none"
    event_step: int | None = None
    final_support = 0.0
    cumulative_support = 0.0
    steps = int(round(config.encounter_time / config.dt))

    for step in range(steps):
        _, _, return_left, return_right, _ = _reciprocal_terms(
            tissue,
            left,
            right,
            theta_left,
            theta_right,
            feedback,
        )
        drive_phase += config.dt * rhythm
        theta_left = _advance_phases(
            config,
            left,
            theta_left,
            return_left,
            drive_phase,
            rhythm,
            rng,
        )
        theta_right = _advance_phases(
            config,
            right,
            theta_right,
            return_right,
            drive_phase,
            rhythm,
            rng,
        )
        _, _, return_left, return_right, _ = _reciprocal_terms(
            tissue,
            left,
            right,
            theta_left,
            theta_right,
            feedback,
        )
        support, alignments, _ = _support(
            left,
            right,
            theta_left,
            theta_right,
            return_left,
            return_right,
        )
        final_support = support
        costs = config.basal_cost + config.mismatch_cost * (1.0 - alignments)
        income = config.metabolic_income * support * np.asarray(
            [left.metabolic_scale, right.metabolic_scale]
        )
        viability = np.clip(viability + config.dt * (income - costs), 0.0, 1.0)
        tension = np.clip(
            tension
            + config.dt
            * (
                config.tension_build * (1.0 - alignments)
                - config.tension_release * support * tension
            ),
            0.0,
            1.5,
        )
        viable_fraction = float(
            np.clip(
                (float(np.min(viability)) - config.viability_failure)
                / (1.0 - config.viability_failure),
                0.0,
                1.0,
            )
        )
        effective_support = support * viable_fraction
        cumulative_support += config.dt * effective_support
        carried += config.dt * (
            config.carrying_gain * effective_support * (1.0 - carried)
            - config.carrying_decay * (1.0 - effective_support) * carried
        )
        carried = float(np.clip(carried, 0.0, 1.0))

        if (
            carried >= config.carrying_cut
            and float(np.min(viability)) >= config.viability_cut_floor
            and float(np.max(tension)) < config.tension_failure
        ):
            status = "enacted"
            event_step = step + 1
        elif float(np.min(viability)) <= config.viability_failure:
            status = "refused"
            failure_mode = "viability"
            event_step = step + 1
        elif float(np.max(tension)) >= config.tension_failure:
            status = "refused"
            failure_mode = "tension"
            event_step = step + 1
        if event_step is not None:
            break

    phase_weights = left.gains * right.gains
    terminal_phase = float(
        np.angle(
            np.sum(
                phase_weights
                * np.exp(1j * theta_left)
                * np.conj(np.exp(1j * theta_right))
            )
        )
    )

    return {
        "status": status,
        "failure_mode": failure_mode,
        "carried": carried,
        "support": final_support,
        "cumulative_support": cumulative_support,
        "min_viability": float(np.min(viability)),
        "max_tension": float(np.max(tension)),
        "terminal_phase": terminal_phase,
        "coordinate_phase": (
            terminal_phase if status == "enacted" else math.nan
        ),
        "coordinate_return": (
            final_support if status == "enacted" else math.nan
        ),
        "event_time": (
            float(event_step * config.dt) if event_step is not None else math.nan
        ),
    }


def _append_probe(
    rows: list[dict[str, object]],
    *,
    seed_index: int,
    condition: str,
    phase: str,
    pair: str,
    tissue: Tissue,
    result: dict[str, float | str],
) -> None:
    rows.append(
        {
            "seed": seed_index,
            "condition": condition,
            "phase": phase,
            "pair": pair,
            **result,
            "tissue_norm": float(np.linalg.norm(tissue.matrix)),
            "capacity": tissue.capacity,
            "capacity_use": float(np.linalg.norm(tissue.matrix) / tissue.capacity),
        }
    )


def run_experiment(
    config: CollectiveConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    probe_rows: list[dict[str, object]] = []
    tissue_rows: list[dict[str, object]] = []
    rhythm_ab = 0.97
    rhythm_bc = 1.06

    for seed_index in range(config.seeds):
        world_seed = config.seed + seed_index * 10_000
        world = make_world(config, world_seed)
        a, b, c = (world.bodies[name] for name in ("A", "B", "C"))
        initial = Tissue(world.initial_tissue.copy(), config.finite_capacity)
        probe_seed_ab = world_seed + 701
        probe_seed_bc = world_seed + 907

        initial_result = probe(
            config, initial, a, b, probe_seed_ab, rhythm_ab
        )
        _append_probe(
            probe_rows,
            seed_index=seed_index,
            condition="initial_tissue",
            phase="before_ab_history",
            pair="AB",
            tissue=initial,
            result=initial_result,
        )

        ab_tissue = initial.copy()
        for episode in range(config.sediment_episodes_ab):
            participate(
                config,
                ab_tissue,
                a,
                b,
                world_seed + 1_100 + episode,
                rhythm_ab,
            )
        ab_before = probe(
            config, ab_tissue, a, b, probe_seed_ab, rhythm_ab
        )
        _append_probe(
            probe_rows,
            seed_index=seed_index,
            condition="ab_history",
            phase="before_bc_participation",
            pair="AB",
            tissue=ab_tissue,
            result=ab_before,
        )

        condition_tissues = {
            "shared_finite": ab_tissue.copy(),
            "partitioned": ab_tissue.copy(),
            "shared_expanded": ab_tissue.copy(capacity=config.expanded_capacity),
        }
        partitioned_bc = initial.copy()
        pre_bc_matrix = ab_tissue.matrix.copy()

        for condition, tissue in condition_tissues.items():
            if condition == "partitioned":
                passive_elapsed_decay(
                    config, tissue, config.sediment_episodes_bc
                )
                bc_target = partitioned_bc
            else:
                bc_target = tissue
            for episode in range(config.sediment_episodes_bc):
                participate(
                    config,
                    bc_target,
                    b,
                    c,
                    world_seed + 2_100 + episode,
                    rhythm_bc,
                )

        for condition, tissue in condition_tissues.items():
            ab_after = probe(
                config, tissue, a, b, probe_seed_ab, rhythm_ab
            )
            _append_probe(
                probe_rows,
                seed_index=seed_index,
                condition=condition,
                phase="after_bc_participation",
                pair="AB",
                tissue=tissue,
                result=ab_after,
            )
            bc_probe_tissue = partitioned_bc if condition == "partitioned" else tissue
            bc_after = probe(
                config, bc_probe_tissue, b, c, probe_seed_bc, rhythm_bc
            )
            _append_probe(
                probe_rows,
                seed_index=seed_index,
                condition=condition,
                phase="after_bc_participation",
                pair="BC",
                tissue=bc_probe_tissue,
                result=bc_after,
            )
            one_way = probe(
                config,
                tissue,
                a,
                b,
                probe_seed_ab,
                rhythm_ab,
                feedback=(1.0, 0.0),
            )
            _append_probe(
                probe_rows,
                seed_index=seed_index,
                condition=condition,
                phase="one_way_after_bc",
                pair="AB",
                tissue=tissue,
                result=one_way,
            )

            denominator = max(
                1.0e-12,
                float(np.linalg.norm(pre_bc_matrix) * np.linalg.norm(tissue.matrix)),
            )
            continuity = float(
                np.real(np.vdot(pre_bc_matrix, tissue.matrix)) / denominator
            )
            tissue_rows.append(
                {
                    "seed": seed_index,
                    "condition": condition,
                    "pre_bc_norm": float(np.linalg.norm(pre_bc_matrix)),
                    "post_bc_norm": float(np.linalg.norm(tissue.matrix)),
                    "capacity": tissue.capacity,
                    "capacity_use": float(np.linalg.norm(tissue.matrix) / tissue.capacity),
                    "pre_post_continuity": continuity,
                    "matrix_change": float(np.linalg.norm(tissue.matrix - pre_bc_matrix)),
                }
            )

    return pd.DataFrame(probe_rows), pd.DataFrame(tissue_rows)


def _paired_table(probes: pd.DataFrame) -> pd.DataFrame:
    selected = probes[
        (probes["pair"] == "AB")
        & (probes["phase"] == "after_bc_participation")
    ]
    pivot = selected.pivot(index="seed", columns="condition", values="carried")
    pivot["shared_minus_partitioned"] = (
        pivot["shared_finite"] - pivot["partitioned"]
    )
    pivot["finite_minus_expanded"] = (
        pivot["shared_finite"] - pivot["shared_expanded"]
    )
    return pivot.reset_index()


def _bootstrap_interval(
    values: np.ndarray, seed: int, draws: int = 5000
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"estimate": math.nan, "low": math.nan, "high": math.nan}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "estimate": float(np.mean(values)),
        "low": float(low),
        "high": float(high),
    }


def summarize(
    config: CollectiveConfig, probes: pd.DataFrame, tissues: pd.DataFrame
) -> dict[str, object]:
    paired = _paired_table(probes)

    def mean_cell(condition: str, phase: str, pair: str, field: str) -> float:
        cell = probes[
            (probes["condition"] == condition)
            & (probes["phase"] == phase)
            & (probes["pair"] == pair)
        ]
        if field == "enacted":
            return float(cell["status"].eq("enacted").mean())
        return float(cell[field].mean())

    initial_ab = mean_cell(
        "initial_tissue", "before_ab_history", "AB", "carried"
    )
    historical_ab = mean_cell(
        "ab_history", "before_bc_participation", "AB", "carried"
    )
    shared_finite_bc = mean_cell(
        "shared_finite", "after_bc_participation", "BC", "carried"
    )
    shared_expanded_bc = mean_cell(
        "shared_expanded", "after_bc_participation", "BC", "carried"
    )
    one_way_rate = float(
        probes[probes["phase"] == "one_way_after_bc"]["status"]
        .eq("enacted")
        .mean()
    )
    shared_effect = paired["shared_minus_partitioned"].to_numpy(float)
    capacity_effect = paired["finite_minus_expanded"].to_numpy(float)
    material_cut = 0.025
    shared_interval = _bootstrap_interval(shared_effect, config.seed + 31)
    shared_absolute_interval = _bootstrap_interval(
        np.abs(shared_effect), config.seed + 37
    )
    capacity_interval = _bootstrap_interval(capacity_effect, config.seed + 41)
    capacity_absolute_interval = _bootstrap_interval(
        np.abs(capacity_effect), config.seed + 43
    )
    summary: dict[str, object] = {
        "question": (
            "Can participation by one pair alter what another pair carries "
            "through one finite shared tissue?"
        ),
        "config": asdict(config),
        "means": {
            "initial_ab_carrying": initial_ab,
            "post_history_ab_carrying": historical_ab,
            "shared_post_bc_ab_carrying": mean_cell(
                "shared_finite", "after_bc_participation", "AB", "carried"
            ),
            "partitioned_post_bc_ab_carrying": mean_cell(
                "partitioned", "after_bc_participation", "AB", "carried"
            ),
            "expanded_post_bc_ab_carrying": mean_cell(
                "shared_expanded", "after_bc_participation", "AB", "carried"
            ),
            "shared_finite_post_bc_bc_carrying": shared_finite_bc,
            "shared_expanded_post_bc_bc_carrying": shared_expanded_bc,
            "one_way_enactment_rate": one_way_rate,
            "mean_shared_minus_partitioned": float(np.mean(shared_effect)),
            "mean_abs_shared_minus_partitioned": float(np.mean(np.abs(shared_effect))),
            "mean_finite_minus_expanded": float(np.mean(capacity_effect)),
            "mean_abs_finite_minus_expanded": float(np.mean(np.abs(capacity_effect))),
        },
        "tissue_diagnostics": {
            condition: {
                "mean_capacity_use": float(frame["capacity_use"].mean()),
                "mean_continuity": float(frame["pre_post_continuity"].mean()),
                "mean_matrix_change": float(frame["matrix_change"].mean()),
            }
            for condition, frame in tissues.groupby("condition")
        },
        "paired_effects": {
            "shared_finite_minus_partitioned": shared_interval,
            "absolute_shared_finite_minus_partitioned": shared_absolute_interval,
            "finite_minus_expanded": capacity_interval,
            "absolute_finite_minus_expanded": capacity_absolute_interval,
            "material_change_cut": material_cut,
            "material_change_prevalence": float(
                np.mean(np.abs(shared_effect) >= material_cut)
            ),
            "shared_effect_negative_prevalence": float(
                np.mean(shared_effect < -1.0e-12)
            ),
            "shared_effect_positive_prevalence": float(
                np.mean(shared_effect > 1.0e-12)
            ),
            "capacity_effect_negative_prevalence": float(
                np.mean(capacity_effect < -1.0e-12)
            ),
        },
        "criteria": {
            "ab_history_changes_carrying": bool(historical_ab - initial_ab > 0.04),
            "bc_participation_establishes_carrying_in_shared_tissue": bool(
                shared_expanded_bc > 0.20
            ),
            "shared_tissue_changes_other_relation": bool(
                np.mean(np.abs(shared_effect)) > 0.025
            ),
            "finite_capacity_changes_collective_effect": bool(
                np.mean(np.abs(capacity_effect)) > 0.015
            ),
            "one_way_never_enacts": bool(one_way_rate == 0.0),
            "no_pair_indexed_memory": True,
        },
    }
    return summary


def write_report(summary: dict[str, object], path: Path) -> None:
    means = summary["means"]
    criteria = summary["criteria"]
    diagnostics = summary["tissue_diagnostics"]
    effects = summary["paired_effects"]
    shared = effects["shared_finite_minus_partitioned"]
    shared_absolute = effects["absolute_shared_finite_minus_partitioned"]
    capacity = effects["finite_minus_expanded"]
    capacity_absolute = effects["absolute_finite_minus_expanded"]
    lines = [
        "# Collective Re-Verb pilot",
        "",
        str(summary["question"]),
        "",
        "## Mean carrying",
        "",
        f"- A--B in initial tissue: {means['initial_ab_carrying']:.3f}.",
        f"- A--B after its own history: {means['post_history_ab_carrying']:.3f}.",
        f"- A--B after B--C in shared finite tissue: {means['shared_post_bc_ab_carrying']:.3f}.",
        f"- A--B after B--C in partitioned tissue: {means['partitioned_post_bc_ab_carrying']:.3f}.",
        f"- A--B after B--C in expanded shared tissue: {means['expanded_post_bc_ab_carrying']:.3f}.",
        f"- B--C after finite shared participation: {means['shared_finite_post_bc_bc_carrying']:.3f}.",
        f"- B--C after expanded shared participation: {means['shared_expanded_post_bc_bc_carrying']:.3f}.",
        "",
        "## Paired collective effects",
        "",
        f"- Shared finite minus partitioned: {shared['estimate']:+.3f} [{shared['low']:+.3f}, {shared['high']:+.3f}].",
        f"- Absolute shared finite minus partitioned change: {shared_absolute['estimate']:.3f} [{shared_absolute['low']:.3f}, {shared_absolute['high']:.3f}].",
        f"- Finite minus expanded capacity: {capacity['estimate']:+.3f} [{capacity['low']:+.3f}, {capacity['high']:+.3f}].",
        f"- Absolute finite minus expanded change: {capacity_absolute['estimate']:.3f} [{capacity_absolute['low']:.3f}, {capacity_absolute['high']:.3f}].",
        f"- Material shared-tissue change (absolute difference >= {effects['material_change_cut']:.3f}): {effects['material_change_prevalence']:.1%} of paired worlds.",
        f"- Shared effect was negative in {effects['shared_effect_negative_prevalence']:.1%} and positive in {effects['shared_effect_positive_prevalence']:.1%} of worlds.",
        f"- Capacity effect was negative in {effects['capacity_effect_negative_prevalence']:.1%} of worlds.",
        f"- One-way enactment rate: {means['one_way_enactment_rate']:.1%}.",
        "",
        "## Tissue diagnostics",
        "",
    ]
    for condition, values in diagnostics.items():
        lines.append(
            f"- {condition}: capacity use {values['mean_capacity_use']:.3f}, "
            f"continuity {values['mean_continuity']:.3f}, matrix change "
            f"{values['mean_matrix_change']:.3f}."
        )
    lines.extend(["", "## Feasibility criteria", ""])
    for name, passed in criteria.items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}.")
    lines.extend(
        [
            "",
            "## Reading boundary",
            "",
            "The direction of change is not a value judgement. This classical "
            "constructed model tests whether a finite tissue without pair-indexed "
            "memory can mediate one relation through another. It does not establish "
            "strong anticipation, ecological validity, quantum non-causality, or "
            "agential realism.",
            "",
            "Bootstrap intervals describe variation across the generated paired "
            "worlds under this fixed model and parameterization. They are not "
            "population-level confidence intervals.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("collective_reverb_output")
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--base-seed", type=int)
    args = parser.parse_args()
    config = CollectiveConfig()
    if args.quick:
        config = replace(
            config,
            seeds=3,
            sediment_episodes_ab=4,
            sediment_episodes_bc=3,
            participation_time=10.0,
            encounter_time=18.0,
        )
    if args.seeds is not None:
        if args.seeds < 1:
            parser.error("--seeds must be positive")
        config = replace(config, seeds=args.seeds)
    if args.base_seed is not None:
        config = replace(config, seed=args.base_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probes, tissues = run_experiment(config)
    summary = summarize(config, probes, tissues)
    probes.to_csv(args.output_dir / "collective_probes.csv", index=False)
    tissues.to_csv(args.output_dir / "tissue_diagnostics.csv", index=False)
    _paired_table(probes).to_csv(
        args.output_dir / "paired_collective_effects.csv", index=False
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_report(summary, args.output_dir / "collective_reverb_report.md")
    print(json.dumps(summary["means"], indent=2))
    print(json.dumps(summary["criteria"], indent=2))


if __name__ == "__main__":
    main()
