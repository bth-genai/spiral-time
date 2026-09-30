"""O1: recurring relations in an overlapping oscillator organization."""

from __future__ import annotations

import argparse
import copy
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

TAU = 2.0 * math.pi
EDGE_NAMES = ("AB", "BC")
BODY_NAMES = ("A", "B", "C")
CHECKPOINTS = ("formation", "maintenance", "withdrawal", "return_1", "return_3", "return_6")


def wrap(values: np.ndarray) -> np.ndarray:
    return (values + math.pi) % TAU - math.pi


def bounded(matrix: np.ndarray, capacity: float) -> np.ndarray:
    norm = float(np.linalg.norm(matrix, ord="fro"))
    return matrix if norm <= capacity else matrix * (capacity / norm)


def cosine_matrix(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left, ord="fro") * np.linalg.norm(right, ord="fro"))
    if denominator <= 1.0e-15:
        return math.nan
    return float(np.sum(left * right) / denominator)


def circular_rms(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.angle(np.exp(1j * (left - right)))
    return float(np.sqrt(np.mean(difference**2)))


@dataclass(frozen=True)
class Config:
    worlds: int = 24
    modes: int = 12
    dt: float = 0.02
    reference_dt: float = 0.01
    episode_duration: float = 2.0
    intrinsic_spread: float = 0.16
    drive_gain: float = 0.72
    local_expression_gain: float = 0.22
    edge_expression_gain: float = 0.48
    phase_noise: float = 0.025
    local_learning: float = 0.0025
    local_decay: float = 0.002
    local_capacity: float = 4.0
    trace_learning: float = 0.20
    trace_decay: float = 0.18
    trace_capacity: float = 6.0
    tissue_learning: float = 0.045
    tissue_decay: float = 0.006
    tissue_capacity: float = 6.0
    trace_expression: float = 0.20
    # Spiral-Time history resolution. The conventional current-state model is
    # recovered exactly when history_expression_gain = 0. The history field is
    # computed directly from archived encounter observations rather than from a
    # recursively compressed state variable.
    history_expression_gain: float = 0.0
    history_fast_fraction: float = 0.35
    history_tau_fast: float = 3.0
    history_tau_slow: float = 18.0
    history_resolution_records: int = 0
    formation_per_edge: int = 12
    maintenance_cycles: int = 4
    withdrawal_episodes: int = 96
    withdrawal_bc_period: int = 4
    return_offers: int = 6
    assay_episodes: int = 4
    rehearsal_episodes: int = 6
    base_seed: int = 202609150

    def __post_init__(self) -> None:
        if self.worlds <= 0 or self.modes <= 1:
            raise ValueError("worlds must be positive and modes must exceed one")
        ratio = self.dt / self.reference_dt
        if ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1.0e-12):
            raise ValueError("dt must be an integer multiple of reference_dt")
        fine_steps = self.episode_duration / self.reference_dt
        if not math.isclose(fine_steps, round(fine_steps), abs_tol=1.0e-12):
            raise ValueError("episode_duration must contain whole reference steps")


@dataclass
class Body:
    name: str
    frequencies: np.ndarray
    gains: np.ndarray
    forcing_offsets: np.ndarray
    phases: np.ndarray
    local_tissue: np.ndarray


@dataclass
class HistoryRecord:
    episode_index: int
    observation: np.ndarray


@dataclass
class Edge:
    name: str
    first: str
    second: str
    rhythm: float
    trace: np.ndarray
    tissue: np.ndarray
    history: list[HistoryRecord] = field(default_factory=list)


@dataclass
class World:
    bodies: dict[str, Body]
    edges: dict[str, Edge]
    episode_index: int = 0

    def clone(self) -> "World":
        return copy.deepcopy(self)


def _centered_offsets(rng: np.random.Generator, size: int) -> np.ndarray:
    offsets = rng.uniform(-math.pi, math.pi, size=size)
    center = float(np.angle(np.mean(np.exp(1j * offsets))))
    return wrap(offsets - center)


def _ring(size: int, strength: float) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=float)
    for index in range(size):
        matrix[index, (index - 1) % size] = strength
        matrix[index, (index + 1) % size] = strength
    return matrix


def make_world(config: Config, seed: int) -> World:
    rng = np.random.default_rng(seed)
    axis = np.linspace(-1.0, 1.0, config.modes)
    bodies: dict[str, Body] = {}
    specifications = (
        ("A", 0.91, axis),
        ("B", 1.00, np.roll(axis[::-1], 2)),
        ("C", 1.10, np.roll(axis, 4)),
    )
    for index, (name, center, profile) in enumerate(specifications):
        frequencies = center + config.intrinsic_spread * profile
        frequencies += rng.normal(0.0, 0.008, size=config.modes)
        x = np.linspace(0.0, 1.0, config.modes)
        gains = np.clip(
            0.68 + 0.28 * np.cos(math.pi * (x - 0.17 * index))
            + rng.normal(0.0, 0.015, size=config.modes),
            0.38,
            1.0,
        )
        bodies[name] = Body(
            name=name,
            frequencies=frequencies,
            gains=gains,
            forcing_offsets=_centered_offsets(rng, config.modes),
            phases=rng.uniform(-math.pi, math.pi, size=config.modes),
            local_tissue=_ring(config.modes, 0.055 + 0.008 * index),
        )
    edges = {}
    for name, first, second in (("AB", "A", "B"), ("BC", "B", "C")):
        rhythm = 0.5 * (
            float(np.mean(bodies[first].frequencies))
            + float(np.mean(bodies[second].frequencies))
        )
        edges[name] = Edge(
            name=name,
            first=first,
            second=second,
            rhythm=rhythm,
            trace=np.zeros((config.modes, config.modes), dtype=float),
            tissue=np.zeros((config.modes, config.modes), dtype=float),
        )
    return World(bodies=bodies, edges=edges)


def _brownian_increments(config: Config, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    fine_steps = round(config.episode_duration / config.reference_dt)
    ratio = round(config.dt / config.reference_dt)
    if fine_steps % ratio:
        raise ValueError("coarse timestep must divide episode duration")
    increments = {}
    for name in BODY_NAMES:
        fine = math.sqrt(config.reference_dt) * rng.normal(
            size=(fine_steps, config.modes)
        )
        increments[name] = fine.reshape(fine_steps // ratio, ratio, config.modes).sum(axis=1)
    return increments


def _history_field(edge: Edge, config: Config, episode_index: int) -> np.ndarray:
    """Return the history-resolved field available at the present encounter.

    The field is evaluated from the archived sequence of encounter observations.
    It is therefore not reconstructible from the current trace/tissue pair in
    general. ``history_resolution_records`` controls how many most recent
    records remain visible: 0 means the complete archived edge history.
    """
    if config.history_expression_gain == 0.0 or not edge.history:
        return np.zeros_like(edge.tissue)
    records = edge.history
    if config.history_resolution_records > 0:
        records = records[-config.history_resolution_records :]
    weighted = np.zeros_like(edge.tissue)
    weight_sum = 0.0
    for record in records:
        age = max(1, episode_index - record.episode_index)
        weight = (
            config.history_fast_fraction * math.exp(-age / config.history_tau_fast)
            + (1.0 - config.history_fast_fraction)
            * math.exp(-age / config.history_tau_slow)
        )
        weighted += weight * record.observation
        weight_sum += weight
    if weight_sum <= 1.0e-15:
        return np.zeros_like(edge.tissue)
    return weighted / weight_sum


def _edge_expression(edge: Edge, config: Config, episode_index: int) -> np.ndarray:
    return (
        edge.tissue
        + config.trace_expression * edge.trace
        + config.history_expression_gain * _history_field(edge, config, episode_index)
    )


def pair_alignment(world: World, edge_name: str) -> float:
    edge = world.edges[edge_name]
    first = world.bodies[edge.first]
    second = world.bodies[edge.second]
    weights = np.outer(first.gains, second.gains)
    first_relative = first.phases - first.forcing_offsets
    second_relative = second.phases - second.forcing_offsets
    difference = second_relative[np.newaxis, :] - first_relative[:, np.newaxis]
    return float(np.sum(weights * np.cos(difference)) / np.sum(weights))


def _endpoint_drive_alignment(body: Body, drive_phase: float) -> float:
    alignment = 0.5 * (
        1.0 + np.cos(drive_phase + body.forcing_offsets - body.phases)
    )
    return float(np.sum(body.gains * alignment) / np.sum(body.gains))


def _encounter_phase(world: World, offered: str | None) -> float:
    if offered is None:
        return 0.0
    edge = world.edges[offered]
    resultant = 0.0j
    for endpoint in (edge.first, edge.second):
        body = world.bodies[endpoint]
        resultant += np.sum(
            body.gains * np.exp(1j * (body.phases - body.forcing_offsets))
        ) / np.sum(body.gains)
    if abs(resultant) <= 1.0e-15:
        return float(world.bodies[edge.first].phases[0])
    return float(np.angle(resultant))


def run_episode(
    world: World,
    config: Config,
    offered: str | None,
    seed: int,
    *,
    plastic: bool = True,
    expression: dict[str, tuple[bool, bool]] | None = None,
    record_trajectory: bool = False,
    record_history: bool | None = None,
    increments_override: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    if offered is not None and offered not in world.edges:
        raise ValueError(f"unknown offered edge: {offered}")
    expression = expression or {name: (True, True) for name in EDGE_NAMES}
    if record_history is None:
        record_history = plastic
    if set(expression) != set(EDGE_NAMES):
        raise ValueError("expression must specify AB and BC")
    increments = (
        _brownian_increments(config, seed)
        if increments_override is None
        else increments_override
    )
    expected_steps = round(config.episode_duration / config.dt)
    if set(increments) != set(BODY_NAMES):
        raise ValueError("increments_override must specify A, B, and C")
    if any(
        values.shape != (expected_steps, config.modes)
        for values in increments.values()
    ):
        raise ValueError("increment arrays do not match episode steps and modes")
    steps = next(iter(increments.values())).shape[0]
    drive_phase = _encounter_phase(world, offered)
    trajectories = {name: [] for name in BODY_NAMES}
    initial_local = {
        name: body.local_tissue.copy() for name, body in world.bodies.items()
    }
    initial_edge = {
        name: edge.tissue.copy() for name, edge in world.edges.items()
    }

    # Expression is fixed within an episode; plasticity acts at its end.
    history_fields = {
        name: _history_field(edge, config, world.episode_index)
        for name, edge in world.edges.items()
    }
    expressed = {
        name: config.edge_expression_gain
        * (
            edge.tissue
            + config.trace_expression * edge.trace
            + config.history_expression_gain * history_fields[name]
        )
        for name, edge in world.edges.items()
    }
    for step in range(steps):
        phases = {name: body.phases.copy() for name, body in world.bodies.items()}
        torques = {name: np.zeros(config.modes, dtype=float) for name in BODY_NAMES}
        for name, body in world.bodies.items():
            difference = phases[name][np.newaxis, :] - phases[name][:, np.newaxis]
            torques[name] += (
                config.local_expression_gain
                * np.sum(body.local_tissue * np.sin(difference), axis=1)
                / config.modes
            )
        for edge_name, edge in world.edges.items():
            first_enabled, second_enabled = expression[edge_name]
            difference = (
                phases[edge.second][np.newaxis, :]
                - phases[edge.first][:, np.newaxis]
            )
            flow = expressed[edge_name] * np.sin(difference)
            if first_enabled:
                torques[edge.first] += np.sum(flow, axis=1) / config.modes
            if second_enabled:
                torques[edge.second] -= np.sum(flow, axis=0) / config.modes
        drive_phase += config.dt * (
            world.edges[offered].rhythm if offered is not None else 1.0
        )
        if offered is not None:
            edge = world.edges[offered]
            for endpoint in (edge.first, edge.second):
                body = world.bodies[endpoint]
                torques[endpoint] += (
                    config.drive_gain
                    * body.gains
                    * np.sin(drive_phase + body.forcing_offsets - phases[endpoint])
                )
        for name, body in world.bodies.items():
            body.phases = wrap(
                phases[name]
                + config.dt * (body.frequencies + torques[name])
                + config.phase_noise * increments[name][step]
            )
            if record_trajectory:
                trajectories[name].append(body.phases.copy())

    alignments = {name: pair_alignment(world, name) for name in EDGE_NAMES}
    offered_observation = None
    if offered is not None:
        offered_edge = world.edges[offered]
        first = world.bodies[offered_edge.first]
        second = world.bodies[offered_edge.second]
        difference = (
            second.phases[np.newaxis, :]
            - first.phases[:, np.newaxis]
        )
        offered_observation = np.cos(difference)
        offered_observation -= float(np.mean(offered_observation))

    if plastic:
        for name, body in world.bodies.items():
            difference = body.phases[np.newaxis, :] - body.phases[:, np.newaxis]
            observation = np.cos(difference)
            observation -= float(np.mean(observation))
            body.local_tissue = bounded(
                (1.0 - config.local_decay) * body.local_tissue
                + config.local_learning * observation,
                config.local_capacity,
            )
        for edge_name, edge in world.edges.items():
            edge.trace *= 1.0 - config.trace_decay
            edge.tissue *= 1.0 - config.tissue_decay
            if edge_name == offered:
                first = world.bodies[edge.first]
                second = world.bodies[edge.second]
                observation = offered_observation
                drive_alignment = math.sqrt(
                    _endpoint_drive_alignment(first, drive_phase)
                    * _endpoint_drive_alignment(second, drive_phase)
                )
                recurrence = cosine_matrix(edge.trace, observation)
                recurrence_gate = 0.0 if math.isnan(recurrence) else max(0.0, recurrence)
                edge.trace = bounded(
                    edge.trace + config.trace_learning * drive_alignment * observation,
                    config.trace_capacity,
                )
                edge.tissue = bounded(
                    edge.tissue
                    + config.tissue_learning * recurrence_gate * edge.trace,
                    config.tissue_capacity,
                )

    if record_history and offered is not None and offered_observation is not None:
        world.edges[offered].history.append(
            HistoryRecord(
                episode_index=world.episode_index,
                observation=offered_observation.copy(),
            )
        )
    world.episode_index += 1

    return {
        "alignment": alignments,
        "history_field_norm": {
            name: float(np.linalg.norm(history_fields[name], ord="fro"))
            for name in EDGE_NAMES
        },
        "history_records_visible": {
            name: (
                len(world.edges[name].history)
                if config.history_resolution_records == 0
                else min(config.history_resolution_records, len(world.edges[name].history))
            )
            for name in EDGE_NAMES
        },
        "trajectory": {
            name: np.asarray(values) for name, values in trajectories.items()
        } if record_trajectory else None,
        "local_change": {
            name: float(np.linalg.norm(body.local_tissue - initial_local[name], ord="fro"))
            for name, body in world.bodies.items()
        },
        "edge_change": {
            name: float(np.linalg.norm(edge.tissue - initial_edge[name], ord="fro"))
            for name, edge in world.edges.items()
        },
    }


def _formation_schedule(config: Config, history: str) -> list[tuple[str, int]]:
    tokens = [("AB", index) for index in range(config.formation_per_edge)]
    tokens += [("BC", index) for index in range(config.formation_per_edge)]
    if history == "blocked":
        return tokens
    if history == "alternating":
        return [
            token
            for index in range(config.formation_per_edge)
            for token in (("AB", index), ("BC", index))
        ]
    raise ValueError(f"unknown history: {history}")


def _token_seed(world_seed: int, phase: int, edge: str | None, token: int) -> int:
    edge_offset = {None: 0, "AB": 100_000, "BC": 200_000}[edge]
    return world_seed + phase * 1_000_003 + edge_offset + token * 1009


def _optional_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    value = cosine_matrix(left, right)
    return None if math.isnan(value) else value


def _checkpoint(
    world: World,
    edge_references: dict[str, np.ndarray],
    body_references: dict[str, np.ndarray],
    config: Config,
) -> dict[str, Any]:
    return {
        "edge": {
            name: {
                "trace_norm": float(np.linalg.norm(edge.trace, ord="fro")),
                "tissue_norm": float(np.linalg.norm(edge.tissue, ord="fro")),
                "orientation_to_formation": _optional_cosine(
                    edge.tissue, edge_references[name]
                ),
                "distance_from_formation": float(np.linalg.norm(
                    edge.tissue - edge_references[name], ord="fro"
                )),
                "capacity_fraction": float(
                    np.linalg.norm(edge.tissue, ord="fro") / config.tissue_capacity
                ),
            }
            for name, edge in world.edges.items()
        },
        "body": {
            name: {
                "local_norm": float(np.linalg.norm(body.local_tissue, ord="fro")),
                "local_orientation_to_formation": _optional_cosine(
                    body.local_tissue, body_references[name]
                ),
                "local_distance_from_formation": float(np.linalg.norm(
                    body.local_tissue - body_references[name], ord="fro"
                )),
                "local_capacity_fraction": float(
                    np.linalg.norm(body.local_tissue, ord="fro") / config.local_capacity
                ),
            }
            for name, body in world.bodies.items()
        },
        "alignment": {name: pair_alignment(world, name) for name in EDGE_NAMES},
    }


def expression_assay(world: World, config: Config, seed_base: int) -> dict[str, Any]:
    sequence = ("AB", "BC", "AB", "BC")
    conditions = {
        "full": {"AB": (True, True), "BC": (True, True)},
        "ab_off": {"AB": (False, False), "BC": (True, True)},
        "bc_off": {"AB": (True, True), "BC": (False, False)},
        "all_off": {"AB": (False, False), "BC": (False, False)},
        "ab_to_a_off": {"AB": (False, True), "BC": (True, True)},
        "ab_to_b_off": {"AB": (True, False), "BC": (True, True)},
    }
    outputs = {}
    for condition, masks in conditions.items():
        branch = world.clone()
        phase_trajectory = {name: [] for name in BODY_NAMES}
        alignment_trajectory = {name: [] for name in EDGE_NAMES}
        for index, offered in enumerate(sequence):
            episode = run_episode(
                branch,
                config,
                offered,
                seed_base + index * 1009,
                plastic=False,
                expression=masks,
                record_trajectory=True,
            )
            for name in BODY_NAMES:
                phase_trajectory[name].append(episode["trajectory"][name])
            for name in EDGE_NAMES:
                alignment_trajectory[name].append(episode["alignment"][name])
        outputs[condition] = {
            "phase": {
                name: np.concatenate(values, axis=0)
                for name, values in phase_trajectory.items()
            },
            "alignment": {
                name: np.asarray(values) for name, values in alignment_trajectory.items()
            },
        }
    erased = world.clone()
    erased.edges["AB"].trace.fill(0.0)
    erased.edges["AB"].tissue.fill(0.0)
    erased.edges["AB"].history.clear()
    erased_output = expression_assay_single(
        erased, config, seed_base, sequence,
        {"AB": (True, True), "BC": (True, True)},
    )
    full = outputs["full"]
    comparisons = {}
    for condition in ("ab_off", "bc_off", "all_off", "ab_to_a_off", "ab_to_b_off"):
        comparisons[condition] = {
            "body_phase_rms": {
                name: circular_rms(full["phase"][name], outputs[condition]["phase"][name])
                for name in BODY_NAMES
            },
            "alignment_rmse": {
                name: float(np.sqrt(np.mean(
                    (
                        full["alignment"][name]
                        - outputs[condition]["alignment"][name]
                    ) ** 2
                )))
                for name in EDGE_NAMES
            },
        }
    comparisons["ab_off_vs_erased"] = {
        "maximum_phase_error": float(max(
            np.max(np.abs(np.angle(np.exp(1j * (
                outputs["ab_off"]["phase"][name] - erased_output["phase"][name]
            ))))) for name in BODY_NAMES
        )),
        "maximum_alignment_error": float(max(
            np.max(np.abs(
                outputs["ab_off"]["alignment"][name]
                - erased_output["alignment"][name]
            )) for name in EDGE_NAMES
        )),
    }
    return comparisons


def expression_assay_single(
    world: World,
    config: Config,
    seed_base: int,
    sequence: tuple[str, ...],
    masks: dict[str, tuple[bool, bool]],
) -> dict[str, dict[str, np.ndarray]]:
    phase_trajectory = {name: [] for name in BODY_NAMES}
    alignment_trajectory = {name: [] for name in EDGE_NAMES}
    for index, offered in enumerate(sequence):
        episode = run_episode(
            world, config, offered, seed_base + index * 1009,
            plastic=False, expression=masks, record_trajectory=True,
        )
        for name in BODY_NAMES:
            phase_trajectory[name].append(episode["trajectory"][name])
        for name in EDGE_NAMES:
            alignment_trajectory[name].append(episode["alignment"][name])
    return {
        "phase": {name: np.concatenate(values, axis=0) for name, values in phase_trajectory.items()},
        "alignment": {
            name: np.asarray(values) for name, values in alignment_trajectory.items()
        },
    }


REHEARSAL_SCHEDULES = {
    "repeat_ab": ("AB",) * 6,
    "repeat_bc": ("BC",) * 6,
    "alternating": ("AB", "BC", "AB", "BC", "AB", "BC"),
    "none": (None,) * 6,
}


def rehearsal(
    world: World,
    config: Config,
    seed_base: int,
    start: str,
) -> dict[str, Any]:
    parent = world.clone()
    if start == "matched_fast":
        rng = np.random.default_rng(seed_base + 777_777)
        for name in BODY_NAMES:
            parent.bodies[name].phases = rng.uniform(-math.pi, math.pi, config.modes)
        for edge in parent.edges.values():
            edge.trace.fill(0.0)
    elif start != "full":
        raise ValueError(f"unknown rehearsal start: {start}")
    output = {}
    for schedule_index, (schedule_name, schedule) in enumerate(REHEARSAL_SCHEDULES.items()):
        branch = parent.clone()
        initial_local = {
            name: body.local_tissue.copy() for name, body in branch.bodies.items()
        }
        initial_edge = {
            name: edge.tissue.copy() for name, edge in branch.edges.items()
        }
        alignments = {name: [] for name in EDGE_NAMES}
        for episode_index, offered in enumerate(schedule):
            result = run_episode(
                branch,
                config,
                offered,
                seed_base + schedule_index * 100_003 + episode_index * 1009,
                plastic=True,
            )
            for name in EDGE_NAMES:
                alignments[name].append(result["alignment"][name])
        output[schedule_name] = {
            "mean_alignment": {
                name: float(np.mean(values)) for name, values in alignments.items()
            },
            "local_tissue_change": {
                name: float(np.linalg.norm(
                    branch.bodies[name].local_tissue - initial_local[name], ord="fro"
                )) for name in BODY_NAMES
            },
            "relation_tissue_change": {
                name: float(np.linalg.norm(
                    branch.edges[name].tissue - initial_edge[name], ord="fro"
                )) for name in EDGE_NAMES
            },
        }
    return output


def _profile_distance(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    alignment_left, alignment_right = [], []
    local_left, local_right = [], []
    edge_left, edge_right = [], []
    for schedule in REHEARSAL_SCHEDULES:
        for edge in EDGE_NAMES:
            alignment_left.append(left[schedule]["mean_alignment"][edge])
            alignment_right.append(right[schedule]["mean_alignment"][edge])
            edge_left.append(left[schedule]["relation_tissue_change"][edge])
            edge_right.append(right[schedule]["relation_tissue_change"][edge])
        for body in BODY_NAMES:
            local_left.append(left[schedule]["local_tissue_change"][body])
            local_right.append(right[schedule]["local_tissue_change"][body])
    return {
        "alignment_rmse": float(np.sqrt(np.mean(
            (np.asarray(alignment_left) - alignment_right) ** 2
        ))),
        "local_tissue_change_rmse": float(np.sqrt(np.mean((np.asarray(local_left) - local_right) ** 2))),
        "relation_tissue_change_rmse": float(np.sqrt(np.mean((np.asarray(edge_left) - edge_right) ** 2))),
    }


def run_history(config: Config, world_seed: int, history_name: str) -> dict[str, Any]:
    world = make_world(config, world_seed)
    checkpoints = {}
    assays = {}
    rehearsals = {}
    edge_references = {
        name: np.zeros((config.modes, config.modes)) for name in EDGE_NAMES
    }
    body_references = {
        name: np.zeros((config.modes, config.modes)) for name in BODY_NAMES
    }

    for edge, token in _formation_schedule(config, history_name):
        run_episode(world, config, edge, _token_seed(world_seed, 1, edge, token))
    edge_references = {
        name: edge.tissue.copy() for name, edge in world.edges.items()
    }
    body_references = {
        name: body.local_tissue.copy() for name, body in world.bodies.items()
    }
    checkpoints["formation"] = _checkpoint(
        world, edge_references, body_references, config
    )
    assays["formation"] = expression_assay(world, config, world_seed + 20_000_000)
    rehearsals["formation"] = {
        start: rehearsal(world, config, world_seed + 30_000_000, start)
        for start in ("full", "matched_fast")
    }

    for cycle in range(config.maintenance_cycles):
        for offset, edge in enumerate(EDGE_NAMES):
            run_episode(
                world, config, edge,
                _token_seed(world_seed, 2, edge, cycle * 2 + offset),
            )
    checkpoints["maintenance"] = _checkpoint(
        world, edge_references, body_references, config
    )
    assays["maintenance"] = expression_assay(world, config, world_seed + 21_000_000)

    for index in range(config.withdrawal_episodes):
        offered = "BC" if index % config.withdrawal_bc_period == 0 else None
        run_episode(world, config, offered, _token_seed(world_seed, 3, offered, index))
    checkpoints["withdrawal"] = _checkpoint(
        world, edge_references, body_references, config
    )
    assays["withdrawal"] = expression_assay(world, config, world_seed + 22_000_000)
    rehearsals["withdrawal"] = {
        start: rehearsal(world, config, world_seed + 31_000_000, start)
        for start in ("full", "matched_fast")
    }

    for index in range(1, config.return_offers + 1):
        run_episode(world, config, "AB", _token_seed(world_seed, 4, "AB", index))
        if index in (1, 3, 6):
            key = f"return_{index}"
            checkpoints[key] = _checkpoint(
                world, edge_references, body_references, config
            )
            assays[key] = expression_assay(
                world, config, world_seed + 23_000_000 + index * 10_000
            )
    rehearsals["return_6"] = {
        start: rehearsal(world, config, world_seed + 32_000_000, start)
        for start in ("full", "matched_fast")
    }

    return {
        "history": history_name,
        "checkpoints": checkpoints,
        "assays": assays,
        "rehearsals": rehearsals,
        "final_state": world,
    }


def _serializable_history(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "final_state"}


def _world_summary(
    config: Config, world_index: int, dt: float | None = None
) -> dict[str, Any]:
    active = Config(**{**asdict(config), "dt": config.dt if dt is None else dt})
    world_seed = active.base_seed + world_index * 10_000_019
    histories = {
        name: run_history(active, world_seed, name)
        for name in ("alternating", "blocked")
    }
    rehearsal_distances = {}
    for checkpoint in ("formation", "withdrawal", "return_6"):
        rehearsal_distances[checkpoint] = {
            start: _profile_distance(
                histories["alternating"]["rehearsals"][checkpoint][start],
                histories["blocked"]["rehearsals"][checkpoint][start],
            ) for start in ("full", "matched_fast")
        }
    return {
        "world": world_index,
        "seed": world_seed,
        "dt": active.dt,
        "histories": {
            name: _serializable_history(result) for name, result in histories.items()
        },
        "paired_rehearsal_distances": rehearsal_distances,
        "checks": {
            "no_ac_edge": all("AC" not in result["final_state"].edges for result in histories.values()),
            "maximum_edge_capacity_fraction": float(max(
                checkpoint["edge"][edge]["capacity_fraction"]
                for result in histories.values()
                for checkpoint in result["checkpoints"].values()
                for edge in EDGE_NAMES
            )),
            "maximum_local_capacity_fraction": float(max(
                checkpoint["body"][body]["local_capacity_fraction"]
                for result in histories.values()
                for checkpoint in result["checkpoints"].values()
                for body in BODY_NAMES
            )),
            "maximum_expression_erasure_error": float(max(
                max(
                    assay["ab_off_vs_erased"]["maximum_phase_error"],
                    assay["ab_off_vs_erased"]["maximum_alignment_error"],
                )
                for result in histories.values()
                for assay in result["assays"].values()
            )),
        },
    }


def _distribution(values: list[float | None]) -> dict[str, float | int | None]:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return {
            "defined": 0,
            "median": None,
            "q25": None,
            "q75": None,
            "minimum": None,
            "maximum": None,
        }
    array = np.asarray(finite, dtype=float)
    return {
        "defined": len(finite),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def aggregate(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "worlds": len(worlds),
        "checks": {
            "all_no_ac_edge": all(world["checks"]["no_ac_edge"] for world in worlds),
            "maximum_edge_capacity_fraction": max(
                world["checks"]["maximum_edge_capacity_fraction"] for world in worlds
            ),
            "maximum_local_capacity_fraction": max(
                world["checks"]["maximum_local_capacity_fraction"] for world in worlds
            ),
            "maximum_expression_erasure_error": max(
                world["checks"]["maximum_expression_erasure_error"] for world in worlds
            ),
        },
        "edge_trajectories": {},
        "body_trajectories": {},
        "expression_effects": {},
        "rehearsal_distances": {},
    }
    for history in ("alternating", "blocked"):
        output["edge_trajectories"][history] = {}
        for edge in EDGE_NAMES:
            output["edge_trajectories"][history][edge] = {}
            for checkpoint in CHECKPOINTS:
                output["edge_trajectories"][history][edge][checkpoint] = {
                    "tissue_norm": _distribution([
                        world["histories"][history]["checkpoints"][checkpoint]["edge"][edge]["tissue_norm"]
                        for world in worlds
                    ]),
                    "orientation_to_formation": _distribution([
                        world["histories"][history]["checkpoints"][checkpoint]["edge"][edge]["orientation_to_formation"]
                        for world in worlds
                    ]),
                    "distance_from_formation": _distribution([
                        world["histories"][history]["checkpoints"][checkpoint]["edge"][edge]["distance_from_formation"]
                        for world in worlds
                    ]),
                }
        output["expression_effects"][history] = {}
        output["body_trajectories"][history] = {}
        for body in BODY_NAMES:
            output["body_trajectories"][history][body] = {}
            for checkpoint in CHECKPOINTS:
                state = [
                    world["histories"][history]["checkpoints"][checkpoint]["body"][body]
                    for world in worlds
                ]
                output["body_trajectories"][history][body][checkpoint] = {
                    "local_norm": _distribution([item["local_norm"] for item in state]),
                    "local_orientation_to_formation": _distribution([
                        item["local_orientation_to_formation"] for item in state
                    ]),
                    "local_distance_from_formation": _distribution([
                        item["local_distance_from_formation"] for item in state
                    ]),
                }
        for checkpoint in CHECKPOINTS:
            output["expression_effects"][history][checkpoint] = {}
            for condition in ("ab_off", "bc_off", "ab_to_a_off", "ab_to_b_off"):
                output["expression_effects"][history][checkpoint][condition] = {
                    body: _distribution([
                        world["histories"][history]["assays"][checkpoint][condition]["body_phase_rms"][body]
                        for world in worlds
                    ]) for body in BODY_NAMES
                }
    for checkpoint in ("formation", "withdrawal", "return_6"):
        output["rehearsal_distances"][checkpoint] = {}
        for start in ("full", "matched_fast"):
            output["rehearsal_distances"][checkpoint][start] = {
                metric: _distribution([
                    world["paired_rehearsal_distances"][checkpoint][start][metric]
                    for world in worlds
                ])
                for metric in (
                    "alignment_rmse",
                    "local_tissue_change_rmse",
                    "relation_tissue_change_rmse",
                )
            }
    return output


def _parallel_worlds(
    config: Config, jobs: list[tuple[int, float | None]], workers: int
) -> list[dict[str, Any]]:
    if workers <= 1:
        return [_world_summary(config, index, dt=dt) for index, dt in jobs]
    completed = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_world_summary, config, index, dt): (index, dt)
            for index, dt in jobs
        }
        for count, future in enumerate(as_completed(futures), start=1):
            completed.append(future.result())
            print(f"completed {count}/{len(jobs)}", flush=True)
    return sorted(completed, key=lambda item: item["world"])


def run(config: Config, workers: int = 1) -> dict[str, Any]:
    worlds = _parallel_worlds(
        config,
        [(index, None) for index in range(config.worlds)],
        workers,
    )
    fine_worlds = _parallel_worlds(
        config,
        [(index, config.reference_dt) for index in range(min(4, config.worlds))],
        min(workers, 4),
    )
    refinements = []
    for fine in fine_worlds:
        index = fine["world"]
        coarse = worlds[index]
        refinements.append({
            "world": index,
            "coarse_dt": coarse["dt"],
            "fine_dt": fine["dt"],
            "coarse": {
                "paired_rehearsal_distances": coarse["paired_rehearsal_distances"],
                "histories": {
                    history: {
                        "checkpoints": coarse["histories"][history]["checkpoints"],
                        "assays": coarse["histories"][history]["assays"],
                    } for history in ("alternating", "blocked")
                },
            },
            "fine": {
                "paired_rehearsal_distances": fine["paired_rehearsal_distances"],
                "histories": {
                    history: {
                        "checkpoints": fine["histories"][history]["checkpoints"],
                        "assays": fine["histories"][history]["assays"],
                    } for history in ("alternating", "blocked")
                },
            },
        })
    return {
        "protocol": asdict(config),
        "worlds": worlds,
        "aggregate": aggregate(worlds),
        "refinement": refinements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--worlds", type=int, default=24)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = Config(worlds=args.worlds)
    result = run(config, workers=max(1, args.workers))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output / "aggregate.json").write_text(
        json.dumps(result["aggregate"], indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["aggregate"]["checks"], indent=2))


if __name__ == "__main__":
    main()
