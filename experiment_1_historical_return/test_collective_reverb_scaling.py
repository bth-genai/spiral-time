import numpy as np

from collective_reverb_experiment import CollectiveConfig, make_world
from collective_reverb_scaling import (
    SweepCell,
    _extend_world,
    build_cells,
    run_scaling,
)


def quick_config() -> CollectiveConfig:
    return CollectiveConfig(
        seeds=1,
        sediment_episodes_ab=1,
        sediment_episodes_bc=1,
        participation_time=1.0,
        encounter_time=1.5,
        seed=20289911,
    )


def test_compressed_tissue_supports_more_modes_than_channels():
    config = CollectiveConfig(mode_count=12, tissue_channels=8)
    world = make_world(config, config.seed)
    assert world.bodies["A"].interface.shape == (8, 12)
    norms = np.linalg.norm(world.bodies["A"].interface, axis=0)
    assert np.allclose(norms, 1.0)


def test_extra_bodies_do_not_change_existing_world():
    config = quick_config()
    world = make_world(config, config.seed)
    extended = _extend_world(config, world, 6, config.seed)
    assert set(extended.bodies) == {"A", "B", "C", "D", "E", "F"}
    assert np.array_equal(world.initial_tissue, extended.initial_tissue)
    for name in ("A", "B", "C"):
        assert np.array_equal(
            world.bodies[name].interface, extended.bodies[name].interface
        )


def test_quick_scaling_is_deterministic_and_paired():
    config = quick_config()
    cell = SweepCell("bodies", 3.0, 12, 3, 12, 0.72)
    trials_a, probes_a, summary_a = run_scaling(config, [cell])
    trials_b, probes_b, summary_b = run_scaling(config, [cell])
    comparable = [column for column in trials_a if column != "runtime_seconds"]
    assert trials_a[comparable].equals(trials_b[comparable])
    assert probes_a.equals(probes_b)
    summary_comparable = [
        column for column in summary_a if column != "mean_runtime_seconds"
    ]
    assert summary_a[summary_comparable].equals(
        summary_b[summary_comparable]
    )
    assert np.allclose(trials_a["distributed_ab"], trials_a["repeated_ab"])
    assert len(build_cells("all")) == 16

