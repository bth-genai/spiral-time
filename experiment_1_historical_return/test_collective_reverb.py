import numpy as np

from collective_reverb_experiment import (
    CollectiveConfig,
    Tissue,
    _project_capacity,
    make_world,
    probe,
    run_experiment,
    summarize,
)


def quick_config() -> CollectiveConfig:
    return CollectiveConfig(
        seeds=2,
        sediment_episodes_ab=2,
        sediment_episodes_bc=2,
        participation_time=2.0,
        encounter_time=3.0,
    )


def test_capacity_projection_is_hermitian_and_bounded():
    matrix = np.asarray([[1 + 2j, 4 - 1j], [2 + 3j, -3 + 1j]])
    projected = _project_capacity(matrix, 0.75)
    assert np.allclose(projected, projected.conj().T)
    assert np.linalg.norm(projected) <= 0.75 + 1.0e-12


def test_one_way_probe_cannot_enact_reciprocal_carrying():
    config = quick_config()
    world = make_world(config, config.seed)
    tissue = Tissue(world.initial_tissue.copy(), config.finite_capacity)
    result = probe(
        config,
        tissue,
        world.bodies["A"],
        world.bodies["B"],
        config.seed + 1,
        0.97,
        feedback=(1.0, 0.0),
    )
    assert result["status"] != "enacted"


def test_quick_protocol_is_paired_and_deterministic():
    config = quick_config()
    probes_a, tissues_a = run_experiment(config)
    probes_b, tissues_b = run_experiment(config)
    assert probes_a.equals(probes_b)
    assert tissues_a.equals(tissues_b)
    assert set(tissues_a["condition"]) == {
        "shared_finite",
        "partitioned",
        "shared_expanded",
    }
    assert len(probes_a) == config.seeds * 11
    summary = summarize(config, probes_a, tissues_a)
    assert summary["criteria"]["no_pair_indexed_memory"] is True

