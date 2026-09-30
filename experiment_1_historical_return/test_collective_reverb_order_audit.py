from dataclasses import replace

from collective_reverb_experiment import CollectiveConfig
from collective_reverb_order_audit import POLICIES, schedule_orders
from collective_reverb_scaling import SweepCell, run_scaled_seed


def test_order_policies_are_permutations_of_the_same_tokens():
    orders = schedule_orders(body_count=6, episodes_per_body=3, seed=42)
    assert tuple(orders) == POLICIES
    expected = list(range(12))
    for order in orders.values():
        assert sorted(order) == expected
    assert len(set(orders.values())) == len(POLICIES)


def test_default_order_reproduces_explicit_blocked_protocol():
    config = replace(
        CollectiveConfig(seeds=1),
        sediment_episodes_ab=2,
        sediment_episodes_bc=2,
        participation_time=2.0,
        encounter_time=3.0,
    )
    cell = SweepCell("order", 4.0, 12, 4, 12, 1.44)
    default, _ = run_scaled_seed(config, cell, 0)
    blocked = schedule_orders(4, config.sediment_episodes_bc, 9)["blocked"]
    explicit, _ = run_scaled_seed(config, cell, 0, secondary_order=blocked)
    ignored = {"runtime_seconds"}
    assert {
        key: value for key, value in default.items() if key not in ignored
    } == {
        key: value for key, value in explicit.items() if key not in ignored
    }
