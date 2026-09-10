from dataclasses import asdict

import pytest

from adaptive_response import (
    Edge,
    Environment,
    IncidentConfig,
    MissionAction,
    MissionAllocation,
    Site,
)


def make_incident(seed: int = 20260910, q: float = 0.25) -> IncidentConfig:
    sites = [
        Site(
            id=f"site_{i:02d}",
            x=float(i),
            y=0.0,
            habitat_score=(i % 5) / 4,
            q_model=q,
        )
        for i in range(20)
    ]
    edges = [
        Edge(
            src=f"site_{i:02d}",
            dst=f"site_{i + 1:02d}",
            distance=1.0,
            connectivity_weight=1.0,
        )
        for i in range(19)
    ]
    return IncidentConfig(
        sites=sites,
        edges=edges,
        initial_detection="site_09",
        budget=30,
        teams=2,
        protocol="binary_detection",
        seed=seed,
        world_model_id="toy_graph_cluster_m1",
    )


def test_step_consumes_budget_and_updates_public_effort() -> None:
    env = Environment(make_incident())
    env.reset(seed=123)
    action = MissionAction(
        allocations=(
            MissionAllocation(site_id="site_08", effort_units=3, team_id="A"),
            MissionAllocation(site_id="site_10", effort_units=2, team_id="B"),
        ),
        total_cost=5,
    )

    batch, done, metrics = env.step(action)
    state = env.current_public_state

    assert batch.round == 1
    assert batch.total_effort == 5
    assert len(batch.observations) == 2
    assert state.remaining_budget == 25
    assert state.round == 1
    assert done is False
    assert metrics["effort_spent"] == 5
    assert metrics["remaining_budget"] == 25
    assert next(site for site in state.sites if site.id == "site_08").observed_effort == 3
    assert next(site for site in state.sites if site.id == "site_10").observed_effort == 2


def test_effort_blocks_for_same_site_are_aggregated() -> None:
    env = Environment(make_incident())
    env.reset(seed=123)
    action = MissionAction(
        allocations=(
            MissionAllocation(site_id="site_08", effort_units=2, team_id="A"),
            MissionAllocation(site_id="site_08", effort_units=3, team_id="B"),
        ),
        total_cost=5,
    )

    batch, _, _ = env.step(action)

    assert len(batch.observations) == 1
    assert batch.observations[0].site_id == "site_08"
    assert batch.observations[0].effort == 5


def test_q_one_detects_known_occupied_initial_site() -> None:
    env = Environment(make_incident(q=1.0))
    env.reset(seed=7)
    action = MissionAction(
        allocations=(MissionAllocation(site_id="site_09", effort_units=1),),
        total_cost=1,
    )

    batch, _, _ = env.step(action)

    assert batch.observations[0].detection is True


def test_unoccupied_site_has_no_false_positive_in_mvp() -> None:
    env = Environment(make_incident(q=1.0))
    env.reset(seed=7)
    # site_00 is graph distance 9 from initial site_09, while the temporary M1
    # hidden-world radius is at most 3, so it is guaranteed unoccupied.
    action = MissionAction(
        allocations=(MissionAllocation(site_id="site_00", effort_units=10),),
        total_cost=10,
    )

    batch, _, _ = env.step(action)

    assert batch.observations[0].detection is False


def test_observation_payload_does_not_leak_hidden_truth() -> None:
    env = Environment(make_incident())
    env.reset(seed=123)
    action = MissionAction(
        allocations=(MissionAllocation(site_id="site_08", effort_units=2),),
        total_cost=2,
    )

    batch, _, metrics = env.step(action)
    payload = asdict(batch)

    assert "occupied" not in repr(payload).lower()
    assert "occupied" not in repr(metrics).lower()
    assert all("occupied" not in key.lower() for key in batch.observations[0].metadata)


def test_same_seed_and_same_action_trajectory_is_reproducible() -> None:
    env_a = Environment(make_incident())
    env_b = Environment(make_incident())
    env_a.reset(seed=456)
    env_b.reset(seed=456)

    actions = [
        MissionAction(
            allocations=(MissionAllocation(site_id="site_08", effort_units=3),),
            total_cost=3,
        ),
        MissionAction(
            allocations=(MissionAllocation(site_id="site_10", effort_units=4),),
            total_cost=4,
        ),
    ]

    results_a = [env_a.step(action)[0] for action in actions]
    results_b = [env_b.step(action)[0] for action in actions]

    assert results_a == results_b
    assert env_a.current_public_state == env_b.current_public_state


def test_over_budget_action_is_rejected_without_state_change() -> None:
    env = Environment(make_incident())
    before = env.reset(seed=123)
    action = MissionAction(
        allocations=(MissionAllocation(site_id="site_08", effort_units=31),),
        total_cost=31,
    )

    with pytest.raises(ValueError, match="remaining budget"):
        env.step(action)

    assert env.current_public_state == before


def test_total_cost_must_match_allocated_effort_in_m1() -> None:
    env = Environment(make_incident())
    env.reset(seed=123)
    action = MissionAction(
        allocations=(MissionAllocation(site_id="site_08", effort_units=3),),
        total_cost=2,
    )

    with pytest.raises(ValueError, match="must equal allocated effort"):
        env.step(action)
