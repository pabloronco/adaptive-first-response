from dataclasses import asdict

import pytest

from adaptive_response import Edge, Environment, IncidentConfig, Site


def make_incident(num_sites: int = 20, seed: int = 20260910) -> IncidentConfig:
    sites = [
        Site(
            id=f"site_{i:02d}",
            x=float(i),
            y=0.0,
            habitat_score=(i % 5) / 4,
            q_model=0.25,
        )
        for i in range(num_sites)
    ]
    edges = [
        Edge(
            src=f"site_{i:02d}",
            dst=f"site_{i + 1:02d}",
            distance=1.0,
            connectivity_weight=1.0,
        )
        for i in range(num_sites - 1)
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


def test_reset_returns_expected_public_incident_state() -> None:
    env = Environment(make_incident())

    state = env.reset()

    assert len(state.sites) == 20
    assert len(state.edges) == 19
    assert state.initial_detection == "site_09"
    assert state.remaining_budget == 30
    assert state.teams == 2
    assert state.protocol == "binary_detection"
    assert state.round == 0


def test_initial_detection_is_explicit_but_preincident_effort_is_not_invented() -> None:
    env = Environment(make_incident())

    state = env.reset()
    initial_site = next(
        site for site in state.sites if site.id == state.initial_detection
    )

    assert initial_site.status == "confirmed_detection"
    assert initial_site.detections == 1
    assert initial_site.observed_effort == 0


def test_public_state_does_not_expose_hidden_occupancy() -> None:
    env = Environment(make_incident())

    state = env.reset()
    payload = asdict(state)

    assert "hidden_world" not in payload
    assert "occupied_by_site" not in payload
    assert all("occupied" not in key.lower() for key in payload)
    assert not hasattr(state, "hidden_world")
    assert not hasattr(state, "occupied_by_site")


def test_same_seed_produces_same_hidden_world_for_evaluator_tests() -> None:
    env_a = Environment(make_incident())
    env_b = Environment(make_incident())

    env_a.reset(seed=12345)
    env_b.reset(seed=12345)

    # Unit tests are allowed to inspect simulator-private state. Planner/public
    # code must not receive this object.
    assert env_a._hidden_world is not None
    assert env_b._hidden_world is not None
    assert env_a._hidden_world == env_b._hidden_world


def test_reset_returns_a_snapshot_not_internal_mutable_state() -> None:
    env = Environment(make_incident())

    returned = env.reset()
    returned.remaining_budget = 0
    returned.sites[0].status = "tampered"

    internal_public = env.current_public_state
    assert internal_public.remaining_budget == 30
    assert internal_public.sites[0].status != "tampered"


def test_invalid_initial_detection_is_rejected() -> None:
    config = make_incident()
    config.initial_detection = "missing_site"

    with pytest.raises(ValueError, match="initial_detection"):
        Environment(config)
