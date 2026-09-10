from dataclasses import asdict

import pytest

from adaptive_response import (
    AdaptiveMissionLoop,
    Edge,
    Environment,
    FrontierPlanner,
    IncidentConfig,
    LoopPhase,
    Site,
)


def build_incident(*, budget: int = 6, seed: int = 20260910) -> IncidentConfig:
    sites = [
        Site(
            id=f"site_{i:02d}",
            x=float(i),
            y=0.0,
            habitat_score=0.5,
            q_model=0.25,
        )
        for i in range(7)
    ]
    edges = [
        Edge(
            src=f"site_{i:02d}",
            dst=f"site_{i + 1:02d}",
            distance=1.0,
            connectivity_weight=1.0,
        )
        for i in range(6)
    ]
    return IncidentConfig(
        sites=sites,
        edges=edges,
        initial_detection="site_03",
        budget=budget,
        teams=2,
        protocol="binary_detection",
        seed=seed,
        world_model_id="toy_graph_cluster_m1",
    )


def build_prior() -> dict[str, float]:
    prior = {f"site_{i:02d}": 0.20 for i in range(7)}
    prior["site_02"] = 0.60
    prior["site_04"] = 0.70
    return prior


def build_loop(*, budget: int = 6, seed: int = 20260910) -> AdaptiveMissionLoop:
    return AdaptiveMissionLoop(
        Environment(build_incident(budget=budget, seed=seed)),
        FrontierPlanner(effort_per_site=3, max_sites=1),
        build_prior(),
    )


def test_reset_builds_observable_belief_and_graph_state() -> None:
    loop = build_loop()

    graph = loop.reset()

    assert loop.phase is LoopPhase.READY_TO_PLAN
    assert loop.current_public_state.round == 0
    assert loop.current_public_state.remaining_budget == 6
    assert loop.current_belief.p_by_site["site_03"] == 1.0
    assert graph.node_ids == tuple(f"site_{i:02d}" for i in range(7))
    assert "occupied_by_site" not in asdict(loop.current_graph_state)


def test_reveal_is_locked_before_episode_completion() -> None:
    loop = build_loop()
    loop.reset()

    with pytest.raises(RuntimeError, match="only available after"):
        loop.reveal()


def test_pending_mission_is_displayed_then_executed_exactly() -> None:
    loop = build_loop()
    loop.reset()

    mission = loop.plan_next()
    assert loop.phase is LoopPhase.MISSION_PLANNED
    assert [(a.site_id, a.effort_units) for a in mission.allocations] == [
        ("site_04", 3)
    ]

    transition = loop.execute_pending()

    assert transition.mission == mission
    assert transition.observations.observations[0].site_id == "site_04"
    assert transition.observations.observations[0].effort == 3


def test_real_simulator_evidence_changes_belief_and_next_mission() -> None:
    loop = build_loop()
    loop.reset()

    transition = loop.run_round()

    assert transition.mission.allocations[0].site_id == "site_04"
    observation = transition.observations.observations[0]
    assert observation.site_id == "site_04"
    assert observation.detection is False
    assert transition.belief_before.p_by_site["site_04"] == pytest.approx(0.70)
    assert transition.belief_after.p_by_site["site_04"] < 0.60
    assert transition.next_mission is not None
    assert transition.next_mission.allocations[0].site_id == "site_02"
    assert transition.public_state_after.remaining_budget == 3
    assert loop.phase is LoopPhase.MISSION_PLANNED


def test_two_round_episode_reaches_complete_and_reveal() -> None:
    loop = build_loop()
    loop.reset()

    first = loop.run_round()
    second = loop.run_round()

    assert first.done is False
    assert second.done is True
    assert second.next_mission is None
    assert second.public_state_after.remaining_budget == 0
    assert loop.phase is LoopPhase.COMPLETE

    hidden = loop.reveal()

    assert loop.phase is LoopPhase.REVEALED
    assert hidden.occupied_by_site["site_03"] is True
    assert set(hidden.occupied_by_site) == set(build_prior())


def test_observable_transition_contains_no_hidden_truth() -> None:
    loop = build_loop()
    loop.reset()

    transition = loop.run_round()
    payload = asdict(transition)

    assert "hidden_world" not in payload
    assert "occupied_by_site" not in payload
    assert "generator_parameters" not in payload


def test_same_seed_produces_same_full_loop_trajectory() -> None:
    first_loop = build_loop(seed=20260910)
    second_loop = build_loop(seed=20260910)
    first_loop.reset()
    second_loop.reset()

    first_transitions = [first_loop.run_round(), first_loop.run_round()]
    second_transitions = [second_loop.run_round(), second_loop.run_round()]

    assert [t.mission for t in first_transitions] == [
        t.mission for t in second_transitions
    ]
    assert [t.observations for t in first_transitions] == [
        t.observations for t in second_transitions
    ]
    assert first_loop.reveal() == second_loop.reveal()


def test_reset_after_reveal_returns_clean_initial_state_and_relocks_truth() -> None:
    loop = build_loop()
    loop.reset()
    loop.run_round()
    loop.run_round()
    loop.reveal()

    loop.reset()

    assert loop.phase is LoopPhase.READY_TO_PLAN
    assert loop.current_public_state.round == 0
    assert loop.current_public_state.remaining_budget == 6
    assert loop.current_belief.observed_history == ()
    assert loop.current_belief.p_by_site["site_04"] == pytest.approx(0.70)
    with pytest.raises(RuntimeError, match="only available after"):
        loop.reveal()
