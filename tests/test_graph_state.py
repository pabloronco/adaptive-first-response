from dataclasses import asdict

import pytest

from adaptive_response import (
    EDGE_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    BeliefEngine,
    Edge,
    Environment,
    GraphStateExporter,
    IncidentConfig,
    Site,
)


def make_incident(num_sites: int = 4) -> IncidentConfig:
    sites = [
        Site(
            id=f"site_{i:02d}",
            x=float(i),
            y=0.0,
            habitat_score=(i - 1) / max(num_sites - 1, 1),
            q_model=0.25,
            access_cost=None if i % 2 else float(i),
        )
        for i in range(1, num_sites + 1)
    ]
    edges = [
        Edge(
            src=f"site_{i:02d}",
            dst=f"site_{i + 1:02d}",
            distance=float(i),
            connectivity_weight=0.5,
        )
        for i in range(1, num_sites)
    ]
    return IncidentConfig(
        sites=sites,
        edges=edges,
        initial_detection="site_01",
        budget=30,
        teams=2,
        protocol="binary_detection",
        seed=20260910,
        world_model_id="toy_graph_cluster_m1",
    )


def make_public_and_belief(num_sites: int = 4):
    env = Environment(make_incident(num_sites))
    public = env.reset(seed=123)
    prior = {site.id: 0.5 for site in public.sites}
    belief = BeliefEngine.initialize(prior, confirmed_sites={public.initial_detection})
    return public, belief


def test_export_has_stable_node_ids_and_feature_order() -> None:
    public, belief = make_public_and_belief()

    graph = GraphStateExporter().export(public, belief)

    assert graph.node_ids == ("site_01", "site_02", "site_03", "site_04")
    assert NODE_FEATURE_NAMES == (
        "belief",
        "uncertainty",
        "observed_effort",
        "detections",
        "habitat_score",
        "access_cost",
        "frontier",
    )
    assert all(len(row) == len(NODE_FEATURE_NAMES) for row in graph.node_features)

    first = dict(zip(NODE_FEATURE_NAMES, graph.node_features[0], strict=True))
    assert first["belief"] == 1.0
    assert first["uncertainty"] == 0.0
    assert first["observed_effort"] == 0.0
    assert first["detections"] == 1.0
    assert first["frontier"] == 0.0


def test_frontier_is_publicly_derived_from_positive_adjacency() -> None:
    public, belief = make_public_and_belief()

    graph = GraphStateExporter().export(public, belief)
    frontier_idx = NODE_FEATURE_NAMES.index("frontier")
    frontier = {
        site_id: graph.node_features[i][frontier_idx]
        for i, site_id in enumerate(graph.node_ids)
    }

    assert frontier == {
        "site_01": 0.0,
        "site_02": 1.0,
        "site_03": 0.0,
        "site_04": 0.0,
    }


def test_edge_index_and_features_are_aligned_and_bidirectional() -> None:
    public, belief = make_public_and_belief()

    graph = GraphStateExporter().export(public, belief)

    assert EDGE_FEATURE_NAMES == ("distance", "connectivity_weight")
    assert graph.edge_index == (
        (0, 1, 1, 2, 2, 3),
        (1, 0, 2, 1, 3, 2),
    )
    assert len(graph.edge_features) == 6
    assert graph.edge_features[0] == (1.0, 0.5)
    assert graph.edge_features[1] == (1.0, 0.5)
    assert graph.edge_features[2] == (2.0, 0.5)


def test_global_features_and_mask_are_explicit() -> None:
    public, belief = make_public_and_belief()

    graph = GraphStateExporter().export(public, belief)

    assert GLOBAL_FEATURE_NAMES == (
        "remaining_budget",
        "round",
        "team_capacity",
        "global_uncertainty",
    )
    globals_by_name = dict(
        zip(GLOBAL_FEATURE_NAMES, graph.global_features, strict=True)
    )
    assert globals_by_name["remaining_budget"] == 30.0
    assert globals_by_name["round"] == 0.0
    assert globals_by_name["team_capacity"] == 2.0
    assert globals_by_name["global_uncertainty"] == pytest.approx(0.75)
    assert graph.feasibility_mask == (True, True, True, True)


def test_no_hidden_truth_or_q_is_embedded_in_graph_state() -> None:
    public, belief = make_public_and_belief()

    graph = GraphStateExporter().export(public, belief)
    payload = asdict(graph)

    assert "hidden_world" not in payload
    assert "occupied_by_site" not in payload
    assert "q_by_site" not in payload
    assert "q" not in NODE_FEATURE_NAMES


def test_planner_constraints_expose_q_separately() -> None:
    public, _ = make_public_and_belief()

    constraints = GraphStateExporter.planner_constraints(public)

    assert constraints == {
        "q_by_site": {
            "site_01": 0.25,
            "site_02": 0.25,
            "site_03": 0.25,
            "site_04": 0.25,
        }
    }


def test_belief_change_is_reflected_in_graph_state_without_raw_evidence_semantics() -> None:
    public, belief = make_public_and_belief()
    exporter = GraphStateExporter()
    before = exporter.export(public, belief)

    from adaptive_response import Observation, ObservationBatch

    obs = Observation(site_id="site_03", effort=10, detection=False, round=1)
    updated = BeliefEngine.update(
        belief,
        ObservationBatch(observations=(obs,), round=1, total_effort=10),
        {site.id: 0.25 for site in public.sites},
    )
    after = exporter.export(public, updated)

    belief_idx = NODE_FEATURE_NAMES.index("belief")
    site_03_idx = before.node_ids.index("site_03")
    assert after.node_features[site_03_idx][belief_idx] < before.node_features[site_03_idx][belief_idx]


def test_graph_size_variation_does_not_break_export() -> None:
    exporter = GraphStateExporter()

    for n in (12, 20, 24, 30):
        public, belief = make_public_and_belief(n)
        graph = exporter.export(public, belief)

        assert len(graph.node_ids) == n
        assert len(graph.node_features) == n
        assert len(graph.feasibility_mask) == n
        assert len(graph.edge_features) == 2 * (n - 1)
        assert len(graph.edge_index[0]) == len(graph.edge_features)
        assert len(graph.edge_index[1]) == len(graph.edge_features)
