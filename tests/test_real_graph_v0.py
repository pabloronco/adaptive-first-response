from __future__ import annotations

from adaptive_response.real_graph_v0 import (
    build_real_graph_v0_audit,
    select_real_graph_v0_edges,
)


def _sites() -> list[dict[str, object]]:
    return [
        {"site_id": "A", "latitude": 48.0, "longitude": -122.00},
        {"site_id": "B", "latitude": 48.0, "longitude": -122.01},
        {"site_id": "C", "latitude": 48.0, "longitude": -122.02},
        {"site_id": "D", "latitude": 48.0, "longitude": -122.03},
    ]


def _rows() -> list[dict[str, object]]:
    return [
        {
            "src": "A",
            "dst": "B",
            "distance_km": 1.0,
            "candidate_local_radius": True,
            "candidate_sparse_review": False,
            "straight_marine_fraction": 1.0,
        },
        {
            "src": "B",
            "dst": "C",
            "distance_km": 1.0,
            "candidate_local_radius": True,
            "candidate_sparse_review": False,
            "straight_marine_fraction": 0.2,
        },
        {
            "src": "C",
            "dst": "D",
            "distance_km": 1.0,
            "candidate_local_radius": True,
            "candidate_sparse_review": False,
            "straight_marine_fraction": 0.0,
        },
        {
            "src": "A",
            "dst": "D",
            "distance_km": 30.0,
            "candidate_local_radius": False,
            "candidate_sparse_review": True,
            "straight_marine_fraction": 1.0,
        },
    ]


def test_primary_keeps_nonzero_local_edges_without_strong_threshold() -> None:
    primary, withheld, review = select_real_graph_v0_edges(_rows())
    pairs = {(row["src"], row["dst"]) for row in primary}
    assert pairs == {("A", "B"), ("B", "C")}
    assert len(withheld) == 1
    assert len(review) == 1


def test_zero_water_local_edge_is_withheld_not_deleted_from_audit() -> None:
    _, withheld, _ = select_real_graph_v0_edges(_rows())
    assert withheld[0]["v0_edge_status"] == "withheld_zero_straight_water_support"
    assert withheld[0]["v0_primary"] is False


def test_sparse_review_edge_never_repairs_primary_connectivity() -> None:
    primary, _, review = select_real_graph_v0_edges(_rows())
    assert all(row["candidate_local_radius"] for row in primary)
    assert review[0]["src"] == "A"
    assert review[0]["dst"] == "D"
    assert review[0]["v0_primary"] is False


def test_audit_reports_primary_topology_and_open_connectivity_weight() -> None:
    _, summary = build_real_graph_v0_audit(_sites(), _rows())
    assert summary["primary_edges"] == 2
    assert summary["withheld_zero_water_local_edges"] == 1
    assert summary["sampling_isolation_review_only_edges"] == 1
    assert summary["primary_graph"]["components"] == 2
    assert summary["primary_graph"]["isolated_sites"] == ["D"]
    assert any("Connectivity weight remains OPEN" in note for note in summary["notes"])
