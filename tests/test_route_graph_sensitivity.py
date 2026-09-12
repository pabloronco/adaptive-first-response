from __future__ import annotations

import pytest

from adaptive_response.route_graph_sensitivity import (
    compare_route_distance_topologies,
    extract_site_snap_distances,
)


def _sites() -> list[dict[str, object]]:
    return [
        {"site_id": "A", "latitude": 48.0, "longitude": -122.0},
        {"site_id": "B", "latitude": 48.0, "longitude": -122.1},
        {"site_id": "C", "latitude": 48.0, "longitude": -122.2},
        {"site_id": "D", "latitude": 48.0, "longitude": -122.3},
    ]


def _rows() -> list[dict[str, object]]:
    return [
        {
            "src": "A",
            "dst": "B",
            "distance_km": 8.0,
            "candidate_local_radius": True,
            "candidate_sparse_review": False,
            "salishseacast_route_found": True,
            "salishseacast_water_route_km": 9.0,
            "salishseacast_detour_ratio": 1.125,
            "salishseacast_src_snap_km": 0.2,
            "salishseacast_dst_snap_km": 0.3,
        },
        {
            "src": "B",
            "dst": "C",
            "distance_km": 10.0,
            "candidate_local_radius": True,
            "candidate_sparse_review": False,
            "salishseacast_route_found": True,
            "salishseacast_water_route_km": 24.0,
            "salishseacast_detour_ratio": 2.4,
            "salishseacast_src_snap_km": 0.3,
            "salishseacast_dst_snap_km": 1.4,
        },
        {
            "src": "C",
            "dst": "D",
            "distance_km": 12.0,
            "candidate_local_radius": True,
            "candidate_sparse_review": False,
            "salishseacast_route_found": True,
            "salishseacast_water_route_km": 42.0,
            "salishseacast_detour_ratio": 3.5,
            "salishseacast_src_snap_km": 1.4,
            "salishseacast_dst_snap_km": 0.4,
        },
        {
            "src": "A",
            "dst": "D",
            "distance_km": 40.0,
            "candidate_local_radius": False,
            "candidate_sparse_review": True,
            "salishseacast_route_found": True,
            "salishseacast_water_route_km": 45.0,
            "salishseacast_detour_ratio": 1.125,
            "salishseacast_src_snap_km": 0.2,
            "salishseacast_dst_snap_km": 0.4,
        },
    ]


def test_route_thresholds_change_only_primary_local_edges() -> None:
    summary = compare_route_distance_topologies(
        _sites(), _rows(), route_thresholds_km=(20.0, 30.0, 50.0)
    )
    variants = summary["topology_variants"]
    assert variants["all_local_with_route"]["edges"] == 3
    assert variants["route_le_20km"]["edges"] == 1
    assert variants["route_le_30km"]["edges"] == 2
    assert variants["route_le_50km"]["edges"] == 3
    assert summary["review_only_edges"] == 1


def test_snap_quality_is_reported_not_used_as_hidden_filter() -> None:
    summary = compare_route_distance_topologies(_sites(), _rows())
    assert summary["site_snap"]["sites_gt_1km"] == 1
    assert summary["local_route"]["edges_with_endpoint_snap_gt_1km"] == 2
    assert summary["topology_variants"]["all_local_with_route"]["edges"] == 3


def test_extract_site_snap_distances_requires_consistency() -> None:
    rows = _rows()
    rows[1]["salishseacast_src_snap_km"] = 0.9
    with pytest.raises(ValueError, match="Inconsistent snap distance"):
        extract_site_snap_distances(rows)


def test_detour_summary_flags_extreme_routes() -> None:
    summary = compare_route_distance_topologies(_sites(), _rows())
    route = summary["local_route"]
    assert route["detour_ratio_gt_2"] == 2
    assert route["detour_ratio_gt_3"] == 1
    assert route["detour_ratio_gt_5"] == 0
    assert route["most_extreme_detours"][0]["src"] == "C"
    assert route["most_extreme_detours"][0]["dst"] == "D"
