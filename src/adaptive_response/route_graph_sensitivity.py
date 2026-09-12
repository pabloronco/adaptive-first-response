from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from .real_graph import graph_summary


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_water_route_edges(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "src",
            "dst",
            "distance_km",
            "candidate_local_radius",
            "candidate_sparse_review",
            "salishseacast_route_found",
            "salishseacast_water_route_km",
            "salishseacast_detour_ratio",
            "salishseacast_src_snap_km",
            "salishseacast_dst_snap_km",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Water-route table missing fields: {sorted(missing)}")
        return [dict(row) for row in reader]


def extract_site_snap_distances(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Recover one consistent SalishSeaCast snap distance per site from edge rows."""
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        values[str(row["src"])].append(float(row["salishseacast_src_snap_km"]))
        values[str(row["dst"])].append(float(row["salishseacast_dst_snap_km"]))

    result: dict[str, float] = {}
    for site_id, distances in values.items():
        reference = distances[0]
        if any(abs(value - reference) > 1e-9 for value in distances[1:]):
            raise ValueError(f"Inconsistent snap distance for site {site_id}")
        result[site_id] = reference
    return dict(sorted(result.items()))


def _primary_local_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if _as_bool(row.get("candidate_local_radius"))
        and _as_bool(row.get("salishseacast_route_found"))
    ]


def _route_filter(rows: list[dict[str, Any]], max_route_km: float) -> list[dict[str, Any]]:
    if max_route_km <= 0:
        raise ValueError("max_route_km must be positive")
    return [
        dict(row)
        for row in _primary_local_rows(rows)
        if float(row["salishseacast_water_route_km"]) <= max_route_km
    ]


def compare_route_distance_topologies(
    sites: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    route_thresholds_km: tuple[float, ...] = (20.0, 25.0, 30.0, 40.0, 60.0),
) -> dict[str, Any]:
    """Compare topology sensitivity to curved water-route distance thresholds.

    Direct geographic <=20 km remains only the candidate-generation envelope.
    The route thresholds below are diagnostics, not ecological constants.
    Sparse-review-only edges never enter these primary variants.
    """
    local = _primary_local_rows(rows)
    variants: dict[str, Any] = {"all_local_with_route": graph_summary(sites, local)}
    for threshold in route_thresholds_km:
        variants[f"route_le_{threshold:g}km"] = graph_summary(
            sites, _route_filter(rows, threshold)
        )

    snaps = extract_site_snap_distances(rows)
    snap_values = list(snaps.values())
    local_detours = [float(row["salishseacast_detour_ratio"]) for row in local]
    local_routes = [float(row["salishseacast_water_route_km"]) for row in local]

    edge_snap_quality = []
    for row in local:
        edge_snap_quality.append(
            max(
                float(row["salishseacast_src_snap_km"]),
                float(row["salishseacast_dst_snap_km"]),
            )
        )

    review_only = [
        row
        for row in rows
        if _as_bool(row.get("candidate_sparse_review"))
        and not _as_bool(row.get("candidate_local_radius"))
    ]

    return {
        "topology_variants": variants,
        "site_snap": {
            "sites": len(snaps),
            "distance_km_min": min(snap_values) if snap_values else None,
            "distance_km_median": median(snap_values) if snap_values else None,
            "distance_km_max": max(snap_values) if snap_values else None,
            "sites_gt_0_5km": sum(value > 0.5 for value in snap_values),
            "sites_gt_1km": sum(value > 1.0 for value in snap_values),
            "sites_gt_2km": sum(value > 2.0 for value in snap_values),
            "worst_sites": sorted(
                ({"site_id": site_id, "snap_km": distance} for site_id, distance in snaps.items()),
                key=lambda item: (-float(item["snap_km"]), str(item["site_id"])),
            )[:12],
        },
        "local_route": {
            "edges": len(local),
            "route_km_median": median(local_routes) if local_routes else None,
            "detour_ratio_median": median(local_detours) if local_detours else None,
            "detour_ratio_gt_2": sum(value > 2.0 for value in local_detours),
            "detour_ratio_gt_3": sum(value > 3.0 for value in local_detours),
            "detour_ratio_gt_5": sum(value > 5.0 for value in local_detours),
            "edges_with_endpoint_snap_gt_1km": sum(value > 1.0 for value in edge_snap_quality),
            "edges_with_endpoint_snap_gt_2km": sum(value > 2.0 for value in edge_snap_quality),
            "most_extreme_detours": sorted(
                (
                    {
                        "src": str(row["src"]),
                        "dst": str(row["dst"]),
                        "direct_km": float(row["distance_km"]),
                        "route_km": float(row["salishseacast_water_route_km"]),
                        "detour_ratio": float(row["salishseacast_detour_ratio"]),
                        "max_endpoint_snap_km": max(
                            float(row["salishseacast_src_snap_km"]),
                            float(row["salishseacast_dst_snap_km"]),
                        ),
                    }
                    for row in local
                ),
                key=lambda item: (-float(item["detour_ratio"]), -float(item["route_km"])),
            )[:15],
        },
        "review_only_edges": len(review_only),
        "notes": [
            "Curved water-route distance is a stronger geometry proxy than straight-line open-water sampling, but it is still not ecological connectivity probability.",
            "Direct <=20 km remains a candidate-generation design default only; route thresholds are sensitivity probes.",
            "Large site-to-grid snap distances indicate that SalishSeaCast is too coarse for some pocket-estuary sites and route values touching those sites should be treated as lower-confidence.",
            "Do not choose a route threshold because it creates a convenient number of connected components.",
            "connectivity_weight remains OPEN.",
        ],
    }
