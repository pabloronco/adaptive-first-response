from __future__ import annotations

from typing import Any

from .real_graph import graph_summary


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def select_real_graph_v0_edges(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a transparent v0 primary edge set from the audited candidate table.

    CURRENT DEFAULT, not ecological truth:
    - only <=20 km local-radius candidates can enter the primary graph;
    - sparse-monitoring review-only candidates never force connectivity;
    - local candidates with zero sampled open-saltwater support are withheld from
      the primary graph but preserved as uncertain alternatives rather than being
      declared ecologically impossible.

    Returns ``(primary, withheld_zero_water, review_only)``.
    """
    primary: list[dict[str, Any]] = []
    withheld: list[dict[str, Any]] = []
    review_only: list[dict[str, Any]] = []

    for row in rows:
        candidate = dict(row)
        is_local = _as_bool(candidate.get("candidate_local_radius"))
        is_review = _as_bool(candidate.get("candidate_sparse_review"))
        water = float(candidate["straight_marine_fraction"])

        if not is_local:
            if is_review:
                candidate["v0_edge_status"] = "sampling_isolation_review_only"
                candidate["v0_primary"] = False
                review_only.append(candidate)
            continue

        if water <= 0.0:
            candidate["v0_edge_status"] = "withheld_zero_straight_water_support"
            candidate["v0_primary"] = False
            withheld.append(candidate)
            continue

        candidate["v0_edge_status"] = "primary_local_nonzero_water_support"
        candidate["v0_primary"] = True
        primary.append(candidate)

    return primary, withheld, review_only


def build_real_graph_v0_audit(
    sites: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary, withheld, review_only = select_real_graph_v0_edges(rows)
    primary_summary = graph_summary(sites, primary)
    all_local = [row for row in rows if _as_bool(row.get("candidate_local_radius"))]
    all_local_summary = graph_summary(sites, all_local)

    summary = {
        "status": "CURRENT_DEFAULT_NOT_FROZEN_ECOLOGICAL_TRUTH",
        "sites": len(sites),
        "primary_edges": len(primary),
        "withheld_zero_water_local_edges": len(withheld),
        "sampling_isolation_review_only_edges": len(review_only),
        "primary_graph": primary_summary,
        "all_local_reference_graph": all_local_summary,
        "topology_stability_vs_all_local": {
            "components_unchanged": primary_summary["components"] == all_local_summary["components"],
            "largest_component_unchanged": (
                primary_summary["largest_component_sites"]
                == all_local_summary["largest_component_sites"]
            ),
            "isolated_sites_unchanged": (
                primary_summary["isolated_sites"] == all_local_summary["isolated_sites"]
            ),
        },
        "withheld_zero_water_edges": [
            {
                "src": str(row["src"]),
                "dst": str(row["dst"]),
                "distance_km": float(row["distance_km"]),
            }
            for row in withheld
        ],
        "notes": [
            "The 20 km local radius is a candidate-generation design choice, not an ecological dispersal threshold.",
            "Zero straight-water support is used conservatively to withhold an edge from the primary v0 graph, not to claim ecological impossibility.",
            "Sparse-monitoring review-only edges are never used to repair connectivity automatically.",
            "The primary graph may be disconnected and may contain isolated sites.",
            "Connectivity weight remains OPEN; this v0 freezes adjacency logic only, not dispersal strength.",
            "Use all-local and stricter water-filter variants as topology sensitivity/OOD alternatives.",
        ],
    }
    return primary, summary
