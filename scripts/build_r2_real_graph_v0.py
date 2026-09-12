from __future__ import annotations

import csv
import json
from pathlib import Path

from adaptive_response.graph_filter_sensitivity import load_adaptive_water_candidates
from adaptive_response.real_graph import load_real_sites
from adaptive_response.real_graph_v0 import build_real_graph_v0_audit


REAL_SITES = Path("data/processed/r2_real_sites.csv")
ADAPTIVE_WATER = Path("data/processed/r2_adaptive_water_candidates.csv")
OUT_EDGES = Path("data/processed/r2_real_graph_v0_edges.csv")
OUT_AUDIT = Path("data/processed/r2_real_graph_v0_audit.json")


def main() -> None:
    if not REAL_SITES.is_file():
        raise SystemExit(f"Missing {REAL_SITES}")
    if not ADAPTIVE_WATER.is_file():
        raise SystemExit(
            f"Missing {ADAPTIVE_WATER}. Run diagnose_r2_adaptive_water.py first."
        )

    sites = load_real_sites(REAL_SITES)
    rows = load_adaptive_water_candidates(ADAPTIVE_WATER)
    edges, summary = build_real_graph_v0_audit(sites, rows)

    OUT_EDGES.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in edges for key in row})
    with OUT_EDGES.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edges)

    OUT_AUDIT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    graph = summary["primary_graph"]
    stability = summary["topology_stability_vs_all_local"]

    print("=== R2 REAL GRAPH V0 BUILD ===")
    print("STATUS: CURRENT DEFAULT, not ecological ground truth.")
    print("Rule: <=20 km local candidates; withhold zero straight-water-support edges; never force sparse-review edges.")
    print()
    print(
        f"sites={summary['sites']} | primary edges={summary['primary_edges']} | "
        f"withheld zero-water local edges={summary['withheld_zero_water_local_edges']} | "
        f"review-only edges excluded={summary['sampling_isolation_review_only_edges']}"
    )
    print(
        f"Primary topology: components={graph['components']} | "
        f"largest={graph['largest_component_sites']}/{summary['sites']} | "
        f"isolated={len(graph['isolated_sites'])} {graph['isolated_sites']} | "
        f"degree min/med/max={graph['degree_min']}/{graph['degree_median']}/{graph['degree_max']}"
    )
    print(
        "Topology vs all-local reference: "
        f"components unchanged={stability['components_unchanged']} | "
        f"largest unchanged={stability['largest_component_unchanged']} | "
        f"isolates unchanged={stability['isolated_sites_unchanged']}"
    )
    print()
    print("Withheld zero-water local edges (kept as uncertainty, not called impossible):")
    for row in summary["withheld_zero_water_edges"]:
        print(
            f"  {row['src']} -- {row['dst']}: d={row['distance_km']:.2f} km"
        )
    print()
    print("GATE:")
    print("  This freezes a transparent PRIMARY ADJACENCY DEFAULT only.")
    print("  connectivity_weight remains OPEN.")
    print("  Next: test incident-subgraph extraction around an initial detection without using future outcomes.")
    print(f"Wrote {OUT_EDGES}")
    print(f"Wrote {OUT_AUDIT}")


if __name__ == "__main__":
    main()
