from __future__ import annotations

import math

from .models import Edge, IncidentConfig, Site


DEMO_NUM_SITES = 20
DEMO_INITIAL_DETECTION = "site_09"
DEMO_BUDGET = 30
DEMO_TEAMS = 2
DEMO_Q = 0.25
DEMO_PRIOR = 0.35
DEMO_SEED = 20260910


def build_demo_incident(*, seed: int = DEMO_SEED) -> IncidentConfig:
    """Build the deterministic 20-site mission-control rehearsal incident.

    DESIGN CHOICE: this is a synthetic demo geometry, not a real coastline or an
    ecological prior. The purpose is to exercise the real adaptive loop with a
    legible 20-site graph while later real-data-constrained scenarios are built.
    """

    sites: list[Site] = []
    for i in range(DEMO_NUM_SITES):
        # Gentle synthetic coastline curve for a readable mission-control graph.
        x = float(i)
        y = float(1.25 * math.sin(i / 2.8) + 0.18 * math.sin(i / 1.25))
        habitat = float(0.50 + 0.12 * math.sin(i / 3.5))
        sites.append(
            Site(
                id=f"site_{i:02d}",
                x=x,
                y=y,
                habitat_score=habitat,
                q_model=DEMO_Q,
                access_cost=1.0,
            )
        )

    edges = [
        Edge(
            src=f"site_{i:02d}",
            dst=f"site_{i + 1:02d}",
            distance=1.0,
            connectivity_weight=1.0,
            travel_cost=1.0,
        )
        for i in range(DEMO_NUM_SITES - 1)
    ]

    return IncidentConfig(
        sites=sites,
        edges=edges,
        initial_detection=DEMO_INITIAL_DETECTION,
        budget=DEMO_BUDGET,
        teams=DEMO_TEAMS,
        protocol="binary_detection",
        seed=seed,
        world_model_id="toy_graph_cluster_m1",
    )


def build_demo_prior() -> dict[str, float]:
    """Return an explicit, deliberately simple demo prior.

    DESIGN CHOICE: every non-confirmed site starts at the same prior occupancy
    belief. This avoids presenting an invented habitat/distance prior as ecology.
    The confirmed initial detection is set to p=1 later by BeliefEngine.
    """

    return {f"site_{i:02d}": DEMO_PRIOR for i in range(DEMO_NUM_SITES)}
