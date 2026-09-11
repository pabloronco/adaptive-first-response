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
DEMO_SCENARIO_NAME = "Synthetic estuary / harbor response theater"

# DESIGN CHOICE: these labels/zones exist only to make the rehearsal UI legible.
# They are not real monitoring stations or ecological classifications.
DEMO_SITE_META: dict[str, dict[str, str]] = {
    "site_00": {"label": "Outer coast N1", "zone": "coast"},
    "site_01": {"label": "North reach", "zone": "coast"},
    "site_02": {"label": "Kelp edge", "zone": "coast"},
    "site_03": {"label": "Breakwater sector", "zone": "coast"},
    "site_04": {"label": "Ferry cut", "zone": "coast"},
    "site_05": {"label": "Inner reach", "zone": "coast"},
    "site_06": {"label": "Channel bend", "zone": "coast"},
    "site_07": {"label": "Harbor west", "zone": "coast"},
    "site_08": {"label": "Harbor mouth W", "zone": "coast"},
    "site_09": {"label": "First detection", "zone": "coast"},
    "site_10": {"label": "Harbor mouth E", "zone": "coast"},
    "site_11": {"label": "South reach", "zone": "coast"},
    "site_12": {"label": "Inner harbor", "zone": "harbor"},
    "site_13": {"label": "Marina basin", "zone": "harbor"},
    "site_14": {"label": "Estuary arm", "zone": "harbor"},
    "site_15": {"label": "Commercial basin", "zone": "harbor"},
    "site_16": {"label": "Outer channel", "zone": "offshore"},
    "site_17": {"label": "Offshore north", "zone": "offshore"},
    "site_18": {"label": "Offshore south", "zone": "offshore"},
    "site_19": {"label": "Shipping corridor", "zone": "offshore"},
}

# Irregular schematic geometry. The main coast is 00..11, while 12..15 form
# an inner-harbor branch and 16..19 an offshore pathway. This keeps the demo
# graph visibly non-linear without pretending that the geometry is a real map.
_DEMO_COORDS: tuple[tuple[float, float], ...] = (
    (0.4, 2.1),
    (1.2, 2.6),
    (2.0, 2.4),
    (2.8, 3.1),
    (3.6, 2.8),
    (4.4, 3.6),
    (5.2, 3.3),
    (6.0, 4.1),
    (6.8, 3.8),
    (7.5, 4.7),
    (8.3, 4.2),
    (9.2, 4.9),
    (7.6, 3.0),
    (8.0, 2.2),
    (8.7, 2.5),
    (9.1, 3.3),
    (7.0, 5.7),
    (6.3, 6.6),
    (8.0, 6.4),
    (9.0, 6.8),
)


def _edge(src: int, dst: int, *, connectivity: float) -> Edge:
    x1, y1 = _DEMO_COORDS[src]
    x2, y2 = _DEMO_COORDS[dst]
    distance = math.hypot(x2 - x1, y2 - y1)
    return Edge(
        src=f"site_{src:02d}",
        dst=f"site_{dst:02d}",
        distance=float(distance),
        connectivity_weight=connectivity,
        travel_cost=float(distance),
    )


def build_demo_incident(*, seed: int = DEMO_SEED) -> IncidentConfig:
    """Build the deterministic 20-site mission-control rehearsal incident.

    DESIGN CHOICE: this is a synthetic response theater, not a real coastline or
    ecological prior. The topology deliberately contains coastal, harbor and
    offshore pathways so replanning can visibly redirect field effort when new
    evidence arrives. Later real-data-constrained scenarios should replace this
    geometry for ecological validation.
    """

    sites: list[Site] = []
    for i, (x, y) in enumerate(_DEMO_COORDS):
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

    edges: list[Edge] = []
    # Main littoral chain.
    edges.extend(_edge(i, i + 1, connectivity=0.90) for i in range(11))
    # Inner-harbor / estuary pathway, reconnecting to the eastern mouth.
    edges.extend(
        [
            _edge(9, 12, connectivity=1.00),
            _edge(12, 13, connectivity=1.00),
            _edge(13, 14, connectivity=0.95),
            _edge(14, 15, connectivity=0.95),
            _edge(15, 10, connectivity=1.00),
        ]
    )
    # Offshore pathway and cross-links.
    edges.extend(
        [
            _edge(9, 16, connectivity=0.80),
            _edge(16, 17, connectivity=0.72),
            _edge(17, 7, connectivity=0.70),
            _edge(16, 18, connectivity=0.78),
            _edge(18, 19, connectivity=0.72),
            _edge(19, 11, connectivity=0.68),
            _edge(18, 10, connectivity=0.76),
        ]
    )

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
