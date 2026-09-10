from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any

from .models import BeliefState, GraphState, PublicState


NODE_FEATURE_NAMES = (
    "belief",
    "uncertainty",
    "observed_effort",
    "detections",
    "habitat_score",
    "access_cost",
    "frontier",
)

EDGE_FEATURE_NAMES = (
    "distance",
    "connectivity_weight",
)

GLOBAL_FEATURE_NAMES = (
    "remaining_budget",
    "round",
    "team_capacity",
    "global_uncertainty",
)


class GraphStateExporter:
    """Convert observable environment + belief state into planner input.

    The exporter contains no simulator/evaluator reference and therefore has no
    route to latent occupancy. It deliberately stays framework-agnostic: Demu's
    learned planner can convert the returned tuples to NumPy/PyTorch tensors.

    CURRENT DEFAULT encoding decisions for this rehearsal:
    - node feature order is fixed by NODE_FEATURE_NAMES;
    - missing access cost is encoded as a configurable neutral unit cost (1.0);
    - frontier = 1 for a non-positive site directly adjacent to a publicly
      detected/confirmed-positive site, otherwise 0;
    - graph edges are treated as undirected and exported in both directions,
      matching the undirected graph semantics currently used by Environment;
    - missing connectivity_weight is encoded as neutral weight 1.0;
    - global uncertainty is the mean Bernoulli entropy across nodes;
    - feasibility is True for every site while at least one budget unit remains.

    These are implementation defaults, not ecological facts. Interface-sensitive
    changes must be reviewed by the team before merge.
    """

    def __init__(
        self,
        *,
        default_access_cost: float = 1.0,
        default_connectivity_weight: float = 1.0,
    ) -> None:
        if default_access_cost < 0:
            raise ValueError("default_access_cost cannot be negative.")
        if default_connectivity_weight < 0:
            raise ValueError("default_connectivity_weight cannot be negative.")
        self._default_access_cost = float(default_access_cost)
        self._default_connectivity_weight = float(default_connectivity_weight)

    def export(self, public_state: PublicState, belief: BeliefState) -> GraphState:
        self._validate_alignment(public_state, belief)

        node_ids = tuple(site.id for site in public_state.sites)
        node_index = {site_id: i for i, site_id in enumerate(node_ids)}
        frontier_by_site = self._frontier_flags(public_state)

        node_features = tuple(
            (
                float(belief.p_by_site[site.id]),
                float(belief.uncertainty_by_site[site.id]),
                float(site.observed_effort),
                float(site.detections),
                float(site.habitat_score),
                float(
                    self._default_access_cost
                    if site.access_cost is None
                    else site.access_cost
                ),
                float(frontier_by_site[site.id]),
            )
            for site in public_state.sites
        )

        src_indices: list[int] = []
        dst_indices: list[int] = []
        edge_features: list[tuple[float, float]] = []

        for edge in public_state.edges:
            connectivity = (
                self._default_connectivity_weight
                if edge.connectivity_weight is None
                else float(edge.connectivity_weight)
            )
            features = (float(edge.distance), connectivity)

            # CURRENT DEFAULT: Environment currently treats the site graph as
            # undirected, so message-passing connectivity is made explicit in
            # both directions. If directed pathway semantics are later added,
            # this encoding decision must be revisited explicitly.
            src_idx = node_index[edge.src]
            dst_idx = node_index[edge.dst]
            src_indices.extend((src_idx, dst_idx))
            dst_indices.extend((dst_idx, src_idx))
            edge_features.extend((features, features))

        global_uncertainty = (
            fmean(belief.uncertainty_by_site.values())
            if belief.uncertainty_by_site
            else 0.0
        )
        global_features = (
            float(public_state.remaining_budget),
            float(public_state.round),
            float(public_state.teams),
            float(global_uncertainty),
        )
        feasible = public_state.remaining_budget > 0
        feasibility_mask = tuple(feasible for _ in node_ids)

        graph_state = GraphState(
            node_ids=node_ids,
            node_features=node_features,
            edge_index=(tuple(src_indices), tuple(dst_indices)),
            edge_features=tuple(edge_features),
            global_features=global_features,
            feasibility_mask=feasibility_mask,
        )
        self._validate_graph_state(graph_state)
        return graph_state

    @staticmethod
    def planner_constraints(public_state: PublicState) -> dict[str, Any]:
        """Return shared observable planner context required by v0 baselines.

        q is kept outside learned node features by default, but Information Gain
        needs it to evaluate prospective detection/non-detection likelihoods.
        """

        q_by_site: dict[str, float] = {}
        for site in public_state.sites:
            if isinstance(site.q_model, bool) or not isinstance(site.q_model, (int, float)):
                raise ValueError(
                    "GraphState v0 planner constraints support scalar q_model only."
                )
            q = float(site.q_model)
            if not 0.0 <= q <= 1.0:
                raise ValueError(f"q_model for {site.id!r} must be in [0, 1].")
            q_by_site[site.id] = q

        return {"q_by_site": q_by_site}

    @staticmethod
    def _frontier_flags(public_state: PublicState) -> dict[str, int]:
        positive_sites = {site.id for site in public_state.sites if site.detections > 0}
        adjacency: defaultdict[str, set[str]] = defaultdict(set)
        for edge in public_state.edges:
            adjacency[edge.src].add(edge.dst)
            adjacency[edge.dst].add(edge.src)

        return {
            site.id: int(
                site.id not in positive_sites
                and any(neighbor in positive_sites for neighbor in adjacency[site.id])
            )
            for site in public_state.sites
        }

    @staticmethod
    def _validate_alignment(public_state: PublicState, belief: BeliefState) -> None:
        site_ids = [site.id for site in public_state.sites]
        if len(site_ids) != len(set(site_ids)):
            raise ValueError("PublicState site ids must be unique.")

        site_id_set = set(site_ids)
        if set(belief.p_by_site) != site_id_set:
            raise ValueError("Belief p_by_site ids must exactly match PublicState sites.")
        if set(belief.uncertainty_by_site) != site_id_set:
            raise ValueError(
                "Belief uncertainty_by_site ids must exactly match PublicState sites."
            )

        for edge in public_state.edges:
            if edge.src not in site_id_set or edge.dst not in site_id_set:
                raise ValueError("PublicState edge references an unknown site.")

    @staticmethod
    def _validate_graph_state(graph_state: GraphState) -> None:
        n = len(graph_state.node_ids)
        if len(graph_state.node_features) != n:
            raise RuntimeError("node_ids and node_features lost index alignment.")
        if len(graph_state.feasibility_mask) != n:
            raise RuntimeError("node_ids and feasibility_mask lost index alignment.")
        if any(len(row) != len(NODE_FEATURE_NAMES) for row in graph_state.node_features):
            raise RuntimeError("Unexpected node feature width.")

        if len(graph_state.edge_index) != 2:
            raise RuntimeError("edge_index must have shape [2, E].")
        edge_count = len(graph_state.edge_features)
        if any(len(row) != edge_count for row in graph_state.edge_index):
            raise RuntimeError("edge_index and edge_features lost edge alignment.")
        if any(len(row) != len(EDGE_FEATURE_NAMES) for row in graph_state.edge_features):
            raise RuntimeError("Unexpected edge feature width.")
        if any(index < 0 or index >= n for row in graph_state.edge_index for index in row):
            raise RuntimeError("edge_index contains an invalid node index.")

        if len(graph_state.global_features) != len(GLOBAL_FEATURE_NAMES):
            raise RuntimeError("Unexpected global feature width.")
