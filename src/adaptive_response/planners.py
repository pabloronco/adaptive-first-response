from __future__ import annotations

from typing import Any, Mapping, Protocol

from .graph_state import NODE_FEATURE_NAMES
from .models import GraphState, MissionAction, MissionAllocation


class Planner(Protocol):
    """Shared planner contract used by heuristic, information-gain and RL planners."""

    def plan(
        self,
        graph_state: GraphState,
        remaining_budget: int,
        constraints: Mapping[str, Any],
    ) -> MissionAction:
        ...


class FrontierPlanner:
    """Transparent frontier-first baseline for rapid-response delimitation.

    CURRENT DEFAULT ranking for M3:
    1. frontier nodes first;
    2. higher occupancy belief;
    3. higher uncertainty;
    4. deterministic site_id tie-break.

    If no feasible frontier node exists, the planner falls back to the same
    belief/uncertainty ranking over all feasible nodes. This is a baseline
    decision rule, not an ecological optimality claim.

    `effort_per_site` is configurable so M3 does not freeze the later RL action
    space. The planner never sees HiddenWorld and uses only GraphState + budget.
    """

    def __init__(self, *, effort_per_site: int = 1, max_sites: int | None = None) -> None:
        if not isinstance(effort_per_site, int) or effort_per_site <= 0:
            raise ValueError("effort_per_site must be a positive integer.")
        if max_sites is not None and (
            not isinstance(max_sites, int) or max_sites <= 0
        ):
            raise ValueError("max_sites must be a positive integer when provided.")
        self.effort_per_site = effort_per_site
        self.max_sites = max_sites

    def plan(
        self,
        graph_state: GraphState,
        remaining_budget: int,
        constraints: Mapping[str, Any],
    ) -> MissionAction:
        del constraints  # Frontier v0 intentionally does not use q or hidden context.

        if not isinstance(remaining_budget, int) or remaining_budget < 0:
            raise ValueError("remaining_budget must be a non-negative integer.")
        if len(graph_state.node_ids) != len(graph_state.node_features):
            raise ValueError("GraphState node_ids and node_features are misaligned.")
        if len(graph_state.node_ids) != len(graph_state.feasibility_mask):
            raise ValueError("GraphState feasibility_mask is misaligned with nodes.")
        if remaining_budget == 0:
            return MissionAction(
                allocations=(),
                total_cost=0,
                diagnostics={
                    "planner": "frontier",
                    "reason": "no_remaining_budget",
                    "fallback_used": False,
                },
            )

        belief_idx = NODE_FEATURE_NAMES.index("belief")
        uncertainty_idx = NODE_FEATURE_NAMES.index("uncertainty")
        frontier_idx = NODE_FEATURE_NAMES.index("frontier")

        candidates: list[tuple[str, float, float, bool]] = []
        for i, site_id in enumerate(graph_state.node_ids):
            if not graph_state.feasibility_mask[i]:
                continue
            features = graph_state.node_features[i]
            if len(features) != len(NODE_FEATURE_NAMES):
                raise ValueError(
                    f"Node {site_id!r} has {len(features)} features; "
                    f"expected {len(NODE_FEATURE_NAMES)}."
                )
            candidates.append(
                (
                    site_id,
                    float(features[belief_idx]),
                    float(features[uncertainty_idx]),
                    bool(features[frontier_idx] > 0.5),
                )
            )

        frontier_candidates = [row for row in candidates if row[3]]
        fallback_used = not frontier_candidates
        ranked = frontier_candidates if frontier_candidates else candidates
        ranked.sort(key=lambda row: (-row[1], -row[2], row[0]))

        if self.max_sites is not None:
            ranked = ranked[: self.max_sites]

        budget_left = remaining_budget
        allocations: list[MissionAllocation] = []
        ranking_diagnostics: list[dict[str, Any]] = []

        for site_id, belief, uncertainty, is_frontier in ranked:
            if budget_left <= 0:
                break
            effort = min(self.effort_per_site, budget_left)
            allocations.append(
                MissionAllocation(site_id=site_id, effort_units=effort)
            )
            ranking_diagnostics.append(
                {
                    "site_id": site_id,
                    "belief": belief,
                    "uncertainty": uncertainty,
                    "frontier": is_frontier,
                    "effort_units": effort,
                }
            )
            budget_left -= effort

        total_cost = sum(a.effort_units for a in allocations)
        return MissionAction(
            allocations=tuple(allocations),
            total_cost=total_cost,
            diagnostics={
                "planner": "frontier",
                "fallback_used": fallback_used,
                "effort_per_site": self.effort_per_site,
                "ranked_selected_sites": tuple(ranking_diagnostics),
            },
        )
