from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .demo_scenario import (
    DEMO_BUDGET,
    DEMO_SEED,
    build_demo_incident,
    build_demo_prior,
)
from .environment import Environment
from .graph_state import NODE_FEATURE_NAMES
from .mission_loop import AdaptiveMissionLoop, LoopPhase, RoundTransition
from .models import HiddenWorld, MissionAction
from .planners import FrontierPlanner, Planner


class MissionControlSession:
    """Thin product adapter over the real adaptive mission loop.

    The UI consumes only observable/public state exposed here. No ecological
    decision rule is duplicated in the frontend: missions come from the active
    Planner, evidence comes from Environment.step, and beliefs come from Bayes.
    Hidden truth appears in snapshots only after the explicit reveal gate.
    """

    def __init__(self, planner: Planner | None = None) -> None:
        self._planner = planner or FrontierPlanner(effort_per_site=3, max_sites=2)
        self._loop: AdaptiveMissionLoop | None = None
        self._mission: MissionAction | None = None
        self._last_transition: RoundTransition | None = None
        self._revealed_world: HiddenWorld | None = None
        self._events: list[dict[str, Any]] = []
        self._seed = DEMO_SEED
        self.reset(seed=DEMO_SEED)

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        resolved_seed = self._seed if seed is None else int(seed)
        self._seed = resolved_seed
        self._loop = AdaptiveMissionLoop(
            Environment(build_demo_incident(seed=resolved_seed)),
            self._planner,
            build_demo_prior(),
        )
        self._loop.reset(seed=resolved_seed)
        self._mission = None
        self._last_transition = None
        self._revealed_world = None
        self._events = [
            {
                "kind": "incident",
                "round": 0,
                "title": "New confirmed detection",
                "detail": "True incursion extent generated and locked from the planner.",
            }
        ]
        return self.snapshot()

    def plan(self) -> dict[str, Any]:
        loop = self._require_loop()
        if loop.phase is not LoopPhase.READY_TO_PLAN:
            raise RuntimeError(f"Cannot plan while loop phase is {loop.phase.value!r}.")
        self._mission = loop.plan_next()
        self._events.append(
            {
                "kind": "mission",
                "round": loop.current_public_state.round,
                "title": "Mission generated",
                "detail": self._mission_text(self._mission),
            }
        )
        return self.snapshot()

    def execute(self) -> dict[str, Any]:
        loop = self._require_loop()
        if loop.phase is not LoopPhase.MISSION_PLANNED:
            raise RuntimeError("A mission must be planned before it can be executed.")
        previous_mission = self._mission
        transition = loop.execute_pending()
        self._last_transition = transition
        self._events.append(
            {
                "kind": "return",
                "round": transition.observations.round,
                "title": "Field return received",
                "detail": self._observation_text(transition),
            }
        )

        if transition.next_mission is not None:
            self._mission = transition.next_mission
            self._events.append(
                {
                    "kind": "replan",
                    "round": transition.observations.round,
                    "title": "Mission updated",
                    "detail": self._mission_text(transition.next_mission),
                }
            )
        else:
            self._mission = None
            self._events.append(
                {
                    "kind": "complete",
                    "round": transition.observations.round,
                    "title": "Field budget exhausted",
                    "detail": "Reveal gate is now available for evaluator/demo use.",
                }
            )

        if previous_mission is None:
            raise RuntimeError("Mission-control adapter lost the pending mission reference.")
        return self.snapshot()

    def reveal(self) -> dict[str, Any]:
        loop = self._require_loop()
        if loop.phase is not LoopPhase.COMPLETE:
            raise RuntimeError("True extent can only be revealed after mission completion.")
        self._revealed_world = loop.reveal()
        self._events.append(
            {
                "kind": "reveal",
                "round": loop.current_public_state.round,
                "title": "True extent revealed",
                "detail": "Evaluator/demo-only latent occupancy is now visible.",
            }
        )
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        loop = self._require_loop()
        public = loop.current_public_state
        belief = loop.current_belief
        graph = loop.current_graph_state

        feature_index = {name: i for i, name in enumerate(NODE_FEATURE_NAMES)}
        site_by_id = {site.id: site for site in public.sites}
        mission_effort = {
            allocation.site_id: allocation.effort_units
            for allocation in (self._mission.allocations if self._mission else ())
        }

        nodes: list[dict[str, Any]] = []
        for i, site_id in enumerate(graph.node_ids):
            site = site_by_id[site_id]
            row = graph.node_features[i]
            node = {
                "id": site_id,
                "x": site.x,
                "y": site.y,
                "belief": belief.p_by_site[site_id],
                "uncertainty": belief.uncertainty_by_site[site_id],
                "effort": site.observed_effort,
                "detections": site.detections,
                "status": site.status,
                "frontier": bool(row[feature_index["frontier"]] > 0.5),
                "feasible": bool(graph.feasibility_mask[i]),
                "mission_effort": mission_effort.get(site_id, 0),
            }
            if self._revealed_world is not None:
                node["true_occupied"] = bool(
                    self._revealed_world.occupied_by_site[site_id]
                )
            nodes.append(node)

        last_round = self._serialize_transition(self._last_transition)
        mission_changed = False
        if self._last_transition is not None and self._last_transition.next_mission is not None:
            mission_changed = self._allocation_signature(
                self._last_transition.mission
            ) != self._allocation_signature(self._last_transition.next_mission)

        return {
            "phase": loop.phase.value,
            "incident": {
                "label": "Marine invasive species — confirmed first detection",
                "initial_detection": public.initial_detection,
                "seed": public.seed,
                "world_model_id": public.world_model_id,
                "truth_locked": self._revealed_world is None,
            },
            "resources": {
                "initial_budget": DEMO_BUDGET,
                "remaining_budget": public.remaining_budget,
                "spent_budget": DEMO_BUDGET - public.remaining_budget,
                "teams": public.teams,
                "round": public.round,
            },
            "mission": self._serialize_mission(self._mission),
            "mission_changed": mission_changed,
            "last_round": last_round,
            "nodes": nodes,
            "edges": [asdict(edge) for edge in public.edges],
            "events": list(self._events),
            "can_plan": loop.phase is LoopPhase.READY_TO_PLAN,
            "can_execute": loop.phase is LoopPhase.MISSION_PLANNED,
            "can_reveal": loop.phase is LoopPhase.COMPLETE,
            "revealed": loop.phase is LoopPhase.REVEALED,
        }

    @staticmethod
    def _serialize_mission(mission: MissionAction | None) -> dict[str, Any] | None:
        if mission is None:
            return None
        return {
            "allocations": [
                {
                    "site_id": allocation.site_id,
                    "effort_units": allocation.effort_units,
                    "team_id": allocation.team_id,
                }
                for allocation in mission.allocations
            ],
            "total_cost": mission.total_cost,
            "planner": mission.diagnostics.get("planner", "unknown"),
        }

    @classmethod
    def _serialize_transition(
        cls, transition: RoundTransition | None
    ) -> dict[str, Any] | None:
        if transition is None:
            return None
        observations = []
        for observation in transition.observations.observations:
            observations.append(
                {
                    "site_id": observation.site_id,
                    "effort": observation.effort,
                    "detection": observation.detection,
                    "belief_before": transition.belief_before.p_by_site[
                        observation.site_id
                    ],
                    "belief_after": transition.belief_after.p_by_site[
                        observation.site_id
                    ],
                }
            )
        return {
            "round": transition.observations.round,
            "observations": observations,
            "done": transition.done,
            "previous_mission": cls._serialize_mission(transition.mission),
            "next_mission": cls._serialize_mission(transition.next_mission),
        }

    @staticmethod
    def _allocation_signature(mission: MissionAction) -> tuple[tuple[str, int], ...]:
        return tuple(
            (allocation.site_id, allocation.effort_units)
            for allocation in mission.allocations
        )

    @staticmethod
    def _mission_text(mission: MissionAction) -> str:
        return ", ".join(
            f"{allocation.site_id}: {allocation.effort_units} checks"
            for allocation in mission.allocations
        )

    @staticmethod
    def _observation_text(transition: RoundTransition) -> str:
        return ", ".join(
            f"{obs.site_id}: {int(obs.detection)} detection / {obs.effort} checks"
            for obs in transition.observations.observations
        )

    def _require_loop(self) -> AdaptiveMissionLoop:
        if self._loop is None:
            raise RuntimeError("Mission-control session has not been initialized.")
        return self._loop
