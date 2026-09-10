from __future__ import annotations

from dataclasses import dataclass

from ..belief import BeliefEngine
from ..environment import Environment
from ..graph_state import GraphStateExporter
from ..models import HiddenWorld, IncidentConfig, MissionAction
from ..planners import Planner
from .round_policy import RoundPolicy


@dataclass(frozen=True)
class EpisodeMetrics:
    """Planner-agnostic task metrics, for comparing RL against FrontierPlanner
    (and later Information Gain) on identical held-out incidents/budgets, per
    the team's benchmark discipline (docs/DECISION_LOG.md, M3 entry, and the
    "RL vs Information Gain on held-out incidents" gate in the handoff)."""

    num_rounds: int
    detections_found: int
    effort_spent: int
    occupied_sites_total: int
    occupied_sites_missed: int


class RLPlannerAdapter:
    """Expose a trained `RoundPolicy` through the shared `Planner` protocol so
    it can be evaluated with the exact same harness as `FrontierPlanner`.

    Always acts deterministically (argmax): evaluation should measure the
    policy's committed behavior, not an exploration sample.
    """

    def __init__(self, policy: RoundPolicy) -> None:
        self._policy = policy

    def plan(self, graph_state, remaining_budget, constraints) -> MissionAction:
        del constraints
        return self._policy.act(graph_state, remaining_budget, deterministic=True).mission


def run_planner_episode(
    planner: Planner, incident_config: IncidentConfig, *, seed: int | None = None
) -> EpisodeMetrics:
    """Run one incident under any `Planner` and report task metrics only.

    Reads Environment's private hidden-world attribute for the missed-extent
    metric; see training_env.py's docstring for why that is legitimate
    evaluator-only access and the TODO to switch to Environment.reveal() once
    M4 merges.
    """

    env = Environment(incident_config)
    public = env.reset(seed=seed)
    prior = {site.id: 0.5 for site in public.sites}
    belief = BeliefEngine.initialize(prior, confirmed_sites={public.initial_detection})
    exporter = GraphStateExporter()

    num_rounds = 0
    detections_found = 0
    effort_spent = 0

    while public.remaining_budget > 0:
        graph_state = exporter.export(public, belief)
        constraints = exporter.planner_constraints(public)
        mission = planner.plan(graph_state, public.remaining_budget, constraints)

        observations, done, metrics = env.step(mission)
        public = env.current_public_state
        q_by_site = exporter.planner_constraints(public)["q_by_site"]
        belief = BeliefEngine.update(belief, observations, q_by_site)

        num_rounds += 1
        detections_found += int(metrics["detections"])
        effort_spent += int(metrics["effort_spent"])

        if done:
            hidden_world: HiddenWorld = env._hidden_world  # noqa: SLF001
            occupied_total = sum(hidden_world.occupied_by_site.values())
            missed = sum(
                1
                for site in public.sites
                if hidden_world.occupied_by_site.get(site.id, False) and site.detections == 0
            )
            return EpisodeMetrics(
                num_rounds=num_rounds,
                detections_found=detections_found,
                effort_spent=effort_spent,
                occupied_sites_total=occupied_total,
                occupied_sites_missed=missed,
            )

    raise RuntimeError("Episode loop exited without reaching done=True.")
