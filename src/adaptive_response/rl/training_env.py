from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..belief import BeliefEngine
from ..environment import Environment
from ..graph_state import GraphStateExporter
from ..models import HiddenWorld, IncidentConfig
from .reward import RewardConfig, round_reward, terminal_missed_extent_penalty
from .round_policy import RoundPolicy


@dataclass
class EpisodeRollout:
    """Everything the trainer needs from one full episode, one round-decision at a time."""

    log_probs: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)
    entropies: list[torch.Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    num_rounds: int = 0
    num_picks_total: int = 0
    detections_found: int = 0
    effort_spent: int = 0
    occupied_sites_missed: int = 0
    occupied_sites_total: int = 0


def run_episode(
    policy: RoundPolicy,
    incident_config: IncidentConfig,
    *,
    reward_config: RewardConfig | None = None,
    seed: int | None = None,
    deterministic: bool = False,
) -> EpisodeRollout:
    """Run one full incident from reset to budget exhaustion under `policy`.

    Reads `Environment`'s private hidden-world attribute ONLY to compute the
    training-time terminal missed-extent term (see reward.py's module
    docstring on the HiddenWorld privileged-access invariant). This never
    touches the policy's inputs: `policy.act` only ever receives a
    `GraphState`. TODO once M4 (`m4/end-to-end-loop`) merges to main and
    exposes `Environment.reveal()`: switch this to the public API instead of
    the private attribute.
    """

    cfg = reward_config or RewardConfig()
    env = Environment(incident_config)
    public = env.reset(seed=seed)
    prior = {site.id: 0.5 for site in public.sites}
    belief = BeliefEngine.initialize(prior, confirmed_sites={public.initial_detection})
    exporter = GraphStateExporter()

    rollout = EpisodeRollout()

    while public.remaining_budget > 0:
        graph_state = exporter.export(public, belief)
        decision = policy.act(graph_state, public.remaining_budget, deterministic=deterministic)

        observations, done, metrics = env.step(decision.mission)
        public_after = env.current_public_state
        q_by_site = exporter.planner_constraints(public_after)["q_by_site"]
        belief_after = BeliefEngine.update(belief, observations, q_by_site)

        reward = round_reward(
            belief_before=belief,
            belief_after=belief_after,
            detections_this_round=int(metrics["detections"]),
            effort_spent_this_round=int(metrics["effort_spent"]),
            config=cfg,
        )

        rollout.log_probs.append(decision.log_prob)
        rollout.values.append(decision.value)
        rollout.entropies.append(decision.entropy)
        rollout.num_rounds += 1
        rollout.num_picks_total += decision.num_picks
        rollout.detections_found += int(metrics["detections"])
        rollout.effort_spent += int(metrics["effort_spent"])

        public = public_after
        belief = belief_after

        if done:
            hidden_world: HiddenWorld = env._hidden_world  # noqa: SLF001 (see docstring)
            terminal_penalty = terminal_missed_extent_penalty(
                public_state=public, hidden_world=hidden_world, config=cfg
            )
            reward += terminal_penalty
            rollout.occupied_sites_total = sum(hidden_world.occupied_by_site.values())
            rollout.occupied_sites_missed = sum(
                1
                for site in public.sites
                if hidden_world.occupied_by_site.get(site.id, False) and site.detections == 0
            )

        rollout.rewards.append(reward)

    return rollout
