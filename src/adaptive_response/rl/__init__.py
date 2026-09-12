"""Learned decision engine (GNN + RL) for the adaptive first-response planner.

This subpackage is deliberately isolated from the core `adaptive_response`
package: importing `adaptive_response` must never require torch. Only code
that explicitly does `from adaptive_response.rl import ...` pays that cost.

Scope as of Checkpoint A (see docs/DECISION_LOG.md, "GNN/RL backbone v0"):
GraphState -> tensors -> message passing -> node embeddings -> graph/global
representation -> per-node policy logits + a critic value estimate.

Explicitly OUT of scope here: action-space semantics, effort representation,
reward, and anything that would require freezing how a MissionAction gets
built from policy output. Those are Checkpoint B decisions and must be made
with the team, not unilaterally in this module.
"""

from .backbone import ActorCriticOutput, GNNActorCritic
from .benchmark import (
    BenchmarkCase,
    BenchmarkRow,
    make_benchmark_cases,
    run_benchmark_suite,
    write_benchmark_csv,
)
from .checkpointing import (
    LoadedCheckpoint,
    PolicyArchitectureConfig,
    load_optimizer_state,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from .eval_utils import EpisodeMetrics, RLPlannerAdapter, run_planner_episode
from .incident_sampler import IncidentSamplerConfig, sample_incident
from .reward import RewardConfig, round_reward, terminal_missed_extent_penalty
from .round_policy import RoundDecision, RoundPolicy
from .tensor_adapter import GraphTensors, graph_state_to_tensors
from .trainer import ActorCriticTrainer, TrainerConfig, UpdateStats
from .training_env import EpisodeRollout, run_episode

__all__ = [
    "ActorCriticOutput",
    "ActorCriticTrainer",
    "BenchmarkCase",
    "BenchmarkRow",
    "EpisodeMetrics",
    "EpisodeRollout",
    "GNNActorCritic",
    "GraphTensors",
    "IncidentSamplerConfig",
    "LoadedCheckpoint",
    "PolicyArchitectureConfig",
    "RLPlannerAdapter",
    "RewardConfig",
    "RoundDecision",
    "RoundPolicy",
    "TrainerConfig",
    "UpdateStats",
    "graph_state_to_tensors",
    "load_optimizer_state",
    "load_policy_checkpoint",
    "make_benchmark_cases",
    "round_reward",
    "run_benchmark_suite",
    "run_episode",
    "run_planner_episode",
    "sample_incident",
    "save_policy_checkpoint",
    "terminal_missed_extent_penalty",
    "write_benchmark_csv",
]
