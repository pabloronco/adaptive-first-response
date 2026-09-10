from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


QModel = float | dict[str, Any]


@dataclass(slots=True)
class Site:
    """Observable survey-site state plus declared site/protocol metadata.

    `q_model` is an explicit observation-model parameter, not hidden occupancy truth.
    """

    id: str
    x: float
    y: float
    habitat_score: float
    q_model: QModel
    observed_effort: int = 0
    detections: int = 0
    status: str = "unsurveyed"
    access_cost: float | None = None


@dataclass(frozen=True, slots=True)
class Edge:
    src: str
    dst: str
    distance: float
    connectivity_weight: float | None = None
    travel_cost: float | None = None


@dataclass(slots=True)
class IncidentConfig:
    sites: list[Site]
    edges: list[Edge]
    initial_detection: str
    budget: int
    teams: int
    protocol: str
    seed: int
    world_model_id: str | None = None


@dataclass(frozen=True, slots=True)
class HiddenWorld:
    """Latent simulator/evaluator-only state.

    This type must never be embedded in PublicState, GraphState, planner inputs,
    UI planner payloads, or planner diagnostics.
    """

    occupied_by_site: dict[str, bool]
    generator_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Observation:
    site_id: str
    effort: int
    detection: bool
    round: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    """Field returns produced by one mission round.

    This object contains observations only. It must never expose latent occupancy.
    """

    observations: tuple[Observation, ...]
    round: int
    total_effort: int


@dataclass(frozen=True, slots=True)
class MissionAllocation:
    site_id: str
    effort_units: int
    team_id: str | None = None


@dataclass(frozen=True, slots=True)
class MissionAction:
    allocations: tuple[MissionAllocation, ...]
    total_cost: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PublicState:
    """Observable environment snapshot returned by reset/step.

    Deliberately contains no HiddenWorld or latent occupancy representation.
    """

    sites: list[Site]
    edges: list[Edge]
    initial_detection: str
    remaining_budget: int
    round: int
    teams: int
    protocol: str
    seed: int
    world_model_id: str | None = None


@dataclass(frozen=True, slots=True)
class BeliefState:
    """Explicit probabilistic occupancy belief derived from field evidence.

    `p_by_site` and `uncertainty_by_site` are observable inference outputs.
    `observed_history` contains only observations, never latent occupancy truth.
    """

    p_by_site: dict[str, float]
    uncertainty_by_site: dict[str, float]
    observed_history: tuple[Observation, ...] = field(default_factory=tuple)
