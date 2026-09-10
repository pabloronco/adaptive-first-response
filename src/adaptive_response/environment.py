from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from typing import Any

import networkx as nx
import numpy as np

from .models import (
    HiddenWorld,
    IncidentConfig,
    MissionAction,
    Observation,
    ObservationBatch,
    PublicState,
    Site,
)


class Environment:
    """Minimal rehearsal environment for M1.

    This implementation validates the system boundary, reset semantics, resource
    accounting, and effort-aware imperfect detection. The hidden-world generator
    remains a *toy model* for M1, not an ecological validity claim and not the
    final multi-family simulator.
    """

    def __init__(self, config: IncidentConfig) -> None:
        self._validate_config(config)
        self._config = deepcopy(config)
        self._rng: np.random.Generator | None = None
        self._hidden_world: HiddenWorld | None = None
        self._public_state: PublicState | None = None

    def reset(self, seed: int | None = None) -> PublicState:
        """Reset the incident and return only observable state.

        Same seed + same config produces the same hidden world. The returned
        object never contains latent occupancy.
        """

        resolved_seed = self._config.seed if seed is None else seed
        self._rng = np.random.default_rng(resolved_seed)
        self._hidden_world = self._generate_toy_hidden_world(self._rng)

        public_sites = [self._reset_site(site) for site in self._config.sites]
        self._public_state = PublicState(
            sites=public_sites,
            edges=deepcopy(self._config.edges),
            initial_detection=self._config.initial_detection,
            remaining_budget=self._config.budget,
            round=0,
            teams=self._config.teams,
            protocol=self._config.protocol,
            seed=resolved_seed,
            world_model_id=self._config.world_model_id,
        )
        return deepcopy(self._public_state)

    def step(
        self, action: MissionAction
    ) -> tuple[ObservationBatch, bool, dict[str, Any]]:
        """Execute one field mission against the hidden incident.

        CURRENT DEFAULT for M1: one effort unit consumes one budget unit. Multiple
        allocations to the same site are aggregated before simulating a single
        field return for that site.

        Detection semantics remain explicit:
        P(detection | occupied, effort=e) = 1 - (1-q)^e.
        An unoccupied site cannot generate a false positive in the MVP.
        """

        self._require_reset()
        assert self._public_state is not None
        assert self._hidden_world is not None
        assert self._rng is not None

        effort_by_site = self._validate_and_aggregate_action(action)
        next_round = self._public_state.round + 1
        observations: list[Observation] = []

        site_lookup = {site.id: site for site in self._public_state.sites}
        for site_id, effort in effort_by_site.items():
            site = site_lookup[site_id]
            q = self._resolve_q(site)
            occupied = self._hidden_world.occupied_by_site[site_id]
            detection_probability = 1.0 - (1.0 - q) ** effort if occupied else 0.0
            detection = bool(self._rng.random() < detection_probability)

            observation = Observation(
                site_id=site_id,
                effort=effort,
                detection=detection,
                round=next_round,
                metadata={
                    "protocol": self._public_state.protocol,
                    "q_used": q,
                },
            )
            observations.append(observation)

            site.observed_effort += effort
            if detection:
                site.detections += 1
                site.status = "detected"
            else:
                site.status = "surveyed_no_detection"

        self._public_state.remaining_budget -= action.total_cost
        self._public_state.round = next_round

        batch = ObservationBatch(
            observations=tuple(observations),
            round=next_round,
            total_effort=sum(effort_by_site.values()),
        )
        done = self._public_state.remaining_budget == 0
        metrics = {
            "round": next_round,
            "effort_spent": action.total_cost,
            "remaining_budget": self._public_state.remaining_budget,
            "detections": sum(obs.detection for obs in observations),
            "sites_surveyed": len(observations),
        }
        return batch, done, metrics

    @property
    def current_public_state(self) -> PublicState:
        self._require_reset()
        assert self._public_state is not None
        return deepcopy(self._public_state)

    def _validate_and_aggregate_action(self, action: MissionAction) -> dict[str, int]:
        assert self._public_state is not None

        if action.total_cost < 0:
            raise ValueError("MissionAction.total_cost cannot be negative.")
        if action.total_cost > self._public_state.remaining_budget:
            raise ValueError("MissionAction exceeds remaining budget.")

        known_sites = {site.id for site in self._public_state.sites}
        effort_by_site: defaultdict[str, int] = defaultdict(int)
        for allocation in action.allocations:
            if allocation.site_id not in known_sites:
                raise ValueError(
                    f"Mission allocation references unknown site {allocation.site_id!r}."
                )
            if not isinstance(allocation.effort_units, int) or allocation.effort_units <= 0:
                raise ValueError("effort_units must be a positive integer.")
            effort_by_site[allocation.site_id] += allocation.effort_units

        allocated_effort = sum(effort_by_site.values())
        if allocated_effort != action.total_cost:
            raise ValueError(
                "For M1, MissionAction.total_cost must equal allocated effort units."
            )
        if allocated_effort == 0:
            raise ValueError("MissionAction must allocate positive effort.")

        return dict(effort_by_site)

    @staticmethod
    def _resolve_q(site: Site) -> float:
        if isinstance(site.q_model, bool) or not isinstance(site.q_model, (int, float)):
            raise ValueError(
                "M1 supports scalar q_model only; richer q models remain future work."
            )
        q = float(site.q_model)
        if not 0.0 <= q <= 1.0:
            raise ValueError("q_model must be between 0 and 1.")
        return q

    def _reset_site(self, site: Site) -> Site:
        public_site = deepcopy(site)
        public_site.observed_effort = 0
        public_site.detections = 0
        public_site.status = "unsurveyed"

        if public_site.id == self._config.initial_detection:
            # The product starts after a confirmed first detection. We represent
            # that fact explicitly without inventing effort for the pre-incident
            # confirmation protocol.
            public_site.detections = 1
            public_site.status = "confirmed_detection"

        return public_site

    def _generate_toy_hidden_world(
        self, rng: np.random.Generator
    ) -> HiddenWorld:
        """Generate a simple connected-ish latent footprint for M1 only.

        MODEL ASSUMPTION: nearby nodes around the confirmed detection are more
        likely to be occupied. This generator is deliberately temporary; M1's
        purpose is to prove hidden/public separation and deterministic execution.
        """

        graph = nx.Graph()
        for site in self._config.sites:
            graph.add_node(site.id)
        for edge in self._config.edges:
            graph.add_edge(edge.src, edge.dst)

        distances = nx.single_source_shortest_path_length(
            graph, self._config.initial_detection
        )
        radius = int(rng.integers(1, 4))

        occupied: dict[str, bool] = {}
        for site in self._config.sites:
            if site.id == self._config.initial_detection:
                occupied[site.id] = True
                continue

            distance = distances.get(site.id)
            if distance is None or distance > radius:
                occupied[site.id] = False
                continue

            habitat = float(np.clip(site.habitat_score, 0.0, 1.0))
            probability = float(
                np.clip(0.72 - 0.16 * distance + 0.18 * habitat, 0.05, 0.95)
            )
            occupied[site.id] = bool(rng.random() < probability)

        return HiddenWorld(
            occupied_by_site=occupied,
            generator_parameters={
                "family": "toy_graph_cluster_m1",
                "radius": radius,
            },
        )

    def _require_reset(self) -> None:
        if self._public_state is None or self._hidden_world is None or self._rng is None:
            raise RuntimeError("Environment has not been reset yet.")

    @staticmethod
    def _validate_config(config: IncidentConfig) -> None:
        if not config.sites:
            raise ValueError("IncidentConfig must contain at least one site.")

        site_ids = [site.id for site in config.sites]
        if len(site_ids) != len(set(site_ids)):
            raise ValueError("Site ids must be unique.")

        if config.initial_detection not in set(site_ids):
            raise ValueError("initial_detection must reference an existing site.")

        if config.budget < 0:
            raise ValueError("budget cannot be negative.")

        if config.teams < 1:
            raise ValueError("teams must be at least 1.")

        known = set(site_ids)
        for edge in config.edges:
            if edge.src not in known or edge.dst not in known:
                raise ValueError(
                    f"Edge {edge.src!r}->{edge.dst!r} references an unknown site."
                )
            if edge.distance < 0:
                raise ValueError("edge distance cannot be negative.")
