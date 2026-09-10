from __future__ import annotations

from copy import deepcopy

import networkx as nx
import numpy as np

from .models import HiddenWorld, IncidentConfig, PublicState, Site


class Environment:
    """Minimal rehearsal environment for M1.

    This first implementation exists to validate the system boundary and reset
    semantics. The hidden-world generator below is intentionally a *toy model*,
    not an ecological validity claim and not the final multi-family simulator.
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

    @property
    def current_public_state(self) -> PublicState:
        if self._public_state is None:
            raise RuntimeError("Environment has not been reset yet.")
        return deepcopy(self._public_state)

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
        purpose is to prove hidden/public separation and deterministic reset.
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
