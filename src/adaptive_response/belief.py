from __future__ import annotations

from math import log2
from typing import Mapping

from .models import BeliefState, Observation, ObservationBatch


class BeliefEngine:
    """Explicit site-level Bayesian occupancy update for the MVP.

    The engine interprets detection/non-detection evidence using effort and the
    declared detection probability q. It deliberately does not learn evidence
    semantics with a neural network.

    CURRENT DEFAULT for this first implementation:
    - false positives are ignored;
    - updates are site-local (no spatial propagation yet);
    - uncertainty is Bernoulli entropy in bits, in [0, 1].

    Priors are supplied explicitly by the caller. The engine does not invent an
    ecological prior from habitat or distance.
    """

    @classmethod
    def initialize(
        cls,
        prior_by_site: Mapping[str, float],
        *,
        confirmed_sites: set[str] | None = None,
    ) -> BeliefState:
        if not prior_by_site:
            raise ValueError("prior_by_site cannot be empty.")

        confirmed = confirmed_sites or set()
        unknown_confirmed = confirmed.difference(prior_by_site)
        if unknown_confirmed:
            raise ValueError(
                f"confirmed_sites contains unknown sites: {sorted(unknown_confirmed)!r}"
            )

        posterior: dict[str, float] = {}
        for site_id, prior in prior_by_site.items():
            cls._validate_probability(prior, name=f"prior[{site_id}]")
            posterior[site_id] = 1.0 if site_id in confirmed else float(prior)

        return BeliefState(
            p_by_site=posterior,
            uncertainty_by_site={
                site_id: cls.bernoulli_entropy(p)
                for site_id, p in posterior.items()
            },
            observed_history=(),
        )

    @classmethod
    def update(
        cls,
        belief: BeliefState,
        observation_batch: ObservationBatch,
        q_by_site: Mapping[str, float],
    ) -> BeliefState:
        cls._validate_belief(belief)

        posterior = dict(belief.p_by_site)
        history = list(belief.observed_history)

        for observation in observation_batch.observations:
            if observation.site_id not in posterior:
                raise ValueError(
                    f"Observation references unknown belief site {observation.site_id!r}."
                )
            if observation.site_id not in q_by_site:
                raise ValueError(
                    f"Missing q for observed site {observation.site_id!r}."
                )

            q = float(q_by_site[observation.site_id])
            cls._validate_probability(q, name=f"q[{observation.site_id}]")
            posterior[observation.site_id] = cls.posterior_after_observation(
                prior=posterior[observation.site_id],
                q=q,
                observation=observation,
            )
            history.append(observation)

        return BeliefState(
            p_by_site=posterior,
            uncertainty_by_site={
                site_id: cls.bernoulli_entropy(p)
                for site_id, p in posterior.items()
            },
            observed_history=tuple(history),
        )

    @classmethod
    def posterior_after_observation(
        cls,
        *,
        prior: float,
        q: float,
        observation: Observation,
    ) -> float:
        cls._validate_probability(prior, name="prior")
        cls._validate_probability(q, name="q")
        if not isinstance(observation.effort, int) or observation.effort <= 0:
            raise ValueError("Observation effort must be a positive integer.")

        if observation.detection:
            if q == 0.0:
                raise ValueError(
                    "Positive detection has zero likelihood when q=0 under the MVP "
                    "no-false-positive observation model."
                )
            # With no false positives, P(occupied | positive detection) = 1.
            return 1.0

        miss_if_occupied = (1.0 - q) ** observation.effort
        evidence_probability = (1.0 - prior) + prior * miss_if_occupied

        if evidence_probability <= 0.0:
            raise ValueError(
                "Observed non-detection has zero probability under the supplied prior/q model."
            )

        return (prior * miss_if_occupied) / evidence_probability

    @staticmethod
    def bernoulli_entropy(p: float) -> float:
        BeliefEngine._validate_probability(p, name="p")
        if p in (0.0, 1.0):
            return 0.0
        return -(p * log2(p) + (1.0 - p) * log2(1.0 - p))

    @staticmethod
    def _validate_belief(belief: BeliefState) -> None:
        if set(belief.p_by_site) != set(belief.uncertainty_by_site):
            raise ValueError(
                "BeliefState p_by_site and uncertainty_by_site must have identical site ids."
            )
        for site_id, p in belief.p_by_site.items():
            BeliefEngine._validate_probability(p, name=f"belief[{site_id}]")

    @staticmethod
    def _validate_probability(value: float, *, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric in [0, 1].")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
