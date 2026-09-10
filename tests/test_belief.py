from dataclasses import asdict

import pytest

from adaptive_response import BeliefEngine, Observation, ObservationBatch


def batch(*observations: Observation, round: int = 1) -> ObservationBatch:
    return ObservationBatch(
        observations=tuple(observations),
        round=round,
        total_effort=sum(obs.effort for obs in observations),
    )


def test_initialize_keeps_explicit_priors_and_marks_confirmed_detection_certain() -> None:
    belief = BeliefEngine.initialize(
        {"site_a": 0.2, "site_b": 0.7},
        confirmed_sites={"site_b"},
    )

    assert belief.p_by_site == {"site_a": 0.2, "site_b": 1.0}
    assert belief.uncertainty_by_site["site_b"] == 0.0
    assert belief.observed_history == ()


def test_zero_after_ten_checks_is_stronger_evidence_than_zero_after_one() -> None:
    prior = BeliefEngine.initialize({"site_a": 0.5})
    q = {"site_a": 0.25}

    after_one = BeliefEngine.update(
        prior,
        batch(Observation("site_a", effort=1, detection=False, round=1)),
        q,
    )
    after_ten = BeliefEngine.update(
        prior,
        batch(Observation("site_a", effort=10, detection=False, round=1)),
        q,
    )

    assert after_ten.p_by_site["site_a"] < after_one.p_by_site["site_a"] < 0.5


def test_non_detection_matches_closed_form_bayes_update() -> None:
    prior = BeliefEngine.initialize({"site_a": 0.4})
    q = {"site_a": 0.3}
    obs = Observation("site_a", effort=3, detection=False, round=1)

    updated = BeliefEngine.update(prior, batch(obs), q)

    miss = (1.0 - 0.3) ** 3
    expected = (0.4 * miss) / ((1.0 - 0.4) + 0.4 * miss)
    assert updated.p_by_site["site_a"] == pytest.approx(expected)


def test_positive_detection_sets_local_belief_to_one_without_false_positives() -> None:
    prior = BeliefEngine.initialize({"site_a": 0.05, "site_b": 0.8})
    obs = Observation("site_a", effort=2, detection=True, round=1)

    updated = BeliefEngine.update(prior, batch(obs), {"site_a": 0.2, "site_b": 0.4})

    assert updated.p_by_site["site_a"] == 1.0
    assert updated.uncertainty_by_site["site_a"] == 0.0
    assert updated.p_by_site["site_b"] == 0.8


def test_q_zero_non_detection_contains_no_information() -> None:
    prior = BeliefEngine.initialize({"site_a": 0.35})
    obs = Observation("site_a", effort=10, detection=False, round=1)

    updated = BeliefEngine.update(prior, batch(obs), {"site_a": 0.0})

    assert updated.p_by_site["site_a"] == pytest.approx(0.35)


def test_history_accumulates_only_observations() -> None:
    belief = BeliefEngine.initialize({"site_a": 0.5})
    first = Observation("site_a", effort=1, detection=False, round=1)
    second = Observation("site_a", effort=2, detection=False, round=2)

    belief = BeliefEngine.update(belief, batch(first, round=1), {"site_a": 0.2})
    belief = BeliefEngine.update(belief, batch(second, round=2), {"site_a": 0.2})

    assert belief.observed_history == (first, second)
    payload = asdict(belief)
    assert "hidden_world" not in payload
    assert "occupied_by_site" not in payload


def test_bernoulli_entropy_is_zero_at_certainty_and_maximal_at_half() -> None:
    assert BeliefEngine.bernoulli_entropy(0.0) == 0.0
    assert BeliefEngine.bernoulli_entropy(1.0) == 0.0
    assert BeliefEngine.bernoulli_entropy(0.5) == pytest.approx(1.0)
    assert BeliefEngine.bernoulli_entropy(0.2) < 1.0


def test_impossible_positive_when_q_zero_is_rejected() -> None:
    belief = BeliefEngine.initialize({"site_a": 0.5})
    obs = Observation("site_a", effort=1, detection=True, round=1)

    with pytest.raises(ValueError, match="zero likelihood"):
        BeliefEngine.update(belief, batch(obs), {"site_a": 0.0})
