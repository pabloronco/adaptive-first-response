from adaptive_response import BeliefEngine, Observation, ObservationBatch


def one_observation(*, effort: int, detection: bool) -> ObservationBatch:
    observation = Observation(
        site_id="site_demo",
        effort=effort,
        detection=detection,
        round=1,
    )
    return ObservationBatch(
        observations=(observation,),
        round=1,
        total_effort=effort,
    )


def main() -> None:
    prior_probability = 0.50
    q = 0.25
    q_by_site = {"site_demo": q}

    prior = BeliefEngine.initialize({"site_demo": prior_probability})

    after_one = BeliefEngine.update(
        prior,
        one_observation(effort=1, detection=False),
        q_by_site,
    )
    after_ten = BeliefEngine.update(
        prior,
        one_observation(effort=10, detection=False),
        q_by_site,
    )
    after_positive = BeliefEngine.update(
        prior,
        one_observation(effort=1, detection=True),
        q_by_site,
    )

    p_one = after_one.p_by_site["site_demo"]
    p_ten = after_ten.p_by_site["site_demo"]
    p_positive = after_positive.p_by_site["site_demo"]

    print("=== BAYESIAN BELIEF / EFFORT SANITY RUN ===")
    print(f"Prior occupancy belief: {prior_probability:.3f}")
    print(f"Detectability q per check if occupied: {q:.3f}")
    print()
    print("FIELD EVIDENCE -> BELIEF")
    print(f"  0 detections / 1 check  -> posterior p = {p_one:.4f}")
    print(f"  0 detections / 10 checks -> posterior p = {p_ten:.4f}")
    print(f"  positive detection       -> posterior p = {p_positive:.4f}")
    print()
    print("Interpretation:")
    print("  A non-detection is not absence.")
    print("  More effort makes the same non-detection stronger evidence.")

    assert 0.0 < p_ten < p_one < prior_probability
    assert p_positive == 1.0


if __name__ == "__main__":
    main()
