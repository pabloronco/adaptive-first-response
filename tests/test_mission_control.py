from adaptive_response.mission_control import MissionControlSession


def test_initial_snapshot_is_20_site_hidden_incident() -> None:
    session = MissionControlSession()

    snapshot = session.snapshot()

    assert snapshot["phase"] == "ready_to_plan"
    assert len(snapshot["nodes"]) == 20
    assert snapshot["resources"]["initial_budget"] == 30
    assert snapshot["resources"]["remaining_budget"] == 30
    assert snapshot["incident"]["truth_locked"] is True
    assert all("true_occupied" not in node for node in snapshot["nodes"])
    assert {node["zone"] for node in snapshot["nodes"]} == {
        "coast",
        "harbor",
        "offshore",
    }


def test_plan_uses_real_planner_and_returns_budget_valid_mission() -> None:
    session = MissionControlSession()

    snapshot = session.plan()

    mission = snapshot["mission"]
    assert snapshot["phase"] == "mission_planned"
    assert mission is not None
    assert mission["planner"] == "frontier"
    assert mission["total_cost"] == 6
    assert len(mission["allocations"]) == 2
    assert sum(row["effort_units"] for row in mission["allocations"]) == 6


def test_execute_surfaces_real_observations_and_belief_shift() -> None:
    session = MissionControlSession()
    planned = session.plan()
    target_ids = {row["site_id"] for row in planned["mission"]["allocations"]}

    snapshot = session.execute()

    assert snapshot["resources"]["remaining_budget"] == 24
    last_round = snapshot["last_round"]
    assert last_round is not None
    assert {row["site_id"] for row in last_round["observations"]} == target_ids
    assert all(row["effort"] == 3 for row in last_round["observations"])
    assert any(
        row["belief_before"] != row["belief_after"]
        for row in last_round["observations"]
    )
    assert snapshot["phase"] == "mission_planned"
    assert snapshot["mission"] is not None


def test_default_demo_first_evidence_physically_changes_next_mission() -> None:
    """Protect the demo's central causal moment from becoming visually static."""

    session = MissionControlSession()
    planned = session.plan()
    first_targets = tuple(
        row["site_id"] for row in planned["mission"]["allocations"]
    )

    updated = session.execute()
    next_targets = tuple(
        row["site_id"] for row in updated["mission"]["allocations"]
    )

    assert first_targets == ("site_08", "site_10")
    assert next_targets == ("site_12", "site_16")
    assert updated["mission_changed"] is True
    assert updated["replan"]["changed"] is True


def test_reveal_remains_unavailable_until_budget_is_exhausted() -> None:
    session = MissionControlSession()
    session.plan()

    first = session.execute()
    assert first["can_reveal"] is False

    while session.snapshot()["can_execute"]:
        session.execute()

    complete = session.snapshot()
    assert complete["phase"] == "complete"
    assert complete["resources"]["remaining_budget"] == 0
    assert complete["can_reveal"] is True
    assert all("true_occupied" not in node for node in complete["nodes"])

    revealed = session.reveal()
    assert revealed["phase"] == "revealed"
    assert revealed["incident"]["truth_locked"] is False
    assert all("true_occupied" in node for node in revealed["nodes"])


def test_reset_after_reveal_hides_truth_and_clears_history() -> None:
    session = MissionControlSession()
    session.plan()
    while session.snapshot()["can_execute"]:
        session.execute()
    session.reveal()

    reset = session.reset()

    assert reset["phase"] == "ready_to_plan"
    assert reset["resources"]["remaining_budget"] == 30
    assert reset["mission"] is None
    assert reset["last_round"] is None
    assert len(reset["events"]) == 1
    assert all("true_occupied" not in node for node in reset["nodes"])


def test_same_seed_produces_same_first_mission_and_field_return() -> None:
    first = MissionControlSession()
    second = MissionControlSession()

    first_mission = first.plan()["mission"]
    second_mission = second.plan()["mission"]
    first_return = first.execute()["last_round"]
    second_return = second.execute()["last_round"]

    assert first_mission == second_mission
    assert first_return == second_return
