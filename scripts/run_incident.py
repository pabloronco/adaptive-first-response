"""Executable M1 sanity run.

This is not the final ecological simulator or the final demo. It is a small,
deterministic end-to-end check that a public incident can be reset, a finite
mission can spend effort, and the simulator can return imperfect field evidence
without exposing hidden occupancy.
"""

from adaptive_response import (
    Edge,
    Environment,
    IncidentConfig,
    MissionAction,
    MissionAllocation,
    Site,
)


def make_sanity_incident() -> IncidentConfig:
    num_sites = 20
    sites = [
        Site(
            id=f"site_{i:02d}",
            x=float(i),
            y=0.0,
            habitat_score=(i % 5) / 4,
            q_model=0.35,
        )
        for i in range(num_sites)
    ]
    edges = [
        Edge(
            src=f"site_{i:02d}",
            dst=f"site_{i + 1:02d}",
            distance=1.0,
            connectivity_weight=1.0,
        )
        for i in range(num_sites - 1)
    ]
    return IncidentConfig(
        sites=sites,
        edges=edges,
        initial_detection="site_09",
        budget=30,
        teams=2,
        protocol="binary_detection",
        seed=20260910,
        world_model_id="toy_graph_cluster_m1",
    )


def main() -> None:
    env = Environment(make_sanity_incident())
    state = env.reset()

    print("=== ADAPTIVE FIRST-RESPONSE / M1 SANITY RUN ===")
    print(f"Incident seed: {state.seed}")
    print(f"Confirmed initial detection: {state.initial_detection}")
    print(f"Field teams: {state.teams}")
    print(f"Initial budget: {state.remaining_budget} effort units")
    print()

    action = MissionAction(
        allocations=(
            MissionAllocation(site_id="site_08", effort_units=3, team_id="A"),
            MissionAllocation(site_id="site_10", effort_units=3, team_id="B"),
            MissionAllocation(site_id="site_11", effort_units=2, team_id="B"),
        ),
        total_cost=8,
        diagnostics={"planner": "manual_sanity_fixture"},
    )

    print("MISSION 1")
    for allocation in action.allocations:
        team = allocation.team_id or "unassigned"
        print(
            f"  Team {team}: {allocation.site_id} -> "
            f"{allocation.effort_units} checks"
        )
    print(f"  Mission cost: {action.total_cost}")
    print()

    batch, done, metrics = env.step(action)

    print("FIELD RETURN")
    for obs in batch.observations:
        detections = 1 if obs.detection else 0
        print(
            f"  {obs.site_id}: {detections} detection(s) / "
            f"{obs.effort} checks"
        )
    print()

    public_after = env.current_public_state
    print("PUBLIC STATE AFTER MISSION 1")
    print(f"  Round: {public_after.round}")
    print(f"  Effort spent: {metrics['effort_spent']}")
    print(f"  Remaining budget: {public_after.remaining_budget}")
    print(f"  Detections returned this round: {metrics['detections']}")
    print(f"  Episode done: {done}")
    print()
    print("Hidden occupancy is intentionally not printed or returned here.")


if __name__ == "__main__":
    main()
