from adaptive_response import (
    AdaptiveMissionLoop,
    Edge,
    Environment,
    FrontierPlanner,
    IncidentConfig,
    Site,
)


def build_incident() -> IncidentConfig:
    sites = [
        Site(
            id=f"site_{i:02d}",
            x=float(i),
            y=0.0,
            habitat_score=0.5,
            q_model=0.25,
        )
        for i in range(7)
    ]
    edges = [
        Edge(
            src=f"site_{i:02d}",
            dst=f"site_{i + 1:02d}",
            distance=1.0,
            connectivity_weight=1.0,
        )
        for i in range(6)
    ]
    return IncidentConfig(
        sites=sites,
        edges=edges,
        initial_detection="site_03",
        budget=6,
        teams=2,
        protocol="binary_detection",
        seed=20260910,
        world_model_id="toy_graph_cluster_m1",
    )


def build_prior() -> dict[str, float]:
    prior = {f"site_{i:02d}": 0.20 for i in range(7)}
    prior["site_02"] = 0.60
    prior["site_04"] = 0.70
    return prior


def format_mission(mission) -> str:
    return ", ".join(
        f"{allocation.site_id}:{allocation.effort_units}"
        for allocation in mission.allocations
    )


def main() -> None:
    loop = AdaptiveMissionLoop(
        Environment(build_incident()),
        FrontierPlanner(effort_per_site=3, max_sites=1),
        build_prior(),
    )
    loop.reset()

    print("=== ADAPTIVE FIRST-RESPONSE / M4 END-TO-END SANITY ===")
    print("Confirmed initial detection: site_03")
    print("Initial field budget: 6 effort units")
    print("Hidden extent is locked.")
    print()

    mission_1 = loop.plan_next()
    print("MISSION 1")
    print(f"  {format_mission(mission_1)}")

    round_1 = loop.execute_pending()
    obs_1 = round_1.observations.observations[0]
    print("FIELD RETURN 1")
    print(
        f"  {obs_1.site_id}: {int(obs_1.detection)} detection(s) / "
        f"{obs_1.effort} checks"
    )
    print("BELIEF UPDATE")
    print(
        f"  p(site_04): {round_1.belief_before.p_by_site['site_04']:.4f} "
        f"-> {round_1.belief_after.p_by_site['site_04']:.4f}"
    )
    print("MISSION UPDATED")
    print(f"  {format_mission(round_1.next_mission)}")
    print(f"  Remaining budget: {round_1.public_state_after.remaining_budget}")
    print()

    round_2 = loop.run_round()
    obs_2 = round_2.observations.observations[0]
    print("FIELD RETURN 2")
    print(
        f"  {obs_2.site_id}: {int(obs_2.detection)} detection(s) / "
        f"{obs_2.effort} checks"
    )
    print(f"  Remaining budget: {round_2.public_state_after.remaining_budget}")
    print(f"  Episode complete: {round_2.done}")
    print()

    hidden = loop.reveal()
    occupied = [site_id for site_id, present in hidden.occupied_by_site.items() if present]
    print("REVEAL TRUE EXTENT — EVALUATOR/DEMO ONLY")
    print(f"  Occupied sites: {', '.join(occupied)}")
    print()
    print("OBSERVE -> INFER -> DECIDE -> SURVEY -> LEARN -> REPLAN")
    print("Planner never received hidden occupancy.")


if __name__ == "__main__":
    main()
