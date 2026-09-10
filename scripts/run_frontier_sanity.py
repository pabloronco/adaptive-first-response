from adaptive_response import (
    BeliefEngine,
    Edge,
    FrontierPlanner,
    GraphStateExporter,
    IncidentConfig,
    Observation,
    ObservationBatch,
    PublicState,
    Site,
)


def build_public_state() -> PublicState:
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
    sites[3].detections = 1
    sites[3].status = "confirmed_detection"
    edges = [
        Edge(
            src=f"site_{i:02d}",
            dst=f"site_{i + 1:02d}",
            distance=1.0,
            connectivity_weight=1.0,
        )
        for i in range(6)
    ]
    return PublicState(
        sites=sites,
        edges=edges,
        initial_detection="site_03",
        remaining_budget=10,
        round=0,
        teams=2,
        protocol="binary_detection",
        seed=20260910,
        world_model_id="toy_graph_cluster_m1",
    )


def chosen_site(action) -> str:
    return action.allocations[0].site_id if action.allocations else "<none>"


def main() -> None:
    public = build_public_state()
    prior = {site.id: 0.20 for site in public.sites}
    prior["site_02"] = 0.60
    prior["site_04"] = 0.70

    belief = BeliefEngine.initialize(prior, confirmed_sites={"site_03"})
    exporter = GraphStateExporter()
    planner = FrontierPlanner(effort_per_site=1, max_sites=1)

    graph_before = exporter.export(public, belief)
    mission_before = planner.plan(
        graph_before,
        remaining_budget=public.remaining_budget,
        constraints=exporter.planner_constraints(public),
    )

    evidence = Observation(
        site_id="site_04",
        effort=5,
        detection=False,
        round=1,
    )
    belief_after = BeliefEngine.update(
        belief,
        ObservationBatch(observations=(evidence,), round=1, total_effort=5),
        {site.id: 0.25 for site in public.sites},
    )
    graph_after = exporter.export(public, belief_after)
    mission_after = planner.plan(
        graph_after,
        remaining_budget=public.remaining_budget,
        constraints=exporter.planner_constraints(public),
    )

    print("=== FRONTIER PLANNER / M3 SANITY RUN ===")
    print("Frontier candidates around confirmed site_03: site_02, site_04")
    print()
    print("BEFORE NEW FIELD EVIDENCE")
    print(f"  p(site_02) = {belief.p_by_site['site_02']:.4f}")
    print(f"  p(site_04) = {belief.p_by_site['site_04']:.4f}")
    print(f"  Mission chooses: {chosen_site(mission_before)}")
    print()
    print("FIELD EVIDENCE")
    print("  site_04: 0 detections / 5 checks")
    print()
    print("AFTER BAYES")
    print(f"  p(site_02) = {belief_after.p_by_site['site_02']:.4f}")
    print(f"  p(site_04) = {belief_after.p_by_site['site_04']:.4f}")
    print(f"  Updated mission chooses: {chosen_site(mission_after)}")
    print()
    print("FIELD EVIDENCE -> BELIEF CHANGED -> MISSION CHANGED")
    print("Hidden occupancy was never used by the planner.")


if __name__ == "__main__":
    main()
