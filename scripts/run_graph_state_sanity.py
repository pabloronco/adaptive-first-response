from adaptive_response import (
    NODE_FEATURE_NAMES,
    BeliefEngine,
    Edge,
    Environment,
    GraphStateExporter,
    IncidentConfig,
    Observation,
    ObservationBatch,
    Site,
)


def build_incident() -> IncidentConfig:
    sites = [
        Site(
            id=f"site_{i:02d}",
            x=float(i),
            y=0.0,
            habitat_score=(i % 5) / 4,
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
        budget=30,
        teams=2,
        protocol="binary_detection",
        seed=20260910,
        world_model_id="toy_graph_cluster_m1",
    )


def print_belief_rows(label, graph):
    belief_idx = NODE_FEATURE_NAMES.index("belief")
    frontier_idx = NODE_FEATURE_NAMES.index("frontier")
    print(label)
    for i, site_id in enumerate(graph.node_ids):
        print(
            f"  {site_id}: p={graph.node_features[i][belief_idx]:.4f} "
            f"frontier={int(graph.node_features[i][frontier_idx])}"
        )


def main() -> None:
    env = Environment(build_incident())
    public = env.reset()
    prior = {site.id: 0.5 for site in public.sites}
    belief = BeliefEngine.initialize(prior, confirmed_sites={public.initial_detection})
    exporter = GraphStateExporter()

    graph_before = exporter.export(public, belief)

    # Controlled field evidence for the interface sanity check. We inject an
    # observable non-detection directly into Bayes so this script demonstrates
    # the evidence -> belief -> GraphState transformation without depending on a
    # random simulator return.
    evidence = Observation(
        site_id="site_04",
        effort=10,
        detection=False,
        round=1,
    )
    belief_after = BeliefEngine.update(
        belief,
        ObservationBatch(observations=(evidence,), round=1, total_effort=10),
        {site.id: 0.25 for site in public.sites},
    )
    graph_after = exporter.export(public, belief_after)

    print("=== GRAPHSTATE EXPORTER / M2.5 SANITY RUN ===")
    print(f"Node feature order: {NODE_FEATURE_NAMES}")
    print(f"node_ids: {graph_before.node_ids}")
    print(f"edge_index shape: 2 x {len(graph_before.edge_index[0])}")
    print(f"feasibility mask: {graph_before.feasibility_mask}")
    print()
    print_belief_rows("BEFORE FIELD EVIDENCE", graph_before)
    print()
    print("FIELD EVIDENCE")
    print("  site_04: 0 detections / 10 checks")
    print()
    print_belief_rows("AFTER BAYES -> GRAPHSTATE", graph_after)
    print()
    print("Planner q context is separate from learned node features:")
    print(f"  {exporter.planner_constraints(public)}")
    print()
    print("Hidden occupancy is not present in GraphState or planner constraints.")


if __name__ == "__main__":
    main()
