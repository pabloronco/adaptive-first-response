# Rehearsal Decision Log

This file mirrors project-relevant decisions made after Project Freeze 3.0 for the rehearsal repository. The canonical project Decision Log remains the governing source of truth.

## 2026-09-10 — M0 interface indexing review

**Status:** APPROVED by team.

**Decision:** Extend `GraphState` with explicit tensor/index mappings required for all planners and GNN message passing:

- add `node_ids`, index-aligned with `node_features` and `feasibility_mask`;
- add `edge_index`, edge-aligned with `edge_features`;
- make `q_by_site` available through the shared planner `constraints` / context so the Information Gain planner can evaluate the observation model without adding `q_i` to learned node features by default.

**Reason:** Demu's interface review identified two structural blockers: planner outputs could not map tensor rows back to real `site_id`s, and a GNN could not perform message passing from edge attributes without graph connectivity indices. The Information Gain baseline also requires explicit detectability for prospective likelihood calculations.

**Impact:** No change to the frozen evidence semantics, feature list, hidden-truth boundary, planner ladder, or product architecture. This makes the existing interface executable rather than redefining it.

**Owner:** Team.

## 2026-09-10 — M2 belief-engine implementation defaults

**Status:** CURRENT DEFAULT for rehearsal implementation; not an ecological fact and not a new frozen project decision.

**Decision:**

- Priors are supplied explicitly to `BeliefEngine`; the engine does not invent occupancy priors from habitat, distance, or the synthetic hidden world.
- A confirmed initial detection may be initialized at occupancy belief 1 under the MVP no-false-positive assumption.
- Site-level evidence updates use the explicit effort-aware binary observation model from the Technical Specification.
- Belief uncertainty is represented for v0 as Bernoulli entropy in bits, with range `[0, 1]`.
- The first belief engine is site-local; it does not propagate a positive detection to connected nodes unless a later explicit spatial/world model is introduced.

**Reason:** Project Freeze 3.0 freezes explicit effort/q Bayesian evidence semantics but does not freeze the numerical prior scheme or exact uncertainty scalar. Keeping priors caller-supplied prevents a design choice from being presented as ecological knowledge. Bernoulli entropy is simple, inspectable, and directly useful to the later Information Gain baseline.

**Impact:** M2 can be tested without coupling inference to the toy simulator. Spatial belief coupling, real-data-informed priors, and q range calibration remain later validation/modeling work.

**Owner:** Pablo + Fede, with cross-team review if these choices change `GraphState` semantics.

## 2026-09-10 — GraphState exporter encoding

**Status:** APPROVED CURRENT DEFAULT after Demu consumer validation.

**Decision:** Implement the already-approved `GraphState` contract with a framework-agnostic numeric encoding:

- node feature order: `belief, uncertainty, observed_effort, detections, habitat_score, access_cost, frontier`;
- edge feature order: `distance, connectivity_weight`;
- global feature order: `remaining_budget, round, team_capacity, global_uncertainty`;
- `edge_index` uses shape `[2, E]` and indexes directly into `node_ids`;
- because the current Environment treats the graph as undirected, each configured edge is exported in both directions for message passing;
- frontier is currently a binary observable indicator: a non-positive node directly adjacent to a publicly detected/confirmed-positive node;
- missing access cost and connectivity weight use configurable neutral defaults of `1.0`;
- global uncertainty is the mean node Bernoulli entropy;
- `q_by_site` remains outside learned node features and is exposed separately through planner constraints;
- GraphState remains Python/serialization friendly; the learned planner owns conversion to PyTorch/PyG tensors.

**Consumer validation:** Demu checked the real branch locally, read the producer contract and implementation, ran the full 30-test suite and GraphState sanity script, and confirmed that the payload can be consumed by a variable-size GNN/PyG adapter without structural changes. He also confirmed the node-id mapping, bidirectional `[2,E]` edge format, separate `q_by_site` planner context, and ML-side normalization ownership.

**Known limitation:** `feasibility_mask` is currently uniform per node while budget remains positive. Per-site closures, effort caps, and richer action-feasibility logic remain OPEN until action-space design and must be reviewed cross-team before training semantics are frozen.

**Reason:** The Technical Specification freezes/candidates the feature families but not their tensor ordering, missing-value encoding, frontier definition, or framework representation. These choices make the producer/consumer contract executable without coupling the environment package to Demu's ML stack.

**Impact:** M2.5 producer/consumer interface is now accepted. Demu may proceed with the GraphState-to-tensor adapter and GNN forward-pass work. Action-space, masking, reward, and evaluation semantics remain cross-team decisions.

**Owner:** Team; implementation by Pablo + Fede, consumer validation by Demu.

## 2026-09-10 — M3 FrontierPlanner baseline

**Status:** APPROVED CURRENT DEFAULT for the interpretable frontier baseline; not an ecological optimality claim and not the competitive benchmark target.

**Decision:** Implement `FrontierPlanner` under the frozen shared planner interface `plan(graph_state, remaining_budget, constraints) -> MissionAction` with deterministic ranking:

- feasible frontier nodes rank before feasible non-frontier nodes;
- within each group, higher occupancy belief ranks first;
- ties are broken by higher uncertainty, then deterministic `site_id` order;
- if budget remains after frontier selections, the planner may allocate to the best remaining feasible non-frontier nodes using the same ranking;
- per-site effort is configurable (`effort_per_site`) so M3 does not freeze the later RL action-space granularity;
- the planner does not use `q_by_site` in v0 and never receives hidden occupancy.

**Validation:** Local suite reported 39/39 tests passing. The M3 sanity script showed a controlled causal replan: before new evidence `site_04` was preferred (`p=0.7000` vs `site_02=0.6000`); after `0 detections / 5 checks` at `site_04`, Bayes reduced its belief to `0.3564` and the next mission switched to `site_02`. Hidden occupancy was not used by the planner.

**Known limitation:** This planner is intentionally simple and myopic. It is an interpretable fallback/integration probe, not the strong competitive baseline. Greedy Information Gain / entropy-VOI remains the main baseline for judging whether GNN+RL adds measurable value.

**Impact:** M3 satisfies the roadmap gate that a heuristic planner returns valid `MissionAction`s and demonstrates `FIELD EVIDENCE -> BELIEF CHANGED -> MISSION CHANGED` through the shared interfaces. The next system milestone is the full no-RL adaptive loop (M4).

**Owner:** Pablo + Fede; planner contract shared with team.