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

## 2026-09-10 — GNN/RL backbone v0 (Checkpoint A)

**Status:** CURRENT DEFAULT implementation detail, isolated to Demu's `adaptive_response.rl` subpackage. Does not touch or freeze action space, reward, or evaluation semantics, which remain OPEN per the M2.5 entry above.

**Decision:** Implement the GraphState -> tensors -> GNN -> per-node logits + critic value forward path as a new `src/adaptive_response/rl/` subpackage, deliberately isolated from the core package so importing `adaptive_response` never requires torch:

- `tensor_adapter.py`: `graph_state_to_tensors(GraphState) -> GraphTensors`. Only accepts `GraphState` (never `PublicState`/`HiddenWorld`), so there is no code path for hidden occupancy to reach the network. Applies a documented, adapter-local `log1p` transform to unbounded count-like features (`observed_effort`, `detections`, `remaining_budget`, `round`); bounded features (`belief`, `uncertainty`, `frontier`, `global_uncertainty`) are left as-is. This is scaling, not a `GraphState` contract change.
- `layers.py` / `backbone.py`: a small custom message-passing GNN in plain PyTorch (no PyTorch Geometric / torch-scatter, to avoid platform-specific wheel issues) — edge-conditioned messages, mean aggregation, residual + LayerNorm update, 2 layers by default. Mean-pooled node embeddings + an encoded `global_features` vector form a graph context vector. An `ActorHead` scores each node with a *shared* per-node MLP (no fixed-N layer anywhere, so the same weights run on any graph size) and applies `feasibility_mask` as `-inf` before returning logits. A `CriticHead` gives one scalar value per graph. Default `hidden_dim=64, num_layers=2` -> ~105k parameters.
- `q_by_site` is not wired into node features here, matching the M0 interface decision; using it would be a separate cross-team benchmarked decision.
- New optional dependency: `torch>=2.2,<3` under a new `rl` extra in `pyproject.toml` (`pip install -e ".[dev,rl]"`), not added to core `dependencies`.

**Validation:** New `tests/test_gnn_backbone.py` (skipped automatically via `pytest.importorskip("torch")` when torch is absent, so it cannot break the non-RL side's `pytest` run). Full local suite: 50/50 passing (39 prior + 11 new) on Python 3.10 CPU-only torch 2.14. Forward pass verified correct and shape-stable for N = 4, 12, 20, 24 with the *same* model instance back-to-back. `scripts/run_gnn_backbone_sanity.py` shows the untrained network's per-node logits shifting after the same `site_04: 0/10 checks` evidence used in the M2.5 sanity script, and reports a live value estimate — end-to-end wiring confirmed, not learning behavior (weights are untrained/random). Confirmed no `q` and no hidden-occupancy path at the tensor boundary (mirrors the M2.5 leakage tests).

**Known limitation:** `masked_action_distribution()` (a convenience helper, not a frozen action interface) raises rather than returning NaN when `feasibility_mask` is all-`False`; callers must already treat an exhausted-budget state as terminal before querying the policy, consistent with `Environment.step`'s existing `done` signal.

**Reason:** Checkpoint A brief: reach a small, robust, graph-size-agnostic forward path (`GraphState -> tensors -> GNN -> logits/value`) without pre-empting action-space/reward decisions reserved for Checkpoint B, and without adding a hard ML dependency to the shared core package.

**Impact:** Non-RL side is unaffected — no changes to `models.py`, `graph_state.py`, `belief.py`, `environment.py`, `planners.py`, or their tests. Demu can proceed toward Checkpoint B (action-space + reward, jointly with the team once `FrontierPlanner`'s full sequential loop exists) and Checkpoint C (training) using this backbone.

**Owner:** Demu; branch `demu/gnn-rl-backbone`.