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

## 2026-09-10 — Action space + reward v0, first training run (Checkpoint B/C, PROPOSED — NOT yet reviewed by Pablo/Fede)

**Status:** PROPOSED. This is explicitly the kind of decision the handoff reserved for a joint Checkpoint B session ("action space + effort semantics + masking + reward"). It was implemented and a training run was started before that review happened, at Demu's discretion under explicit time pressure, once `m4/end-to-end-loop` showed the full sequential loop existed for real. Flag this entry to Pablo/Fede for review; nothing here should be treated as frozen.

**Decision:**

- **Action space**: per round, the policy autoregressively picks sites one at a time (`effort_per_pick=1`, matching `FrontierPlanner`'s existing default rather than inventing a new granularity), each pick adding one `MissionAllocation` and marking that site ineligible for further picks in the same round. A learned STOP token (extra logit from a small head on the graph context) lets the policy end the round early to preserve budget for future evidence; STOP is masked out on a round's first pick so a round can never legally produce an empty `MissionAction`. `node_embeddings`/`graph_context`/`node_logits` are computed once per round and reused across picks (the underlying `GraphState` does not change mid-round). Implementation: `src/adaptive_response/rl/round_policy.py` (`RoundPolicy`).
- **Credit assignment**: one round's full autoregressive pick sequence (including the final STOP) is treated as one factorized joint action, credited with one round-level reward via a single summed log-probability. No finer-grained (per-pick) credit assignment is implemented.
- **Reward** (`src/adaptive_response/rl/reward.py`), directly instantiating the reward candidate already named in the Technical Specification ("information gain / useful frontier coverage / detections - unnecessary effort - travel; terminal penalty per missed extent"):
  - per-round: `+2.0 * (mean uncertainty before − mean uncertainty after)` + `+1.0 * detections this round` − `0.02 * effort spent this round` (all from publicly observable `BeliefState`/`ObservationBatch`, no hidden truth);
  - terminal-only: `−1.0 * count(occupied sites with zero detections)`, which does need `HiddenWorld` — read via `Environment._hidden_world` (a private attribute) because `main` does not yet have M4's `Environment.reveal()` public API. This is training/evaluator-only code; it is never on the policy's observation path (`RoundPolicy.act` only ever receives `GraphState`). **TODO once `m4/end-to-end-loop` merges to `main`: switch `training_env.py` and `eval_utils.py` from the private attribute to `Environment.reveal()`.**
- **Algorithm**: on-policy actor-critic (REINFORCE + learned baseline, full-episode discounted returns, advantage normalization, entropy bonus), not clipped PPO — both are named as candidates in the Technical Specification, and the simpler formulation was chosen because a single gradient step per freshly-collected rollout batch has no stale-rollout importance-sampling problem for PPO's clip to solve. `src/adaptive_response/rl/trainer.py` (`ActorCriticTrainer`).
- **Domain randomization for training incidents** (`src/adaptive_response/rl/incident_sampler.py`): still only the M1 toy single-generator family (Pablo/Fede's multi-family simulator does not exist yet), but sites (12–24), budget (20–40), teams (1–3), habitat/q per site, and topology (random recursive-attachment tree + ~15% extra edges, not just a chain) are all randomized per episode.
- **Evaluation harness** (`src/adaptive_response/rl/eval_utils.py`): a planner-agnostic episode runner + `RLPlannerAdapter` so the trained policy can be benchmarked through the exact same code path as `FrontierPlanner`, on a fixed held-out incident/seed set (seeded independently of training draws) — this is the harness the "RL vs Information Gain on identical held-out incidents" gate from the handoff will eventually run through, once Information Gain exists.

**Validation:** New `tests/test_round_policy_training.py` (11 tests, same `pytest.importorskip("torch")` isolation). Full local suite: 56/56 passing. Confirmed: sampled incidents respect configured ranges and produce a connected, resettable `Environment`; `RoundPolicy` never double-picks a site within one round and always returns a mission `Environment.step()` accepts; a full episode's rollout lengths match `num_rounds` and always spends exactly the incident's budget; one `ActorCriticTrainer.update()` call produces finite losses and measurably changes parameters; `FrontierPlanner` and `RLPlannerAdapter` produce comparable `EpisodeMetrics` through the same harness on the same incident/seed. `scripts/train_gnn_policy.py` was smoke-tested end to end (tiny run: files/checkpoints/CSVs all written correctly) and timed at the intended default config (~1.8s/update at `episodes_per_update=32`, hidden_dim=64/num_layers=2) before starting a real run.

**Known limitations / explicitly not decided here:**
- Per-pick (rather than per-round) credit assignment was not attempted.
- `effort_per_pick=1` and the reward weights are defaults carried over from existing precedent (Frontier's default) or lifted directly from the Spec's named reward terms with round numbers filled in — none of them have had a sensitivity analysis, as the Spec itself requires before treating them as settled.
- Training happens only against the M1 toy generator; per the Data/Simulator/Validation Spec this must not be read as anything beyond "the loop and gradients work," and specifically not as evidence of real-world or even cross-simulator-family effectiveness.
- The `Environment._hidden_world` private-attribute access is a known, flagged, temporary workaround, not a new public contract.

**Reason:** Explicit user instruction to start a long (~7.5h) training run immediately, while stepping away and unable to review a design proposal first. Given `m4/end-to-end-loop` already demonstrates the real sequential loop the handoff wanted Checkpoint B to look at, filling in the open action-space/reward parameters now (rather than blocking) was judged lower-risk than leaving a "come back to us" checkpoint unresolved for hours with no training happening at all. This entry exists specifically so Pablo/Fede see exactly what was decided unilaterally and can override any of it.

**Impact:** A training run was launched in the background (see terminal transcript / `runs/<run-name>_<timestamp>/` for the specific run). Nothing here changes `GraphState`, `MissionAction`, `Planner`, or any file outside `src/adaptive_response/rl/` and `scripts/`. If Pablo/Fede reject any part of this design at review, the affected run(s) should be treated as disposable — checkpoints are cheap to regenerate once the design is corrected, per the project's own "don't count reward-curve-only progress" discipline.

**Owner:** Demu (unilateral, PROPOSED); requires Pablo + Fede review before being treated as CURRENT DEFAULT or FROZEN.

## 2026-09-11 — First empirical read: reward reweighting, three parallel runs

**Status:** Still PROPOSED/unreviewed; this is a data point for that review, not a new decision.

**What was compared**, all on the same fixed 20-incident held-out eval set (`eval_rng` seeded independently of `--seed`), same code otherwise:
- **v0** `checkpointB_v0_unattended` (seed=0): original weights (`missed_extent_weight=1.0`, `gamma=0.99`). Ran ~7.5h, 5800+ updates.
- **v1** `checkpointB_v1_reward_reweight` (seed=1): `missed_extent_weight=8.0`, `gamma=0.99` (isolates the reward-weight change).
- **v2** `checkpointB_v2_reward_reweight_lessdiscount` (seed=2): `missed_extent_weight=8.0`, `gamma=0.995`.

(Return normalization, see the entry above this one, was required first: raising `missed_extent_weight` alone without it pushed the loss into the hundreds in a smoke test.)

**Result at v0's end (~5800 updates) vs v1/v2 at ~2500-2600 updates each:**
- v0 never beat FrontierPlanner's held-out missed_fraction (0.2543, constant): best 0.390 at update 3600, and its last-20-point trend actually reversed (correlation +0.65 — got noisier/worse late, not better) despite entropy having collapsed to ~1.5-2.
- v1: missed_fraction 0.209 (first 10 evals avg) -> 0.183 (last 10) -> 0.155 (latest, also its best), consistently *below* Frontier's 0.254 essentially throughout, trend correlation -0.59 (still improving, not flat).
- v2: 0.251 -> 0.199 -> 0.184 latest (best 0.178), also consistently below Frontier once past its first few evals, trend correlation -0.53.

**Why this is more than "one lucky number"**: two independent seeds (v1, v2) sharing only the `missed_extent_weight` change both show the same qualitative shift - consistent, still-improving outperformance of Frontier on the fixed eval set - while v0 (same seed family, old weight) never got there in more total updates. If this were purely seed-init luck rather than the reward change, both would not be expected to move the same way. It does NOT yet separate "the weight change" from "the gamma change" (v1 vs v2 differ on both seed and gamma), and it does not yet say whether v1's slight edge over v2 is the smaller discounting or seed luck.

**What this does NOT show** (per the project's own claim discipline): validity beyond this one M1 toy-generator family and this one fixed 20-incident eval set; that RL beats Information Gain (doesn't exist yet); that this generalizes to different graph-size ranges, budgets, or a real simulator family. It is a same-generator, same-eval-set result on early-in-training checkpoints (v1/v2 are ~20% through their planned 7.5h budget), not a benchmark-grade claim.

**Decision:** Let v1 and v2 keep running to use their full compute budget rather than stopping early on a promising-but-partial trend; v0 is allowed to finish and then retired (it has clearly plateaued/regressed, not worth further wall-clock). No further reward/hyperparameter changes made mid-run — comparing more variants now would re-introduce the "changed too many things at once" problem this entry is trying to avoid.

**Owner:** Demu (unilateral, PROPOSED); flagging to Pablo + Fede that the Checkpoint B reward design looks directionally validated on this narrow test, pending their review and, eventually, benchmarking against Information Gain rather than only Frontier.

## 2026-09-11 — Follow-up read at ~5700-6000 updates: plateau, not continued improvement, and v2 instability

**Status:** Still PROPOSED/unreviewed; refines the entry above rather than replacing it.

**What changed since the last read:** v1/v2 ran ~3 more hours (v1: update 5600, v2: update 6000; both roughly half their 7.5h budget).

**v1**: stabilized into a consistent 0.156-0.199 missed_fraction band, every single one of the last 15 eval points below Frontier's 0.254. Trend correlation over the last 20 points is ~0.05 (flat) - it has stopped improving, but it has not degraded either.

**v2**: noticeably less stable. In the update-4900-5600 window it repeatedly matched or exceeded Frontier (0.299, 0.254, 0.294, 0.256), before recovering to 0.18-0.21 in its last few evals. Best single point (0.156 at update 4700) is comparable to v1's, but the path there is far noisier.

**Root cause, from `metrics.csv`**: entropy for both v1 and v2 plateaued around 8-13 nats by roughly update 1500-2000 and never continued collapsing toward a confident policy the way v0's did (v0 reached ~1.5-2 by its end). Working hypothesis: multiplying `missed_extent_weight` by 8x also multiplies the *variance* of the terminal reward term - the underlying detection outcome is still a stochastic Bernoulli draw (`q` in the 0.15-0.45 range), so a bigger weight on a noisy binary-ish outcome makes the policy-gradient signal noisier, which plausibly stalls the natural entropy reduction rather than only reweighting it. Not confirmed, just the most consistent explanation for what's observed.

**Interpretation:** the win over Frontier established in the previous entry holds up over 3 more hours (v1 especially - it did not regress into a v0-style late decline), but neither variant is still improving; both found a plateau early and are oscillating around it. v2's extra instability adds more (still seed-confounded) weight to "the gamma change is not obviously helping, and may hurt" from the previous entry.

**Decision:** No mid-run changes (same reasoning as before - don't add more moving pieces to an already-confounded comparison). Let both finish their 7.5h. Flag for the actual Checkpoint C training run (once Pablo/Fede review this and it becomes real, not disposable): reward-scale changes should probably be accompanied by variance-reduction measures (more episodes per update, and/or an entropy-coefficient or reward-scale schedule) rather than assuming more wall-clock time alone will finish the convergence a raw weight bump stalls.

**Owner:** Demu (unilateral, PROPOSED).