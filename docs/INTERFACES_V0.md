# Interfaces v0

Status: M0 working contract derived from Project Freeze 3.0 and approved team review on 2026-09-10.

The goal of this document is to keep environment, belief engine, planners, evaluation, and UI compatible. It does not redefine the ecological model.

## Pipeline

`IncidentConfig -> Environment -> Observation/Belief -> GraphState -> Planner -> MissionAction -> Simulator -> Observation -> repeat`

Hidden truth is available only to the simulator/evaluator and must never be included in planner inputs.

## Core entities

### Site
Minimum fields:
- `id: str`
- `x: float`
- `y: float`
- `habitat_score: float`
- `q_model: float | dict`
- `observed_effort: int`
- `detections: int`
- `status: str`

Optional/current-default fields:
- `access_cost: float | None`

### Edge
Minimum fields:
- `src: str`
- `dst: str`
- `distance: float`

Optional/current-default fields:
- `connectivity_weight: float | None`
- `travel_cost: float | None`

### IncidentConfig
- `sites`
- `edges`
- `initial_detection`
- `budget`
- `teams`
- `protocol`
- `seed`
- `world_model_id`

### HiddenWorld
- latent occupancy state per site
- generator parameters

**Invariant:** this object is simulator/evaluator-only.

### Observation
- `site_id: str`
- `effort: int`
- `detection: bool`
- `round: int`
- `metadata: dict`

Binary detection is sufficient for MVP. Count/CPUE remains optional/stretch.

### BeliefState
- `p_i`: occupancy belief per site
- `uncertainty_i`: uncertainty per site
- `observed_history`

Optional:
- posterior/model weights if later implemented

### GraphState
Required structural fields:
- `node_ids: list[str]`
- `node_features`
- `edge_index`
- `edge_features`
- `global_features`
- `feasibility_mask`

Indexing invariants:
- `node_ids[i]` identifies the real `site_id` represented by `node_features[i]`.
- `feasibility_mask[i]` refers to the same node/site as `node_ids[i]`.
- `edge_index` stores graph connectivity using node indices into `node_ids`.
- `edge_index[:, j]` (or equivalent source/destination pair representation) identifies the endpoints of `edge_features[j]`.
- No planner may infer site identity from row position without using `node_ids`.

Frozen node features to support:
- occupancy belief
- uncertainty
- observed effort
- detections / last outcome

Current-default node features:
- habitat score
- access/travel cost
- frontier indicator / distance from positive

Frozen edge/global features to support:
- geographic/coastal distance
- remaining budget
- round / remaining horizon

Current-default edge/global features:
- connectivity/pathway weight
- team capacity
- global uncertainty

### Planner context / constraints
Planner signature remains:

`plan(graph_state, remaining_budget, constraints) -> MissionAction`

For v0, `constraints` is the shared observable planner context and may contain operational constraints plus explicit observation-model parameters needed by non-learned planners.

Required v0 field:
- `q_by_site: dict[str, float] | equivalent indexed structure`

Semantic note:
- `q_by_site` is an observation-model parameter, not hidden ecological truth.
- It is exposed so an Information Gain planner can evaluate future detection/non-detection likelihoods.
- The learned planner is not required to use `q_i` as a node feature in v0.
- If later experiments justify giving `q_i` directly to the learned policy, that is a separate cross-team interface decision and must be benchmarked rather than assumed.

Other candidate constraint fields, only when needed:
- maximum effort per site
- unavailable/closed sites
- team-specific feasibility
- round-level effort limits

### MissionAction
```text
allocations = [
  {site_id, effort_units, team_id?}, ...
]
total_cost <= remaining_budget
diagnostics = optional planner scores / confidence / rationale hooks
```

The same contract applies to:
- `FrontierPlanner`
- `InformationGainPlanner`
- `RLGraphPlanner`

### EpisodeMetrics
Candidate evaluation fields:
- occupied sites missed
- frontier coverage
- extent reconstruction score
- final uncertainty
- effort / wasted effort
- detections found
- travel/access cost

## Hard invariants

1. `HiddenWorld` is never present in `GraphState`, planner inputs, UI planner payloads, or policy diagnostics.
2. `MissionAction.total_cost <= remaining_budget`.
3. Consumed effort cannot be reused.
4. The frontend does not duplicate Bayesian inference or planning logic.
5. All planners consume the same observable state and return the same action schema.
6. Same seed + same action trajectory must be reproducible in deterministic demo mode.
7. Reveal is evaluator/demo-only and cannot be called through the planner path.
8. `node_ids`, `node_features`, and `feasibility_mask` must remain index-aligned.
9. `edge_index` and `edge_features` must remain edge-aligned.

## Bayesian observation semantics

For site `i`:

- `p_i`: current occupancy belief
- `q_i`: detection probability per effort unit if occupied
- `e_i`: effort

`P(no detection | occupied, e_i) = (1 - q_i)^e_i`

Therefore a zero after one effort unit must be weaker evidence than a zero after ten effort units at the same `q_i`.

The learned planner receives the belief state after this explicit evidence update; it does not learn the semantics of detection from raw observations.
