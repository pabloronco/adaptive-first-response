# Interfaces v0

Status: M0 working contract derived from Project Freeze 3.0.

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
- `node_features`
- `edge_features`
- `global_features`
- `feasibility_mask`

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

### MissionAction
```text
allocations = [
  {site_id, effort_units, team_id?}, ...
]
total_cost <= remaining_budget
diagnostics = optional planner scores / confidence / rationale hooks
```

Planner contract:

`plan(graph_state, remaining_budget, constraints) -> MissionAction`

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

## Bayesian observation semantics

For site `i`:

- `p_i`: current occupancy belief
- `q_i`: detection probability per effort unit if occupied
- `e_i`: effort

`P(no detection | occupied, e_i) = (1 - q_i)^e_i`

Therefore a zero after one effort unit must be weaker evidence than a zero after ten effort units at the same `q_i`.

The learned planner receives the belief state after this explicit evidence update; it does not learn the semantics of detection from raw observations.
