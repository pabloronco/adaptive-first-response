# Data-First Training & Validation Freeze

**Date:** 2026-09-11  
**Status:** TEAM-APPROVED METHODOLOGICAL FREEZE for the rehearsal project.  
**Scope:** ecological data, simulator/world models, Bayesian belief, learned-policy training, evaluation, and claim discipline.

## North star

The project remains an ecological first-response decision loop:

`FIELD EVIDENCE -> BELIEF CHANGED -> MISSION CHANGED`

A confirmed marine invasive-species detection starts the incident. The true extent is hidden, detection is imperfect, resources are limited, and each new observation changes what is plausible and therefore what the next field mission should be.

The GNN/RL is a decision planner inside this loop. It must not become a substitute for ecological modelling or Bayesian evidence interpretation.

## 1. Real data are the anchor

Use all relevant, accessible, methodologically compatible real datasets that materially improve the central loop. Real data should be used wherever the real world can actually provide the quantity of interest, including when available:

- monitoring-site coordinates and real coastal/estuarine geometry;
- visit dates and season;
- survey effort (for example traps, trap-days/hours, checks, soak time when available);
- detection / non-detection;
- counts and CPUE when comparable;
- sampling protocol / trap type;
- habitat information;
- access / operational information when defensible;
- environmental covariates when they are sourced, sufficiently covered, and ecologically justified.

Candidate environmental covariates to audit include temperature, salinity, substrate/sediment, tidal elevation or depth, wave exposure, eelgrass/saltmarsh/vegetation, estuary/harbor class, and other variables supported by the selected data sources. A variable is NOT included merely because it exists or sounds scientific.

A variable enters the final model only if:

1. it has an ecological/observation-process role that can be defended;
2. it can be obtained reproducibly for enough relevant site/visit records;
3. its provenance and missingness are documented;
4. it materially improves calibration, robustness, or simulator realism, or is necessary to represent the observation process.

## 2. Synthetic data have one legitimate role

Synthetic data are NOT used to invent convenient evidence or to make the learned policy win.

Synthetic hidden incidents are used because complete latent occupancy and counterfactual outcomes do not exist in historical field records. If a historical team did not survey site B, the result that would have occurred at B is unknowable.

Therefore:

- real data describe/constrain the ecological and observation process;
- fitted or constrained world models generate complete plausible hidden incidents;
- synthetic truth provides the full counterfactual ground truth needed for sequential policy training and planner comparison.

The final claim is never `trained on synthetic -> works in reality`.

## 3. Real-data split before final modelling

The real-data workflow must preserve data that were not used to fit/tune the ecological model.

Target structure (exact split depends on the datasets):

- **calibration set:** estimate or constrain priors, q/detectability, habitat/environment relationships, spatial dependence, effort/observation distributions;
- **validation set:** compare candidate ecological/world models and reject unrealistic simulator behaviour;
- **locked real test set:** final reality check that is not used to tune the chosen model.

Temporal, geographic, or site-based splitting should be chosen according to the actual data structure. Do not decide the split solely to maximize final scores.

## 4. Ecological model before learned policy

Evidence semantics remain explicit.

The current site-local Bayesian engine is a software-kernel milestone, not the final spatial ecological model. The final belief layer should allow an observation at one site to affect beliefs at other sites when the fitted ecological/world model supports that dependence.

Preferred direction for the rehearsal:

- maintain an ensemble/particle set of plausible hidden ecological worlds or an equivalent explicit spatial probabilistic model;
- update world/model weights with the observation likelihood using effort and detectability;
- derive site-level occupancy beliefs and uncertainty from the updated posterior;
- expose only that observable posterior-derived state to the planner.

The GNN must learn **what to do with the belief state**, not learn secretly what a detection/non-detection means.

## 5. Multiple world-model families are mandatory for serious RL results

Do not train the final policy against one procedural generator.

Before serious learned-policy benchmarking, Pablo + Fede must provide at least:

- >=3 plausible training world-model candidates;
- >=1 held-out world-model family not used during policy training;
- documented parameters and data provenance/constraints for each family;
- simulator-criticism summaries comparing simulated vs real observation statistics.

Candidate families include connected/graph spread, distance-decay spatial clusters, habitat-driven occupancy, cluster + satellite populations, and patchy/fragmented extent. Exact families may change after the real-data audit.

Use leave-one-world-model-out and stress tests where feasible.

## 6. Simulator criticism is a gate, not decoration

A simulator family is rejected if it cannot reproduce basic real-data statistics relevant to the selected monitoring system.

Candidate checks include:

- frequency of zero-detection visits;
- detections/counts as a function of effort;
- CPUE distribution where comparable;
- spatial clustering / recurrence patterns;
- seasonal patterns;
- habitat/environment associations;
- effort distributions;
- plausible detectability ranges.

Do not keep a simulator because it produces attractive policy behaviour.

## 7. Real-data evaluation lane

Whenever the selected dataset supports it, evaluate the probabilistic observation/belief model on held-out real survey visits.

Potential metrics include Brier score, predictive log loss / likelihood, and calibration diagnostics. Baselines should be simple and honest (for example global/site historical rates or simpler non-spatial models, depending on available data).

This lane asks: **does the probabilistic surveillance model predict real held-out observations sensibly?**

It does NOT prove that a counterfactual field mission would have been better.

## 8. Planner evaluation lane

Frontier, Information Gain, and GNN/RL must face the same held-out incidents, same budget, same information, same detectability/observation process, and same operational constraints.

The primary competitive baseline remains myopic Information Gain / entropy-VOI. Frontier remains an interpretable fallback/integration baseline.

Reward and evaluation metrics remain separate. Reward weights are design choices, never agency preferences.

The learned policy earns its place only through reproducible benchmark results. If it does not outperform or materially complement the strong baseline, use the best baseline in the demo.

## 9. Historical replay

Prepare at least one documented historical incident replay with strict separation between information available at t0 and later evidence.

Historical replay is a plausibility/reality check. It cannot establish `we would have beaten the operators`, because unobserved counterfactual field outcomes are unavailable.

## 10. Demu / RL training gate

**Checkpoint A (GraphState -> tensors -> GNN actor/critic backbone) is accepted as useful infrastructure.** It is independent of the ecological generator and may remain stable.

Action-space, masking, reward, and training choices implemented after Checkpoint A remain **PROPOSED** until cross-team review.

Any current training runs on the toy single-generator environment are classified as **SMOKE / ENGINEERING EXPERIMENTS ONLY**. They may be allowed to finish for debugging and algorithm-stability information, but:

- their scores are not project results;
- they do not validate ecology;
- they must not determine final reward/evaluation choices;
- they must not be used to claim RL superiority.

**Serious/final RL training is blocked until the team freezes:**

1. canonical real-data pack and data dictionary;
2. real/real-derived graph inputs;
3. q/effort/observation-process treatment with provenance or explicit uncertainty;
4. multiple real-data-constrained world-model families;
5. held-out/OOD split;
6. final action-space/masking contract;
7. reward + independent evaluation metrics.

Demu can continue engineering work that does not depend on these ecological assumptions, including backbone robustness, logging, checkpointing, deterministic inference, training-pipeline reliability, and smoke tests.

## 11. Claim discipline

Allowed interpretations:

- real-data-constrained simulation;
- predictive performance/calibration on held-out real observations, if demonstrated;
- robustness to held-out simulator assumptions / model misspecification;
- historical replay plausibility;
- planner improvement within the frozen benchmark, if demonstrated.

Not allowed without separate evidence:

- proven real-world effectiveness;
- optimal ecological response;
- beats field experts/agencies;
- AI predicts invasions;
- a synthetic benchmark presented as real-world ground truth.

## 12. What remains frozen from the project

- hidden truth never enters the planner observation path;
- non-detection is not absence;
- effort and detectability determine evidential strength;
- observations -> explicit Bayes/spatial probabilistic belief -> GraphState -> planner -> MissionAction;
- planner interface remains interchangeable;
- AI is not the project; the sequential ecological decision loop is the project.

## Immediate next work

Pablo + Fede: pause cosmetic UI work after the current shell and execute a real-data acquisition/audit, data dictionary, environmental-covariate audit, real graph construction, q/effort analysis, spatial belief/world-model design, simulator criticism, and validation splits.

Demu: preserve Checkpoint A, classify ongoing B/C/D/E/F toy-generator runs as smoke experiments, report them as engineering diagnostics only, and wait for the cross-team ecological/data freeze before final training or benchmark claims.
