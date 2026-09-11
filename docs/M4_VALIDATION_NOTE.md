# M4 validation note

The no-RL adaptive kernel has been validated locally with 47 passing tests and an executable sanity trajectory.

The demonstrated claim is intentionally narrow: a mission is planned from observable state, executed against the hidden simulator, field evidence updates Bayesian belief, the updated belief is re-exported to GraphState, and the next mission can change. Reveal remains gated until completion.

This does not establish ecological realism or real-world effectiveness. The current hidden-world generator is still a toy family; later validation must use real-data-constrained multiple world models, held-out/OOD testing, and the strong Information Gain baseline.
