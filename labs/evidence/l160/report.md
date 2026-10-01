# Year 4 exit evidence replay

L160 Year 4 exit evidence replay

Replay **PASS**; Year 4 exit **INCOMPLETE**; learner **PENDING_WRITTEN_DEFENSE**.

98,918 prediction rows rescored; 45 validation-selection checks; 226 frozen inputs.

| Task | Test experiment | Matched FE | Human effort | Temporal sign-off |
|---|---|---|---|---|
| rel-trial/study-outcome | COMPLETE | NOT_RUN | NOT_OBSERVED | NOT_ESTABLISHED |
| rel-f1/driver-position | COMPLETE | COMPLETE | NOT_OBSERVED | NOT_ESTABLISHED |
| rel-trial/site-sponsor-run | INCOMPLETE | NOT_RUN | NOT_OBSERVED | NOT_ESTABLISHED |

Coverage: 2/3 completed tasks; 1/3 matched FE tasks; 0/3 observed effort ratios; 0/3 temporal sign-offs.

F1 test MAE: FE 3.948917, RDL 4.013141; FE − RDL -0.064225 positions. Conditional driver-bootstrap 95% interval [-0.312329, +0.172961]. No superiority or equivalence established.

## Gate verdicts
- task_coverage: INCOMPLETE
- matched_fe: INCOMPLETE
- human_effort: INCOMPLETE
- temporal_audit: INCOMPLETE
- honest_failures: PASS

## Evidence boundaries
- replay: Saved-prediction CPU replay; no fresh training
- temporal: Policy verdicts and gradient/SQL audits inherited, hash-pinned; no new raw SQL or sampling audit
- uncertainty: L155 driver-cluster interval conditional on fitted models; not database-level or race/time uncertainty
- independence: Same task/database across F1 repeats; packet identity does not establish statistical independence
- graph_construction_157b: NOT_RUN; lesson absent
- human_study: NOT_RUN
- whole_paper: NOT_RUN
- historical_identity: NOT_ESTABLISHED
- public_contribution: PENDING_PUBLICATION
- learner: PENDING_WRITTEN_DEFENSE
- live_colab: NOT_CHECKED
- deployment: NOT_CHECKED

98,918 counts repeated seed/split predictions including validation pilot; not independent observations. Extra paired-loss checks reuse 12,590 rows.

USD0 additional cloud spending; no fresh fits. Original human study and whole-paper reproduction NOT_RUN.
