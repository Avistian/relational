# Year 4 thesis evidence replay

L158 Year 4 thesis evidence replay — PASS

98,918 prediction rows rescored; 45 selection checks; 220 frozen inputs.

| Evidence | Measured result | Permitted interpretation |
|---|---|---|
| L151 classification | 69.2588% AUROC | Execution evidence; fresh FE missing |
| L152 regression | 3.970840 MAE | Execution evidence; use L155 for matched FE |
| L153 recommendation | Test NOT_RUN | INCOMPLETE; validation pilot cannot fill test |
| L155 matched F1 | FE 3.948917; RDL 4.013141 MAE | Point estimate favors FE |
| L155 benefit FE − RDL | -0.064225; conditional 95% [-0.312329, +0.172961] | Neither superiority nor equivalence established |
| Human effort | NOT_OBSERVED | No local effort-saving ratio |
| L156 released → fixed horizon | 4.128265 → 4.200381 MAE | Policy verdict FAIL → NOT_ESTABLISHED; no leak-free sign-off |
| L157 released → fixed horizon | 4.015191 → 4.073480 MAE | Policy verdict FAIL → NOT_ESTABLISHED; no leak-free sign-off |

## Bounded interim verdict
The selected pipelines are executable and their saved predictions are auditable. The one matched FE comparison does not establish an RDL advantage, and local human-effort savings remain unmeasured. The broad undervaluation thesis remains a research hypothesis.

Coverage: 2 completed tasks on 2 databases; one matched FE task. Repeated F1 seeds and reports add no new database.

## Limits
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

Additional cloud spend: USD0. Full replay is not fresh training or whole-paper reproduction.
