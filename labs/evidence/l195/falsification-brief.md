# Lesson 195 — falsification brief

**Verdict: narrow the thesis; broad superiority and undervaluation remain unestablished.** This is an author reference defense, not the learner submission.

## Replayed evidence

All 33,650 stored predictions were rescored: L149 five runs per pipeline on 499 validation and 760 test queries, and L182 three arms × ten support draws × 702 test queries. Hash chains retain original receipts. Reusing these results is not an independent replication.

L149 test GNN MAE 4.123071; engineered-feature LightGBM MAE 3.948917. GNN advantage (FE minus GNN loss) -0.174155, conditional 95% driver-cluster interval [-0.449621, 0.085850]. This interval crosses zero. It establishes neither superiority nor equivalence. Both pipelines use relational information. The basic GNN is not the boosted GNN in the paper's Figure 3.

L182 RDB-PFN minus TabICL mean +0.004369 AUROC; positive on 6/10 support draws. Both consume released DFS features. These observations compare saved systems, not the presence versus absence of relational data. Same task, reused test population and uncertain historical feature availability limit generalization.

L194 retains all 21 tasks: 17 favorable, 3 unfavorable and 1 equal published reference signs against AutoGluon+DFS; every fresh result remains missing. The preprocessing stop is a reproducibility obstacle, not a measured performance defeat. Model reproduction INCOMPLETE_SOURCE_PREPROCESSING_GATE.

## Strongest objections, replies and revision conditions

| Claim | Strongest objection from this evidence | What survives | What would change the verdict |
|---|---|---|---|
| C1: flattening can lose useful signal | A collision in one feature map does not show every feasible feature map loses task-relevant signal. | Information loss can occur; predictive impact is task-dependent. | Predeclared same-backend comparison of target-only versus legal relational features on new tasks; label access and budget matched. |
| C2: learned structure recovers value | Engineered relational features have lower mean error in L149; RDBLearn offers a strong aggregation-plus-tabular alternative. | A learned prior can still help; L182 has a positive mean against TabICL on this one task. | Compare strong relational FE and learned systems with matched information access, validation search and declared compute; separate architecture and prior interventions. |
| C3: the advantage is fair and robust | L149 uncertainty crosses zero; L182 support draws are not databases; L194 has no fresh scores. | Scoped replay is reproducible. | New held-out databases and temporal shifts, point-in-time features, complete runs, predeclared practical margin and uncertainty unit. |
| C4: the field undervalues it | Predictive scores contain no adoption, total engineering cost or economic valuation measurement. | Undervaluation remains a research hypothesis. | Measure total human/compute/serving cost and downstream utility against a specified adoption or investment benchmark. |

## Guard against a moving target

Do not repair an unfavorable result by redefining the comparator, switching the unit of analysis, dropping a task, or moving a practical margin after seeing the score. The interactive margin sweep is explicitly retrospective sensitivity, not a preregistered test. A future confirmatory study must specify its target population, minimally useful benefit, measurement unit and stop rule before test access.

Observational rows may share entities, events and time. One-table serialization does not create independent observations. The driver bootstrap preserves within-driver dependence, but not shared race/time or new-database uncertainty. Legal event cutoffs alone do not recover unknown historical arrival times.

## Next decisive study — NOT_RUN

On a new untouched task set, freeze three information/representation conditions: target-only features; legal relational summaries; learned relational processing with the same available information. Tune only on validation with a declared search budget. Measure task-specific predictive benefit, uncertainty at the appropriate group/database level, and total engineering/serving cost. Predeclare the practical margin in domain units. A valid interval wholly below the required benefit challenges a practical-superiority claim; missing runs never count as a loss. This is a study design, not an executed intervention.

## Complete published reference inventory

| Task | Metric | RDBLearn | AutoGluon+DFS | Fresh score |
|---|---|---:|---:|---|
| relbench/rel-amazon/item-churn | AUROC | 0.8188 | 0.7953 | NOT_RUN |
| relbench/rel-amazon/user-churn | AUROC | 0.6823 | 0.6666 | NOT_RUN |
| relbench/rel-avito/user-clicks | AUROC | 0.6709 | 0.6191 | NOT_RUN |
| relbench/rel-avito/user-visits | AUROC | 0.6569 | 0.6064 | NOT_RUN |
| relbench/rel-hm/user-churn | AUROC | 0.6802 | 0.6802 | NOT_RUN |
| relbench/rel-stack/user-badge | AUROC | 0.8430 | 0.8470 | NOT_RUN |
| relbench/rel-stack/user-engagement | AUROC | 0.8935 | 0.8928 | NOT_RUN |
| relbench/rel-trial/study-outcome | AUROC | 0.7167 | 0.6740 | NOT_RUN |
| relbench/rel-amazon/item-ltv | MAE | 48.5044 | 57.0000 | NOT_RUN |
| relbench/rel-amazon/user-ltv | MAE | 14.5290 | 16.2047 | NOT_RUN |
| relbench/rel-avito/ad-ctr | MAE | 0.0341 | 0.0460 | NOT_RUN |
| relbench/rel-event/user-attendance | MAE | 0.2393 | 0.2590 | NOT_RUN |
| relbench/rel-hm/item-sales | MAE | 0.0630 | 0.0678 | NOT_RUN |
| relbench/rel-stack/post-votes | MAE | 0.0676 | 0.0709 | NOT_RUN |
| relbench/rel-trial/site-success | MAE | 0.4179 | 0.4396 | NOT_RUN |
| relbench/rel-trial/study-adverse | MAE | 44.5186 | 43.6232 | NOT_RUN |
| 4dbinfer/amazon/churn | AUROC | 0.7777 | 0.7291 | NOT_RUN |
| 4dbinfer/outbrain/ctr-100k | AUROC | 0.5499 | 0.5494 | NOT_RUN |
| 4dbinfer/retailrocket/cvr | AUROC | 0.8609 | 0.7343 | NOT_RUN |
| 4dbinfer/stackexchange/post-upvote | AUROC | 0.8736 | 0.8849 | NOT_RUN |
| 4dbinfer/stackexchange/user-churn | AUROC | 0.8774 | 0.8396 | NOT_RUN |

## Sources and limits

[RelBench v1](https://arxiv.org/html/2407.20060v1) · [RDB-PFN v5](https://arxiv.org/html/2603.03805v5) · [RDBLearn v1](https://arxiv.org/html/2602.18495v1). Source revisions, original training protocols and numerical deviations are preserved in packet/contracts/. Complete selected replay; no fresh training, raw database reconstruction, whole-paper reproduction, causal architecture attribution, economic valuation or learner mastery. $0 cloud/API; 1800 seconds aggregate local execution cap.
