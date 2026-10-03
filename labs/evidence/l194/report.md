# RDBLearn reproduction — analysis report

**Model reproduction: INCOMPLETE_SOURCE_PREPROCESSING_GATE**. Completed tasks: 0/21; model evaluations: 0/630.

## Question and protocol

Can the complete RDBLearn v1 Tables 1–3 column be reproduced? The frozen L193 plan uses all 21 tasks, three course seeds, depths 2/3/4 and TabPFNv2/v2.5/LimiX-16M: 567 validation candidates, 63 selected tests. Source commit b5b03ebf8091547285a6e06cba53d2d1a40cb171, release 0.1.2, FastDFS 0.2.1; official splits and support cap 10000. Course seeds are a disclosed extension.

## Results and uncertainty

All fresh scores and their seed SDs remain null. Positive reference gaps favor RDBLearn; AUROC uses model minus comparator, MAE uses comparator minus model. Comparator fixed before analysis: AutoGluon+DFS. These are rounded published values, not fresh or paired measurements. No statistical significance or causal claim follows. No raw-MAE or mixed-metric average is computed.

| Task | Metric | Published RDBLearn | Published AutoGluon+DFS | Oriented reference gap | Fresh result | Attribution |
|---|---|---:|---:|---:|---|---|
| relbench/rel-amazon/item-churn | AUROC | 0.8188 | 0.7953 | +0.0235 | NOT_RUN | NOT_ESTABLISHED |
| relbench/rel-amazon/user-churn | AUROC | 0.6823 | 0.6666 | +0.0157 | NOT_RUN | NOT_ESTABLISHED |
| relbench/rel-avito/user-clicks | AUROC | 0.6709 | 0.6191 | +0.0518 | NOT_RUN | NOT_ESTABLISHED |
| relbench/rel-avito/user-visits | AUROC | 0.6569 | 0.6064 | +0.0505 | NOT_RUN | NOT_ESTABLISHED |
| relbench/rel-hm/user-churn | AUROC | 0.6802 | 0.6802 | +0.0000 | NOT_RUN | NOT_ESTABLISHED |
| relbench/rel-stack/user-badge | AUROC | 0.8430 | 0.8470 | -0.0040 | NOT_RUN | NOT_ESTABLISHED |
| relbench/rel-stack/user-engagement | AUROC | 0.8935 | 0.8928 | +0.0007 | NOT_RUN | NOT_ESTABLISHED |
| relbench/rel-trial/study-outcome | AUROC | 0.7167 | 0.6740 | +0.0427 | NOT_RUN | NOT_ESTABLISHED |
| relbench/rel-amazon/item-ltv | MAE | 48.5044 | 57.0000 | +8.4956 | NOT_RUN | NOT_ESTABLISHED |
| relbench/rel-amazon/user-ltv | MAE | 14.5290 | 16.2047 | +1.6757 | NOT_RUN | NOT_ESTABLISHED |
| relbench/rel-avito/ad-ctr | MAE | 0.0341 | 0.0460 | +0.0119 | NOT_RUN | NOT_ESTABLISHED |
| relbench/rel-event/user-attendance | MAE | 0.2393 | 0.2590 | +0.0197 | NOT_RUN | NOT_ESTABLISHED |
| relbench/rel-hm/item-sales | MAE | 0.0630 | 0.0678 | +0.0048 | NOT_RUN | NOT_ESTABLISHED |
| relbench/rel-stack/post-votes | MAE | 0.0676 | 0.0709 | +0.0033 | NOT_RUN | NOT_ESTABLISHED |
| relbench/rel-trial/site-success | MAE | 0.4179 | 0.4396 | +0.0217 | NOT_RUN | NOT_ESTABLISHED |
| relbench/rel-trial/study-adverse | MAE | 44.5186 | 43.6232 | -0.8954 | NOT_RUN | NOT_ESTABLISHED |
| 4dbinfer/amazon/churn | AUROC | 0.7777 | 0.7291 | +0.0486 | NOT_RUN | NOT_ESTABLISHED |
| 4dbinfer/outbrain/ctr-100k | AUROC | 0.5499 | 0.5494 | +0.0005 | NOT_RUN | NOT_ESTABLISHED |
| 4dbinfer/retailrocket/cvr | AUROC | 0.8609 | 0.7343 | +0.1266 | NOT_RUN | NOT_ESTABLISHED |
| 4dbinfer/stackexchange/post-upvote | AUROC | 0.8736 | 0.8849 | -0.0113 | NOT_RUN | NOT_ESTABLISHED |
| 4dbinfer/stackexchange/user-churn | AUROC | 0.8774 | 0.8396 | +0.0378 | NOT_RUN | NOT_ESTABLISHED |

At printed precision: 17 favorable, 3 unfavorable, 1 equal. Counts give each task one vote; dependent tasks and heterogeneous methods prevent a causal or significance interpretation.

## Diagnostic and explanation boundary

Recorded synthetic source observations change known category codes in two of four interventions; numeric controls are unchanged. This justifies the declared source-admission stop. It does not establish occurrence or score impact on any real task. L194 replays these observations; it does not freshly execute the upstream preprocessor.

## Testable follow-up studies (all NOT_RUN)

### label_coverage

**Hypothesis:** More available labeled support improves prediction
**Vary:** Nested support sizes sampled only from labels available at each cutoff
**Hold fixed:** Same queries, features, checkpoint, depth, preprocessing, seed pairing and evaluation rule
**Measure:** Per-task paired AUROC or MAE change across complete seeds
**Limitation:** Context sensitivity is not pretraining scale or cross-database superiority

### schema_information

**Hypothesis:** A specified join path supplies useful predictive information
**Vary:** Remove that path under a declared feature intervention
**Hold fixed:** Same queries, support labels, checkpoint, preprocessing policy, seed pairing and remaining paths
**Measure:** Per-task paired score change with and without the path
**Limitation:** Ablation tests path information under this pipeline, not arbitrary schema robustness

### cold_start

**Hypothesis:** Performance differs for entities unseen in support history
**Vary:** Prespecified cold/warm strata using only past entity membership
**Hold fixed:** Same fitted predictor, task cutoff policy and scoring implementation
**Measure:** Stratum sizes, label counts and separate scores; AUROC undefined for single-class strata
**Limitation:** Observational subgroup comparison; degree, time and labels may confound the difference

## Limitations and resumption

Resolve source/preprocessing identity in a separately named repair protocol; audit raw data, labels, features and temporal availability; authenticate checkpoint bytes; establish regression denominators; implement and validate the backend executor; measure full-grid cost before dispatch. Preserve all tasks, seeds and candidates. No backend pretraining or comparator retraining is included. Whole-paper reproduction remains NOT_RUN and historical identity NOT_ESTABLISHED.

L194 approved budget: $0 cloud/API and 1800 aggregate local execution seconds. L193’s $10 ceiling was not a demonstrated full-run cost estimate. Learner defense remains PENDING_WRITTEN_DEFENSE.

Primary source: [RDBLearn v1](https://arxiv.org/html/2602.18495v1). Input hashes: [manifest](input-manifest.json). Structured result: [report](report.json).
