## A strong closed-task score leaves several contracts untested

Your goal is to identify which assumption broke before trying to repair a foundation model. Read [Cheng et al., Sections 4–5](https://arxiv.org/html/2505.16226v1). Its open-environment evaluation separates emerging classes, changing features, changing data distributions and changing learning objectives. These are different failure axes, so one aggregate “robustness” number can conceal the actual problem.

A closed classification task usually assumes a fixed feature schema, target definition and class vocabulary. Deployment can violate any of them. The model may return probabilities of the expected shape while answering the wrong question. An honest baseline report therefore records the **prediction contract** as well as the score: what rows mean, which features exist, which labels are possible and how the prediction will be used.

## Unsupported classes require a visible policy

Suppose context contains only classes A and B, and the classifier emits `[.2,.8]`. A later target belongs to new class C. The original two-class probability vector cannot assign calibrated probability to C. Dropping that row from evaluation falsely improves apparent performance, while relabeling C as B changes the task.

The lab's diagnostic assigns unsupported labels zero probability, then clips at a declared epsilon only to obtain a finite log-loss value. With epsilon `1e−12`, a wholly unsupported target contributes about 27.63 nats. This is a diagnostic scoring convention, not a calibrated open-set detector. A deployment system might abstain, route the case or update its class vocabulary; each requires its own evaluation and coverage report.

<!--figure:mechanism-->

Keep **known-class performance**, **unsupported-class frequency**, **abstention coverage** and **performance on all required rows** separate. A high accuracy on the accepted subset is not enough if the model silently rejects most difficult cases. Thresholds for abstention must be chosen on validation evidence that represents the intended unknown-class process.

## A feature change has several meanings

Removing a column is a schema change. Replacing it by a training-derived imputation value is one explicit fallback; its usefulness depends on how the model was trained. Adding a new feature can change dimensional limits and representation behavior. Changing a sensor's units without changing the schema is a distributional change that may look superficially like ordinary numerical variation.

The measured intervention holds trained models and test rows fixed, then alters one column selected using training variation only. One condition replaces it with its training mean; another scales its deviation from that mean by three. The clean condition remains visible. This isolates sensitivity to two corruptions and gives each arm identical altered data. It does not reproduce natural temporal drift, missing-not-at-random mechanisms or every feature-addition scenario in the paper.

<!--figure:results-->

The compared arms are pinned Nature-v2 inference and a small XGBoost recipe on three real binary datasets, with three seeds. The test corruption is a prespecified intervention, not a selected attack optimized against the test labels. A model's smaller degradation here supports a narrower statement than “robust in open environments.” Inspect both absolute error and the paired change from clean performance; a poor clean model can have little room to degrade.

## A new objective is not necessarily a new model

Suppose the original task optimized log loss, but deployment now values a recall constraint, asymmetric cost or a calibrated probability. Those requirements can change the decision rule without changing the underlying probability model. For a binary action, predicting a calibrated probability and choosing a threshold are separate operations. If probabilities cease to be calibrated under shift, moving the threshold alone may fail.

Define the target label before discussing metrics. Predicting “will churn within 30 days” differs from “will churn eventually,” even if both produce a binary column. Feature availability and label delay also change the valid context. The paper's objective-change axis is a reminder to inspect the semantic task, not merely switch from accuracy to AUROC after viewing results.

## Build a failure matrix with controlled rows

Use one row per failure axis and columns for changed assumption, held-fixed quantities, intervention, measured output, fallback and untested claim. For an emerging class, the class-support contract changes. For a missing feature, the available input changes. For temporal concept shift, the conditional target rule changes. For a new utility function, the action criterion changes. A single corruption experiment cannot populate all four rows with “verified.”

Ask what observation would distinguish your leading explanation from an alternative. If a missing-feature fallback fails, compare retraining under the same missingness to a frozen model with imputation. If a context lacks a class, distinguish representation failure from impossible class support. If a shift changes calibration, report both discrimination and a proper scoring rule. These follow-up experiments must use fresh or appropriately nested evaluation evidence.

## Exit: report one failure without overclaiming

Submit the schema/class contract, all-row unsupported-class diagnostic and clean-versus-corrupted paired losses. Write a failure report that names the broken assumption, the evidence, a competing explanation and the next discriminating experiment. Include one case where the model did not fail, so the conclusion is not a curated collection of bad examples.

For the relational mission, extend the matrix to new entities, new tables, foreign-key changes and delayed labels. Relational context can add information, but it also adds ways to violate availability and schema assumptions. The later model must be tested against those additional contracts rather than credited with robustness because it has a graph.
