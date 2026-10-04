# B19a → B23 metric and calibration contract

Learner submission: PENDING. Complete before looking at final evaluation labels. This template has not run the proposed comparison.

1. **Decision and units:** What action needs a prediction? Name the target, units, horizon and accessible relational information. Explain the cost of central versus tail errors.
2. **Paired comparison:** Name a version-pinned TFM and a probabilistic/quantile-tree baseline. Give full dataset/entity/cutoff keys, pretraining exposure audit, legal support and common evaluation rows. Choose at least two independent datasets if making cross-dataset claims; one task supports only a task-specific conclusion.
3. **Split and fit:** Freeze train/validation/calibration/test IDs. Fit preprocessing on training only. Specify candidate counts, seeds, equal tuning budgets and validation-only primary-score selection. Freeze predictors before calibration. Never reuse calibration labels for adaptive selection without a valid separate protocol.
4. **Distribution interface:** Describe native bins/quantiles/samples, inverse target transforms, CDF/quantile interpolation, crossing repair, tail support/extrapolation, clipping and normalization. Freeze identical evaluation conventions for both models. State approximation accuracy and failure policy.
5. **Scores:** Primary metric and direction; report RMSE, CRPS, 90%/95% interval score, empirical coverage and mean width. Specify closed/open boundaries, generalized inverse convention and whether RMSE uses the predictive mean. Log/CRLS are distinct; do not infer a density from atoms without an explicit modeling change.
6. **Aggregation:** Average paired fold/seed effects inside each dataset; show raw unitful differences per dataset and a preregistered training-only scale normalization if comparing magnitudes. Report common-support roster and missing runs; do not cherry-pick score-specific winners or silently replace missing scores.
7. **Calibration claim:** Separate before/after calibration results on the same untouched test rows. State exchangeability assumptions, marginal versus conditional scope and likely group/time shift. Coverage alone does not measure informativeness.
8. **Cost and falsification:** Total all-in cap, runtime cutoff, failed-attempt accounting; what result would change your thesis? A point-only thesis must justify omitting distribution quality explicitly.

Exit defense: Explain why a higher-coverage model can be worse, why a proper score can have a different finite-sample winner, and why B19a's exact table reconstruction does not establish fresh training or historical checkpoint identity. Ask the teacher to review this contract. Retrieve again after1/7/30days.
