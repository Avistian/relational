# B19b → L201 forecasting task contract (learner worksheet)

- Decision and forecast issue timestamp:
- Target, series/entity key, unit, horizon and timestamp frequency:
- Label event time versus label availability time:
- Each covariate: event time, arrival/revision time, known/forecast/observed status:
- Missingness and revision policy (as-of snapshot required):
- Frozen training, validation origins and untouched final origins:
- Seasonal period and train-only preprocessing/support/period estimation:
- Seasonal-naive and stronger task-specific baseline; challenger/checkpoint identity:
- Primary metric (MASE or WQL), quantiles, denominators and zero policy:
- Same information access for every competitor; covariate forecasts fitted separately:
- Aggregation across series, horizons and datasets; paired uncertainty unit:
- Selection/seed budget, aggregate cost cap and cutoff:
- Falsification test: change all post-issue unavailable values; legal predictions must stay fixed:
- Inclusion/exclusion decision for L201 and evidence needed to revise it:

Keep forecasting, missing-value imputation and causal effects separate. A promotion forecast conditional on the planned promotion is not the effect of intervening on promotion. Ask the teacher to review this contract before running your untouched final evaluation.
