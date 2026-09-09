## The checkpoint is a defensible comparison

Your goal is to compare XGBoost, CatBoost, an MLP, RealMLP and TabM under a protocol you can explain and rerun. The passing outcome is an accurate evidence report, including a failed neural gain if that is what you measure. This is the single-table baseline the later relational work must beat.

The core sources are [TabM](https://arxiv.org/abs/2410.24210), [RealMLP](https://arxiv.org/abs/2407.04491), [TabArena](https://arxiv.org/abs/2506.16791) and [TabReD](https://arxiv.org/abs/2406.19380). They motivate the arms and protocol questions. Our small comparison does not reproduce their original tuning budgets, data rosters or paper tables.

## Freeze the experiment before fitting

Eight public tables cover small numeric tasks and mixed-type business tables: diabetes, blood transfusion, kc1, phoneme, German credit, churn, bank marketing and adult income. Three additional TabReD releases—Ecom Offers, Homesite Insurance and Sberbank Housing—provide random and temporal evaluation. That is eleven underlying datasets, not fourteen independent ones: the two split regimes of each TabReD task are related observations.

Random public-table runs use at most 900 rows with a fixed 60/20/20 train/validation/test split and split seed 60. TabReD uses released split 0, label-blind caps of 540 training and 180 validation/test rows, and the earlier release loader. The measured task is binary log loss except Sberbank's released regression target, scored by root mean squared error. Preserve the release's target interpretation from lesson 55. Numeric and binary TabReD columns are included; categorical columns there are omitted and counted in provenance.

The two candidates per fitted arm vary depth or learning rate. Trees use 100 boosting rounds; neural candidates use 24 epochs and width 64; TabM-mini has eight members. Validation selects candidates and neural epochs. Model seeds 0, 1 and 2 are crossed with all arms on the same rows. Equal candidate counts do not imply equal compute or equal effective search effort: a neural candidate also selects an epoch. Record both rather than claiming perfect budget equality.

## Preserve each baseline's important recipe

Training-only encoders median-impute numeric inputs and one-hot encode categorical values. Unknown categories at inference map to the declared encoder's unknown representation. This common input treatment removes CatBoost's native categorical advantage; the report must call that a limitation. It is a controlled numeric/one-hot comparison, not the strongest possible mixed-type CatBoost benchmark.

MLP and TabM inputs are standardized from training rows. RealMLP uses training-derived robust scaling and smooth clipping, learned feature scales, neural tangent parameterization, its oscillating learning-rate schedule and parameter-group multipliers. Its classification objective includes label smoothing. The local width, epochs, numeric encoding, validation metric and candidate choice differ from the full paper defaults; retain the name **reduced RealMLP-TD-S recipe**. TabM averages member probabilities, with separate member losses during fitting. These mechanisms were implemented in lessons 53 and 54; the notebook exposes the reused source and the live selection/reporting code.

<!--figure:mechanism-->

## Test labels enter once the choice is frozen

The fitting function receives training features/targets and validation features/targets. It returns a predictor, its validation error and its selected epoch. Only after comparing those validation errors does the runner supply test features; only the metric routine receives test labels. This interface makes the boundary reviewable. A checksum alone cannot prove the boundary, but it can identify exactly which implementation was used.

Keep every selected prediction, target, candidate error, chosen index, split ID and measured duration. The lab checks that the chosen index minimizes validation error and reconstructs every reported test metric from saved predictions. A wrong row order can still produce a plausible score, which is why row identity and partition disjointness are separate checks.

## Read the result at three resolutions

First inspect each dataset's raw metric, three seed points, sample standard deviation and paired gaps against the tree baseline. These seed intervals are conditional on fixed rows and splits; three seeds provide a fragile estimate of training randomness. They do not measure uncertainty over future enterprises or temporal regimes.

Next average seed errors within each dataset, rank the arms there and average ranks. Keep random and temporal summaries separate. A Friedman test treats datasets as blocks; the Nemenyi critical difference supplies an exploratory pairwise rank comparison. With three temporal tasks, power is weak. A nonsignificant result neither proves equivalence nor rescues an unsupported superiority claim.

Finally inspect fitting/selection time and inference time. These CPU measurements include the declared local pipeline, not a GPU-serving benchmark. A method with a tiny score advantage and much larger latency may be a poor deployment baseline. The task distribution, tolerance for error and latency budget determine the practical choice.

<!--figure:results-->

<!--figure:comparison-->

The paired intervals condition on the fixed split and only three downstream seeds; they do not cover split uncertainty or pretraining variation. A negative loss gap favors the named model over XGBoost. Inspect each task in its own metric units.

<!--figure:ranks-->

The critical-distance bar belongs to the complete declared method pool. It is an exploratory multiple-comparison threshold over dataset-level ranks, not a confidence interval around a model score.

## Exit: write a report a skeptic can rerun

Submit the complete crossed result table, separate random/temporal rank summaries, paired dataset gaps, timing table and a reproduction ledger. Identify one dataset where the methods disagree and trace the disagreement to a hypothesis you can test—sample size, feature representation, irregular target behavior or temporal shift. The current run cannot establish that hypothesis causally.

State which arm you would deploy under a stated budget, what evidence could reverse that decision, and which published claim remains untested. The broader `closer` runner increases resource budgets using the same implementation. Its output remains a new local experiment; there is no “paper” switch that turns eleven reduced tasks into four original paper protocols.
