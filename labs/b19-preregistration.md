# B19 untouched-data preregistration — learner template

Status: UNFILLED / NOT_RUN. An empty template is not a frozen experiment.

## Question and exposure

- Intended deployment population and decision:
- Dataset name, source, license, snapshot/hash and complete row key:
- Why this test has not already guided model/pipeline choices:
- Dataset relationship to previous development/pretraining data (known / excluded by which checks / unknown):
- Related datasets/tasks that prevent treating this as an independent draw:
- Provider involvement and independently available evidence:
- Task mix, categorical fraction, semantic types, imbalance, missingness:

## Information and splits

- Entity/group key, event time, arrival time and label-availability time:
- Exact outer training/validation/test IDs or deterministic split-generation code and hashes:
- Exact inner split/selection protocol matching deployment:
- Features, support labels and preprocessing available identically to both arms:
- Protected test storage/access rule:

## Frozen comparison

- Strong baseline and challenger, full code versions/checkpoint hashes:
- Preprocessing fit scope, categorical/text handling, missing values:
- Candidate settings, seed IDs, early stopping, ensemble/refit rule:
- Primary metric, orientation, scoring unit and paired support:
- Secondary metrics (clearly secondary), minimum meaningful effect:
- Missing/failing runs: preserve missingness and label any fallback:
- Dataset-level uncertainty plan (one dataset does not support across-dataset inference):
- All-in preparation/seed/retry/validation budget and hard cutoff:
- Falsification criterion and action if it holds:

## Freeze before test access

- Freeze timestamp and SHA256 of completed contract/input manifests:
- Who/what may access test labels, at which stage:
- Deviations ledger (record before/after, reason and whether outcomes were already seen):
- Final report must separate source parity, saved replay, fresh execution, historical identity and learner defense.

Do not fill missing identities with plausible guesses. Return this completed contract to the teacher for review before running the untouched test.
