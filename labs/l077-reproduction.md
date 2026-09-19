# L077 full reproduction contract

## Target and scope

This is the curriculum's original **single-table-ceiling synthesis experiment**, not a new-model paper reproduction. Target: exactly 0.5 test accuracy and exact ceiling from `[own,count,sum,mean,max]`, exactly 1.0 from those columns plus eligible last-minus-first, on every one of five full runs. No borrowed benchmark score is used as a target. This is a controlled historical-pattern classification task, not forecasting.

Primary grounding: [Fey et al., ICML2024 position paper](https://proceedings.mlr.press/v235/fey24a.html); [Deep Sets](https://arxiv.org/abs/1703.06114) for permutation-invariant functions; [RelBench v1](https://arxiv.org/html/2407.20060v1) for the separate empirical question. None supplies this synthetic experiment. Its generator, proof and protocol are authored locally.

## Full protocol

- Tier C generator: 1,000 pairs/seed; 2,000 customers and 10,000 events. Base integer uniform in 10–100 inclusive; step in 1–20; own feature in 0–9. A pair shares these values; the two labels correspond to descending/ascending eligible history. IDs and storage order randomized independently of their predictive use, excluded from model features.
- Three history events on days1,2,3, available immediately. Future event on12 and late event on2 arriving11 are excluded at cutoff10. Generated amount arithmetic is integer-valued, so exact vector equality is defined without a floating tolerance.
- Generator seeds0–4; split seeds10000–10004. Pair-group permutation and 60/20/20 split gives 1,200/400/400 rows. All members of a customer history remain with that entity. Pair IDs are split metadata only.
- Fixed features: own/count/sum/mean/max, plus delta in restoration arm. Aggregation is per customer with no learned preprocessing state and no cross-entity information. Empty aggregates and delta are zero. Tied eligible event times are rejected.
- Classifier: complete visible exhaustive decision-stump trainer. Enumerate sorted observed training-value midpoints plus constant-prediction boundaries, both orientations, every column. Maximize training accuracy with first-candidate tie rule. No optimizer, epochs, initialization/checkpoint or external model weights apply. Hyperparameters are fixed; validation is diagnostic, never used for selection.
- Model freezes before validation/test metrics and ceiling audit. Accuracy uses hard predictions. The audit's test labels never enter fitting. Five runs are repetitions under one generator, not independent real datasets. Report each run and sample SD; exact zero spread follows from paired symmetry.

## Re-run everything

From the repository root:

```bash
.venv/bin/python labs/_verify_l077.py
.venv/bin/python labs/_build_l077.py
.venv/bin/python labs/_execute_l077.py
.venv/bin/python labs/_browser_l077.py
.venv/bin/python labs/_delivery_l077.py
```

Minimal numeric environment is Python3.12 and `numpy==2.5.0`; independent library control uses `scikit-learn==1.9.0`. Course authoring dependencies (nbformat, nbclient, nbconvert, matplotlib, BeautifulSoup and Playwright) are already in the workspace environment. Exact observed versions and source hash are saved in the verification report. No external dataset/model download is needed. Dataset hashes for all five seeds accompany the results.

The notebook contains all functions, exposes three live TODOs, runs the same full five-seed experiment and writes `l077-student-results.json`. Its Colab bootstrap follows the existing course path. Browser execution of Colab is NOT_CHECKED; local notebook execution is checked separately. Since the complete numeric experiment runs on CPU in seconds, there is no deferred GPU scale-up or Modal requirement for this construction.

## Evidence ledger and independent checks

| Bucket | Status | Evidence |
|---|---|---|
| Complete local construction | MATCH when assertions pass | `_verify_l077_results.json`, five full run hashes/scores, exact bound |
| Live inline solution | MATCH when parity passes | `_execution_l077_results.json`; all functions match canonical source |
| Independent controls | PASS when verifier passes | SQL aggregation and event-order oracle; depth-three sklearn tree |
| Published benchmark | NOT_RUN in L077 | See separate historical release below |
| Larger construction | Not required | Full declared size already runs; extra pairs do not establish real-data benefit |
| Live Colab / deployment | NOT_CHECKED | Local rendering/execution do not establish those frontends |

Interventions: change excluded amounts, reverse physical storage/customer ordering, check disjoint pair groups, handle empty history, reject ambiguous time ordering. The empirical ceiling also has an imbalanced fixture and a unique-ID boundary check. Notebook-delivery validation injects broken live helpers and confirms that the resulting evidence/check fails.

## Separate real-data reproduction

The [L076 contract](l076-reproduction.md) contains the named **RelBench v1 Table6 rel-f1 driver-dnf** target, complete original model/trainer source, historical source pin, environment and protocol deviations, and runnable local/Colab/Modal operators. Its saved benchmark status is NOT_RUN. The [L076 notebook](html/0076-encoder-predictor-stack.html) contains the archived implementation inline. Use that aligned replay for a real-data sequel; neither this construction nor an enlarged version reproduces its AUROC or the expert feature-engineering study.

## Failure diagnosis

A deterministic flat score above 0.5 means the claimed interface, pair balance, metric or labels differ. Check accidental ID/position/target columns, lost pair members and time-order features. Restored performance below1 can indicate wrong sorting, eligibility, routing, threshold fitting or live TODO replacement. A ceiling of1 on unique continuous rows is an empirical lookup possibility, not a generalization certificate.
