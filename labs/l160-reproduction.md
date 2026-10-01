# L160 Year 4 exit evidence replay

Approved 2026-10-01. Named experiment: **L160 Year 4 exit evidence replay**. USD0 cloud spending; no new fitting, paid retries or training downloads. Bound each complete replay, verification and solution execution to 600 seconds, one numerical CPU thread. Stop on failure or timeout, preserve evidence, and investigate; do not shrink the declared scope.

## Exact replay scope

The immutable 226-file manifest includes the complete L158 220-file inclusion set plus its manifest/report and replay/check implementations. Verify all hashes before scoring. Never re-freeze changed artifacts to make a check pass. L151 reference seeds0–4 and selected seeds10–14 remain separate; L152 seeds0–4; L153 validation-only full-catalog pilot; L155 five complete FE searches and five GNN fits; L156/L157 five seeds in each released/fixed-horizon lane. No test-led lane selection. L154/L158 are views of this evidence; L159 is a conceptual callback. Notebook-validation fits are excluded from primary estimates.

Replay 98,918 saved validation/test prediction rows, including the recommendation pilot, and 45 validation-selection decisions. These are repeated populations, not independent observations. Preserve full (entity, cutoff) keys and original source identities. Existing source-SQL, feature-arrival, sampling and gradient audits remain inherited. Independent verification rescans all prediction packets, recomputes AUROC/MAE/MAP@10 and the 2,000-draw driver-cluster interval. The three new exam functions also check 12,590 already-counted FE/RDL prediction rows; this does not increase the main row count.

The notebook contains all inputs and readable replay functions. Three learner functions operate in the real adapter; student implementations remain blank. No model is introduced or fitted in this evaluation-only lesson.

## Commands

From the repository root, using the existing environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 600 .venv/bin/python labs/_replay_l160.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 600 .venv/bin/python labs/_verify_l160.py
.venv/bin/python labs/_build_l160.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 600 .venv/bin/python labs/_execute_l160.py
.venv/bin/python labs/_delivery_l160.py
```

`_freeze_l160.py` is a once-only source manifest initializer; subsequent invocation checks equality and refuses drift. The standalone notebook has embedded input hashes. Python3.10+ and NumPy suffice for the portable replay. The local independent oracle additionally uses scikit-learn; notebook author execution uses nbformat/nbclient/nbconvert. Record actual versions and code hashes in the verification and execution receipts. No dependency installation occurs inside the notebook.

## Published experiments and full-training paths

[RelBench v1 Tables6–8, Section6 and AppendixC](https://arxiv.org/html/2407.20060v1). Full model/trainer, archive/source hashes, splits, preprocessing, validation selection, epochs, seeds and deviations remain in the original per-lesson contracts and visible notebooks:

| Lane | Named published/released target | Executable contract | Current boundary |
|---|---|---|---|
| Classification | Table6 study-outcome RDL | [L151](l151-reproduction.md), [notebook](0151-classification-portfolio.ipynb) | Five reference fits completed; matched FE missing |
| Regression | Table7 driver-position RDL | [L152](l152-reproduction.md), [notebook](0152-regression-portfolio.ipynb) | Five complete fits; historical identity unestablished |
| Recommendation | Table8 site-sponsor-run GraphSAGE | [L153](l153-reproduction.md), [notebook](0153-recommendation-portfolio.ipynb) | Pilot only; full test NOT_RUN |
| Matched manual FE | Released F1 SQL/LightGBM versus basic RDL | [L155](l155-reproduction.md), [notebook](0155-compare-manual-fe.ipynb) | Five fits/searches per method; human effort NOT_OBSERVED |
| Temporal correction | Released versus fixed fitting horizon | [L156](l156-reproduction.md), [L157](l157-reproduction.md) | Correction is a separately declared extension; strict sign-off NOT_ESTABLISHED |

For recommendation the existing guarded full-run operator is `modal run --detach modal/l153_repro.py --phase full`. **Do not launch under this approval.** Its measured forecast was USD51.854174 including safety and notebook allowance, above its original USD10 cap; this is a recorded scenario estimate, not a refreshed quote or invoice. The operator retains its STOP guard. A new experiment needs an approved source/protocol/cost plan. The notebooks expose the original full-training lane for user-operated execution; L160 does not claim to have rerun it.

## Remediation specification

1. Declare the same ≥3 tasks before testing. Freeze source/archive identities, full query populations, all seeds/epochs/trials, metric direction and validation selection. A changed third task is a new protocol, not an invisible replacement of the failed run.
2. Resolve the temporal policy and required feature-arrival evidence. Produce reviewer-verifiable audit reports per task. Do not turn missing arrival histories into PASS by changing a report flag.
3. Pre-register effort scope, operator experience, work order, assistance and shared-infrastructure treatment. Use the blank L160 effort template for actual prospective logs. Repeating released SQL measures execution effort, not original feature discovery; keep those claims separate.
4. Supply missing matched FE baselines and the complete third-task experiment. Budget preparation, all runs, retries, checks and margin together. Stop if the full schedule exceeds the approved cap; do not shorten it while retaining the full-reproduction label.
5. Rescore held-out evidence, preserve adverse results, and submit the 700–1,000-word defense. Teacher review requires ≥8/10 with no zero. All evidence gates remain conjunctive.

## Evidence boundaries

A successful saved-evidence replay is computational audit evidence. It is not fresh training, historical identity, a rerun of the expert-user study, a whole-paper reproduction, a temporal sign-off, or learner mastery. Matching population and preprocessing assumptions need explicit review even when the comparison is executable. The conditional driver bootstrap omits race/time and training uncertainty. Distinct files or seeds do not establish independent databases. No publication was requested. Live Colab/deployment remain NOT_CHECKED until separately tested.
